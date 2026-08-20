"""Search-then-imitate training: NPV descent, GA search, then supervision.

One epoch =
    A  20 gradient steps on continuous NPV        (local, exact objective)
    B  GA search for a fixed wall-clock budget, warm-started from the network,
       then a dominance sweep on its result       (global, combinatorial)
    C  20 gradient steps on NPV + a pairwise ranking loss against the GA
       + the dominance cut                        (long-range, corrective)

WHY THREE SIGNALS, AND WHY THEY DIFFER IN REACH. The NPV gradient is exact but
LOCAL: continuous_time smooths the ordering with a compactly supported kernel,
so d sigma_j / d s_i is EXACTLY zero once |s_i - s_j| > h. A block that needs to
move a long way in score order receives no gradient at all -- which is precisely
the situation where training stalls. Both corrective terms are therefore written
on the RAW SCORES, not on sigma, because a pairwise term on scores has global
reach and is unaffected by the bandwidth.

  ranking    imitation of the GA's order. Pairwise rather than a regression on
             sigma, because sigma is invariant to affine maps of the score, so
             there is no well-posed target for s itself -- only an order.
             Weighted by |dNPV/dsigma| = delta v_j psi_j exp(-delta sigma_j),
             which prices a misplacement in currency. That weighting is the one
             non-redundant use of the sensitivity: as a loss term of its own it
             is exactly the NPV gradient (verified, cosine 1.0000000000).
  dominance  a PROVEN condition, needing no teacher. For two blocks adjacent in
             the sequence and unrelated by precedence, an exchange argument
             gives `i before j` optimal iff v_i/tau_i > v_j/tau_j. Any schedule
             holding such a pair the other way is improvable, so the violation
             is a certain error rather than an imitation of someone's guess.

THE CERTIFICATE IS COMPUTED ONCE. The dual bound is a property of the INSTANCE,
not of the schedule -- edge duals from different primals correlate ~0.95, and
primal-informed dual estimates are measurably worse than a cold start. So it is
solved once at startup and every epoch just reports (bound - NPV) / bound for
both the network and the GA. Re-certifying per epoch would be pure waste.

Run: python train_loop.py [epochs] [crop] [ga_seconds]
"""

import sys
import time

import numpy as np
import torch

import continuous_time as ct
import ga_schedule as G
import train as T
from anchor_interpolation import (clamp_torch, dag_levels,
                                  interpolate_precedence_torch)
from decoders import children_csr, schedule_priority_kahn
from dual_certificate import certify
from kernel_projection import sequence_violations
from lp_bound import reachability_bounds
from model import SimpleTransformer

EPOCHS = 10
CROP = 6
GA_SECONDS = 10.0
NPV_STEPS = 20
SUP_STEPS = 20
LR = 1e-3
CLIP = 1.0
DISCOUNT = 0.90
T_PERIODS = 10
SEED = 0

W_RANK = 0.3
W_DOM = 0.3
# The two corrective terms act on pairs at completely different score scales and
# need their own temperatures. A RANDOM pair differs by about the score std; two
# ADJACENT blocks in the decoded order differ by about std/n. Using one
# temperature for both pins the dominance loss at softplus(0) = 0.693 -- it
# cannot see the pairs it is supposed to fix.
TEMP_RANK = 0.25               # x std(s)
TEMP_DOM = 3.0                 # x std(s) / n
PAIRS = 4                      # sampled ranking pairs per block per step
CERT_ITERS = 3000
CERT_K = 4


def pairwise(s, a, b, temp, weight):
    """softplus penalty for every pair the teacher says should read a before b."""
    d = (s[a] - s[b]) / temp
    return (weight * torch.nn.functional.softplus(-d)).sum() / weight.sum().clamp(min=1e-9)


def main(epochs=EPOCHS, crop=CROP, ga_seconds=GA_SECONDS, proj="clamp"):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    P = G.load_instance(crop=crop, t_periods=T_PERIODS)
    st, n = P["static"], P["n"]
    tau, value, scale, adjc = P["tau"], P["value"], P["scale"], P["adj"]
    w = st["tonnage"]
    par, chi = np.asarray(P["par"]), np.asarray(P["chi"])
    keys = G.edge_keys(par, chi, n)
    csr = children_csr(n, par, chi)
    topo = np.empty(n, np.int64)
    topo[P["order"]] = np.arange(n)
    dens = torch.tensor(value / np.maximum(tau, 1e-300), device=dev)
    delta = ct.delta_from_discount(DISCOUNT)
    psi_np = ct.within_block_shape(tau, delta)

    def npv_of(seq):
        return float(ct.npv(ct.start_times_from_order(seq, tau), tau, value,
                            discount=DISCOUNT, scale=scale))

    # ---------------- the certificate, once ----------------
    print(f"instance crop-{crop}: n={n} blocks, {len(par)} edges, "
          f"horizon {tau.sum():.2f} periods, device {dev}, feasibility={proj}")
    t0 = time.perf_counter()
    e, l = reachability_bounds(n, par, chi, P["order"], w, w.sum() / T_PERIODS,
                               float(T_PERIODS))
    cert = certify(P["order"], tau, value, w, par, chi, iters=CERT_ITERS,
                   subperiods=CERT_K, earliest=e, latest=l)
    BOUND = cert["bound"] / scale
    print(f"certificate (solver-free, K={CERT_K} + reachability): "
          f"bound {BOUND:+.5f}  computed once in {time.perf_counter()-t0:.0f}s")
    print(f"  topological baseline {npv_of(P['order']):+.5f}   "
          f"gap {(BOUND - npv_of(P['order']))/BOUND*100:.2f}%\n")

    # ---------------- the network ----------------
    x = torch.tensor(T.block_features(st), dtype=torch.float32,
                     device=dev).unsqueeze(0)
    gram = T.build_gram(st) if proj == "kernel" else None
    _, grp = dag_levels(n, par, chi, P["order"])
    groups = [(torch.tensor(a, device=dev), torch.tensor(b, device=dev))
              for a, b in grp]
    pr = ct.prepare(w, w.sum() / T_PERIODS, value, discount=DISCOUNT,
                    device=dev, dtype=torch.float32)
    net = SimpleTransformer(in_dim=x.shape[-1], d_model=T.D_MODEL,
                            nhead=T.N_HEAD, num_layers=T.N_LAYERS,
                            dim_feedforward=T.FF, dropout=T.DROPOUT,
                            out_dim=1, use_posenc=False).to(dev)
    # eval mode: a single fixed instance, so dropout only injects noise that
    # makes consecutive forward passes disagree and the reported NPV jitter
    net.eval()
    opt = torch.optim.Adam(net.parameters(), lr=LR)

    def forward():
        """Returns (raw score, feasible score, NPV of the feasible one).

        The two are kept apart on purpose. NPV is computed on the PROJECTED
        field, so the objective always prices a precedence-feasible schedule.
        The corrective terms are applied to the RAW field, so they never pass
        through the projection's Jacobian.

        That split is the whole fix. The projection passes dense gradient
        (1242/1242 blocks) but it cannot pass a direction that would SEPARATE
        an active tight pair -- and separating near-tied adjacent blocks is
        exactly what the ranking and dominance terms ask for, so routing them
        through it had them pulling against the operator itself. Measured with
        both losses on the projected field, inversions rose as often as they
        fell over ten epochs.
        """
        s_raw = net(x, pool=None).squeeze(-1).squeeze(0)
        if proj == "kernel":
            s = interpolate_precedence_torch(s_raw, gram, par, chi)
        elif proj == "clamp":
            s = clamp_torch(s_raw, groups, n)
        else:
            s = s_raw
        sig = ct.start_times_soft(s.unsqueeze(0), pr["tau"], value=pr["value"],
                                  window=pr["window"])
        npv = ct.npv_soft(sig, pr["tau"], pr["value"], psi=pr["psi"],
                          scale=pr["scale"]).sum()
        return s_raw, s, npv

    def decode(s):
        return schedule_priority_kahn(
            s.detach().cpu().numpy().astype(np.float64), csr,
            tiebreak=topo.astype(float))

    best_ga, best_ga_npv = None, -np.inf
    hist = []
    print(f"{'ep':>3} {'phase':<10} {'NPV start':>10} {'NPV end':>10} "
          f"{'best':>10} {'gap':>8}   detail")
    print("-" * 96)

    for ep in range(1, epochs + 1):
        # ---- A: NPV descent ----
        a0 = None
        for i in range(NPV_STEPS):
            opt.zero_grad()
            _, s, npv = forward()
            (-npv).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP)
            opt.step()
            if a0 is None:
                a0 = float(npv)
        with torch.no_grad():
            _, s_now, npv_now = forward()
        nn_seq = decode(s_now)
        nn_npv = npv_of(nn_seq)
        print(f"{ep:>3} {'A npv':<10} {a0:>10.5f} {float(npv_now):>10.5f} "
              f"{nn_npv:>10.5f} {(BOUND-nn_npv)/BOUND*100:>7.2f}%   "
              f"hard-decoded transformer schedule, {sequence_violations(nn_seq,par,chi)} viol")

        # ---- B: GA, warm-started from the network, then the dominance sweep ----
        seeds = [nn_seq] if best_ga is None else [nn_seq, best_ga]
        ga, ga_fit, info = G.run_ga(seeds, tau, value, scale, adjc, n, rng,
                                    generations=10**9, population=128,
                                    label="", every=10**9, seconds=ga_seconds,
                                    quiet=True)
        raw = npv_of(ga)
        ga = G.dominance_sweep(ga, value, tau, keys, n)
        swept = npv_of(ga)
        if swept > best_ga_npv:
            best_ga, best_ga_npv = ga.copy(), swept
        inv_before, _ = G.count_inversions(nn_seq, value, tau, keys, n)
        print(f"{'':>3} {'B ga':<10} {raw:>10.5f} {swept:>10.5f} "
              f"{best_ga_npv:>10.5f} {(BOUND-best_ga_npv)/BOUND*100:>7.2f}%   "
              f"{info['distinct']:,} distinct in {info['seconds']:.0f}s, "
              f"sweep {swept-raw:+.5f}, {sequence_violations(ga,par,chi)} viol")

        # ---- C: NPV + ranking against the GA + the dominance cut ----
        teacher = np.empty(n, np.int64)
        teacher[best_ga] = np.arange(n)
        tt = torch.tensor(teacher, device=dev)
        sig_t = ct.start_times_from_order(best_ga, tau)
        sens = torch.tensor(np.abs(delta * value * psi_np
                                   * np.exp(-delta * sig_t)),
                            dtype=torch.float32, device=dev)
        c0 = rank0 = dom0 = None
        for i in range(SUP_STEPS):
            opt.zero_grad()
            s_raw, s, npv = forward()
            sd = s_raw.detach().std().clamp(min=1e-6)
            temp_rank = sd * TEMP_RANK
            temp_dom = sd * TEMP_DOM / n

            # ranking: sampled pairs, oriented by the teacher's order
            idx = torch.randint(0, n, (2, PAIRS * n), device=dev)
            u, v = idx[0], idx[1]
            first = torch.where(tt[u] < tt[v], u, v)
            second = torch.where(tt[u] < tt[v], v, u)
            keep = tt[u] != tt[v]
            wgt = (sens[first] + sens[second])[keep]
            l_rank = pairwise(s_raw, first[keep], second[keep], temp_rank, wgt)

            # dominance: adjacent precedence-free inversions in OUR schedule
            cur = decode(s)
            a, b = cur[:-1], cur[1:]
            k = a * n + b
            j = np.clip(np.searchsorted(keys, k), 0, keys.size - 1)
            free = keys[j] != k
            dn = value / np.maximum(tau, 1e-300)
            bad = free & (dn[a] < dn[b] - 1e-12)
            if bad.any():
                ta = torch.tensor(a[bad], device=dev)
                tb = torch.tensor(b[bad], device=dev)
                l_dom = pairwise(s_raw, tb, ta, temp_dom,
                                 sens[ta] + sens[tb])       # b should come first
            else:
                l_dom = torch.zeros((), device=dev)

            loss = -npv + W_RANK * l_rank + W_DOM * l_dom
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP)
            opt.step()
            if c0 is None:
                c0, rank0, dom0 = float(npv), float(l_rank), float(l_dom)
        with torch.no_grad():
            _, s_now, npv_now = forward()
        nn_seq = decode(s_now)
        nn_npv = npv_of(nn_seq)
        inv_after, nfree = G.count_inversions(nn_seq, value, tau, keys, n)
        print(f"{'':>3} {'C sup':<10} {c0:>10.5f} {float(npv_now):>10.5f} "
              f"{nn_npv:>10.5f} {(BOUND-nn_npv)/BOUND*100:>7.2f}%   "
              f"rank {rank0:.3f}->{float(l_rank):.3f}, dom {dom0:.3f}->"
              f"{float(l_dom):.3f}, inversions {inv_before}->{inv_after}")
        hist.append({"epoch": ep, "nn": nn_npv, "ga": best_ga_npv,
                     "nn_gap": (BOUND-nn_npv)/BOUND,
                     "ga_gap": (BOUND-best_ga_npv)/BOUND})
        print()

    print("=" * 96)
    print(f"{'epoch':>6} {'transformer':>13} {'gap':>8} {'GA best':>12} {'gap':>8}")
    for h in hist:
        print(f"{h['epoch']:>6} {h['nn']:>13.5f} {h['nn_gap']*100:>7.2f}% "
              f"{h['ga']:>12.5f} {h['ga_gap']*100:>7.2f}%")
    print(f"\ncertified bound {BOUND:+.5f} (computed once; it is a property of "
          f"the instance, not the schedule)")
    return hist


if __name__ == "__main__":
    a = sys.argv[1:]
    main(epochs=int(a[0]) if len(a) > 0 else EPOCHS,
         crop=int(a[1]) if len(a) > 1 else CROP,
         ga_seconds=float(a[2]) if len(a) > 2 else GA_SECONDS,
         proj=(a[3] if len(a) > 3 else "clamp"))


# --------------------------------------------------------------------------

def sample_instances(windows, t_periods=T_PERIODS, min_pos=0.05,
                     min_headroom=0.02, verbose=True):
    """Screen candidate crops and keep only ones where SEQUENCING MATTERS.

    Two degeneracy traps, both met the hard way on this deposit:

      all-waste      a 102-block cone of pure overburden, every value negative
                     and spread only -81k to -54k. Every method scores the same
                     there and the instance teaches nothing. Rejected by
                     requiring a minimum fraction of positive-value blocks.
      no headroom    an instance where the topological order is already almost
                     as good as anything else. Rejected by requiring the
                     value-density order (the exact UNCONSTRAINED optimum, so a
                     ceiling) to beat topological by a margin -- if there is no
                     room between them there is nothing for a scheduler to win.
    """
    import ga_schedule as _G
    keep = []
    if verbose:
        print(f"{'window':>12} {'n':>6} {'pos %':>7} {'topo':>9} {'density':>9}"
              f" {'headroom':>9}  verdict")
    for wnd in windows:
        try:
            P = _G.load_instance(crop=wnd, t_periods=t_periods)
        except Exception as exc:
            if verbose:
                print(f"{str(wnd):>12} {'-':>6}  load failed: {type(exc).__name__}")
            continue
        n, tau, value = P["n"], P["tau"], P["value"]
        scale = P["scale"]
        pos = float((value > 0).mean())
        topo = float(ct.npv(ct.start_times_from_order(P["order"], tau), tau,
                            value, discount=DISCOUNT, scale=scale))
        dens = float(ct.npv(ct.start_times(value / tau, tau, value=value), tau,
                            value, discount=DISCOUNT, scale=scale))
        head = (dens - topo) / abs(topo) if topo != 0 else 0.0
        ok = pos >= min_pos and head >= min_headroom
        if verbose:
            why = "keep" if ok else ("all-waste" if pos < min_pos
                                     else "no headroom")
            print(f"{str(wnd):>12} {n:>6} {pos:>6.1%} {topo:>+9.4f} "
                  f"{dens:>+9.4f} {head:>8.1%}  {why}")
        if ok:
            keep.append(P)
    return keep


def main_multi(windows=((0, 6), (6, 12), (12, 18), (18, 24)), epochs=8,
               ga_seconds=5.0, proj="clamp", cert_k=1, holdout=(),
               w_npv=1.0, w_rank=W_RANK, w_dom=W_DOM, verbose=True,
               eval_every=0, tag="net", step_every=1):
    """The same loop, but cycling over SEVERAL instances with one network.

    This is the only setting in which the transformer can earn its cost. On a
    single instance it has one input and 400k parameters -- it is parameterising
    one score vector, and the GA simply does that better. Amortisation across
    instances is the claim worth testing: solve many slowly, then get a schedule
    in one forward pass.
    """
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    insts = sample_instances(list(windows), verbose=verbose)
    print()

    prep = []
    for P in insts:
        n = P["n"]
        par, chi = np.asarray(P["par"]), np.asarray(P["chi"])
        w = P["static"]["tonnage"]
        e, l = reachability_bounds(n, par, chi, P["order"], w,
                                   w.sum() / T_PERIODS, float(T_PERIODS))
        cert = certify(P["order"], P["tau"], P["value"], w, par, chi,
                       iters=2000, subperiods=cert_k, earliest=e, latest=l)
        _, grp = dag_levels(n, par, chi, P["order"])
        prep.append({
            "P": P, "par": par, "chi": chi, "w": w, "n": n,
            "bound": cert["bound"] / P["scale"],
            "keys": G.edge_keys(par, chi, n),
            "csr": children_csr(n, par, chi),
            "groups": [(torch.tensor(a, device=dev), torch.tensor(b, device=dev))
                       for a, b in grp],
            "x": torch.tensor(T.block_features(P["static"]),
                              dtype=torch.float32, device=dev).unsqueeze(0),
            "pr": ct.prepare(w, w.sum() / T_PERIODS, P["value"],
                             discount=DISCOUNT, device=dev, dtype=torch.float32),
            "best_ga": None, "best_ga_npv": -np.inf})
        topo = np.empty(n, np.int64)
        topo[P["order"]] = np.arange(n)
        prep[-1]["topo"] = topo
        print(f"  instance n={n:<6} certified bound {prep[-1]['bound']:+.5f}"
              f"   topological gap "
              f"{(prep[-1]['bound'] - float(ct.npv(ct.start_times_from_order(P['order'], P['tau']), P['tau'], P['value'], discount=DISCOUNT, scale=P['scale']))) / prep[-1]['bound'] * 100:5.2f}%")

    in_dim = prep[0]["x"].shape[-1]
    print("=" * 92)
    net, opt, ep0, best_seen = build_or_load(in_dim, dev, tag=tag or "net",
                                             verbose=True)
    print(f"  MODEL: {'RESUMED a trained model' if ep0 else 'STARTING FROM SCRATCH (zero model)'}"
          f"   epochs already done: {ep0}")
    print("=" * 92)

    def ep_label(ep):
        return f"{ep0 + ep}" if ep0 else f"{ep}"
    delta = ct.delta_from_discount(DISCOUNT)

    held = ([_prep_one(Ph, dev, cert_k)
             for Ph in sample_instances(list(holdout), verbose=False)]
            if holdout else [])
    if verbose:
        print(f"  {len(prep)} train instances, {len(held)} held out")
    print("")
    log = []
    for ep in range(1, epochs + 1):
        I = prep[(ep - 1) % len(prep)]
        P, n = I["P"], I["n"]
        tau, value, scale = P["tau"], P["value"], P["scale"]
        psi_np = ct.within_block_shape(tau, delta)

        def fwd():
            sr = net(I["x"], pool=None).squeeze(-1).squeeze(0)
            s = clamp_torch(sr, I["groups"], n) if proj == "clamp" else sr
            sg = ct.start_times_soft(s.unsqueeze(0), I["pr"]["tau"],
                                     value=I["pr"]["value"],
                                     window=I["pr"]["window"])
            return sr, s, ct.npv_soft(sg, I["pr"]["tau"], I["pr"]["value"],
                                      psi=I["pr"]["psi"],
                                      scale=I["pr"]["scale"]).sum()

        def dec(s):
            return schedule_priority_kahn(
                s.detach().cpu().numpy().astype(np.float64), I["csr"],
                tiebreak=I["topo"].astype(float))

        def npv_of(q):
            return float(ct.npv(ct.start_times_from_order(q, tau), tau, value,
                                discount=DISCOUNT, scale=scale))

        B = I["bound"]
        if verbose:
            print("")
            print(f"epoch {ep_label(ep)}  instance {(ep-1)%len(prep)}  "
                  f"n={n}  certified bound {B:+.5f}")
            print(f"  phase A -- NPV descent ({NPV_STEPS} steps)")
        for st in range(1, NPV_STEPS + 1):
            opt.zero_grad(); _, s, v = fwd(); (-v).backward()
            gn = float(torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP))
            opt.step()
            if verbose and step_every and st % step_every == 0:
                hard = npv_of(dec(s))
                print(f"    step {st:>3}  softNPV {float(v):+.5f}  "
                      f"hardNPV {hard:+.5f}  gap {(B-hard)/B*100:6.2f}%  "
                      f"|g| {gn:.2e}")
        with torch.no_grad():
            _, s_now, _ = fwd()
        nn_a = npv_of(dec(s_now))

        seeds = [dec(s_now)] + ([I["best_ga"]] if I["best_ga"] is not None else [])
        ga, _, info = G.run_ga(seeds, tau, value, scale, P["adj"], n, rng,
                               generations=10**9, population=128, label="",
                               every=10**9, seconds=ga_seconds, quiet=True)
        raw_ga = npv_of(ga)
        ga = G.dominance_sweep(ga, value, tau, I["keys"], n)
        if npv_of(ga) > I["best_ga_npv"]:
            I["best_ga"], I["best_ga_npv"] = ga.copy(), npv_of(ga)
        if verbose:
            print(f"  phase B -- GA {ga_seconds:.0f}s: {raw_ga:+.5f} "
                  f"-> sweep {npv_of(ga):+.5f}  best {I['best_ga_npv']:+.5f}  "
                  f"gap {(B-I['best_ga_npv'])/B*100:6.2f}%  "
                  f"{info['distinct']:,} distinct")
            print(f"  phase C -- rank + dominance ({SUP_STEPS} steps)")

        teacher = np.empty(n, np.int64); teacher[I["best_ga"]] = np.arange(n)
        tt = torch.tensor(teacher, device=dev)
        sens = torch.tensor(np.abs(delta * value * psi_np * np.exp(
            -delta * ct.start_times_from_order(I["best_ga"], tau))),
            dtype=torch.float32, device=dev)
        dn = value / np.maximum(tau, 1e-300)
        for st in range(1, SUP_STEPS + 1):
            opt.zero_grad(); sr, s, v = fwd()
            sd = sr.detach().std().clamp(min=1e-6)
            idx = torch.randint(0, n, (2, PAIRS * n), device=dev)
            u, vv = idx[0], idx[1]
            f1 = torch.where(tt[u] < tt[vv], u, vv)
            f2 = torch.where(tt[u] < tt[vv], vv, u)
            k = tt[u] != tt[vv]
            l_rank = pairwise(sr, f1[k], f2[k], sd * TEMP_RANK,
                              (sens[f1] + sens[f2])[k])
            cur = dec(s); a, b = cur[:-1], cur[1:]
            kk = a * n + b
            j = np.clip(np.searchsorted(I["keys"], kk), 0, I["keys"].size - 1)
            bad = (I["keys"][j] != kk) & (dn[a] < dn[b] - 1e-12)
            if bad.any():
                ta = torch.tensor(a[bad], device=dev); tb = torch.tensor(b[bad], device=dev)
                l_dom = pairwise(sr, tb, ta, sd * TEMP_DOM / n, sens[ta] + sens[tb])
            else:
                l_dom = torch.zeros((), device=dev)
            (-w_npv * v + w_rank * l_rank + w_dom * l_dom).backward()
            gn = float(torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP))
            opt.step()
            if verbose and step_every and st % step_every == 0:
                hard = npv_of(cur)
                print(f"    step {st:>3}  softNPV {float(v):+.5f}  "
                      f"hardNPV {hard:+.5f}  gap {(B-hard)/B*100:6.2f}%  "
                      f"rank {float(l_rank):7.4f}  dom {float(l_dom):8.4f}  "
                      f"inv {int(bad.sum()):>4}  |g| {gn:.2e}")
        with torch.no_grad():
            _, s_now, _ = fwd()
        q = dec(s_now); nn_c = npv_of(q)
        inv, _ = G.count_inversions(q, value, tau, I["keys"], n)
        if verbose:
            print(f"  EPOCH {ep_label(ep)} SUMMARY  transformer A {nn_a:+.5f} "
                  f"-> C {nn_c:+.5f}   certified gap {(B-nn_c)/B*100:6.2f}%   "
                  f"GA {I['best_ga_npv']:+.5f} ({(B-I['best_ga_npv'])/B*100:.2f}%)"
                  f"   inv {inv}   {sequence_violations(q, I['par'], I['chi'])} viol")
        log.append({"ep": ep, "inst": (ep-1) % len(prep), "nn": nn_c,
                    "gap": (B-nn_c)/B})
        is_best = (B - nn_c) / B < (1 - best_seen if best_seen > -np.inf else np.inf)
        if is_best:
            best_seen = 1 - (B - nn_c) / B
        save_checkpoint(net, opt, ep0 + ep, in_dim, tag=tag or "net",
                        best=best_seen, is_best=is_best)
        if eval_every and held and ep % eval_every == 0:
            hz = _zero_shot(net, held, proj)
            print(f"    held-out zero-shot mean gap "
                  f"{np.mean([h['nn_gap'] for h in hz])*100:6.2f}%   "
                  f"(topological {np.mean([h['topo_gap'] for h in hz])*100:6.2f}%)")
    tail = log[-len(prep):]
    return {"log": log, "net": net, "train": prep, "held": held,
            "train_gap": float(np.mean([t["gap"] for t in tail])),
            "holdout": _zero_shot(net, held, proj) if held else []}

# --------------------------------------------------------------------------
# held-out evaluation: one forward pass, no GA, no training
# --------------------------------------------------------------------------

def _prep_one(P, dev, cert_k=1, cert_iters=2000):
    """Everything constant for one instance, including its certified bound."""
    n = P["n"]
    par, chi = np.asarray(P["par"]), np.asarray(P["chi"])
    w = P["static"]["tonnage"]
    e, l = reachability_bounds(n, par, chi, P["order"], w,
                               w.sum() / T_PERIODS, float(T_PERIODS))
    cert = certify(P["order"], P["tau"], P["value"], w, par, chi,
                   iters=cert_iters, subperiods=cert_k, earliest=e, latest=l)
    _, grp = dag_levels(n, par, chi, P["order"])
    topo = np.empty(n, np.int64)
    topo[P["order"]] = np.arange(n)
    return {"P": P, "par": par, "chi": chi, "w": w, "n": n, "topo": topo,
            "bound": cert["bound"] / P["scale"],
            "keys": G.edge_keys(par, chi, n),
            "csr": children_csr(n, par, chi),
            "groups": [(torch.tensor(a, device=dev), torch.tensor(b, device=dev))
                       for a, b in grp],
            "x": torch.tensor(T.block_features(P["static"]),
                              dtype=torch.float32, device=dev).unsqueeze(0),
            "pr": ct.prepare(w, w.sum() / T_PERIODS, P["value"],
                             discount=DISCOUNT, device=dev, dtype=torch.float32),
            "best_ga": None, "best_ga_npv": -np.inf}


def _zero_shot(net, insts, proj="clamp"):
    """The amortisation test: schedule an UNSEEN instance in one forward pass.

    No GA, no gradient steps, no tuning on this instance. If the gap here beats
    the topological baseline, the network has learnt something transferable; if
    it does not, it was only ever fitting the training instances one score
    vector at a time -- which is what a single-instance run cannot distinguish.
    """
    out = []
    for I in insts:
        P = I["P"]
        with torch.no_grad():
            sr = net(I["x"], pool=None).squeeze(-1).squeeze(0)
            s = clamp_torch(sr, I["groups"], I["n"]) if proj == "clamp" else sr
        q = schedule_priority_kahn(s.cpu().numpy().astype(np.float64), I["csr"],
                                   tiebreak=I["topo"].astype(float))
        f = lambda seq: float(ct.npv(ct.start_times_from_order(seq, P["tau"]),
                                     P["tau"], P["value"], discount=DISCOUNT,
                                     scale=P["scale"]))
        nn, topo = f(q), f(P["order"])
        out.append({"n": I["n"], "bound": I["bound"], "nn": nn, "topo": topo,
                    "nn_gap": (I["bound"] - nn) / I["bound"],
                    "topo_gap": (I["bound"] - topo) / I["bound"],
                    "viol": sequence_violations(q, I["par"], I["chi"])})
    return out


# --------------------------------------------------------------------------
# checkpointing
# --------------------------------------------------------------------------

MODEL_DIR = "models"


def _arch(in_dim):
    return {"in_dim": in_dim, "d_model": T.D_MODEL, "nhead": T.N_HEAD,
            "num_layers": T.N_LAYERS, "dim_feedforward": T.FF,
            "dropout": T.DROPOUT, "out_dim": 1, "use_posenc": False}


def build_or_load(in_dim, dev, tag="net", lr=LR, directory=MODEL_DIR,
                  verbose=True):
    """Make the network, resuming from `models/<tag>_latest.pt` when present.

    Resuming is refused rather than forced when the saved architecture differs
    from the one requested -- silently loading a mismatched checkpoint would
    either throw deep inside load_state_dict or, worse, partially succeed. The
    optimiser state travels with the weights, because Adam's moments matter as
    much as the parameters for picking up mid-run.
    """
    import os
    arch = _arch(in_dim)
    net = SimpleTransformer(**arch).to(dev)
    net.eval()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    path = os.path.join(directory, f"{tag}_latest.pt")
    start, best = 0, -np.inf
    if os.path.isfile(path):
        ck = torch.load(path, map_location=dev, weights_only=False)
        if ck.get("arch") != arch:
            raise RuntimeError(
                f"{path} was trained with a different architecture:\n"
                f"  saved   {ck.get('arch')}\n  requested {arch}\n"
                "Delete it or pass a different tag to start fresh.")
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optim"])
        start, best = ck.get("epoch", 0), ck.get("best", -np.inf)
        if verbose:
            print(f"  resumed {path}: epoch {start}, best train gap "
                  f"{(1-best)*100 if best > -np.inf else float('nan'):.2f}%")
    elif verbose:
        print(f"  no checkpoint in {directory}/ -- starting from scratch")
    return net, opt, start, best


def save_checkpoint(net, opt, epoch, in_dim, tag="net", best=None,
                    directory=MODEL_DIR, is_best=False):
    import os
    os.makedirs(directory, exist_ok=True)
    blob = {"model": net.state_dict(), "optim": opt.state_dict(),
            "epoch": epoch, "arch": _arch(in_dim), "best": best}
    torch.save(blob, os.path.join(directory, f"{tag}_latest.pt"))
    if is_best:
        torch.save(blob, os.path.join(directory, f"{tag}_best.pt"))
