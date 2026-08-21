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

Run: python train_loop.py                     sensible defaults, resumes if a
                                            checkpoint exists
     python train_loop.py --help            every knob
     python train_loop.py --fresh --epochs 100 --step-every 5
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
                                    quiet=True, keys=keys)
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
        # nothing here needs the heuristic yet; _prep_one computes it once
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
               ga_seconds=5.0, proj="kernel", cert_k=1, holdout=(),
               npv_steps=NPV_STEPS, sup_steps=SUP_STEPS,
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

    prep = [_prep_one(P, dev, cert_k, need_gram=(proj == "kernel"))
            for P in insts]
    for I in prep:
        P = I["P"]
        topo = float(ct.npv(ct.start_times_from_order(P["order"], P["tau"]),
                            P["tau"], P["value"], discount=DISCOUNT,
                            scale=P["scale"]))
        if verbose:
            print(f"  instance n={I['n']:<6} certified bound {I['bound']:+.5f}"
                  f"   topological gap {(I['bound']-topo)/I['bound']*100:5.2f}%")

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

    held = ([_prep_one(Ph, dev, cert_k, need_gram=(proj == "kernel"))
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
            s = feasible_scores(sr, I, proj)
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
            print(f"  phase A -- NPV descent ({npv_steps} steps)")
        for st in range(1, npv_steps + 1):
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
                               every=10**9, seconds=ga_seconds, quiet=True,
                               keys=I["keys"])
        raw_ga = npv_of(ga)
        ga = G.dominance_sweep(ga, value, tau, I["keys"], n)
        if npv_of(ga) > I["best_ga_npv"]:
            I["best_ga"], I["best_ga_npv"] = ga.copy(), npv_of(ga)
        if verbose:
            print(f"  phase B -- GA {ga_seconds:.0f}s: {raw_ga:+.5f} "
                  f"-> sweep {npv_of(ga):+.5f}  best {I['best_ga_npv']:+.5f}  "
                  f"gap {(B-I['best_ga_npv'])/B*100:6.2f}%  "
                  f"{info['distinct']:,} distinct")
            print(f"  phase C -- rank + dominance ({sup_steps} steps)")

        teacher = np.empty(n, np.int64); teacher[I["best_ga"]] = np.arange(n)
        tt = torch.tensor(teacher, device=dev)
        sens = torch.tensor(np.abs(delta * value * psi_np * np.exp(
            -delta * ct.start_times_from_order(I["best_ga"], tau))),
            dtype=torch.float32, device=dev)
        dn = value / np.maximum(tau, 1e-300)
        for st in range(1, sup_steps + 1):
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

def _prep_one(P, dev, cert_k=1, cert_iters=2000, need_gram=False):
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
    # The honest baseline. Descendant-cone efficiency -- value reachable below
    # a block divided by the periods of work to clear it -- decoded through the
    # ready set. Closed form, no search, and measured to beat a 5-second GA on
    # three of five crops (mean certified gap 8.66% against topological's
    # 19.66%). Reporting against topological was flattering every result.
    csr_ = children_csr(n, par, chi)
    heur = schedule_priority_kahn(
        np.asarray(T.efficiency_columns(P["static"])[3], dtype=np.float64),
        csr_, tiebreak=topo.astype(float))
    heur_npv = float(ct.npv(ct.start_times_from_order(heur, P["tau"]),
                            P["tau"], P["value"], discount=DISCOUNT,
                            scale=P["scale"]))
    return {"P": P, "par": par, "chi": chi, "w": w, "n": n, "topo": topo,
            "heur": heur, "heur_npv": heur_npv,
            "bound": cert["bound"] / P["scale"],
            "keys": G.edge_keys(par, chi, n),
            "csr": csr_,
            "groups": [(torch.tensor(a, device=dev), torch.tensor(b, device=dev))
                       for a, b in grp],
            "x": torch.tensor(T.block_features(P["static"]),
                              dtype=torch.float32, device=dev).unsqueeze(0),
            "pr": ct.prepare(w, w.sum() / T_PERIODS, P["value"],
                             discount=DISCOUNT, device=dev, dtype=torch.float32),
            "gram": T.build_gram(P["static"]) if need_gram else None,
            "best_ga": None, "best_ga_npv": -np.inf}


def feasible_scores(s_raw, I, proj):
    """Make a raw score field precedence-feasible.

    One function, used by every call site. The previous arrangement had
    `clamp_torch(...) if proj == "clamp" else s_raw` written out three times,
    which meant --proj kernel silently applied NO projection in the
    multi-instance path -- the gram was never built there. A flag that quietly
    does nothing is worse than one that errors.
    """
    if proj == "kernel":
        if I.get("gram") is None:
            raise RuntimeError("proj='kernel' needs a gram; prepare the "
                               "instance with need_gram=True")
        return interpolate_precedence_torch(s_raw, I["gram"], I["par"],
                                            I["chi"])
    if proj == "clamp":
        return clamp_torch(s_raw, I["groups"], I["n"])
    if proj == "none":
        return s_raw
    raise ValueError(f"unknown proj {proj!r}")


def _fwd(net, I, proj):
    """x -> raw score -> feasible score -> soft NPV, for one instance."""
    s_raw = net(I["x"], pool=None).squeeze(-1).squeeze(0)
    s = feasible_scores(s_raw, I, proj)
    sg = ct.start_times_soft(s.unsqueeze(0), I["pr"]["tau"],
                             value=I["pr"]["value"], window=I["pr"]["window"])
    return s_raw, s, ct.npv_soft(sg, I["pr"]["tau"], I["pr"]["value"],
                                 psi=I["pr"]["psi"],
                                 scale=I["pr"]["scale"]).sum()


def _decode(s, I):
    """Feasible mining sequence from a score field, via the ready set."""
    return schedule_priority_kahn(s.detach().cpu().numpy().astype(np.float64),
                                  I["csr"], tiebreak=I["topo"].astype(float))


def _zero_shot(net, insts, proj="kernel"):
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
            s = feasible_scores(sr, I, proj)
        q = schedule_priority_kahn(s.cpu().numpy().astype(np.float64), I["csr"],
                                   tiebreak=I["topo"].astype(float))
        f = lambda seq: float(ct.npv(ct.start_times_from_order(seq, P["tau"]),
                                     P["tau"], P["value"], discount=DISCOUNT,
                                     scale=P["scale"]))
        nn, topo = f(q), f(P["order"])
        out.append({"n": I["n"], "bound": I["bound"], "nn": nn, "topo": topo,
                    "heur": I["heur_npv"],
                    "nn_gap": (I["bound"] - nn) / I["bound"],
                    "topo_gap": (I["bound"] - topo) / I["bound"],
                    "heur_gap": (I["bound"] - I["heur_npv"]) / I["bound"],
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


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------

def make_windows(width, n_train, n_hold, stride=None, buffer=5, start=0):
    """Training and held-out crop windows, generated rather than typed out.

    Training windows step by `stride` (default width - 1, so consecutive crops
    overlap slightly and the set is varied without being disjointly tiny).
    Held-out windows start `buffer` columns past the last training column and
    are laid end to end, so nothing they contain was seen in training -- the
    buffer is what stops a held-out crop sharing a slope cone with a training
    one and quietly leaking.
    """
    stride = stride or max(1, width - 1)
    train = [(start + i * stride, start + i * stride + width)
             for i in range(n_train)]
    h0 = train[-1][1] + buffer
    held = [(h0 + i * width, h0 + i * width + width) for i in range(n_hold)]
    return train, held


def _cli():
    import argparse
    p = argparse.ArgumentParser(
        description="Search-then-imitate training on the block model. "
                    "Resumes from models/<tag>_latest.pt when present.")
    g = p.add_argument_group("what to train on")
    g.add_argument("--width", type=int, default=5,
                   help="crop width in columns; 5 is ~1035 blocks (default 5)")
    g.add_argument("--n-train", type=int, default=10,
                   help="number of training instances (default 10)")
    g.add_argument("--n-hold", type=int, default=3,
                   help="number of held-out instances, 0 to skip (default 3)")
    g.add_argument("--stride", type=int, default=None,
                   help="columns between training windows (default width - 1)")
    g.add_argument("--single", type=int, metavar="CROP", default=None,
                   help="train on ONE crop of this width instead, and skip "
                        "the held-out evaluation")

    g = p.add_argument_group("schedule of the loop")
    g.add_argument("--epochs", type=int, default=30)
    g.add_argument("--phased", action="store_true",
                   help="old structure: 20 NPV steps, one fixed-budget GA, "
                        "then 20 corrective steps. Default is interleaved.")
    g.add_argument("--improvements", type=int, default=10,
                   help="GA improvements chased per epoch (interleaved mode); "
                        "each one yields --steps-per-improvement steps")
    g.add_argument("--fit-steps", type=int, default=100,
                   help="MAX gradient steps spent fitting each teacher. The "
                        "loop stops early when the net matches it or stops "
                        "improving, so this is a ceiling, not a cost.")
    g.add_argument("--fit-patience", type=int, default=15,
                   help="stop fitting a teacher after this many steps with no "
                        "improvement in the net's hard NPV")
    g.add_argument("--fit-w-npv", type=float, default=0.1,
                   help="NPV weight WHILE fitting a teacher. Lower than "
                        "--w-npv on purpose: NPV's gradient is 1e3-1e5 against "
                        "an O(1) ranking term and swamps it. Full-weight NPV "
                        "descent still runs in each epoch's prelude.")
    g.add_argument("--margin", type=float, default=0.005,
                   help="relative improvement the GA must achieve over the net "
                        "to count as a teacher. Without it the GA stops at the "
                        "first epsilon-better schedule and, once the net has "
                        "caught up, teaches it noise.")
    g.add_argument("--npv-steps", type=int, default=NPV_STEPS,
                   help="NPV-descent steps at the start of each epoch, before "
                        "any GA teacher exists. The dominance cut runs in "
                        "these too; only ranking needs a teacher. 0 to skip.")
    g.add_argument("--sup-steps", type=int, default=SUP_STEPS,
                   help="phase C steps, --phased only")
    g.add_argument("--ga-cap", type=float, default=20.0,
                   help="seconds an epoch spends chasing improvements")
    g.add_argument("--ga-seconds", type=float, default=5.0,
                   help="fixed GA budget, --phased only")
    g.add_argument("--eval-every", type=int, default=10,
                   help="epochs between held-out evaluations, 0 to disable")
    g.add_argument("--step-every", type=int, default=1,
                   help="print every Nth gradient step; 0 prints none")

    g = p.add_argument_group("loss weights (defaults are the ablation winner)")
    g.add_argument("--w-npv", type=float, default=1.0,
                   help="soft NPV in the SUPERVISED phase. 0 by default: the "
                        "clamp's ties inflate the surrogate and it drags "
                        "(ablation 13.77%% with it, 10.49%% without) -- but "
                        "that was under the CLAMP, whose ties inflate the "
                        "surrogate. Under the kernel projection soft and hard "
                        "NPV agree, so 1.0 is the default again.")
    g.add_argument("--w-rank", type=float, default=0.3)
    g.add_argument("--w-dom", type=float, default=0.3)

    g = p.add_argument_group("model")
    g.add_argument("--tag", default="net",
                   help="checkpoint name under models/ (default net)")
    g.add_argument("--fresh", action="store_true",
                   help="ignore any checkpoint and start from a zero model")
    g.add_argument("--proj", choices=("kernel", "clamp", "none"),
                   default="kernel",
                   help="how the score field is made precedence-feasible. "
                        "kernel is anchor interpolation: exact, dense gradient "
                        "(1242/1242 blocks) and ~129 ms. clamp is exact and "
                        "0.8 ms but collapses subtrees onto shared values, so "
                        "only 64/1242 blocks get gradient and the reachable "
                        "schedule set shrinks. none is fastest but leaves the "
                        "score field infeasible at loss time.")
    g.add_argument("--cert-k", type=int, default=1,
                   help="sub-periods in the certificate; higher is tighter "
                        "and slower (default 1)")
    return p.parse_args()




# --------------------------------------------------------------------------
# interleaved loop: the GA advances until it beats the network, then one step
# --------------------------------------------------------------------------

IMPROVEMENTS = 10
GA_SLICE = 0.3             # seconds of GA per attempt before re-checking
GA_CAP = 20.0              # seconds an epoch will spend chasing improvements


def _losses(net, I, proj, teacher, sens, dens, w_npv, w_rank, w_dom, dev, n):
    """NPV + ranking + dominance, all at once.

    The old phased arrangement ran NPV for 20 steps, then the GA, then the
    corrective terms for 20 more. Only the RANKING term ever needed that
    ordering -- it needs a teacher, which the GA phase produces. The dominance
    cut needs no teacher at all: it is derived from the schedule itself, so
    keeping it out of the early steps was an artefact of the batch structure
    rather than a decision. Here every step carries every term whose inputs
    exist.
    """
    s_raw, s, npv = _fwd(net, I, proj)
    cur = _decode(s, I)
    sd = s_raw.detach().std().clamp(min=1e-6)

    l_rank = torch.zeros((), device=dev)
    if teacher is not None and sens is not None:
        idx = torch.randint(0, n, (2, PAIRS * n), device=dev)
        u, v = idx[0], idx[1]
        f1 = torch.where(teacher[u] < teacher[v], u, v)
        f2 = torch.where(teacher[u] < teacher[v], v, u)
        k = teacher[u] != teacher[v]
        l_rank = pairwise(s_raw, f1[k], f2[k], sd * TEMP_RANK,
                          (sens[f1] + sens[f2])[k])

    a, b = cur[:-1], cur[1:]
    kk = a * n + b
    j = np.clip(np.searchsorted(I["keys"], kk), 0, I["keys"].size - 1)
    bad = (I["keys"][j] != kk) & (dens[a] < dens[b] - 1e-12)
    if bad.any():
        ta = torch.tensor(a[bad], device=dev)
        tb = torch.tensor(b[bad], device=dev)
        wt = (torch.ones(ta.shape[0], device=dev) if sens is None
              else sens[ta] + sens[tb])
        l_dom = pairwise(s_raw, tb, ta, sd * TEMP_DOM / n, wt)
    else:
        l_dom = torch.zeros((), device=dev)

    loss = -w_npv * npv + w_rank * l_rank + w_dom * l_dom
    return loss, npv, l_rank, l_dom, cur, int(bad.sum())


def main_interleaved(windows=((0, 5), (4, 9), (8, 13)), epochs=10,
                     improvements=IMPROVEMENTS, proj="kernel", cert_k=1,
                     holdout=(), w_npv=1.0, w_rank=0.3, w_dom=0.3,
                     eval_every=5, tag="net", ga_cap=GA_CAP,
                     ga_slice=GA_SLICE, verbose=True, steps_per_imp=1,
                     npv_steps=NPV_STEPS, margin=0.005, fit_steps=100,
                     fit_patience=15, fit_w_npv=0.1):
    """One epoch = up to `improvements` rounds of (GA beats the net -> 1 step).

    No fixed GA budget. The GA advances in slices, carrying its population
    forward, until its best schedule beats the network's current one; that
    schedule becomes the teacher and the network takes exactly ONE gradient
    step against it. Then the GA continues from where it left off.

    So the teacher is regenerated between every student step instead of once
    per twenty, which is what expert iteration is meant to do. `ga_cap` bounds
    the chase: if the GA cannot beat the network inside it the epoch ends
    rather than hanging, and that failure is itself the signal worth having --
    it means the network has caught up.
    """
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    insts = sample_instances(list(windows), verbose=verbose)
    prep = [_prep_one(P, dev, cert_k, need_gram=(proj == "kernel"))
            for P in insts]
    held = ([_prep_one(Ph, dev, cert_k, need_gram=(proj == "kernel"))
             for Ph in sample_instances(list(holdout), verbose=False)]
            if holdout else [])
    in_dim = prep[0]["x"].shape[-1]
    print("=" * 100)
    net, opt, ep0, best_seen = build_or_load(in_dim, dev, tag=tag, verbose=True)
    state = "RESUMED a trained model" if ep0 else "STARTING FROM SCRATCH (zero model)"
    print(f"  MODEL: {state}   epochs already done: {ep0}")
    print("=" * 100)
    delta = ct.delta_from_discount(DISCOUNT)
    log = []

    for ep in range(1, epochs + 1):
        I = prep[(ep - 1) % len(prep)]
        P, n, B = I["P"], I["n"], I["bound"]
        tau, value, scale = P["tau"], P["value"], P["scale"]
        dens = value / np.maximum(tau, 1e-300)
        psi_np = ct.within_block_shape(tau, delta)

        def npv_of(q):
            return float(ct.npv(ct.start_times_from_order(q, tau), tau, value,
                                discount=DISCOUNT, scale=scale))

        with torch.no_grad():
            _, s0, _ = _fwd(net, I, proj)
        nn_npv = npv_of(_decode(s0, I))
        print("")
        print(f"epoch {ep0+ep}  instance {(ep-1)%len(prep)}  n={n}  "
              f"bound {B:+.5f}  net starts {nn_npv:+.5f} "
              f"(gap {(B-nn_npv)/B*100:.2f}%)")
        print(f"         baselines: cone-efficiency {I['heur_npv']:+.5f} "
              f"(gap {(B-I['heur_npv'])/B*100:.2f}%)   topological "
              f"{npv_of(P['order']):+.5f} "
              f"(gap {(B-npv_of(P['order']))/B*100:.2f}%)")
        print(f"  {'round':>5} {'GA best':>10} {'net after':>10} {'gap':>8} "
              f"{'rank':>8} {'dom':>8} {'|grad|':>9} {'inv':>5} {'GA s':>6} "
              f"{'fit':>5} {'stop':<8}")

        # NPV descent prelude. Interleaving alone cut the NPV budget from 40
        # steps an epoch to one per improvement, which is not what "optimise
        # NPV as well" should mean. The dominance cut rides along because it
        # needs no teacher; only the ranking term has to wait for the GA.
        for st in range(1, npv_steps + 1):
            opt.zero_grad()
            loss, npv, _, l_dom, cur, ninv = _losses(
                net, I, proj, None, None, dens, w_npv, 0.0, w_dom, dev, n)
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP))
            opt.step()
            if verbose and st % max(1, npv_steps // 4) == 0:
                hard = npv_of(cur)
                print(f"    npv  {st:>3} {'':>10} {hard:>10.5f} "
                      f"{(B-hard)/B*100:>7.2f}% {'-':>8} "
                      f"{float(l_dom):>8.4f} {gn:>9.2e} {ninv:>5}")
        if npv_steps:
            with torch.no_grad():
                _, s0, _ = _fwd(net, I, proj)
            nn_npv = npv_of(_decode(s0, I))

        pop, ga_best, ga_npv = None, None, -np.inf
        t_ga, info = 0.0, {"distinct": 0}
        used_heur = False
        for r in range(1, improvements + 1):
            beat = False
            need = nn_npv + margin * abs(nn_npv)
            # The closed-form heuristic is a teacher in its own right, and a
            # free one. If it already clears the margin there is no reason to
            # spend GA time rediscovering it.
            if not used_heur and I["heur_npv"] > max(ga_npv, need):
                ga_best, ga_npv, beat, used_heur = (I["heur"], I["heur_npv"],
                                                   True, True)
            while not beat and t_ga < ga_cap:
                # seed the GA with the heuristic as well as the network: it
                # lands roughly where cold search takes 5 s to climb to
                seeds = ([_decode(s0, I), I["heur"]] if pop is None else [])
                _, _, info = G.run_ga(seeds, tau, value, scale, P["adj"], n,
                                      rng, generations=10**9, population=128,
                                      label="", every=10**9, seconds=ga_slice,
                                      quiet=True, keys=I["keys"], init_pop=pop)
                pop, t_ga = info["pop"], t_ga + info["seconds"]
                top = pop[int(np.argmax(G.evaluate(pop, tau, value, scale)))]
                cand = G.dominance_sweep(top, value, tau, I["keys"], n)
                # A MARGIN, not just "better". Without one the GA stops at the
                # first epsilon-better schedule, and once the network has caught
                # up that margin was measured at 0.047% -- so the teacher was
                # the student's own output plus noise, and epochs went net
                # NEGATIVE (10.00% -> 10.21% while the net wobbled 0.52609 ->
                # 0.52137 -> 0.52570). Failing to clear the margin is the
                # useful signal, not a problem: it means the net has caught up
                # on this instance at this GA budget.
                if npv_of(cand) > max(ga_npv, need):
                    ga_best, ga_npv, beat = cand, npv_of(cand), True
                    break
            if not beat:
                print(f"  {r:>5}  GA could not beat the net by {margin:.1%} "
                      f"within {ga_cap:.0f}s -- net has caught up, ending epoch")
                break

            teacher = np.empty(n, np.int64)
            teacher[ga_best] = np.arange(n)
            tt = torch.tensor(teacher, device=dev)
            sens = torch.tensor(np.abs(
                delta * value * psi_np
                * np.exp(-delta * ct.start_times_from_order(ga_best, tau))),
                dtype=torch.float32, device=dev)

            # FIT this teacher, do not merely nod at it. The fixed-teacher
            # test showed the network reaches 99.7% of a schedule with rank
            # correlation 0.998 given enough steps; one step per teacher was
            # starving it. Stop on any of three conditions: the net matches the
            # teacher, it stops improving for `fit_patience` steps, or
            # `fit_steps` is exhausted -- so an easy teacher is cheap and a hard
            # one gets the budget.
            best_fit, since, used, why = nn_npv, 0, 0, "cap"
            for st in range(1, max(1, fit_steps) + 1):
                opt.zero_grad()
                # NPV is kept but DOWN-WEIGHTED while fitting a teacher. Its
                # gradient runs 1e3-1e5 against an O(1) ranking term, so at full
                # weight it swamps the imitation and nothing is memorised -- the
                # net oscillated 0.52099 -> 0.54581 -> 0.51274 and every teacher
                # ended in `plateau`, never `matched`. Full-weight NPV descent
                # still happens, in the prelude at the top of each epoch.
                loss, npv, l_rank, l_dom, cur, ninv = _losses(
                    net, I, proj, tt, sens, dens, fit_w_npv, w_rank, w_dom,
                    dev, n)
                loss.backward()
                gn = float(torch.nn.utils.clip_grad_norm_(net.parameters(),
                                                          CLIP))
                opt.step()
                used = st
                with torch.no_grad():
                    _, s0, _ = _fwd(net, I, proj)
                nn_npv = npv_of(_decode(s0, I))
                if nn_npv > best_fit + 1e-9:
                    best_fit, since = nn_npv, 0
                else:
                    since += 1
                if nn_npv >= ga_npv:
                    why = "matched"
                    break
                if since >= fit_patience:
                    why = "plateau"
                    break
            print(f"  {r:>5} {ga_npv:>10.5f} {nn_npv:>10.5f} "
                  f"{(B-nn_npv)/B*100:>7.2f}% {float(l_rank):>8.4f} "
                  f"{float(l_dom):>8.4f} {gn:>9.2e} {ninv:>5} {t_ga:>6.1f} "
                  f"{used:>5} {why:<8}")

        q = _decode(s0, I)
        inv, _ = G.count_inversions(q, value, tau, I["keys"], n)
        gtxt = f"{ga_npv:+.5f} (gap {(B-ga_npv)/B*100:.2f}%)" if ga_best is not None else "none"
        print(f"  EPOCH {ep0+ep} SUMMARY  net {nn_npv:+.5f} "
              f"(gap {(B-nn_npv)/B*100:.2f}%)   GA {gtxt}   "
              f"vs heuristic {nn_npv - I['heur_npv']:+.5f}   inv {inv}   "
              f"{sequence_violations(q, I['par'], I['chi'])} viol")
        log.append({"ep": ep, "nn": nn_npv, "gap": (B - nn_npv) / B})
        save_checkpoint(net, opt, ep0 + ep, in_dim, tag=tag, best=best_seen)
        if eval_every and held and ep % eval_every == 0:
            hz = _zero_shot(net, held, proj)
            print(f"  held-out zero-shot mean gap "
                  f"{np.mean([h['nn_gap'] for h in hz])*100:6.2f}%   "
                  f"(cone-efficiency "
                  f"{np.mean([h['heur_gap'] for h in hz])*100:6.2f}%,  "
                  f"topological "
                  f"{np.mean([h['topo_gap'] for h in hz])*100:6.2f}%)")
    return {"log": log, "net": net, "held": held,
            "holdout": _zero_shot(net, held, proj) if held else []}


if __name__ == "__main__":
    args = _cli()
    if args.fresh:
        import os
        for suffix in ("latest", "best"):
            f = os.path.join(MODEL_DIR, f"{args.tag}_{suffix}.pt")
            if os.path.isfile(f):
                os.remove(f)
                print(f"  --fresh: removed {f}")
    if args.single is not None:
        train_w, held_w = [(0, args.single)], []
    else:
        train_w, held_w = make_windows(args.width, args.n_train, args.n_hold,
                                       stride=args.stride)
    print(f"train windows {train_w}")
    if held_w:
        print(f"held out      {held_w}")
    if not args.phased:
        main_interleaved(windows=train_w, holdout=held_w, epochs=args.epochs,
                         improvements=args.improvements, proj=args.proj,
                         cert_k=args.cert_k, w_npv=args.w_npv,
                         w_rank=args.w_rank, w_dom=args.w_dom,
                         eval_every=args.eval_every, tag=args.tag,
                         ga_cap=args.ga_cap,
                         npv_steps=args.npv_steps, margin=args.margin,
                         fit_steps=args.fit_steps,
                         fit_patience=args.fit_patience,
                         fit_w_npv=args.fit_w_npv)
        raise SystemExit
    main_multi(windows=train_w, holdout=held_w, epochs=args.epochs,
               ga_seconds=args.ga_seconds, proj=args.proj,
               cert_k=args.cert_k, w_npv=args.w_npv, w_rank=args.w_rank,
               w_dom=args.w_dom, eval_every=args.eval_every, tag=args.tag,
               step_every=args.step_every, npv_steps=args.npv_steps,
               sup_steps=args.sup_steps)
