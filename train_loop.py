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
from anchor_interpolation import interpolate_precedence_torch
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


def main(epochs=EPOCHS, crop=CROP, ga_seconds=GA_SECONDS, project=True):
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
          f"horizon {tau.sum():.2f} periods, device {dev}")
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
    gram = T.build_gram(st)
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
        s = net(x, pool=None).squeeze(-1).squeeze(0)
        if project:
            s = interpolate_precedence_torch(s, gram, par, chi)
        sig = ct.start_times_soft(s.unsqueeze(0), pr["tau"], value=pr["value"],
                                  window=pr["window"])
        npv = ct.npv_soft(sig, pr["tau"], pr["value"], psi=pr["psi"],
                          scale=pr["scale"]).sum()
        return s, npv

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
            s, npv = forward()
            (-npv).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP)
            opt.step()
            if a0 is None:
                a0 = float(npv)
        with torch.no_grad():
            s_now, npv_now = forward()
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
            s, npv = forward()
            sd = s.detach().std().clamp(min=1e-6)
            temp_rank = sd * TEMP_RANK
            temp_dom = sd * TEMP_DOM / n

            # ranking: sampled pairs, oriented by the teacher's order
            idx = torch.randint(0, n, (2, PAIRS * n), device=dev)
            u, v = idx[0], idx[1]
            first = torch.where(tt[u] < tt[v], u, v)
            second = torch.where(tt[u] < tt[v], v, u)
            keep = tt[u] != tt[v]
            wgt = (sens[first] + sens[second])[keep]
            l_rank = pairwise(s, first[keep], second[keep], temp_rank, wgt)

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
                l_dom = pairwise(s, tb, ta, temp_dom,
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
            s_now, npv_now = forward()
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
         project=(len(a) < 4 or a[3] != "noproj"))
