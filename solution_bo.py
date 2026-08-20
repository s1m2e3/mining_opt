"""
Stage 3a: BO that screens candidate SCHEDULES, not knob settings.

The loop, all of it transformer-free so that it is the control the learned
proposer has to beat later:

    1. propose a pool of P candidate score fields          ~microseconds each
    2. describe each by phi_score = cluster-mean rank      ~microseconds each
    3. GP over (phi_score, NPV) from history -> logEI, take the top q
    4. evaluate those q for real                           ~0.3-1.7s each
    5. append and repeat

Step 1 is where the transformer goes in Stage 3b. Everything else stays.

Why this is worth building, from descriptor_test.py: a GP predicts held-out
NPV from phi_score at Spearman 0.730 against 0.603 from the 15 knobs, and
phi_score is available BEFORE the projection. That combination -- better
predictor, computable early -- is what makes screening real. A pool of 800
costs about as much as one true evaluation, so the GP gets to reject 792
candidates for free.

phi_period was the better predictor still (0.983) but is useless here: it needs
start_period, which only exists after the evaluation it was supposed to avoid.
It is the right descriptor for Variant 2 (regress a policy toward a target
profile) and the wrong one for screening.

THE CONTROL. The random arm draws q from the SAME pool with the same budget of
true evaluations. Only the selection rule differs, so the measured gap is the
value of screening and nothing else.

    python solution_bo.py --crop 14 --rounds 12 --pool 800 --q 4
"""

import argparse
import json
import time

import numpy as np
import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from torch.quasirandom import SobolEngine

from bo_search import DTYPE, KnobSpace, fit_gp
from capacity_cuts import evaluate_cuts_multi
from cluster_profile import fit_clusters, profile, score_profile
from mine_problem import (DISCOUNT, KNOBS, evaluate, knob_active, load_static,
                          priority_from_knobs)


def P(*a):
    print(*a, flush=True)


def propose_pool(static, space, size, seed, k, labels):
    """Cheap candidates: a knob draw, its score field, and its descriptor.

    No projection, no decode, no cuts. priority_from_knobs is a handful of
    vector ops over n and the cone aggregates are memoised, so a pool of
    hundreds costs less than a single true evaluation. This is the function the
    transformer replaces in 3b -- it needs only to emit a score field.
    """
    sob = SobolEngine(space.dim, scramble=True, seed=seed)
    U = sob.draw(size).numpy()
    knobs, psis = [], np.empty((size, k))
    for i, u in enumerate(U):
        kn = space.from_unit(u)
        s_raw = priority_from_knobs(static, kn)
        knobs.append(kn)
        psis[i] = score_profile(s_raw, labels, k)
    return U, knobs, psis


def true_eval(static, knobs, labels, k):
    r = evaluate(static, knobs, transform="pocs", decoder="kahn", check=False)
    return r["npv"], profile(r["start_period"], labels, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", type=int, default=14)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--pool", type=int, default=800)
    ap.add_argument("--q", type=int, default=4)
    ap.add_argument("--n-init", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="outputs/solution_bo.json")
    args = ap.parse_args()
    crop = args.crop if args.crop > 0 else None

    t_all = time.perf_counter()
    S = load_static(crop=crop)
    names = [kk for kk in KNOBS if knob_active(kk, "pocs")]
    space = KnobSpace(names)
    labels = fit_clusters(S, k=args.k)["labels"]
    budget = args.rounds * args.q
    P(f"n={S['n']:,}  k={args.k}  pool={args.pool}  q={args.q}  "
      f"rounds={args.rounds}  -> {budget} true evaluations per arm")
    P(f"seeds={args.seeds}\n")

    report = {"crop": args.crop, "k": args.k, "pool": args.pool, "q": args.q,
              "rounds": args.rounds, "budget": budget, "runs": []}
    gaps = []

    for seed in range(args.seeds):
        torch.manual_seed(seed)
        # shared start: same initial true evaluations for both arms
        _, init_knobs, init_psi = propose_pool(S, space, args.n_init,
                                               seed=1000 + seed, k=args.k,
                                               labels=labels)
        init_y = []
        for kn in init_knobs:
            y, _ = true_eval(S, kn, labels, args.k)
            init_y.append(y)
        base = max(init_y)

        arms = {}
        for arm in ("bo", "random"):
            X = [p for p in init_psi]
            Y = list(init_y)
            rng = np.random.default_rng(7000 + seed)
            t0 = time.perf_counter()
            for rnd in range(args.rounds):
                _, pool_knobs, pool_psi = propose_pool(
                    S, space, args.pool, seed=seed * 1000 + rnd, k=args.k,
                    labels=labels)
                if arm == "bo":
                    gp, _, Yt = fit_gp(X, Y)
                    acq = LogExpectedImprovement(model=gp, best_f=Yt.max())
                    with torch.no_grad():
                        # discrete acquisition maximisation: the candidates are
                        # given, so this is a sort, not an optimisation
                        vals = acq(torch.tensor(pool_psi, dtype=DTYPE
                                                ).unsqueeze(1)).numpy()
                    pick = np.argsort(-vals)[:args.q]
                else:
                    pick = rng.choice(len(pool_knobs), size=args.q, replace=False)
                for i in pick:
                    y, _ = true_eval(S, pool_knobs[i], labels, args.k)
                    X.append(pool_psi[i])
                    Y.append(y)
            arms[arm] = {"best": float(max(Y)), "trace": [float(v) for v in Y],
                         "seconds": time.perf_counter() - t0}

        gap = (arms["bo"]["best"] / arms["random"]["best"] - 1.0) * 100
        gaps.append(gap)
        report["runs"].append({"seed": seed, "init_best": float(base),
                               "bo": arms["bo"]["best"],
                               "random": arms["random"]["best"],
                               "gap_pct": gap,
                               "bo_seconds": arms["bo"]["seconds"],
                               "random_seconds": arms["random"]["seconds"]})
        P(f"seed {seed}:  init {base:>15,.0f}   BO {arms['bo']['best']:>15,.0f}"
          f"   random {arms['random']['best']:>15,.0f}   gap {gap:>+6.2f}%"
          f"   ({arms['bo']['seconds']:.0f}s / {arms['random']['seconds']:.0f}s)")

    gaps = np.asarray(gaps)
    report["summary"] = {"mean_gap_pct": float(gaps.mean()),
                         "sd_gap_pct": float(gaps.std(ddof=1)) if len(gaps) > 1 else 0.0,
                         "wins": int((gaps > 0).sum()), "n": int(gaps.size)}
    P(f"\nBO screening - random, over {gaps.size} seeds: "
      f"mean {gaps.mean():+.2f}%  "
      f"sd {gaps.std(ddof=1) if len(gaps) > 1 else 0:.2f}%  "
      f"wins {int((gaps > 0).sum())}/{gaps.size}")

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    P(f"wrote {args.out}   total {time.perf_counter()-t_all:.0f}s")


if __name__ == "__main__":
    main()
