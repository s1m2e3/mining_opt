"""
A schedule, described by ~k numbers instead of 12,213.

The GP in the solution-space loop needs a descriptor of a whole mining
schedule. A permutation will not do: we measured pairwise position correlation
between POCS-projected schedules at rho = 0.998, so a Kendall-tau kernel would
see every candidate as the same point. What actually differs between two plans
economically is WHEN each region gets mined, so the descriptor is

    phi(schedule)[c] = mean start period of the blocks in cluster c

with the clusters fixed per instance and the profile normalised to [0, 1].

Clusters are k-means over five standardised columns:

    x, y, z                geometry
    income_above           gross revenue in the FULL ancestor cone
    cost_above             mining + haulage + processing cost in that cone

The last two are the pushback idea at block level. Two blocks in the same place
can have very different stripping burdens above them -- one under barren
overburden, one under ore that pays for its own removal -- and they belong to
different pushbacks even though a purely spatial k-means would merge them.
Splitting on ancestor-cone economics is what makes a cluster mean something a
scheduler would recognise.

cost_above comes free. The cost deck makes value = income - cost blockwise, and
cone sums are linear, so cost_above = income_above - value_above with no extra
pass. The cones are FULL depth, not the truncated `cone_levels` used for the
scoring features -- "everything that must come off before this block" is only
meaningful all the way to the surface.

UNMINED BLOCKS. evaluate_cuts_multi leaves start_period = -1 for anything past
the horizon, and at the default knobs 6,898 of 12,213 blocks are unmined. Left
as -1 those would average in as if they were mined before period 0, which
inverts their meaning. They are encoded as T instead: unmined is the latest
possible outcome, not the earliest.

    python cluster_profile.py --crop 14 --k 16,32,64,128
"""

import argparse
import time

import numpy as np
from sklearn.cluster import KMeans

from block_lookahead import cone_sums
from mine_problem import T_PERIODS, evaluate, load_static


def P(*a):
    print(*a, flush=True)


def ancestor_cone_full(static):
    """Income and cost summed over every block above, to the surface.

    cone_sums truncates at `levels`; full depth is levels = nz - 1, which for
    this model is 22 and costs sum_l (2l+1)^2 ~ 16k vectorised gathers. Cached
    on the static dict because it depends on nothing the knobs can change.
    """
    if "_anc_full" in static:
        return static["_anc_full"]
    levels = max(int(static["nz"]) - 1, 1)
    t0 = time.perf_counter()
    la = cone_sums(static["ix"], static["iy"], static["iz"],
                   {"income": static["income"], "value": static["value"]},
                   levels=levels, direction="above")
    static["_anc_full"] = {
        "income_above": la["income"],
        # value = income - cost blockwise, and the sum is linear
        "cost_above": la["income"] - la["value"],
        "count_above": la["count"],
        "levels": levels,
        "seconds": time.perf_counter() - t0,
    }
    return static["_anc_full"]


def pushback_features(static, econ_weight=1.0, net_value=False):
    """(n, 5) standardised: x, y, z, income_above, cost_above.

    `net_value=True` collapses the two economic columns into one, income_above
    - cost_above, giving the 4-column variant. `econ_weight` scales the
    economic block against the spatial one after standardisation; 0 recovers a
    purely spatial k-means, which is the control.
    """
    anc = ancestor_cone_full(static)
    spatial = np.column_stack([static["x"], static["y"], static["z"]])
    if net_value:
        econ = (anc["income_above"] - anc["cost_above"])[:, None]
    else:
        econ = np.column_stack([anc["income_above"], anc["cost_above"]])

    def z(A):
        mu = A.mean(axis=0)
        sd = A.std(axis=0)
        return (A - mu) / np.where(sd > 0, sd, 1.0)

    return np.column_stack([z(spatial), float(econ_weight) * z(econ)])


def fit_clusters(static, k=64, seed=0, econ_weight=1.0, net_value=False):
    """k-means over the pushback features. Fit once per instance and cached."""
    key = f"_clusters_{k}_{econ_weight}_{net_value}_{seed}"
    if key in static:
        return static[key]
    F = pushback_features(static, econ_weight=econ_weight, net_value=net_value)
    km = KMeans(n_clusters=int(k), n_init=10, random_state=int(seed))
    labels = km.fit_predict(F).astype(np.int64)
    counts = np.bincount(labels, minlength=int(k))
    out = {"labels": labels, "k": int(k), "counts": counts,
           "features": F, "centers": km.cluster_centers_}
    static[key] = out
    return out


def profile(start_period, labels, k, t_periods=T_PERIODS):
    """Mean start period per cluster, normalised to [0, 1].

    Unmined blocks (start_period < 0) are encoded as t_periods -- past the
    horizon is the latest outcome, not the earliest.
    """
    p = np.asarray(start_period, dtype=float)
    p = np.where(p < 0, float(t_periods), p)
    sums = np.bincount(labels, weights=p, minlength=k)
    counts = np.bincount(labels, minlength=k).astype(float)
    return sums / np.maximum(counts, 1.0) / float(t_periods)


def score_profile(s, labels, k):
    """Cluster-mean of the RANK-normalised score. The cheap sibling of
    `profile`, and the difference decides whether a GP can screen at all.

    `profile` is computed from start_period, which only exists after the
    projection, the decode and the capacity cuts -- that is, after the whole
    1.7s evaluation. A GP over it therefore cannot filter candidates: by the
    time the descriptor is known, so is the NPV, and there is nothing left to
    predict. It can only serve as a target to aim at.

    This one is computed from the raw priority, before the projection, in
    microseconds. If it predicts NPV even moderately well, candidate screening
    becomes real: propose a thousand score fields, rank them with the GP, pay
    the 1.7s only for the handful worth evaluating.

    Ranks rather than raw values because only the ORDER of s affects the
    schedule -- any monotone rescaling decodes identically, so a mean of raw
    scores would vary with a quantity the objective cannot see.
    """
    s = np.asarray(s, dtype=float)
    r = np.empty(s.shape[0], dtype=float)
    r[np.argsort(-s, kind="stable")] = np.arange(s.shape[0])
    r /= max(s.shape[0] - 1, 1)
    sums = np.bincount(labels, weights=r, minlength=k)
    counts = np.bincount(labels, minlength=k).astype(float)
    return sums / np.maximum(counts, 1.0)


def within_cluster_r2(start_period, labels, k, t_periods=T_PERIODS):
    """How much of the per-block period variation the profile retains.

    1 - within/total. This is the number that picks k: the profile is a lossy
    summary, and if blocks inside a cluster scatter across many periods then
    two very different schedules can share a profile and the GP is modelling a
    quantity that does not determine NPV.
    """
    p = np.asarray(start_period, dtype=float)
    p = np.where(p < 0, float(t_periods), p)
    means = np.bincount(labels, weights=p, minlength=k) / np.maximum(
        np.bincount(labels, minlength=k), 1)
    within = float(np.mean((p - means[labels]) ** 2))
    total = float(p.var())
    return 1.0 - within / total if total > 0 else 0.0


def spatial_compactness(static, labels, k):
    """Mean within-cluster spatial spread, in block widths."""
    step = float(np.median(np.diff(np.unique(static["x"]))))
    xyz = np.column_stack([static["x"], static["y"], static["z"]])
    tot, n = 0.0, 0
    for c in range(k):
        m = labels == c
        if m.sum() < 2:
            continue
        d = xyz[m] - xyz[m].mean(axis=0)
        tot += float(np.sqrt((d ** 2).sum(axis=1)).mean()) * m.sum()
        n += int(m.sum())
    return tot / max(n, 1) / max(step, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", type=int, default=14)
    ap.add_argument("--k", default="16,32,64,128")
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--econ-weight", type=float, default=1.0)
    ap.add_argument("--source", default="gumbel", choices=("gumbel", "knobs"),
                    help="'knobs' replays BO-found settings, which span a far "
                         "wider NPV band than gumbel jitter around one setting")
    ap.add_argument("--knobs-json", default="outputs/bo_replicate.json")
    args = ap.parse_args()
    ks = [int(v) for v in args.k.split(",")]
    crop = args.crop if args.crop > 0 else None

    S = load_static(crop=crop)
    anc = ancestor_cone_full(S)
    P(f"n={S['n']:,}  ancestor cones to depth {anc['levels']} "
      f"in {anc['seconds']:.1f}s")
    P(f"income_above  min {anc['income_above'].min():,.0f}  "
      f"max {anc['income_above'].max():,.0f}")
    P(f"cost_above    min {anc['cost_above'].min():,.0f}  "
      f"max {anc['cost_above'].max():,.0f}")
    assert np.all(anc["cost_above"] >= -1e-6), "negative cost in ancestor cone"

    # a bag of schedules to test the descriptor against
    runs = []
    if args.source == "knobs":
        # Knob settings BO actually found, which span a far wider band than
        # gumbel noise around one setting: the transfer matrix ran -23% to
        # +23%. This is the range the descriptor will really operate over, so
        # a correlation measured on gumbel jitter alone would be optimistic
        # about a narrow neighbourhood and silent about everything else.
        import json
        rep = json.load(open(args.knobs_json))
        settings = [row["bo_knobs"] for runs_ in rep["A"].values() for row in runs_]
        settings += [v["knobs"] for v in rep["B"].values()]
        if "C_knobs" in rep:
            settings.append(rep["C_knobs"])
        settings = settings[:args.samples] if args.samples < len(settings) else settings
        P(f"\nevaluating {len(settings)} knob settings from {args.knobs_json}")
        for kn in settings:
            runs.append(evaluate(S, kn, transform="pocs", decoder="kahn",
                                 check=False))
    else:
        P(f"\ngenerating {args.samples} schedules (gumbel-perturbed priorities)")
        for i in range(args.samples):
            g = np.random.default_rng(500 + i).gumbel(size=S["n"])
            tau = (0.1, 0.3, 1.0)[i % 3]
            runs.append(evaluate(S, {}, transform="pocs", decoder="kahn",
                                 noise=tau * g, check=False))
    npvs = np.array([r["npv"] for r in runs])
    P(f"NPV spread {npvs.min():,.0f} .. {npvs.max():,.0f} "
      f"({(npvs.max()/npvs.min()-1)*100:.1f}% range)")

    # the degenerate baseline this descriptor has to beat
    pos = [r["pos"] for r in runs]
    rho = [np.corrcoef(pos[a], pos[b])[0, 1]
           for a in range(len(pos)) for b in range(a + 1, len(pos))]
    P(f"pairwise POSITION correlation: mean {np.mean(rho):.4f} "
      f"min {np.min(rho):.4f}   <- what a permutation kernel would see")

    def pair_corr(vecs, npvs):
        """corr(descriptor distance, |dNPV|) -- can a GP over this see NPV?"""
        d, dn = [], []
        for a in range(len(vecs)):
            for b in range(a + 1, len(vecs)):
                d.append(float(np.linalg.norm(vecs[a] - vecs[b])
                               / np.sqrt(vecs.shape[1])))
                dn.append(abs(npvs[a] - npvs[b]))
        d, dn = np.array(d), np.array(dn)
        if d.std() <= 0 or dn.std() <= 0:
            return float("nan"), float(d.mean())
        return float(np.corrcoef(d, dn)[0, 1]), float(d.mean())

    P(f"\n{'k':>5}{'sizes min/med/max':>20}{'compact':>9}{'period R2':>11}"
      f"{'d(period)':>11}{'corr PERIOD':>13}{'d(score)':>10}{'corr SCORE':>12}")
    P("-" * 91)
    for k in ks:
        cl = fit_clusters(S, k=k, econ_weight=args.econ_weight)
        lab, cnt = cl["labels"], cl["counts"]
        assert lab.shape[0] == S["n"]
        assert cnt.sum() == S["n"]
        assert (cnt > 0).all(), f"k={k} produced an empty cluster"

        r2 = float(np.mean([within_cluster_r2(r["start_period"], lab, k)
                            for r in runs]))
        comp = spatial_compactness(S, lab, k)

        phis = np.array([profile(r["start_period"], lab, k) for r in runs])
        assert phis.shape == (len(runs), k)
        assert np.all((phis >= 0) & (phis <= 1.0 + 1e-9)), "profile out of [0,1]"
        # the cheap descriptor: available from s_raw, before the projection
        psis = np.array([score_profile(r["s_raw"], lab, k) for r in runs])

        c_per, d_per = pair_corr(phis, npvs)
        c_sco, d_sco = pair_corr(psis, npvs)
        P(f"{k:>5}{f'{cnt.min()}/{int(np.median(cnt))}/{cnt.max()}':>20}"
          f"{comp:>8.1f}b{r2:>11.3f}{d_per:>11.4f}{c_per:>13.3f}"
          f"{d_sco:>10.4f}{c_sco:>12.3f}")

    # determinism: same schedule, same profile
    cl = fit_clusters(S, k=ks[0], econ_weight=args.econ_weight)
    p1 = profile(runs[0]["start_period"], cl["labels"], ks[0])
    p2 = profile(runs[0]["start_period"], cl["labels"], ks[0])
    assert np.array_equal(p1, p2), "profile is not deterministic"
    # responsiveness: different schedules, different profiles
    p3 = profile(runs[1]["start_period"], cl["labels"], ks[0])
    assert not np.allclose(p1, p3), "profile does not distinguish schedules"
    P("\nassertions passed: coverage, no empty clusters, range [0,1], "
      "determinism, responsiveness")


if __name__ == "__main__":
    main()
