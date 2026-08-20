"""
Exact period cuts for a FIXED mining order, by dynamic programming.

Given an order pi, the only remaining freedom is where to place the T-1 period
boundaries. The greedy fill in capacity_cuts packs each period until a resource
binds; that is not optimal, because with mixed-sign values you sometimes want to
DELAY material -- for v < 0, gamma^t * v is less negative later. Greedy always
pulls forward, which is right for ore and wrong for waste.

    V[j]     = prefix sum of value along pi
    S_r[j]   = prefix sum of resource r along pi
    jmax(i,t)= largest j with S_r[j] - S_r[i] <= C_r,t for every r

    f(i, T) = 0
    f(i, t) = max_{i <= j <= jmax(i,t)}  gamma^t (V[j] - V[i]) + f(j, t+1)

Answer f(0, 0); blocks past the final cut are unmined and contribute nothing.
jmax is nondecreasing in i, so all windows come from one two-pointer sweep per
period. Cost O(n*T*W) with W the blocks that fit in a period.

Greedy is one of the solutions the DP considers, so DP >= greedy always.

Caveat: cuts here are integral. The greedy evaluator splits the straddling
block pro rata, so a like-for-like comparison needs npv_of_cuts on both --
compare_greedy_dp does exactly that.
"""

import numpy as np

from capacity_cuts import evaluate_cuts_multi


def _dp_core(V, S, caps, gam_pow, n, R, T):
    NEG = -1.0e300
    f = np.full((T + 1, n + 1), NEG)
    arg = np.zeros((T + 1, n + 1), dtype=np.int64)
    f[T, :] = 0.0

    for t in range(T - 1, -1, -1):
        g = gam_pow[t]
        j = 0
        for i in range(n + 1):
            if j < i:
                j = i
            # extend the window while every resource still fits
            while j + 1 <= n:
                ok = True
                for r in range(R):
                    if S[r, j + 1] - S[r, i] > caps[r, t]:
                        ok = False
                        break
                if not ok:
                    break
                j += 1
            best = NEG
            bj = i
            for k in range(i, j + 1):
                cand = g * (V[k] - V[i]) + f[t + 1, k]
                if cand > best:
                    best = cand
                    bj = k
            f[t, i] = best
            arg[t, i] = bj
    return f, arg


try:                                            # pragma: no cover
    from numba import njit
    _dp_core_fast = njit(cache=True)(_dp_core)
    HAVE_NUMBA = True
except Exception:                               # pragma: no cover
    _dp_core_fast = None
    HAVE_NUMBA = False


def dp_cuts(seq, value, weights, caps, discount=0.90):
    """Optimal integral period boundaries for the order `seq`.

    seq      : (n,) int      mining order
    value    : (n,) float    per-block value, indexed by block id
    weights  : (R, n) float  per-resource consumption, by block id
    caps     : (R, T) float  per-resource per-period capacity

    Returns (npv, cuts, start_period) with cuts[t] the position where period t
    ends (0-based, exclusive) and start_period[-1] = -1 for unmined blocks.
    """
    seq = np.asarray(seq, dtype=np.int64)
    n = seq.shape[0]
    v = np.ascontiguousarray(np.asarray(value, dtype=np.float64)[seq])
    w = np.ascontiguousarray(np.asarray(weights, dtype=np.float64)[:, seq])
    caps = np.ascontiguousarray(np.asarray(caps, dtype=np.float64))
    R, T = caps.shape

    V = np.zeros(n + 1)
    np.cumsum(v, out=V[1:])
    S = np.zeros((R, n + 1))
    np.cumsum(w, axis=1, out=S[:, 1:])
    gam_pow = np.power(float(discount), np.arange(T, dtype=np.float64))

    fn = _dp_core_fast if HAVE_NUMBA else _dp_core
    f, arg = fn(V, S, caps, gam_pow, n, R, T)

    cuts = np.zeros(T, dtype=np.int64)
    # indexed by BLOCK ID, not by position: seq may be a subset of the blocks,
    # in which case seq[p] can exceed len(seq)
    start_period = np.full(np.asarray(value).shape[0], -1, dtype=np.int64)
    i = 0
    for t in range(T):
        j = int(arg[t, i])
        cuts[t] = j
        for p in range(i, j):
            start_period[seq[p]] = t
        i = j
    return float(f[0, 0]), cuts, start_period


def npv_of_cuts(seq, value, cuts, discount=0.90):
    """Value of an explicit integral cut vector, for like-for-like comparison."""
    seq = np.asarray(seq, dtype=np.int64)
    v = np.asarray(value, dtype=np.float64)[seq]
    V = np.concatenate([[0.0], np.cumsum(v)])
    total, i = 0.0, 0
    for t, j in enumerate(np.asarray(cuts, dtype=np.int64)):
        total += (discount ** t) * (V[j] - V[i])
        i = int(j)
    return float(total)


def greedy_cuts(seq, weights, caps):
    """Integral version of the fill-until-a-resource-binds rule, so the DP can
    be compared against greedy under identical (integral) accounting."""
    seq = np.asarray(seq, dtype=np.int64)
    w = np.asarray(weights, dtype=np.float64)[:, seq]
    caps = np.asarray(caps, dtype=np.float64)
    R, T = caps.shape
    n = seq.shape[0]
    cuts = np.zeros(T, dtype=np.int64)
    i = 0
    for t in range(T):
        used = np.zeros(R)
        j = i
        while j < n and np.all(used + w[:, j] <= caps[:, t]):
            used += w[:, j]
            j += 1
        cuts[t] = j
        i = j
    return cuts


def compare_greedy_dp(seq, value, weights, caps, resources, discount=0.90):
    """greedy (integral), DP (integral), and the fractional evaluator, together."""
    gc_ = greedy_cuts(seq, weights, caps)
    g_int = npv_of_cuts(seq, value, gc_, discount)
    dp_val, dp_c, _ = dp_cuts(seq, value, weights, caps, discount)
    frac = evaluate_cuts_multi(seq, resources, value=value, discount=discount)
    return {"greedy_integral": g_int, "dp_integral": dp_val,
            "greedy_fractional": frac["npv"],
            "lift_dp_vs_greedy": dp_val / g_int - 1.0 if g_int != 0 else float("nan"),
            "greedy_cuts": gc_, "dp_cuts": dp_c}
