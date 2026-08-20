"""LP dual bound on the continuous-NPV schedule: how far from optimal are we?

The value-density schedule is a valid upper bound but a useless one -- it
throws away precedence, which on this deposit is doing most of the work, and
reads 0.870 against a 0.535 topological baseline. This is the real instrument.

THE MODEL is the LP relaxation of the CPIT formulation already in
problems.LowerScheduleMIP: a per-period fraction for each block, precedence as
cumulative sums, a per-period tonnage capacity and beta^k discounting. It is
built here from mine_problem's arrays rather than by reusing that class,
because three details have to differ for the bound to be VALID for our
continuous problem rather than merely similar:

  non-strict precedence   LowerScheduleMIP requires a predecessor in a STRICTLY
                          earlier period. Continuous time lets a parent and its
                          child both be mined inside one period, so that
                          version excludes schedules we can actually run, and a
                          bound that excludes feasible points is not a bound.
                          Here it is sum_{t<=k} x_i <= sum_{t<=k} x_j.
  mine-everything         LowerScheduleMIP allows a block to go unmined
                          (sum_k x <= 1), which is a relaxation and only
                          loosens the bound. We mine the lot, so this is an
                          equality.
  sign-aware discounting  a block mined during period k starts somewhere in
                          [k, k+1), so exp(-delta sigma) lies in
                          (gamma^(k+1), gamma^k]. For a POSITIVE block the
                          largest possible contribution uses gamma^k; for a
                          NEGATIVE one the least negative uses gamma^(k+1).
                          Using gamma^k for everything, as the MIP does, is
                          wrong in the negative case -- it charges waste more
                          than any real schedule would and quietly turns the
                          upper bound into something that is not one.

WHY IT IS A BOUND. Take any continuous feasible schedule and set x[i][k] to the
fraction of block i mined during period k. Then sum_k x[i][k] = 1; the tonnage
mined in period k is exactly the capacity, so the capacity rows hold; block i
starts only once its predecessors are finished, so the cumulative precedence
rows hold; and the sign-aware coefficients dominate that schedule's own
discount block by block. So every continuous schedule maps to an LP-feasible
point whose LP objective is at least its continuous NPV, and the LP optimum is
therefore an upper bound on the continuous optimum. Relaxing the integrality
only raises it further.

The gap this reports is a genuine optimality gap: (bound - achieved) / bound.

Run: python lp_bound.py [crop]        (crop 14 by default, "full" for all blocks)
"""

import sys
import time

import numpy as np
from ortools.math_opt.python import mathopt

import continuous_time as ct

DISCOUNT = 0.90
T_PERIODS = 10


def reachability_bounds(n, par, chi, order, tonnage, capacity, horizon):
    """Earliest start and latest finish for every block, in periods.

    Before any of block j can be mined, ALL of its transitive ancestors must be
    finished, which takes their tonnage divided by the capacity rate. After j is
    finished, all of its transitive descendants still have to be mined. So

        e_j = W_ancestors(j) / C          l_j = horizon - W_descendants(j) / C

    and every feasible schedule obeys both. The LP does NOT imply these: with
    fractional variables it can mine a sliver of every block at once, consuming
    almost no capacity, and so never pays for the overburden it has to move
    first. Fixing y accordingly is a genuine strengthening, not a restatement,
    and on crop-6 it pins 67% of the variables.

    Dense boolean reachability, n^2 bits. Fine at LP scale (8 MB at n=2898,
    149 MB at n=12213); a bitset would be needed well beyond that.
    """
    w = np.asarray(tonnage, dtype=float)
    par, chi = np.asarray(par), np.asarray(chi)
    pre = [[] for _ in range(n)]
    suc = [[] for _ in range(n)]
    for a, b in zip(par.tolist(), chi.tolist()):
        pre[b].append(a)
        suc[a].append(b)

    anc = np.zeros((n, n), dtype=bool)
    for j in order:
        for a in pre[j]:
            anc[j] |= anc[a]
            anc[j, a] = True
    des = np.zeros((n, n), dtype=bool)
    for j in order[::-1]:
        for b in suc[j]:
            des[j] |= des[b]
            des[j, b] = True
    e = (anc @ w) / capacity
    l = horizon - (des @ w) / capacity
    return e, l


def build(n, par, chi, tonnage, value, capacity, T, discount,
          subperiods=1, earliest=None, latest=None):
    """The LP. Cumulative variables y[i][k] = fraction of i mined by end of k.

    Cumulative rather than per-period variables because precedence is then a
    two-term row, y_i_k <= y_j_k, instead of a row summing 2(k+1) terms. With
    22,000 edges over 10 periods that is 220,000 rows either way, but building
    them is the difference between seconds and minutes.
    """
    model = mathopt.Model(name="cpit_lp_bound")
    K = int(subperiods)
    TK = T * K
    # a finer grid shrinks the discretisation slack: a block charged gamma^(k/K)
    # actually starts somewhere inside a sub-period of length 1/K, so the
    # overcharge falls from gamma^-1 = 1.111 to gamma^(-1/K)
    gam = float(discount) ** (1.0 / K)
    cap_k = float(capacity) / K

    lo_fix = np.zeros(n, dtype=np.int64)
    hi_fix = np.full(n, TK, dtype=np.int64)
    if earliest is not None:
        # y[i,k] = 0 while the whole sub-period ends before the earliest start
        lo_fix = np.clip(np.floor(np.asarray(earliest) * K).astype(np.int64),
                         0, TK)
    if latest is not None:
        hi_fix = np.clip(np.ceil(np.asarray(latest) * K).astype(np.int64) - 1,
                         0, TK)

    y = np.empty((n, TK), dtype=object)
    fixed = 0
    for i in range(n):
        for k in range(TK):
            if k < lo_fix[i]:
                lb = ub = 0.0
                fixed += 1
            elif k >= hi_fix[i]:
                lb = ub = 1.0
                fixed += 1
            else:
                lb, ub = 0.0, 1.0
            y[i, k] = model.add_variable(lb=lb, ub=ub, name=f"y_{i}_{k}")
    build.fixed = fixed
    T = TK

    # every block is mined inside the horizon
    for i in range(n):
        model.add_linear_constraint(y[i, T - 1] == 1.0)

    # monotone: the per-period fraction x = y_k - y_{k-1} is non-negative
    for i in range(n):
        for k in range(1, T):
            model.add_linear_constraint(y[i, k - 1] <= y[i, k])

    # precedence, NON-STRICT: parent and child may share a period
    for p, c in zip(par.tolist(), chi.tolist()):
        for k in range(T):
            model.add_linear_constraint(y[c, k] <= y[p, k])

    # per-period mining capacity
    w = np.asarray(tonnage, dtype=float)
    for k in range(T):
        expr = sum(float(w[i]) * (y[i, k] - (y[i, k - 1] if k else 0.0))
                   for i in range(n))
        model.add_linear_constraint(expr <= cap_k)

    # sign-aware discount: gamma^k for value we want early, gamma^(k+1) for
    # cost we would rather defer -- each is the best a real schedule could do
    v = np.asarray(value, dtype=float)
    obj = 0.0
    for i in range(n):
        d = [gam ** k if v[i] > 0 else gam ** (k + 1) for k in range(T)]
        for k in range(T):
            xk = y[i, k] - (y[i, k - 1] if k else 0.0)
            obj += float(v[i] * d[k]) * xk
    model.maximize(obj)
    return model, y


def solve(model, seconds=1800, solver=mathopt.SolverType.HIGHS, verbose=False):
    params = mathopt.SolveParameters()
    params.time_limit = __import__("datetime").timedelta(seconds=seconds)
    params.enable_output = verbose
    res = mathopt.solve(model, solver_type=solver, params=params)
    return res


def targets(crop=14, subperiods=2, tighten=True, seconds=1800, t_periods=None,
            cache=True):
    """Solve the LP once and return what training can actually use.

    NOTE ON ORDERING: the LP sees only the INSTANCE -- blocks, tonnage, value,
    precedence, capacity, discount. Nothing about the network enters it, so it
    is solved once, offline, and everything below is a constant for the whole
    training run. It is not estimated from the pipeline's output.

    Returns
    -------
    bound      LP optimum in currency. As a loss denominator this makes the
               objective the fraction of the ACHIEVABLE value captured, so 1.0
               is provably optimal, where sum |v| only ever gave the fraction
               of gross value -- a quantity that says more about how much waste
               a crop happens to contain than about scheduling quality.
    sigma_lp   (n,) expected start time per block, in periods, from the primal.
               A relaxed-optimal profile, so it is INFEASIBLE: use it as a
               ranking target behind the projection, never as the sole target.
    """
    import ga_schedule as G
    T = t_periods or T_PERIODS
    P = G.load_instance(crop=crop, t_periods=T)
    n, par, chi = P["n"], P["par"], P["chi"]
    tonnage, value = P["static"]["tonnage"], P["value"]
    capacity = tonnage.sum() / T

    e = l = None
    if tighten:
        e, l = reachability_bounds(n, par, chi, P["order"], tonnage, capacity,
                                   float(T))
    model, y = build(n, par, chi, tonnage, value, capacity, T, DISCOUNT,
                     subperiods=subperiods, earliest=e, latest=l)
    res = solve(model, seconds=seconds)
    if not getattr(res, "solutions", None):
        raise RuntimeError(f"LP did not solve: {res.termination.reason}")
    val = res.solutions[0].primal_solution.variable_values
    TK = T * subperiods

    Y = np.empty((n, TK))
    for i in range(n):
        for k in range(TK):
            Y[i, k] = val[y[i, k]]
    X = np.diff(Y, axis=1, prepend=0.0)              # fraction mined per slot
    X = np.clip(X, 0.0, None)
    mass = X.sum(axis=1, keepdims=True)
    grid = np.arange(TK) / float(subperiods)         # slot start, in periods
    sigma_lp = (X * grid).sum(axis=1) / np.where(mass[:, 0] > 0, mass[:, 0], 1.0)

    return {"bound": float(res.objective_value()), "sigma_lp": sigma_lp,
            "n": n, "crop": crop, "subperiods": subperiods,
            "scale_sumabs": float(ct.npv_scale(value))}


def main(crop=14, seconds=1800, verbose=False, subperiods=1, tighten=True):
    import ga_schedule as G

    P = G.load_instance(crop=crop, t_periods=T_PERIODS)
    n, tau, value, scale = P["n"], P["tau"], P["value"], P["scale"]
    par, chi, tonnage = P["par"], P["chi"], P["static"]["tonnage"]
    capacity = tonnage.sum() / T_PERIODS

    def sc(seq):
        return float(ct.npv(ct.start_times_from_order(seq, tau), tau, value,
                            discount=DISCOUNT, scale=scale))

    base = sc(P["order"])
    density = float(ct.npv(ct.start_times(value / tau, tau, value=value), tau,
                           value, discount=DISCOUNT, scale=scale))

    print(f"crop-{crop}: n={n} blocks, {len(par)} edges, T={T_PERIODS} periods")

    e = l = None
    if tighten:
        e, l = reachability_bounds(n, par, chi, P["order"], tonnage, capacity,
                                   float(T_PERIODS))
    t0 = time.perf_counter()
    model, _ = build(n, par, chi, tonnage, value, capacity, T_PERIODS, DISCOUNT,
                     subperiods=subperiods, earliest=e, latest=l)
    t_build = time.perf_counter() - t0
    nv = n * T_PERIODS * subperiods
    print(f"  K={subperiods} sub-periods, tighten={tighten}: "
          f"{nv:,} variables, {build.fixed:,} fixed ({build.fixed/nv:.0%})")
    print(f"  built in {t_build:.1f}s, solving ...")

    t0 = time.perf_counter()
    res = solve(model, seconds=seconds, verbose=verbose)
    t_solve = time.perf_counter() - t0
    print(f"  {res.termination.reason} in {t_solve:.1f}s")
    if not getattr(res, "solutions", None):
        print("  no solution returned")
        return None
    bound = res.objective_value() / scale

    print(f"\n  LP DUAL BOUND            {bound:+.5f}")
    print(f"  value-density bound      {density:+.5f}   "
          f"({(density - bound) / abs(bound) * 100:+.1f}% looser)")
    print(f"  topological baseline     {base:+.5f}   "
          f"gap {(bound - base) / abs(bound) * 100:5.1f}%")

    # the bound must dominate anything we can actually build
    assert bound >= base - 1e-9, "LP bound below a feasible schedule -- invalid"
    return {"bound": float(bound), "topological": base, "density": density,
            "build_s": t_build, "solve_s": t_solve, "fixed": build.fixed,
            "subperiods": subperiods, "tighten": tighten}


if __name__ == "__main__":
    a = sys.argv[1:]
    crop = None if (a and a[0] == "full") else (int(a[0]) if a else 14)
    main(crop=crop, verbose=len(a) > 1)
