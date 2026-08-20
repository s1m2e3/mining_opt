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


def build(n, par, chi, tonnage, value, capacity, T, discount):
    """The LP. Cumulative variables y[i][k] = fraction of i mined by end of k.

    Cumulative rather than per-period variables because precedence is then a
    two-term row, y_i_k <= y_j_k, instead of a row summing 2(k+1) terms. With
    22,000 edges over 10 periods that is 220,000 rows either way, but building
    them is the difference between seconds and minutes.
    """
    model = mathopt.Model(name="cpit_lp_bound")
    gam = float(discount)

    y = np.empty((n, T), dtype=object)
    for i in range(n):
        for k in range(T):
            y[i, k] = model.add_variable(lb=0.0, ub=1.0, name=f"y_{i}_{k}")

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
        model.add_linear_constraint(expr <= float(capacity))

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


def main(crop=14, seconds=1800, verbose=False):
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
    print(f"  LP size: {n * T_PERIODS:,} variables, "
          f"{len(par) * T_PERIODS + n * (T_PERIODS - 1) + n + T_PERIODS:,} rows")

    t0 = time.perf_counter()
    model, _ = build(n, par, chi, tonnage, value, capacity, T_PERIODS, DISCOUNT)
    t_build = time.perf_counter() - t0
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
            "build_s": t_build, "solve_s": t_solve}


if __name__ == "__main__":
    a = sys.argv[1:]
    crop = None if (a and a[0] == "full") else (int(a[0]) if a else 14)
    main(crop=crop, verbose=len(a) > 1)
