"""An optimality certificate for a schedule, without solving an LP.

Weak duality says ANY dual-feasible point bounds the optimum. Optimality is
what makes a bound tight, not what makes it valid -- so a certificate can be
manufactured from a schedule we already have, in closed form, and refined by
subgradient steps. Nothing here calls a solver.

THE DUAL. For the CPIT formulation in lp_bound (x_ik = fraction of block i
mined in period k, sign-aware coefficients c_ik):

    max  sum_ik c_ik x_ik
    s.t. sum_k x_ik = 1                              all i        alpha_i free
         sum_{t<=k} (x_jt - x_it) <= 0               edge i->j, k mu_ijk >= 0
         sum_i w_i x_ik <= C                         all k        lambda_k >= 0

the dual is  min sum_i alpha_i + C sum_k lambda_k  subject to

    alpha_i + w_i lambda_k + G_i(k) >= c_ik          for every i, k

    G_i(k) = sum_{p->i} sum_{k'>=k} mu_pik'  -  sum_{i->j} sum_{k'>=k} mu_ijk'

alpha is FREE and separable, so for any lambda, mu >= 0 the choice

    alpha_i = max_k ( c_ik - w_i lambda_k - G_i(k) )

is dual-feasible by construction and

    UB(lambda, mu) = sum_i alpha_i + C sum_k lambda_k

is a valid upper bound. Cost O(mT + nT) -- 176k operations on crop-6.

THE PER-BLOCK GAP. Let k_i be where the schedule actually mines block i. The
amount alpha_i must exceed its complementary-slackness value is

    g_i = max_k(c_ik - w_i lam_k - G_i(k)) - (c_ik_i - w_i lam_k_i - G_i(k_i))

which is >= 0, is in currency, and sums to the bulk of the optimality gap. It
says WHICH blocks are misplaced and by how much, and argmax_k gives each one a
target period -- a supervised signal derived from the schedule itself.

THE SUBGRADIENTS both read off the argmax solution k*, which is the relaxed
problem's own answer:

    d UB / d lambda_k = C - sum_i w_i [k*_i = k]      capacity slack
    d UB / d mu_ijk   = [k*_i <= k] - [k*_j <= k]     minus the violation

so lambda rises where the relaxed solution overruns capacity and mu rises where
it puts a child before its parent. Polyak steps, using the primal NPV as the
known lower bound.

WHAT CAN GO WRONG: validity is free, usefulness is not. Poor multipliers give a
true but vacuous bound. `certify` reports the bound against the primal so that
is visible rather than assumed.

THE PRIMAL CONTRIBUTES NOTHING TO THE DUAL. This was tested directly and the
answer is no, three ways:

  the duals are instance-determined   edge duals computed from a GA schedule
                                      correlate 0.9476 with those from the
                                      topological one, two primals 4 NPV points
                                      apart.
  per-block attribution is thin       g_i captures 3-4% of the gap, unchanged
                                      by sub-period refinement.
  primal-informed starts HURT         on crop-6 at 4000 iterations, cold mu = 0
                                      reaches 0.66336; restricting mu to the
                                      primal's tight set gives 0.68104; adding
                                      a pressure-based magnitude gives 0.70726.
                                      Cold wins at 100, 400, 1000 and 4000.

The reason is that complementary slackness pairs an OPTIMAL primal with an
OPTIMAL dual. Our schedules are feasible but ~12% off, and integral where the
LP optimum is fractional, so their tight set is not the LP's: 95.8% of
precedence rows are tight in our schedule, and the 4.2% that are slack are
exactly the rows the LP wants a positive mu on. Masking them forbids mu where
it is needed. Weak duality bounds the OPTIMUM, not the solution handed in.

The practical consequence is good news for cost rather than bad: since the dual
is a property of the instance, it is computed ONCE per instance and cached, not
once per training step. And a schedule-specific learning signal has to come
from a better PRIMAL -- a search -- not from duals.
"""

import numpy as np

import continuous_time as ct

DISCOUNT = 0.90
T_PERIODS = 10


def coefficients(value, discount, T):
    """c_ik: the best a real schedule could do for block i mined in period k.

    A block mined during period k starts in [k, k+1), so exp(-delta sigma) lies
    in (gamma^(k+1), gamma^k]. Positive value takes gamma^k, negative takes
    gamma^(k+1) -- the same sign-aware argument that makes lp_bound valid.
    """
    v = np.asarray(value, dtype=float)
    g = np.power(float(discount), np.arange(T + 1, dtype=float))
    return np.where(v[:, None] > 0, v[:, None] * g[None, :T],
                    v[:, None] * g[None, 1:T + 1])


def _G(mu_suffix, par, chi, n, T):
    """G_i(k) from the per-edge suffix sums of mu."""
    G = np.zeros((n, T))
    for k in range(T):
        G[:, k] = (np.bincount(chi, weights=mu_suffix[:, k], minlength=n)
                   - np.bincount(par, weights=mu_suffix[:, k], minlength=n))
    return G


def bound(c, w, C, lam, mu, par, chi, n, T, allowed=None):
    """UB and the argmax assignment, for any lambda, mu >= 0.

    `allowed` is a boolean (n, T) mask of the periods each block may occupy,
    from reachability: a block cannot start before its ancestors are dug out,
    nor finish so late that its descendants no longer fit. Ruling those periods
    out only shrinks the primal's feasible set, and every real schedule still
    lies inside it, so the bound stays valid and gets tighter -- the relaxed
    solution can no longer park a deep block in period 0 while the multipliers
    are still catching up.
    """
    mu_suffix = np.cumsum(mu[:, ::-1], axis=1)[:, ::-1] if mu is not None \
        else np.zeros((par.size, T))
    A = c - w[:, None] * lam[None, :] - _G(mu_suffix, par, chi, n, T)
    if allowed is not None:
        A = np.where(allowed, A, -np.inf)
    kstar = np.argmax(A, axis=1)
    alpha = A[np.arange(n), kstar]
    return float(alpha.sum() + C * lam.sum()), kstar, A, alpha


def certify(seq, tau, value, tonnage, par, chi, T=T_PERIODS,
            discount=DISCOUNT, iters=300, rho=1.5, verbose=False,
            subperiods=1, earliest=None, latest=None):
    """Certified upper bound for a feasible schedule, by dual subgradient.

    `seq` is a mining sequence. Returns the bound, the primal value, the gap,
    and the per-block gap attribution g_i with its target period.
    """
    n = tau.shape[0]
    par, chi = np.asarray(par), np.asarray(chi)
    w = np.asarray(tonnage, dtype=float)
    K = int(subperiods)
    horizon_C = w.sum() / T
    # a finer grid shrinks the slack between c_ik and the true continuous
    # discount, which is otherwise charged to the gap and looks like
    # suboptimality the schedule cannot actually fix
    TS = T * K
    C = horizon_C / K
    c = coefficients(value, float(discount) ** (1.0 / K), TS)

    sigma = ct.start_times_from_order(seq, tau)
    primal = float(ct.npv(sigma, tau, value, discount=discount))
    k_prim = np.clip(np.floor(sigma * K).astype(int), 0, TS - 1)
    T = TS

    allowed = None
    if earliest is not None and latest is not None:
        kk = np.arange(T)[None, :]
        lo = np.floor(np.asarray(earliest) * K)[:, None]
        hi = np.maximum(np.ceil(np.asarray(latest) * K)[:, None] - 1, lo)
        allowed = (kk >= lo) & (kk <= hi)
        allowed |= ~allowed.any(axis=1, keepdims=True)

    lam = np.zeros(T)
    mu = np.zeros((par.size, T))
    best = np.inf
    best_state = None
    for t in range(iters):
        ub, kstar, A, alpha = bound(c, w, C, lam, mu, par, chi, n, T,
                                    allowed=allowed)
        if ub < best:
            best, best_state = ub, (lam.copy(), mu.copy(), kstar.copy(),
                                    A.copy(), alpha.copy())
        # subgradients, read off the relaxed solution k*
        usage = np.bincount(kstar, weights=w, minlength=T)
        g_lam = C - usage
        below = (kstar[:, None] <= np.arange(T)[None, :])
        g_mu = below[chi].astype(float) - below[par].astype(float)
        # PER-BLOCK Polyak steps. A single shared step cannot move both: g_lam
        # is in tonnes (~1e7 per entry) while g_mu is a 0/+-1 indicator, so one
        # step size sized for capacity leaves mu at zero, precedence never gets
        # priced, and the bound collapses to the no-precedence relaxation. That
        # is exactly what a shared step produced -- a true but useless bound.
        gap = max(ub - primal, 1e-9)
        decay = 1.0 / (1.0 + t / 50.0)
        n_lam = float((g_lam ** 2).sum())
        n_mu = float((g_mu ** 2).sum())
        if n_lam <= 0 and n_mu <= 0:
            break
        if n_lam > 0:
            lam = np.maximum(0.0, lam - (rho * decay * gap / n_lam) * g_lam)
        if n_mu > 0:
            mu = np.maximum(0.0, mu + (rho * decay * gap / n_mu) * g_mu)
        if verbose and t % max(1, iters // 10) == 0:
            print(f"    it {t:>4}  UB {ub:>14,.0f}  gap "
                  f"{(ub - primal) / abs(ub) * 100:6.2f}%  "
                  f"|viol| {np.abs(g_mu).sum():>8.0f}")

    lam, mu, kstar, A, alpha = best_state
    g_block = alpha - A[np.arange(n), k_prim]
    return {"bound": best, "primal": primal,
            "gap": (best - primal) / abs(best) if best != 0 else np.nan,
            "lam": lam, "kstar": kstar, "k_primal": k_prim,
            "g_block": g_block, "target_period": kstar, "mu": mu}
