"""
Precedence by GP conditioning: the constraint as a noiseless observation.

The projection in kernel_projection.py asks "what is the nearest point of the
cone". This module asks a different question with the same kernel: given a prior
GP over the score field, what does the posterior look like once you are TOLD
that certain parent/child pairs are tied?

    prior        s ~ N(s0, K)          s0 = the raw (transformer) score
                                       K  = the sparse Wendland Gram
    observation  B_A s = 0             one row per tight edge, +1 at the parent
                                       and -1 at the child
    posterior    mean  s0 - K B_A' (B_A K B_A')^-1 B_A s0
                 cov   K - K B_A' (B_A K B_A')^-1 B_A K

This is the standard noiseless-linear-observation formula, and it is exact: the
observation is satisfied to solver precision in ONE linear solve rather than to
a tolerance after ~1300 relaxation sweeps.

What makes it interpolation rather than repair is the K in front of B_A'. The
correction is not applied to the two endpoints of a violated edge; it is applied
to K B_A' lam, which is the constraint residual pushed through the kernel. A
block that appears in no violated edge still moves, by exactly as much as the
kernel says it is correlated with the blocks that do. That is the same mechanism
by which a GP posterior updates away from its observations, and it is why the
cone-projection literature and the GP literature are describing one object:

    argmin (s - s0)' K^-1 (s - s0)  s.t.  B s >= 0
        ==  MAP of  N(s0, K)  conditioned on  B s >= 0

so project_qp is the mode of an inequality-truncated posterior, and what is
below is the mean of an equality-conditioned one. The equality set is not known
in advance, which is what the active-set loop in `condition_precedence` is for.

ORIENTATION. Feasibility is s[parent] >= s[child], because blocks are mined
highest-score-first (see kernel_projection's module docstring). An edge is
violated when a child outscores its parent, i.e. when the schedule would mine a
block before something that must come off above it.

A NOTE ON WHICH END MOVES. Conditioning on the DIFFERENCE functional being zero
does not say where the two scores meet. The posterior moves both endpoints
toward each other, weighted by their prior (co)variances -- the child falls and
the parent rises. If what is wanted is specifically "drag the child down to the
parent and leave the parent alone", that is a different observation: pin the
child's VALUE at the parent's prior score, which is `mode='lower'` below. The
two agree on the constraint and disagree on the correction; `equalize` is the
GP-natural one, `lower` is the one the hard clamp approximates.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg, splu

from kernel_projection import (project_hard_clamp, slack,
                               topological_order)


def gram_to_csr(gram):
    """(indptr, indices, data) -> scipy CSR. The Gram is symmetric PSD."""
    indptr, indices, data = gram
    n = int(indptr.shape[0]) - 1
    return sp.csr_matrix((np.asarray(data, dtype=np.float64),
                          np.asarray(indices), np.asarray(indptr)),
                         shape=(n, n))


def incidence(par, chi, active, n):
    """B_A: one row per active edge, +1 at the parent, -1 at the child."""
    a = np.asarray(active, dtype=np.int64)
    m = a.size
    if m == 0:
        return sp.csr_matrix((0, n))
    rows = np.repeat(np.arange(m, dtype=np.int64), 2)
    cols = np.empty(2 * m, dtype=np.int64)
    cols[0::2], cols[1::2] = par[a], chi[a]
    vals = np.empty(2 * m, dtype=np.float64)
    vals[0::2], vals[1::2] = 1.0, -1.0
    return sp.csr_matrix((vals, (rows, cols)), shape=(m, n))


def condition_equality(s0, K, B, jitter=1e-8, tol=1e-12, maxiter=5000,
                       direct_below=4000):
    """Posterior mean of N(s0, K) given the noiseless observation B s = 0.

        lam = (B K B')^-1 B s0
        s   = s0 - K B' lam

    B K B' is formed explicitly -- B has two nonzeros per row and K about
    fourteen per row, so the product stays sparse and a direct factorisation is
    affordable well past the sizes here. It is singular whenever the active
    edges are linearly dependent (three mutually tied blocks give a redundant
    row), which is common, so a jitter is added: that is a ridge on the
    observation, i.e. conditioning on B s = 0 with a tiny observation noise
    rather than none. The residual reported by the caller is what that costs.
    """
    if B.shape[0] == 0:
        return np.array(s0, dtype=np.float64, copy=True), {"lam": None,
                                                           "iters": 0}
    KBt = (K @ B.T).tocsc()
    M = (B @ KBt).tocsc()
    M = M + jitter * sp.eye(M.shape[0], format="csc")
    rhs = B @ s0

    if M.shape[0] <= direct_below:
        lam = splu(M).solve(rhs)
        iters = -1
    else:
        lam, info = cg(M, rhs, rtol=tol, maxiter=maxiter)
        iters = int(info)
    return s0 - KBt @ lam, {"lam": lam, "iters": iters}


def condition_lower(s0, K, par, chi, active, jitter=1e-8):
    """The asymmetric variant: pin each violated child's VALUE at the lowest
    prior score among its parents, and let the kernel interpolate the rest.

    Observation is A s = y with A a selector on the offending children, so only
    the child is targeted -- the parent is free to stay where it was. This is
    what "reduce the former to match the other" says literally; the equality
    version above instead lets them meet.
    """
    n = s0.shape[0]
    a = np.asarray(active, dtype=np.int64)
    kids = np.unique(chi[a])
    if kids.size == 0:
        return np.array(s0, dtype=np.float64, copy=True), {}
    # target: min over ALL parents of that child, in the prior score
    target = np.full(n, np.inf)
    np.minimum.at(target, chi, s0[par])
    y = target[kids]

    rows = np.arange(kids.size, dtype=np.int64)
    A = sp.csr_matrix((np.ones(kids.size), (rows, kids)), shape=(kids.size, n))
    KAt = (K @ A.T).tocsc()
    M = (A @ KAt).tocsc() + jitter * sp.eye(kids.size, format="csc")
    lam = splu(M).solve(A @ s0 - y)
    return s0 - KAt @ lam, {"n_pinned": int(kids.size)}


def condition_precedence(s0, gram, par, chi, mode="equalize", max_rounds=60,
                         tol=1e-9, dual_tol=1e-12, jitter=1e-10, verbose=False):
    """Enforce s[par] >= s[child] by conditioning on the edges that are TIGHT.

    Which edges are tight is not known in advance -- that is the entire content
    of an inequality constraint, and getting it wrong in either direction is
    fatal:

      too few    condition on nothing and the constraint is not enforced.
      too many   condition on every violated edge and the kernel correction
                 breaks neighbouring edges, which get added, which breaks more.
                 Measured on the transformer field: the working set snowballs
                 from 12,639 to 21,133 of 22,000 edges in three rounds and the
                 posterior collapses to a near-constant, 2,892 of 2,898 blocks
                 tied. Equality is a strong statement and most violated edges
                 do not deserve it -- they want to end up strictly satisfied.

    So the working set is grown AND shrunk. Writing the KKT conditions of

        min (s - s0)' K^-1 (s - s0)   s.t.   B s >= 0

    gives s* = s0 + K B' mu with mu >= 0, and `condition_equality` returns
    s = s0 - K B' lam, so mu = -lam. An edge held tight whose lam came out
    POSITIVE has a negative multiplier: the constraint is pushing the wrong way
    and would be strictly satisfied if released. Those get dropped. Iterating
    add-violated / drop-wrong-signed is the classical primal active-set method,
    and its fixed point satisfies primal feasibility, dual feasibility and
    complementary slackness simultaneously -- i.e. it is the MAP of the
    inequality-truncated posterior, reached through a sequence of equality
    conditionings.

    Cycling is possible in degenerate cases, as always with active sets. If a
    round makes no progress the drop is narrowed to the single worst edge,
    which is the standard remedy.

    Returns (s, info). `s` satisfies B s >= 0 up to `tol`; callers still need a
    topological tiebreak to get a permutation, because equality conditioning
    produces tied pairs BY DESIGN and a weak inequality cannot order them.
    """
    s0 = np.asarray(s0, dtype=np.float64)
    n = s0.shape[0]
    K = gram_to_csr(gram)

    if mode == "lower":
        active = np.flatnonzero(slack(s0, par, chi) < -tol)
        s, sub = condition_lower(s0, K, par, chi, active, jitter=jitter)
        g = slack(s, par, chi)
        return s, {"rounds": 1, "n_active": int(active.size),
                   "n_violated_start": int(active.size),
                   "n_violated_end": int((g < -tol).sum()),
                   "min_slack": float(g.min()), "history": [], "mode": mode}
    if mode != "equalize":
        raise ValueError("mode must be 'equalize' or 'lower'")

    n_start = int((slack(s0, par, chi) < -tol).sum())
    W = np.flatnonzero(slack(s0, par, chi) < -tol)
    s, hist, stalled = s0.copy(), [], 0

    for rounds in range(1, max_rounds + 1):
        B = incidence(par, chi, W, n)
        s, sub = condition_equality(s0, K, B, jitter=jitter)
        lam = sub["lam"]
        g = slack(s, par, chi)

        viol = np.flatnonzero(g < -tol)                 # primal: must be added
        add = np.setdiff1d(viol, W)
        # dual: mu = -lam >= 0, so lam > 0 means the edge is held wrongly
        drop = W[lam > dual_tol] if lam is not None else np.empty(0, np.int64)

        hist.append({"round": rounds, "n_active": int(W.size),
                     "n_add": int(add.size), "n_drop": int(drop.size),
                     "min_slack": float(g.min()),
                     "obs_residual": float(np.abs(B @ s).max()) if W.size else 0.0})
        if verbose:
            print(f"  round {rounds:>2}: |W|={W.size:>6,}  +{add.size:<6,} "
                  f"-{drop.size:<6,}  min slack {g.min():+.3e}")

        if add.size == 0 and drop.size == 0:
            break
        if add.size == 0 and drop.size:
            stalled += 1
            if stalled >= 3:            # narrow to the single worst multiplier
                drop = W[[int(np.argmax(lam))]]
        else:
            stalled = 0
        W = np.union1d(np.setdiff1d(W, drop), viol)

    g = slack(s, par, chi)
    return s, {"rounds": rounds, "n_active": int(W.size),
               "n_violated_start": n_start,
               "n_violated_end": int((g < -tol).sum()),
               "min_slack": float(g.min()), "history": hist, "mode": mode}


def condition_precedence_damped(s0, gram, par, chi, eta=0.3, max_iters=200,
                                tol=1e-9, jitter=1e-10, verbose=False):
    """Enforce s[par] >= s[child] by REPEATED PARTIAL equality conditioning.

    The alternative to deciding which edges are truly tight is to never commit
    to the decision. Each iteration conditions on whatever is violated RIGHT
    NOW, takes only an `eta` fraction of the resulting correction, and looks
    again:

        A     = {e : s[par_e] < s[chi_e]}
        s_c   = s - K B_A' (B_A K B_A')^-1 B_A s        full conditioning
        s     <- (1 - eta) s + eta s_c

    Why this avoids the collapse that add-only suffers: the working set is
    rebuilt from scratch every iteration rather than accumulated, so an edge
    that stops being violated simply stops being conditioned on. Nothing is
    ever asserted permanently, and the fixed point has every edge either
    satisfied strictly (never in A) or tight -- which is complementary
    slackness, arrived at without ever computing a multiplier.

    The prior is the CURRENT score, not s0. That is what makes it a relaxation
    of the constraint residual rather than a sequence of re-interpretations of
    the same prior: each step reduces B_A s toward zero by a factor of eta in
    the K metric.

    Relation to POCS: identical in spirit, different granularity. POCS visits
    one edge at a time with under-relaxation omega, so a sweep is m sequential
    rank-one updates and it takes ~1,100 of them. This solves ALL currently
    violated edges simultaneously and exactly, so an iteration is one sparse
    linear solve. Fewer, more expensive steps.

    eta = 1.0 recovers undamped simultaneous conditioning, which overshoots:
    correcting every violated edge at once moves the kernel neighbourhood of
    each, and the overshoot manifests as new violations elsewhere.
    """
    s0 = np.asarray(s0, dtype=np.float64)
    n = s0.shape[0]
    K = gram_to_csr(gram)
    s = s0.copy()
    hist = []
    n_start = int((slack(s0, par, chi) < -tol).sum())
    iters = 0

    for iters in range(1, max_iters + 1):
        g = slack(s, par, chi)
        active = np.flatnonzero(g < -tol)
        if active.size == 0:
            break
        B = incidence(par, chi, active, n)
        s_c, _ = condition_equality(s, K, B, jitter=jitter)
        s = (1.0 - eta) * s + eta * s_c
        hist.append({"iter": iters, "n_active": int(active.size),
                     "min_slack": float(g.min())})
        if verbose and (iters <= 5 or iters % 20 == 0):
            print(f"  iter {iters:>3}: |A|={active.size:>6,}  "
                  f"min slack {g.min():+.3e}")

    g = slack(s, par, chi)
    return s, {"iters": iters, "eta": eta, "n_violated_start": n_start,
               "n_violated_end": int((g < -tol).sum()),
               "min_slack": float(g.min()), "history": hist,
               "converged": bool((g < -tol).sum() == 0)}


def schedule_from_conditioned(s, par, chi, order=None, topo_rank=None,
                              return_snap=False):
    """Permutation from a conditioned score: mine highest first, ties by
    topological rank.

    Two safeguards, both load-bearing, and both the same ones
    kernel_projection.schedule_from_scores applies for the same reasons:

    SNAP. The conditioning solves a linear system, so the observation holds to
    solver precision and no better -- measured min slack -9.6e-10 on the
    transformer field. A sort reads the sign of -9.6e-10 literally and emits
    the child first, which turned 0 violated EDGES into 17,014 violated
    SEQUENCE positions before this pass was added. One topological clamp
    snaps them to exact equality; it moves the score by at most the solver
    residual.

    TIEBREAK. Equality conditioning creates tied pairs deliberately -- that is
    what the observation says. A weak inequality carries no information about
    the order within a tie, so the topological rank has to supply it. This is
    not rounding cleanup; it is structurally required.
    """
    s = np.asarray(s, dtype=np.float64)
    n = s.shape[0]
    if order is None:
        order = topological_order(n, par, chi)
    if topo_rank is None:
        topo_rank = np.empty(n, dtype=np.int64)
        topo_rank[order] = np.arange(n)
    s_snap = project_hard_clamp(s, par, chi, order=order)
    seq = np.lexsort((topo_rank, -s_snap))
    if return_snap:
        return seq, float(np.abs(s_snap - s).max())
    return seq
