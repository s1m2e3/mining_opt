"""
Precedence by dynamic anchor interpolation.

Not a projection and not a clamp. The constraint is enforced by repeatedly
asking a GP the ordinary interpolation question -- "given corrections observed
at these points, what is the correction everywhere?" -- where the observation
points, the observed values AND the kernel itself are all recomputed from the
current state each round.

Per iteration, with s the current score field:

    r_i   = min_{p in pa(i)} s_p - s_i          worst offending precedent.
                                                r_i < 0 <=> block i outranks a
                                                block that must precede it.
    c_i   = 1[r_i < 0]                          the constraint-satisfaction
                                                feature: does this block respect
                                                precedence, yes or no.
    A     = {i : c_i = 1}                       anchor points -- the offenders.
    K     = k([z_i ; lam c_i], [z_j ; lam c_j]) kernel over features AUGMENTED
                                                with the satisfaction state.
    dhat  = K[:,A] (K[A,A] + sig2 I)^-1 r_A     posterior mean of the correction
    s    <- s + eta * dhat

and repeat until A is empty.

Three things are dynamic, which is the whole point:

  the anchor SET      a block stops being an anchor the moment it complies, so
                      nothing is ever permanently asserted about it.
  the anchor VALUES   r_A is recomputed from the current scores, so the size of
                      the correction shrinks as the field approaches feasibility
                      instead of being fixed at the first estimate.
  the KERNEL          c enters the feature vector, so two blocks are near each
                      other only if they are alike geometrically AND in the same
                      constraint state. A correction therefore spreads among
                      fellow offenders and is damped across the boundary into
                      already-satisfied ground, which is what stops the fix from
                      knocking compliant blocks back out.

r is signed and the correction is applied as it comes; there is no raise-only or
lower-only rule and no clamp anywhere. Feasibility is the fixed point.

WHY c AUGMENTS THE FEATURES RATHER THAN REPLACING THEM. If K depended on c
alone, every anchor would sit at distance zero from every other, K[A,A] would be
the all-ones matrix, and K[j,A] would take one of two values over j -- so dhat
would be piecewise constant with two levels across the entire deposit and could
not express a precedence fix. Augmenting keeps the spatial geometry that makes
the interpolation meaningful and uses c to separate the two populations.

WHAT THE KERNEL CAN AND CANNOT CARRY. Measured on this block model,
K[parent, child] ~ 0 at the default radius -- one bench of separation exhausts
the z lengthscale, and build_features spends it twice by carrying both z and
level. So a correction does NOT travel up the DAG through the kernel. It does
not need to: depth is carried by the ITERATION, since a block whose parent was
just corrected becomes next round's anchor. The kernel supplies lateral
spreading, which is what it is actually good at, and the recursion supplies
vertical propagation. Separating those two jobs is what this method gets right
and the edge-functional conditioning in gp_precedence.py got wrong.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from kernel_projection import slack, topological_order

try:                                                # pragma: no cover
    import torch
    HAVE_TORCH = True
except Exception:                                   # pragma: no cover
    HAVE_TORCH = False

WENDLAND_POWER = 5


def parent_csr(par, chi, n):
    """child -> its immediate parents, in CSR form."""
    deg = np.bincount(chi, minlength=n)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(deg, out=indptr[1:])
    indices = par[np.argsort(chi, kind="stable")].astype(np.int64)
    return indptr, indices


def worst_offender(s, par, chi, n):
    """r_i = min over parents p of (s_p - s_i).

    Root blocks have no parent and can never violate, so they get +inf and are
    never anchors. Vectorised with minimum.at over the edge list, which is one
    pass regardless of how ragged the parent sets are.
    """
    r = np.full(n, np.inf)
    np.minimum.at(r, chi, s[par] - s[chi])
    return r


def base_radius(gram, sigma2=1.0):
    """Recover the kernel ARGUMENT r from stored Wendland values.

    phi(r) = sigma2 (1 - r)^5 is invertible on [0, 1], so r = 1 - (K/sigma2)^(1/5)
    per stored nonzero. Keeping r rather than K is what makes the augmented
    kernel cheap: adding a feature column adds in quadrature to r, so each
    iteration is a vectorised pass over the nonzeros instead of a full rebuild
    of the Gram from the feature matrix.
    """
    indptr, indices, data = gram
    ratio = np.clip(np.asarray(data, dtype=np.float64) / sigma2, 0.0, 1.0)
    r = 1.0 - ratio ** (1.0 / WENDLAND_POWER)
    return indptr, indices, r


def augmented_gram(base, c, lam, sigma2=1.0):
    """K_ij = sigma2 (1 - sqrt(r_ij^2 + (lam (c_i - c_j))^2))_+^5.

    The satisfaction flag enters as one more feature column, so it adds in
    quadrature to the kernel argument exactly as any other column would. With
    lam >= 1 a mismatched pair is pushed past the compact support and the entry
    becomes exactly zero -- i.e. large lam makes the kernel refuse to spread a
    correction from an offender into satisfied ground at all, and lam = 0
    recovers the plain geometric kernel.
    """
    indptr, indices, r = base
    n = indptr.shape[0] - 1
    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
    dc = lam * (c[rows] - c[indices])
    r_aug = np.sqrt(r * r + dc * dc)
    v = np.where(r_aug < 1.0, sigma2 * (1.0 - r_aug) ** WENDLAND_POWER, 0.0)
    return sp.csr_matrix((v, indices, indptr), shape=(n, n))


def interpolate_precedence(s0, gram, par, chi, eta=0.5, sig2=1e-2, lam=1.0,
                           max_iters=200, tol=1e-9, sigma2=1.0,
                           direct_below=6000, verbose=False):
    """Dynamic anchor interpolation. Returns (s, info).

    `lam` weights the satisfaction feature: 0 ignores it, >=1 makes the kernel
    effectively block-diagonal in the two populations.
    `sig2` is the observation noise on the anchors; without it the interpolant
    hits every anchor exactly and the field collapses onto few levels, the same
    failure noiseless equality conditioning showed.
    `eta` damps the step.
    """
    s0 = np.asarray(s0, dtype=np.float64)
    n = s0.shape[0]
    base = base_radius(gram, sigma2=sigma2)
    s = s0.copy()
    hist = []
    n_start = int((slack(s0, par, chi) < -tol).sum())
    it = 0

    for it in range(1, max_iters + 1):
        r = worst_offender(s, par, chi, n)
        A = np.flatnonzero(r < -tol)
        if A.size == 0:
            break
        c = (r < -tol).astype(np.float64)
        K = augmented_gram(base, c, lam, sigma2=sigma2)

        KAA = K[A, :][:, A].tolil()
        KAA.setdiag(KAA.diagonal() + sig2)
        KAA = KAA.tocsc()
        rhs = r[A]
        if A.size <= direct_below:
            w = splu(KAA).solve(rhs)
        else:
            from scipy.sparse.linalg import cg as _cg
            w, _ = _cg(KAA, rhs, rtol=1e-10, maxiter=5000)
        delta = K[:, A] @ w
        s = s + eta * delta

        hist.append({"iter": it, "n_anchor": int(A.size),
                     "min_r": float(r[A].min()),
                     "max_step": float(np.abs(eta * delta).max())})
        if verbose and (it <= 5 or it % 25 == 0):
            print(f"  iter {it:>3}: |A|={A.size:>6,}  min r={r[A].min():+.3e}  "
                  f"max step={np.abs(eta*delta).max():.3e}", flush=True)

    g = slack(s, par, chi)
    return s, {"iters": it, "eta": eta, "sig2": sig2, "lam": lam,
               "n_violated_start": n_start,
               "n_violated_end": int((g < -tol).sum()),
               "min_slack": float(g.min()),
               "converged": bool((g < -tol).sum() == 0), "history": hist}


# --------------------------------------------------------------------------
# differentiable version
# --------------------------------------------------------------------------

class _SparseSolve(torch.autograd.Function if HAVE_TORCH else object):
    """x = M^-1 b for a CONSTANT sparse symmetric M, differentiable in b.

    M is constant because it is built from the anchor set and the satisfaction
    feature, both of which are piecewise constant in s and therefore detached --
    the same treatment ReLU's mask gets, exact almost everywhere. That makes the
    solve LINEAR in b, so the adjoint is another solve with M^T = M rather than
    anything involving dM. The factorisation is computed once in forward and
    reused in backward, so the backward pass costs a triangular solve, not a
    refactorisation.
    """

    @staticmethod
    def forward(ctx, b, lu):
        x = lu.solve(b.detach().cpu().numpy().astype(np.float64))
        ctx.lu = lu
        return torch.as_tensor(x, dtype=b.dtype, device=b.device)

    @staticmethod
    def backward(ctx, g):
        y = ctx.lu.solve(g.detach().cpu().numpy().astype(np.float64), trans="T")
        return torch.as_tensor(y, dtype=g.dtype, device=g.device), None


def _torch_csr_cols(K, A, device, dtype):
    """K[:, A] as a torch sparse COO tensor (constant, no grad)."""
    sub = K[:, A].tocoo()
    idx = torch.tensor(np.vstack([sub.row, sub.col]), dtype=torch.long,
                       device=device)
    val = torch.tensor(sub.data, dtype=dtype, device=device)
    return torch.sparse_coo_tensor(idx, val, sub.shape,
                                   device=device).coalesce()


def interpolate_precedence_torch(s, gram, par, chi, eta=1.0, sig2=1e-2,
                                 lam=1.0, max_iters=200, tol=1e-9, sigma2=1.0,
                                 big=1e30, return_info=False):
    """Differentiable dynamic anchor interpolation.

    Same operator as `interpolate_precedence`, with a gradient. `s` is a torch
    tensor of shape (n,); everything else matches the numpy signature.

    The graph per iteration is

        r     = amin over parents of (s_par - s_chi)     differentiable
        A, c  = f(r)                                     DETACHED
        M     = K_AA + sig2 I,  K = K(c)                 constant, factorised
        w     = M^-1 r_A                                 _SparseSolve
        s    <- s + eta K[:,A] w                         sparse matvec

    Nothing dense is ever formed: K stays CSR, K[:,A] becomes a sparse COO
    matvec, and M is factorised sparsely. That is what lets this run at the full
    block model, where the dense form would need ~1.2 GB for K alone.

    Gradient note. Measured retention is ~1.0 for a random direction against
    ~0.66 for the projection, because this is I + correction rather than a
    projector. The one thing it cannot pass gradient for is a direction that
    would separate an ACTIVE tight pair -- at the fixed point that gap is ~0 for
    every input in a neighbourhood, so its derivative is 0 by construction. No
    feasibility layer can do better; the constraint set has no gradient pointing
    out of itself.
    """
    if not HAVE_TORCH:
        raise RuntimeError("torch is not available")
    n = s.shape[0]
    device, dtype = s.device, s.dtype
    base = base_radius(gram, sigma2=sigma2)
    par_t = torch.as_tensor(np.asarray(par), dtype=torch.long, device=device)
    chi_t = torch.as_tensor(np.asarray(chi), dtype=torch.long, device=device)
    hist = []
    it = 0

    for it in range(1, max_iters + 1):
        d = s[par_t] - s[chi_t]
        r = torch.full((n,), big, dtype=dtype, device=device)
        r = r.scatter_reduce(0, chi_t, d, reduce="amin", include_self=True)

        with torch.no_grad():
            mask = r < -tol
            if not bool(mask.any()):
                break
            A_t = torch.nonzero(mask).squeeze(-1)
            A = A_t.cpu().numpy()
            c = mask.to(torch.float64).cpu().numpy()
            K = augmented_gram(base, c, lam, sigma2=sigma2)
            M = K[A, :][:, A].tolil()
            M.setdiag(M.diagonal() + sig2)
            lu = splu(M.tocsc())
            KcolA = _torch_csr_cols(K, A, device, dtype)

        w = _SparseSolve.apply(r[A_t], lu)
        delta = torch.sparse.mm(KcolA, w.unsqueeze(1)).squeeze(1)
        s = s + eta * delta
        hist.append({"iter": it, "n_anchor": int(A.size)})

    if not return_info:
        return s
    sd = s.detach().cpu().numpy()
    g = slack(sd, np.asarray(par), np.asarray(chi))
    return s, {"iters": it, "n_violated_end": int((g < -tol).sum()),
               "min_slack": float(g.min()),
               "converged": bool((g < -tol).sum() == 0), "history": hist}


def schedule_from_scores_tiebreak(s, par, chi, order=None, topo_rank=None):
    """Permutation: mine highest first, ties broken by topological rank.

    No clamp. If the field is not exactly feasible the sequence will show it --
    sequence_violations is the honest report, and it is left to the caller
    rather than hidden behind a repair pass.
    """
    s = np.asarray(s, dtype=np.float64)
    n = s.shape[0]
    if topo_rank is None:
        if order is None:
            order = topological_order(n, par, chi)
        topo_rank = np.empty(n, dtype=np.int64)
        topo_rank[order] = np.arange(n)
    return np.lexsort((topo_rank, -s))
