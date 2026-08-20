"""
The precedence projection as a differentiable torch layer.

Forward is the exact minimum-norm projection (Dykstra, numpy/numba). Backward
uses the analytic Jacobian: on the active face the projection is the
K^-1-orthogonal projector onto ker(B_A),

    J = P_A = I - K B_A^T (B_A K B_A^T)^-1 B_A
    J^T g   = g - B_A^T lam,      where  (B_A K B_A^T) lam = B_A K g

P_A is piecewise constant in s -- it changes only when the active set changes --
so this is exact almost everywhere, and undefined on a measure-zero set, the
same situation as ReLU.

The dense |A| x |A| system is never formed: M lam is applied as
B_A (K (B_A^T lam)), all sparse, and solved by conjugate gradients.
"""

import numpy as np
import torch

from kernel_projection import project_dykstra_sparse, project_pocs_sparse


def _csr_matvec(gram, v):
    indptr, indices, data = gram
    n = indptr.shape[0] - 1
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        a, b = indptr[i], indptr[i + 1]
        if b > a:
            out[i] = float(np.dot(data[a:b].astype(np.float64), v[indices[a:b]]))
    return out


def _cg(apply_M, rhs, tol=1e-10, max_iter=400):
    x = np.zeros_like(rhs)
    r = rhs - apply_M(x)
    p = r.copy()
    rs = float(r @ r)
    if rs <= tol ** 2:
        return x
    for _ in range(max_iter):
        Mp = apply_M(p)
        denom = float(p @ Mp)
        if abs(denom) < 1e-300:
            break
        a = rs / denom
        x += a * p
        r -= a * Mp
        rs_new = float(r @ r)
        if np.sqrt(rs_new) <= tol * max(1.0, np.linalg.norm(rhs)):
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x


class PrecedenceProjection(torch.autograd.Function):
    """s -> Pi_K(s), differentiable. Set `projector='pocs'` for the fast
    feasibility-only forward; the backward still uses the min-norm Jacobian at
    the resulting active set, which is an approximation in that case."""

    @staticmethod
    def forward(ctx, s, gram, par, chi, order, projector, omega, max_sweeps):
        s_np = s.detach().cpu().numpy().astype(np.float64)
        if projector == "dykstra":
            sp, lam, info = project_dykstra_sparse(s_np, gram, par, chi, order,
                                                   max_sweeps=max_sweeps)
            act = np.flatnonzero(lam > 1e-9)
        else:
            sp, info = project_pocs_sparse(s_np, gram, par, chi, order,
                                           max_sweeps=max_sweeps, omega=omega)
            # active face = edges that ended up tight
            g = sp[par] - sp[chi]
            act = np.flatnonzero(np.abs(g) <= 1e-9)
        ctx.saved = (gram, par, chi, act)
        ctx.n_active = int(act.size)
        return torch.as_tensor(sp, dtype=s.dtype, device=s.device)

    @staticmethod
    def backward(ctx, grad_out):
        gram, par, chi, act = ctx.saved
        g = grad_out.detach().cpu().numpy().astype(np.float64)
        if act.size == 0:
            return (grad_out, None, None, None, None, None, None, None)

        pa, ca = par[act], chi[act]
        n = g.shape[0]

        def BA(v):
            return v[pa] - v[ca]

        def BAt(l):
            return (np.bincount(pa, weights=l, minlength=n)
                    - np.bincount(ca, weights=l, minlength=n))

        def apply_M(l):
            return BA(_csr_matvec(gram, BAt(l)))

        rhs = BA(_csr_matvec(gram, g))
        lam = _cg(apply_M, rhs)
        out = g - BAt(lam)
        return (torch.as_tensor(out, dtype=grad_out.dtype, device=grad_out.device),
                None, None, None, None, None, None, None)


def project_torch(s, gram, par, chi, order=None, projector="dykstra",
                  omega=0.2, max_sweeps=30000):
    """Convenience wrapper around PrecedenceProjection.apply."""
    return PrecedenceProjection.apply(s, gram, par, chi, order, projector,
                                      omega, max_sweeps)
