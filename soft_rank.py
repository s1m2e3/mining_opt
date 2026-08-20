"""
Differentiable soft-rank surrogate for the schedule NPV.

The hard pipeline breaks at exactly one operation, the argsort in
schedule_from_scores. The soft-rank construction replaces it without ever
forming an n x n permutation matrix:

    M[i,j] = sigmoid( (s_j - s_i) / tau )        prob. j is mined before i
    W_r[i] = sum_j w_r[j] * M[i,j]               resource r consumed before i
    p_r[i] = sum_t sigmoid( (W_r[i] - C_r,t) / tau_w )     soft period index
    p[i]   = max_r p_r[i]                        whichever resource binds latest
    d[i]   = gamma^p[i] * sigmoid( (T - p[i]) / tau_h )    discount x horizon gate
    NPV    = sum_i v_i d[i]

O(n) memory (M is materialised but is n x n only in compute -- see note) and
O(n^2) work. Monotone in s, so if s already satisfies s_parent >= s_child the
soft ranks inherit it: precedence survives the relaxation.

Everything here is a *surrogate*. Report the hard NPV from
schedule_from_scores + evaluate_cuts_multi; the gap between them is the
relaxation error.
"""

import torch


def soft_npv(s, value, weights, caps, discount=0.90, tau=0.5, tau_w=None,
             tau_h=0.25, chunk=None, tau_r=0.5):
    """Differentiable NPV surrogate.

    Parameters
    ----------
    s : (n,) tensor            scores, higher = mined earlier
    value : (n,) tensor        undiscounted per-block value
    weights : (R, n) tensor    per-resource consumption
    caps : (R, T) tensor       per-resource per-period capacity
    tau : float                ordering temperature
    tau_w : float              capacity temperature; defaults to 2% of mean cap
    tau_h : float              horizon-gate temperature, in periods
    chunk : int or None        rows per block if the n x n term must be tiled

    Returns (npv, aux) where aux carries the soft period index.
    """
    n = s.shape[0]
    R, T = caps.shape
    cum = torch.cumsum(caps, dim=1)                       # (R, T)
    if tau_w is None:
        tau_w = 0.02 * float(caps.mean())

    if chunk is None:
        M = torch.sigmoid((s.unsqueeze(0) - s.unsqueeze(1)) / tau)   # (n,n)
        W = M @ weights.T                                            # (n,R)
    else:
        W = torch.zeros(n, R, dtype=s.dtype, device=s.device)
        for a in range(0, n, chunk):
            b = min(a + chunk, n)
            Mb = torch.sigmoid((s.unsqueeze(0) - s[a:b].unsqueeze(1)) / tau)
            W[a:b] = Mb @ weights.T

    # q[r,i,t] = P(block i is still unfinished after period t under resource r)
    q = torch.sigmoid((W.T.unsqueeze(-1) - cum.unsqueeze(1)) / tau_w)   # (R,n,T)
    # a block is unfinished if ANY resource has run out -- a union, which is a
    # smooth max that keeps every resource on the graph (a hard max(dim=0)
    # routes the gradient to one resource and zeroes every block with no weight
    # in it; with processing binding everywhere that killed 842 of 1035 blocks)
    Q = 1.0 - torch.prod(1.0 - q, dim=0)                                # (n,T)

    ones = torch.ones(n, 1, dtype=s.dtype, device=s.device)
    Qprev = torch.cat([ones, Q[:, :-1]], dim=1)                         # Q_{t-1}
    member = Qprev - Q                                                  # P(in period t)

    gam = torch.as_tensor(discount, dtype=s.dtype, device=s.device)
    gpow = gam ** torch.arange(T, dtype=s.dtype, device=s.device)
    # E[gamma^t] over the period distribution -- NOT gamma^E[t], which is
    # convex and so inflates NPV by the Jensen gap. Unmined mass, Q[:, -1],
    # simply carries no term, so the horizon needs no separate gate.
    delta = (member * gpow).sum(dim=1)                                  # (n,)
    return (value * delta).sum(), {"member": member, "delta": delta,
                                   "W": W, "unmined": Q[:, -1]}
