"""Continuous scheduled time from projected scores, and continuous NPV.

No periods anywhere. Capacity is a rate, so cumulative tonnage and time are the
same axis rescaled, and the start time of a block is just the tonnage that
outranks it divided by that rate:

    tau_j    = w_j / C                                fraction of a period block j occupies
    sigma_j  = sum_i tau_i * 1[s_i > s_j]             start time, in periods
    t_j      = sigma_j + tau_j                        completion time
    psi(tau) = (1 - exp(-delta tau)) / (delta tau)    within-block discount shape
    NPV      = sum_j v_j psi(tau_j) exp(-delta sigma_j)

with delta = -log(gamma) for a per-period discount factor gamma. psi is the
exact continuum limit of the tonnage-weighted overlap discount that
capacity_cuts computes by splitting straddling blocks -- same quantity, closed
form, nothing to split. Deferring a negative-value block raises NPV, which is
the right economics and needs no special case.

Everything that made the discrete version awkward is gone with the periods: the
O(nT) capacity-crossing sigmoid stack, its temperature tau_w, the horizon gate
and tau_h, and the pro-rata straddle split. One temperature remains, the
ordering bandwidth, and it is chosen from the problem rather than tuned (see
`window_for_blur`).

PRECEDENCE IS INHERITED, NOT ENFORCED. sigma = G(s) with G non-increasing, so
on any precedence edge i -> j, s_i >= s_j implies sigma_i <= sigma_j. If the
score field is precedence-feasible the time field is too, at no cost. See
`time_violations`.

TWO PATHS.

  hard  start_times / npv          argsort + cumsum. Exact, no bandwidth, no
                                   surrogate gap, O(n log n), and batched over
                                   many score vectors at once. This is the fast
                                   sequence evaluator -- use it for BO, beam
                                   search and reporting.

  soft  start_times_soft / npv_soft  torch, differentiable in s. Replaces the
                                   indicator with a COMPACTLY SUPPORTED kernel
                                   CDF (Epanechnikov). A logistic never
                                   saturates, so every one of the n blocks
                                   leaks a little mass across the whole score
                                   line and the errors accumulate with n rather
                                   than cancelling. With compact support blocks
                                   further than h apart in score contribute
                                   exactly tau_i or exactly 0, and only genuine
                                   near-ties are smoothed -- which is the only
                                   place smoothing was ever wanted.

The soft path costs O(n log n + n k), not O(n^2): sort once, and the sum splits
into a prefix sum over everything above the window plus a banded window of 2k+1
terms. Nothing n x n is ever formed.

DEVICE. Every torch entry point follows the device and dtype of its inputs;
`prepare` moves the fixed data across once. Measured on the RTX 2050 in this
machine, at n = 8280 (the size of the 8280-binary MIP in large_run.log):

    hard, B = 1        GPU 0.6 ms vs numpy 1.1 ms     no real gain
    hard, B = 4096     GPU 93 ms vs numpy 4908 ms     53x, 23 us per sequence
    soft, B = 1        GPU 6.0 ms vs CPU 5.9 ms       no gain
    soft, B = 32       GPU 17 ms vs CPU 221 ms        13x

so the GPU only pays when the work is BATCHED -- a single (n,) call is far too
little to cover the kernel launches, and at these block counts the CPU is
already fast. Two hardware notes. This is a consumer card, where fp64 runs at
1/64 of fp32, so float32 roughly doubles throughput; its cost is cumsum
round-off that grows with n, measured against float64 at 3e-6 periods for
n = 8280 and 7e-4 for n = 20000, both far under a blur of 2e-2, but check it
rather than assume it if n grows much beyond that. And the card has only 4 GiB,
which the band overruns at n = 20000, B = 128 -- hence the automatic tiling in
`_auto_chunk`, which splits the sum by rows and is exact, not approximate.

TIE-BREAK. Equal scores are ordered by descending value, as specified. This is
free: permute into value-descending order once, then a STABLE sort by -s
preserves it among ties. Note the conflict this can create -- kernel_projection
creates parent/child score ties in bulk on active faces, and breaking those by
value can order a child before its parent. `time_violations` will show it if it
happens. Pass tie_break="topo" with a topological rank to rule it out instead.
"""

import numpy as np

try:                                                # pragma: no cover
    import torch
    import torch.nn.functional as F
    HAVE_TORCH = True
except Exception:                                   # pragma: no cover
    HAVE_TORCH = False


# --------------------------------------------------------------------------
# occupancy and discounting
# --------------------------------------------------------------------------

def occupancy(tonnage, capacity):
    """tau_j = w_j / C, the fraction of a period block j occupies.

    Parameters
    ----------
    tonnage : (n,) float      per-block tonnage, everything moved
    capacity : float          mining / strip capacity, tonnes per period

    Returns (n,) float. sum(tau) is the horizon needed to mine the lot.
    """
    w = np.asarray(tonnage, dtype=np.float64)
    C = float(capacity)
    if C <= 0:
        raise ValueError("capacity must be positive")
    if np.any(w < 0):
        raise ValueError("negative tonnage")
    return w / C


def delta_from_discount(discount):
    """delta such that exp(-delta * u) == gamma ** u, for gamma = discount."""
    g = float(discount)
    if not 0.0 < g <= 1.0:
        raise ValueError("discount must be in (0, 1]")
    return -np.log(g)


def npv_scale(value):
    """Reference magnitude for the objective: sum |v_j|, the total money on the
    table irrespective of sign.

    NPV on a real deposit runs to 1e7-1e9, and an unscaled objective hands the
    optimiser gradients of the same order -- measured at 9.8e7 falling to 9.4e5
    on an 84-block cone, and growing 60x over 200 steps once a precedence
    projection is in the chain. That is not a learning-rate problem to be tuned
    around per instance: it is a units problem. Dividing by this scale makes
    the objective the FRACTION of available value captured, a dimensionless
    number in [-1, 1], so one learning rate transfers across deposits, crops
    and price decks, and gradient clipping thresholds mean the same thing
    everywhere.

    sum |v_j| rather than sum max(v_j, 0) because waste is real work the
    schedule has to place, and rather than |sum v_j| because that can be near
    zero on a marginal pit and would blow the scaling up.
    """
    v = np.abs(np.asarray(value, dtype=np.float64)).sum()
    return float(v) if v > 0 else 1.0


def within_block_shape(tau, delta):
    """psi(tau) = (1 - exp(-delta tau)) / (delta tau), the discount a block

    earns by being mined over an interval rather than at an instant. Depends
    only on fixed data, so it is computed once and never enters the gradient.
    psi in (0, 1], -> 1 as tau -> 0 or delta -> 0.
    """
    tau = np.asarray(tau, dtype=np.float64)
    x = delta * tau
    out = np.ones_like(tau)
    big = x > 1e-12
    out[big] = -np.expm1(-x[big]) / x[big]
    return out


# --------------------------------------------------------------------------
# ordering
# --------------------------------------------------------------------------

def _value_order(value, n):
    """Blocks in descending value -- the tie-break, applied once up front."""
    if value is None:
        return np.arange(n)
    v = np.asarray(value, dtype=np.float64)
    return np.argsort(-v, kind="stable")


def mining_order(s, value=None, topo_rank=None, tie_break="value"):
    """Permutation: highest score mined first.

    tie_break : "value"  equal scores ordered by descending value (default)
                "topo"   equal scores ordered by topological rank, which can
                         never violate precedence but ignores value
                "none"   equal scores left in block-index order
    """
    s = np.asarray(s, dtype=np.float64)
    n = s.shape[0]
    if tie_break == "topo":
        if topo_rank is None:
            raise ValueError("tie_break='topo' needs topo_rank")
        return np.lexsort((np.asarray(topo_rank), -s))
    if tie_break == "none":
        return np.argsort(-s, kind="stable")
    if tie_break != "value":
        raise ValueError("tie_break must be 'value', 'topo' or 'none'")
    vp = _value_order(value, n)
    return vp[np.argsort(-s[vp], kind="stable")]


# --------------------------------------------------------------------------
# hard path -- exact, fast, batched
# --------------------------------------------------------------------------

def start_times(s, tau, value=None, topo_rank=None, tie_break="value",
                return_order=False):
    """Exact sigma_j: the tonnage-time that outranks block j, in periods.

    O(n log n), no bandwidth, no surrogate error. Returns (n,) float indexed by
    block id, or (sigma, order) if return_order.
    """
    tau = np.asarray(tau, dtype=np.float64)
    order = mining_order(s, value=value, topo_rank=topo_rank,
                         tie_break=tie_break)
    sigma = np.empty_like(tau)
    # exclusive cumulative sum along the mining order
    sigma[order] = np.cumsum(tau[order]) - tau[order]
    return (sigma, order) if return_order else sigma


def start_times_batch(S, tau, value=None):
    """sigma for a batch of score vectors. S is (B, n), returns (B, n).

    Fully vectorised -- one stable argsort and one cumsum over the batch, no
    Python loop. This is the throughput path: evaluating B candidate sequences
    costs one sort each and nothing else.
    """
    S = np.asarray(S, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    B, n = S.shape
    vp = _value_order(value, n)
    Sv = S[:, vp]                                   # value-descending columns
    tv = tau[vp]
    ordv = np.argsort(-Sv, axis=1, kind="stable")   # ties keep value order
    tsort = tv[ordv]                                # (B, n)
    excl = np.cumsum(tsort, axis=1) - tsort
    sigma_v = np.empty_like(excl)
    np.put_along_axis(sigma_v, ordv, excl, axis=1)
    sigma = np.empty_like(sigma_v)
    sigma[:, vp] = sigma_v
    return sigma


def npv(sigma, tau, value, discount=0.90, psi=None, scale=None):
    """sum_j v_j psi(tau_j) exp(-delta sigma_j). Accepts sigma of (n,) or (B, n).

    `scale` divides the result; pass `npv_scale(value)` to get the fraction of
    available value captured instead of currency. None leaves it in currency.
    """
    d = delta_from_discount(discount)
    tau = np.asarray(tau, dtype=np.float64)
    v = np.asarray(value, dtype=np.float64)
    if psi is None:
        psi = within_block_shape(tau, d)
    coef = v * psi
    out = np.einsum("...j,j->...", np.exp(-d * np.asarray(sigma)), coef)
    return out if scale is None else out / scale


def evaluate(s, tonnage, capacity, value, discount=0.90, **kw):
    """One-shot: scores -> sigma, t, NPV. Convenience over the pieces above."""
    tau = occupancy(tonnage, capacity)
    sigma, order = start_times(s, tau, value=value, return_order=True, **kw)
    return {"tau": tau, "sigma": sigma, "t": sigma + tau, "order": order,
            "npv": npv(sigma, tau, value, discount), "horizon": float(tau.sum())}


def time_violations(sigma, par, chi):
    """Count precedence edges whose start times are out of order.

    Should be 0 whenever the score field was precedence-feasible: this checks
    the inheritance claim rather than assuming it. par/chi are the edge arrays
    used elsewhere in the codebase (par[e] must precede chi[e]).
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    par = np.asarray(par, dtype=np.int64)
    chi = np.asarray(chi, dtype=np.int64)
    bad = sigma[par] > sigma[chi]
    return int(bad.sum())


# --------------------------------------------------------------------------
# soft path -- differentiable, compact support, banded
# --------------------------------------------------------------------------

def window_for_blur(n, horizon, blur=0.02, lo=8, hi=64):
    """Half-window k, chosen from the problem rather than tuned.

    The window spans 2k blocks carrying roughly 2k * horizon / n periods of
    tonnage, and that mass IS the time-blur the smoothing introduces. Fixing a
    target blur in PERIODS and solving for k,

        k = ceil(blur * n / (2 * horizon)),     blur(k) = 2 k horizon / n

    makes the relaxation error interpretable and scale-free instead of a tuned
    temperature. At the default 0.02 periods the discount moves by about 0.2%
    at gamma = 0.9, well under any modelling error in the block values.

    The two clamps are not symmetric, which is worth knowing:

      hi  is a pure cost cap (cost is O(n k)). Capping k BELOW the target only
          makes the blur smaller, so it costs gradient reach and never
          accuracy. Safe to lower.
      lo  is a gradient-quality floor -- with too few neighbours the gradient
          is dominated by one or two blocks. When lo binds, the realised blur
          EXCEEDS the target, which is the one case to keep an eye on. It binds
          for small n: at n = 2000 over a 10-period horizon, k = 8 gives 0.08
          periods, about 0.8% in discount.

    `blur_of_window` reports what you actually got.
    """
    if horizon <= 0:
        return lo
    k = max(lo, int(np.ceil(blur * n / (2.0 * horizon))))
    # a window must also be a small fraction of the deposit: at n = 102 the lo
    # floor alone gives 2.85 periods of blur on an 18-period horizon, a window
    # spanning 16% of every block's competition. Below n ~ 1600 this cap is
    # what binds; above it, hi does, so the normal regime is untouched.
    return max(1, min(k, hi, n // 25, n // 2))


def blur_of_window(n, horizon, window):
    """Realised time-blur in periods for a given half-window."""
    return 2.0 * window * horizon / max(1, n)


def _epanechnikov_cdf(u):
    """Compactly supported CDF: exactly 0 below -1, exactly 1 above +1.

    K(u) = 1/2 + 3u/4 - u^3/4 on [-1, 1]. C^1, and the clamp is what makes the
    saturation exact -- a block outside the window contributes precisely its
    full tau or precisely nothing, with no leaked mass and no gradient.
    """
    uc = u.clamp(-1.0, 1.0)
    return 0.5 + 0.75 * uc - 0.25 * uc * uc * uc


BAND_TILE = 1 << 23        # elements per band tile; ~67 MB in f64, 34 MB in f32


def prepare(tonnage, capacity, value, discount=0.90, device="cpu",
            dtype=None, blur=0.02):
    """Move the fixed problem data to a device once, with psi precomputed.

    Everything here is constant across a training run or a search, so building
    it per step is pure waste -- psi in particular only depends on tau.

    Returns a dict with tau, value, psi (tensors), plus delta, horizon and the
    half-window k implied by `blur`.
    """
    if not HAVE_TORCH:                              # pragma: no cover
        raise RuntimeError("prepare needs torch")
    dtype = dtype or torch.get_default_dtype()
    tau_np = occupancy(tonnage, capacity)
    d = delta_from_discount(discount)
    n, H = tau_np.shape[0], float(tau_np.sum())
    k = window_for_blur(n, H, blur=blur)
    tt = torch.as_tensor(tau_np, dtype=dtype, device=device)
    return {"tau": tt,
            "value": torch.as_tensor(np.asarray(value), dtype=dtype,
                                     device=device),
            "psi": within_block_shape_torch(tt, d),
            "scale": npv_scale(value),
            "delta": d, "discount": discount, "horizon": H, "n": n,
            "window": k, "blur": blur_of_window(n, H, k)}


def _auto_chunk(B, n, k, itemsize):
    """Rows per tile so one band stays inside BAND_TILE elements.

    A 4 GiB card runs out at n=20000, B=128, k=20 without this: the band alone
    is 105M elements and autograd keeps several of them alive. Tiling is exact
    -- the sum splits by rows -- so this costs nothing but a Python loop.
    """
    per_row = max(1, n * (2 * k + 1))
    if B * per_row <= BAND_TILE:
        return None
    return max(1, min(n, BAND_TILE // max(1, B * (2 * k + 1))))


def _order_torch(S, value):
    """(B, n) sorted-position -> block id. Descending score, ties by value."""
    B, n = S.shape
    if value is not None:
        vp = torch.argsort(value.reshape(-1).to(S.device), descending=True,
                           stable=True)
    else:
        vp = torch.arange(n, device=S.device)
    ordv = torch.argsort(S.detach()[:, vp], dim=1, descending=True, stable=True)
    return vp[ordv]


def start_times_torch(S, tau, value=None):
    """Exact sigma in torch, no bandwidth. Accepts (n,) or (B, n).

    The GPU throughput path for evaluating many candidate sequences: one
    batched argsort and one batched cumsum, nothing else. Not differentiable in
    S (it is the hard rank), so use it for search and reporting, not training.
    """
    if not HAVE_TORCH:                              # pragma: no cover
        raise RuntimeError("start_times_torch needs torch")
    flat = S.dim() == 1
    S = S.reshape(1, -1) if flat else S
    tau = tau.reshape(-1).to(dtype=S.dtype, device=S.device)
    perm = _order_torch(S, value)
    ts = tau[perm]
    excl = torch.cumsum(ts, dim=1) - ts
    sigma = torch.empty_like(excl).scatter_(1, perm, excl)
    return sigma.reshape(-1) if flat else sigma


def start_times_soft(s, tau, value=None, window=None, blur=0.02,
                     eps=1e-12, chunk=None):
    """Differentiable sigma. torch in, torch out, gradient flows to s.

    Parameters
    ----------
    s : (n,) or (B, n) tensor  projected scores, higher = mined earlier
    tau : (n,) tensor          occupancy fractions from `occupancy`
    value : (n,) tensor        used only to order exact ties
    window : int or None       half-window k; None -> window_for_blur
    blur : float               target time-blur in periods, if window is None
    chunk : int, None or -1    rows per band tile; None picks one automatically
                               from BAND_TILE, -1 disables tiling

    Batched: pass (B, n) to score B candidate fields in one go, which is what
    makes this worth putting on a GPU -- a single (n,) call is far too little
    work to cover the kernel launches.

    The bandwidth is per-block and adaptive: h_j is the SMALLER of the two
    one-sided distances to the window edge, which puts the kernel support
    strictly inside the window. The banded sum is therefore not an
    approximation of the smoothed G, it is exactly the smoothed G.

    h carries gradient. It is a function of the scores and d sigma / d h is the
    same order as the direct term, so detaching it gives an analytic gradient
    that does not match the function being computed -- a finite-difference
    check catches that at once. Only the argsort is detached, which is free: a
    permutation has no gradient, while the order statistics it selects are
    differentiable and min() is continuous where the k-th neighbour changes.
    """
    if not HAVE_TORCH:                              # pragma: no cover
        raise RuntimeError("start_times_soft needs torch")
    flat = s.dim() == 1
    s = s.reshape(1, -1) if flat else s
    B, n = s.shape
    tau = tau.reshape(-1).to(dtype=s.dtype, device=s.device)

    if window is None:
        window = window_for_blur(n, float(tau.sum().item()), blur=blur)
    k = int(max(1, min(window, max(1, n - 1))))

    perm = _order_torch(s, value)                   # (B, n)
    ss = torch.gather(s, 1, perm)                   # descending score
    ts = tau[perm]

    # ---- adaptive bandwidth: support strictly inside the window -----------
    idx = torch.arange(n, device=s.device)
    li, ri = idx - k, idx + k
    inf = torch.full((), float("inf"), dtype=ss.dtype, device=s.device)
    dl = torch.where(li >= 0, ss[:, li.clamp(min=0)] - ss, inf)
    dr = torch.where(ri < n, ss - ss[:, ri.clamp(max=n - 1)], inf)
    h = torch.minimum(dl, dr)
    span = ((ss[:, :1] - ss[:, -1:]).abs().detach() if n > 1
            else torch.ones_like(ss[:, :1]))
    h = torch.where(torch.isfinite(h), h, span.clamp(min=eps))
    h = h.clamp(min=eps * span.clamp(min=1.0) + eps)

    # ---- prefix mass above the window ------------------------------------
    P = F.pad(torch.cumsum(ts, dim=1), (1, 0))      # P[:, m] = mass of first m
    base = P[:, (idx - k).clamp(min=0)]

    # ---- banded window ----------------------------------------------------
    # tau padded with zeros makes out-of-range neighbours weightless, so the
    # padded scores they carry never matter.
    s_pad = F.pad(ss, (k, k))
    t_pad = F.pad(ts, (k, k))
    inv_h = h.reciprocal()

    def band(a, b):
        sw = s_pad[:, a:b + 2 * k].unfold(1, 2 * k + 1, 1)
        tw = t_pad[:, a:b + 2 * k].unfold(1, 2 * k + 1, 1)
        u = (sw - ss[:, a:b].unsqueeze(2)) * inv_h[:, a:b].unsqueeze(2)
        return (tw * _epanechnikov_cdf(u)).sum(2)

    if chunk is None:
        chunk = _auto_chunk(B, n, k, s.element_size())
    if chunk is None or chunk == -1 or chunk >= n:
        local = band(0, n)
    else:
        local = torch.cat([band(a, min(a + chunk, n))
                           for a in range(0, n, chunk)], dim=1)

    # the centre column is the block against itself, where K(0) = 1/2; a block
    # never outranks itself, so drop it here rather than masking the whole band
    sigma_sorted = base + local - 0.5 * ts
    sigma = torch.empty_like(sigma_sorted).scatter_(1, perm, sigma_sorted)
    return sigma.reshape(-1) if flat else sigma


def npv_soft(sigma, tau, value, discount=0.90, psi=None, scale=None):
    """Differentiable NPV. Accepts sigma of (n,) -> scalar, or (B, n) -> (B,).

    psi depends only on tau, so it is constant across a training run -- build
    it once with `within_block_shape_torch` and pass it in.

    `scale` divides the result. For TRAINING always pass one (`prepare` returns
    `npv_scale(value)` ready to use): an unscaled objective produces gradients
    of order 1e7-1e9 and forces a per-instance learning rate. See `npv_scale`.
    """
    if not HAVE_TORCH:                              # pragma: no cover
        raise RuntimeError("npv_soft needs torch")
    d = delta_from_discount(discount)
    if psi is None:
        psi = within_block_shape_torch(tau, d, dtype=sigma.dtype,
                                       device=sigma.device)
    coef = value.reshape(-1).to(dtype=sigma.dtype, device=sigma.device) * psi
    out = (coef * torch.exp(-d * sigma)).sum(-1)
    return out if scale is None else out / scale


def within_block_shape_torch(tau, delta, dtype=None, device=None):
    """psi on the device, without a numpy round trip."""
    tau = tau.reshape(-1)
    if dtype is not None or device is not None:
        tau = tau.to(dtype=dtype, device=device)
    x = delta * tau
    return torch.where(x > 1e-12, -torch.expm1(-x) / x.clamp(min=1e-12),
                       torch.ones_like(x))


# --------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    rng = np.random.default_rng(0)
    n = 20000
    tonnage = rng.uniform(2e4, 6e4, n)
    value = rng.normal(0.0, 1.0, n) * 5e4
    s = rng.normal(0.0, 1.0, n)
    capacity = tonnage.sum() / 10.0                 # 10-period horizon

    res = evaluate(s, tonnage, capacity, value, discount=0.90)
    print(f"n={n}  horizon={res['horizon']:.3f} periods  npv={res['npv']:,.0f}")

    t0 = time.perf_counter()
    for _ in range(20):
        start_times(s, res["tau"], value=value)
    print(f"hard  sigma      {1e3 * (time.perf_counter() - t0) / 20:7.2f} ms")

    S = np.repeat(s[None, :], 64, axis=0) + rng.normal(0, 0.1, (64, n))
    t0 = time.perf_counter()
    sb = start_times_batch(S, res["tau"], value=value)
    dt = time.perf_counter() - t0
    print(f"hard  batch x64  {1e3 * dt:7.2f} ms  ({1e3 * dt / 64:.3f} ms/seq)")
    assert np.allclose(sb[0], start_times(S[0], res["tau"], value=value))

    # tie-break: equal scores must come out in descending value
    s_tie = np.zeros(6)
    v_tie = np.array([1.0, 6.0, 3.0, 2.0, 5.0, 4.0])
    assert np.array_equal(mining_order(s_tie, value=v_tie),
                          np.argsort(-v_tie, kind="stable"))

    if HAVE_TORCH:
        st = torch.tensor(s, dtype=torch.float64, requires_grad=True)
        tt = torch.tensor(res["tau"], dtype=torch.float64)
        vt = torch.tensor(value, dtype=torch.float64)
        k = window_for_blur(n, res["horizon"])
        for _ in range(2):                          # torch cold start is ~1.8 s
            start_times_soft(st, tt, value=vt).sum().backward()
            st.grad = None
        t0 = time.perf_counter()
        sig = start_times_soft(st, tt, value=vt)
        obj = npv_soft(sig, tt, vt, discount=0.90)
        obj.backward()
        print(f"soft  k={k:<4d}      {1e3 * (time.perf_counter() - t0):7.2f} ms"
              f"  (fwd+bwd, blur target "
              f"{blur_of_window(n, res['horizon'], k):.4f} periods)")
        gap = abs(obj.item() - res["npv"]) / max(1.0, abs(res["npv"]))
        blur = np.abs(sig.detach().numpy() - res["sigma"]).max()
        print(f"soft  npv gap  {gap:.3e}   max |sigma_soft - sigma| = {blur:.4f} periods")
        print(f"soft  grad     finite={bool(torch.isfinite(st.grad).all())}"
              f"  nnz={int((st.grad != 0).sum())}/{n}")

        # gradient must agree with finite differences -- detaching the adaptive
        # bandwidth silently breaks this, so it is checked rather than assumed
        m = 300
        pr = prepare(tonnage[:m], tonnage[:m].sum() / 10.0, value[:m],
                     dtype=torch.float64)
        s0 = s[:m].copy()

        def f(x):
            sg = start_times_soft(x, pr["tau"], value=pr["value"], window=16)
            return npv_soft(sg, pr["tau"], pr["value"], psi=pr["psi"])

        xt = torch.tensor(s0, dtype=torch.float64, requires_grad=True)
        f(xt).backward()
        g, e = xt.grad.numpy(), 1e-6
        err = max(abs((f(torch.tensor(np.where(np.arange(m) == j, s0 + e, s0))).item()
                       - f(torch.tensor(np.where(np.arange(m) == j, s0 - e, s0))).item())
                      / (2 * e) - g[j]) / max(1.0, abs(g[j]))
                  for j in rng.choice(m, 8, replace=False))
        print(f"soft  grad vs finite differences  max rel err {err:.2e}")

        if torch.cuda.is_available():
            dev = torch.device("cuda")
            pg = prepare(tonnage, capacity, value, dtype=torch.float32,
                         device=dev)
            B = 512
            Sg = torch.as_tensor(np.repeat(s[None, :], B, 0), dtype=torch.float32,
                                 device=dev)
            for _ in range(3):
                start_times_torch(Sg, pg["tau"], value=pg["value"])
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            sg = start_times_torch(Sg, pg["tau"], value=pg["value"])
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            ref = start_times(s, res["tau"], value=value)
            err = np.abs(sg[0].cpu().numpy() - ref).max()
            print(f"cuda  {torch.cuda.get_device_name(0)}  f32  hard x{B}"
                  f"  {1e3 * dt:7.2f} ms  ({1e6 * dt / B:.1f} us/seq,"
                  f" max|dsigma| {err:.1e} periods)")
