"""
Precedence-feasible score projection by kernel interpolation.

Feasibility convention: blocks are mined highest-score-first, so the schedule
argsort(-s) is precedence-feasible iff

    s[parent] >= s[child]   for every precedence edge (parent -> child)

which makes the feasible set a polyhedral convex cone  {s : B s >= 0}, where B
is the signed incidence matrix of the precedence DAG.

Two projections onto that cone are provided, both in the RKHS metric induced by
a Matern-1/2 Gram matrix K over an augmented feature vector:

    project_qp          exact minimum-norm projection, argmin (s-s0)' K^-1 (s-s0)
                        solved in the dual, s* = s0 + K B' lam, lam >= 0.
    project_sequential  the "fuzzy rule in topological order" -- repeated sweeps
                        of exact single-halfspace projections. Feasible, cheap,
                        but not the minimum-norm point (cyclic projection needs
                        Dykstra corrections to land on it).

project_hard_clamp is the K = I baseline: one topological pass of
s[child] <- min(s[child], min over parents).

The feature vector fed to the kernel carries the score itself, so K depends on
the point being projected. It is frozen at the pre-projection score, which keeps
each solve a genuine convex projection in a fixed metric.
"""

import numpy as np


# --------------------------------------------------------------------------
# Dykstra inner sweep -- numba if available, numpy otherwise
# --------------------------------------------------------------------------

def _dykstra_sweeps_numpy(s, K, par, chi, d, c, edge_order, max_sweeps, tol):
    sweeps = 0
    for sweep in range(max_sweeps):
        sweeps = sweep + 1
        shift = 0.0
        for e in edge_order:
            de = d[e]
            if de <= 0.0:
                continue
            p, ch = par[e], chi[e]
            t = s[p] - s[ch] + c[e] * de
            cn = t / de if t < 0.0 else 0.0
            delta = c[e] - cn
            if delta != 0.0:
                s += delta * (K[p] - K[ch])
                shift = max(shift, abs(delta) * de)
            c[e] = cn
        if shift <= tol:
            break
    return sweeps


def _dykstra_sweeps_scalar(s, K, par, chi, d, c, edge_order, max_sweeps, tol):
    # K is symmetric, so K[p, i] == K[i, p]; indexing by row keeps both reads
    # contiguous, which is worth several-fold over the column form.
    n = s.shape[0]
    m = edge_order.shape[0]
    sweeps = 0
    for sweep in range(max_sweeps):
        sweeps = sweep + 1
        shift = 0.0
        for k in range(m):
            e = edge_order[k]
            de = d[e]
            if de <= 0.0:
                continue
            p = par[e]
            ch = chi[e]
            t = s[p] - s[ch] + c[e] * de
            cn = t / de if t < 0.0 else 0.0
            delta = c[e] - cn
            if delta != 0.0:
                kp = K[p]
                kc = K[ch]
                for i in range(n):
                    s[i] += delta * (kp[i] - kc[i])
                ad = abs(delta) * de
                if ad > shift:
                    shift = ad
            c[e] = cn
        if shift <= tol:
            break
    return sweeps


def _dykstra_sweeps_sparse(s, indptr, indices, data, par, chi, d, c,
                           edge_order, max_sweeps, tol):
    """Dykstra over a sparse symmetric Gram matrix in CSR form.

    The update needs only columns p and c of K; symmetry means column p is
    row p, so each edge touches 2 * nnz_per_row entries instead of 2n.
    """
    m = edge_order.shape[0]
    sweeps = 0
    for sweep in range(max_sweeps):
        sweeps = sweep + 1
        shift = 0.0
        for k in range(m):
            e = edge_order[k]
            de = d[e]
            if de <= 0.0:
                continue
            p = par[e]
            ch = chi[e]
            t = s[p] - s[ch] + c[e] * de
            cn = t / de if t < 0.0 else 0.0
            delta = c[e] - cn
            if delta != 0.0:
                for jj in range(indptr[p], indptr[p + 1]):
                    s[indices[jj]] += delta * data[jj]
                for jj in range(indptr[ch], indptr[ch + 1]):
                    s[indices[jj]] -= delta * data[jj]
                ad = abs(delta) * de
                if ad > shift:
                    shift = ad
            c[e] = cn
        if shift <= tol:
            break
    return sweeps


def _pocs_sweeps_flat(s, indptr, indices, data, par, chi, d,
                      max_sweeps, tol, omega):
    """POCS with par/chi/d already materialised in sweep order.

    Identical arithmetic to _pocs_sweeps_sparse; the only change is that the
    edge loop reads three contiguous arrays instead of chasing edge_order[k]
    to index par/chi/d. One less dependent load per edge.
    """
    m = par.shape[0]
    sweeps = 0
    for sweep in range(max_sweeps):
        sweeps = sweep + 1
        shift = 0.0
        for k in range(m):
            de = d[k]
            if de <= 0.0:
                continue
            p = par[k]
            ch = chi[k]
            t = s[p] - s[ch]
            if t < 0.0:
                delta = omega * (-t / de)
                for jj in range(indptr[p], indptr[p + 1]):
                    s[indices[jj]] += delta * data[jj]
                for jj in range(indptr[ch], indptr[ch + 1]):
                    s[indices[jj]] -= delta * data[jj]
                ad = abs(delta) * de
                if ad > shift:
                    shift = ad
        if shift <= tol:
            break
    return sweeps


def _pocs_worklist(s, indptr, indices, data, par, chi, d,
                   inc_ptr, inc_idx, active, n_active, touched, mark,
                   max_sweeps, tol, omega):
    """POCS restricted to edges that can plausibly have become violated.

    An edge is a no-op unless one of its endpoints moved, so skipping satisfied
    edges is exact. The bookkeeping: firing edge (p,ch) perturbs every block in
    kernel rows p and ch, so all edges incident to those blocks must be
    re-queued. `touched` and `mark` use a generation counter (the sweep index)
    rather than being cleared, which would cost O(n)+O(m) per sweep and defeat
    the point.

    The rebuilt queue is not in the original sweep order, so the limit point
    differs from _pocs_sweeps_flat -- acceptable here because POCS is already
    order-dependent and used only as a feasibility correction.
    """
    n = s.shape[0]
    m = par.shape[0]
    na = n_active
    sweeps = 0
    for sweep in range(1, max_sweeps + 1):
        sweeps = sweep
        shift = 0.0
        fired = 0
        for a in range(na):
            k = active[a]
            de = d[k]
            if de <= 0.0:
                continue
            p = par[k]
            ch = chi[k]
            t = s[p] - s[ch]
            if t < 0.0:
                delta = omega * (-t / de)
                for jj in range(indptr[p], indptr[p + 1]):
                    j = indices[jj]
                    s[j] += delta * data[jj]
                    touched[j] = sweep
                for jj in range(indptr[ch], indptr[ch + 1]):
                    j = indices[jj]
                    s[j] -= delta * data[jj]
                    touched[j] = sweep
                ad = abs(delta) * de
                if ad > shift:
                    shift = ad
                fired += 1
        if fired == 0 or shift <= tol:
            break
        na = 0
        for b in range(n):
            if touched[b] == sweep:
                for kk in range(inc_ptr[b], inc_ptr[b + 1]):
                    e = inc_idx[kk]
                    if mark[e] != sweep:
                        mark[e] = sweep
                        active[na] = e
                        na += 1
        if na == 0:
            break
    return sweeps


def _pocs_sweeps_sparse(s, indptr, indices, data, par, chi, d,
                        edge_order, max_sweeps, tol, omega):
    """Cyclic projection (POCS) over a sparse Gram matrix -- Dykstra without the
    stored increments. Converges to *a* point of the cone, chosen by sweep
    order, not to the nearest one. Sweep count stays flat in n, which is where
    the runtime difference comes from."""
    m = edge_order.shape[0]
    sweeps = 0
    for sweep in range(max_sweeps):
        sweeps = sweep + 1
        shift = 0.0
        for k in range(m):
            e = edge_order[k]
            de = d[e]
            if de <= 0.0:
                continue
            p = par[e]
            ch = chi[e]
            t = s[p] - s[ch]
            if t < 0.0:
                delta = omega * (-t / de)
                for jj in range(indptr[p], indptr[p + 1]):
                    s[indices[jj]] += delta * data[jj]
                for jj in range(indptr[ch], indptr[ch + 1]):
                    s[indices[jj]] -= delta * data[jj]
                ad = abs(delta) * de
                if ad > shift:
                    shift = ad
        if shift <= tol:
            break
    return sweeps


def _gram_count(Zs, lut, ix, iy, iz, nx, ny, nz, offs, counts):
    """Pass 1: nonzeros per row, so indptr can be built before filling."""
    n = ix.shape[0]
    nf = Zs.shape[1]
    no = offs.shape[0]
    for i in range(n):
        c = 0
        for o in range(no):
            jx = ix[i] + offs[o, 0]
            jy = iy[i] + offs[o, 1]
            jz = iz[i] + offs[o, 2]
            if jx < 0 or jx >= nx or jy < 0 or jy >= ny or jz < 0 or jz >= nz:
                continue
            j = lut[(jx * ny + jy) * nz + jz]
            if j < 0:
                continue
            r2 = 0.0
            for f in range(nf):
                d = Zs[i, f] - Zs[j, f]
                r2 += d * d
                if r2 >= 1.0:
                    break
            if r2 < 1.0:
                c += 1
        counts[i] = c


def _gram_fill(Zs, lut, ix, iy, iz, nx, ny, nz, offs, indptr, indices, data,
               sigma2, jitter):
    """Pass 2: write the CSR payload. Wendland C0, phi(r) = sigma2 (1-r)^5."""
    n = ix.shape[0]
    nf = Zs.shape[1]
    no = offs.shape[0]
    for i in range(n):
        w = indptr[i]
        for o in range(no):
            jx = ix[i] + offs[o, 0]
            jy = iy[i] + offs[o, 1]
            jz = iz[i] + offs[o, 2]
            if jx < 0 or jx >= nx or jy < 0 or jy >= ny or jz < 0 or jz >= nz:
                continue
            j = lut[(jx * ny + jy) * nz + jz]
            if j < 0:
                continue
            r2 = 0.0
            for f in range(nf):
                d = Zs[i, f] - Zs[j, f]
                r2 += d * d
                if r2 >= 1.0:
                    break
            if r2 < 1.0:
                t = 1.0 - np.sqrt(r2)
                v = sigma2 * t * t * t * t * t
                if i == j:
                    v += jitter
                indices[w] = j
                data[w] = v
                w += 1


try:                                            # pragma: no cover
    from numba import njit
    _gram_count_fast = njit(cache=True)(_gram_count)
    _gram_fill_fast = njit(cache=True)(_gram_fill)
    _dykstra_sweeps_fast = njit(cache=True)(_dykstra_sweeps_scalar)
    _dykstra_sparse_fast = njit(cache=True)(_dykstra_sweeps_sparse)
    _pocs_sparse_fast = njit(cache=True)(_pocs_sweeps_sparse)
    _pocs_flat_fast = njit(cache=True)(_pocs_sweeps_flat)
    _pocs_worklist_fast = njit(cache=True)(_pocs_worklist)
    HAVE_NUMBA = True
except Exception:                               # pragma: no cover
    _dykstra_sweeps_fast = None
    _dykstra_sparse_fast = None
    _pocs_sparse_fast = None
    _pocs_flat_fast = None
    _pocs_worklist_fast = None
    _gram_count_fast = None
    _gram_fill_fast = None
    HAVE_NUMBA = False


# --------------------------------------------------------------------------
# features and kernel
# --------------------------------------------------------------------------

def build_features(x, y, z, level, score, value_now, value_future, tonnage=None,
                   extra=None):
    """Stack the kernel argument, one row per block. Column order is fixed:
    (x, y, z, level, score, value_now, value_future[, tonnage]).

    tonnage matters because two blocks alike in geometry and economics still
    consume capacity differently, and the projection should not treat them as
    interchangeable. If it is uniform across blocks, minmax_normalize maps the
    column to the midpoint and it contributes exactly zero to every pairwise
    distance -- inert, not harmful, and live as soon as tonnage varies.

    `extra` appends further columns (cone look-ahead aggregates, grade, ...);
    ard_lengthscales gives every column past index 3 the same `other`
    lengthscale, so extras need no special handling there.
    """
    cols = [x, y, z, level, score, value_now, value_future]
    if tonnage is not None:
        cols.append(tonnage)
    if extra:
        cols.extend(extra)
    return np.column_stack([np.asarray(c, dtype=float).ravel() for c in cols])


def minmax_normalize(Z, lo=-100.0, hi=100.0):
    """Per-column min-max rescale to [lo, hi]. Constant columns map to the
    midpoint so they contribute nothing to the distance."""
    Z = np.asarray(Z, dtype=float)
    zmin = Z.min(axis=0)
    zmax = Z.max(axis=0)
    span = zmax - zmin
    flat = span <= 0
    span = np.where(flat, 1.0, span)
    out = lo + (hi - lo) * (Z - zmin) / span
    if flat.any():
        out[:, flat] = 0.5 * (lo + hi)
    return out


def matern12_gram(Z, lengthscale=50.0, sigma2=1.0, jitter=1e-6):
    """Matern-1/2 (exponential / OU) Gram matrix: sigma2 * exp(-||dz|| / l).

    Sample paths are continuous but nowhere differentiable, so this does not
    smooth across ore/waste or bench discontinuities the way an RBF would.
    `lengthscale` may be a scalar or a per-column array (ARD).
    """
    Z = np.asarray(Z, dtype=float)
    l = np.asarray(lengthscale, dtype=float)
    W = Z / l if l.ndim else Z / float(l)

    sq = (W * W).sum(axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (W @ W.T)
    np.maximum(d2, 0.0, out=d2)
    K = sigma2 * np.exp(-np.sqrt(d2))
    K[np.diag_indices_from(K)] += jitter
    return K


def ard_lengthscales(nf, n_grid_x, n_grid_y, n_grid_z, radius_xy=6.0,
                     radius_z=2.0, other=100.0, span=200.0):
    """Per-column lengthscales in the min-max normalised feature space.

    Why this is needed: min-max normalising x,y,z to [-100,100] ties the
    kernel's reach to the *deposit extent* rather than the block size. A scalar
    lengthscale that spans 6 blocks on an 11x11 grid spans 158 blocks on a
    317x317 one, so nnz/row grows with n and memory goes back to O(n^2).

    Scaling the geometric columns by the grid spacing fixes the smoothing
    radius in block-widths instead, which keeps nnz/row roughly constant in n
    and memory at O(n). The measured optimum on synthetic data was ~5-7 block
    widths laterally.

    Assumes build_features column order: x, y, z, level, then the rest.
    `level` shares z's spacing since it is z re-indexed.
    """
    ell = np.full(int(nf), float(other), dtype=float)
    sx = span / max(int(n_grid_x) - 1, 1)
    sy = span / max(int(n_grid_y) - 1, 1)
    sz = span / max(int(n_grid_z) - 1, 1)
    ell[0] = radius_xy * sx
    ell[1] = radius_xy * sy
    if nf > 2:
        ell[2] = radius_z * sz
    if nf > 3:
        ell[3] = radius_z * sz          # 'level' duplicates z; keep them consistent
    return ell


def kernel_interpolate(s, gram, alpha=1.0):
    """Nadaraya-Watson smoothing of a score field against the same sparse Gram.

        s_smooth[i] = sum_j K[i,j] s[j] / sum_j K[i,j]
        out         = (1 - alpha) * s + alpha * s_smooth

    This is the *interpolation* half of the module's title, and it is the
    non-contracting counterpart to the projections below. Both use K, but they
    do opposite things to the degrees of freedom in s:

      project_*  moves s onto the cone {s_par >= s_chi}. Feasibility is exact,
                 but the operator is a contraction that levels scores into
                 exact ties (count_ties measures the damage), so many distinct
                 inputs collapse onto one output and one schedule.
      this       is a convex average over a compact neighbourhood. It smooths
                 spatial noise -- the reason the projection was reached for --
                 while remaining injective for alpha < 1, so distinct inputs
                 stay distinct. It does NOT produce a feasible point, and must
                 be paired with a decoder that enforces precedence itself
                 (decoders.schedule_priority_kahn, or the hard clamp inside
                 schedule_from_scores).

    alpha=0 is the identity, alpha=1 is a full smoothing pass.
    """
    indptr, indices, data = gram
    s = np.asarray(s, dtype=np.float64)
    n = s.shape[0]
    if s.shape[0] != indptr.shape[0] - 1:
        raise ValueError("s and gram disagree on n")

    w = np.asarray(data, dtype=np.float64)
    # reduceat over the starts of the NON-EMPTY rows only: an empty row has
    # indptr[i] == indptr[i+1] and contributes no elements, so skipping it
    # leaves every retained segment ending exactly at its true row end.
    lens = np.diff(indptr)
    nz = lens > 0
    num = np.zeros(n)
    den = np.zeros(n)
    if nz.any():
        starts = indptr[:-1][nz]
        num[nz] = np.add.reduceat(w * s[indices], starts)
        den[nz] = np.add.reduceat(w, starts)

    out = np.where(den > 0, num / np.where(den > 0, den, 1.0), s)
    a = float(alpha)
    return (1.0 - a) * s + a * out


def sparse_gram_masking_error(Z, gram, lengthscale, sigma2=1.0, n_sample=64,
                              seed=0):
    """Largest kernel value the sparsity pattern actually discarded.

    wendland_c0_sparse_gram's neighbourhood is defined on *grid offsets* while
    the kernel distance runs over all feature columns, and the non-geometric
    columns are not monotone in grid offset. So two blocks far apart on the
    grid can be close in feature space, and the sparse pattern can drop a
    nonzero entry. This samples rows, computes them densely, and reports the
    largest dropped value -- the honest version of the `boundary_max` field,
    which only inspects the boundary shell and cannot see past it.

    A value of ~0 means the pattern is exact. Small values are benign: by
    Weyl's inequality the eigenvalues move by at most n * max_dropped.
    """
    Z = np.asarray(Z, dtype=float)
    n, nf = Z.shape
    ell = np.asarray(lengthscale, dtype=float)
    ell = np.full(nf, float(ell)) if ell.ndim == 0 else ell
    indptr, indices, data = gram
    rng = np.random.default_rng(seed)
    rows = rng.choice(n, size=min(n_sample, n), replace=False)

    W = Z / ell
    worst = 0.0
    for i in rows:
        r = np.sqrt(np.maximum(((W - W[i]) ** 2).sum(axis=1), 0.0))
        phi = sigma2 * np.maximum(0.0, 1.0 - r) ** 5
        phi[indices[indptr[i]:indptr[i + 1]]] = 0.0     # drop what we did keep
        worst = max(worst, float(phi.max(initial=0.0)))
    return worst


def morton_order(ix, iy, iz):
    """Permutation sorting blocks along a Z-order (Morton) curve.

    Blocks arrive in meshgrid order, so vertical neighbours are adjacent in
    memory but lateral ones sit ny*nz apart -- and the Wendland support is
    overwhelmingly lateral. Interleaving the coordinate bits puts spatial
    neighbours close in index space, which helps the scattered accesses in the
    POCS/Dykstra inner loop once the score vector stops fitting in cache.

    Returns `perm` with perm[k] = old index of the k-th block in the new order.
    """
    def spread(v):
        v = np.asarray(v, dtype=np.uint64)
        out = np.zeros_like(v)
        for b in range(21):                       # 21 bits per axis is ample
            out |= ((v >> np.uint64(b)) & np.uint64(1)) << np.uint64(3 * b)
        return out

    key = (spread(ix) | (spread(iy) << np.uint64(1)) | (spread(iz) << np.uint64(2)))
    return np.argsort(key, kind="stable").astype(np.int64)


def permute_csr(gram, perm):
    """Reindex a symmetric CSR matrix by `perm`: K_new = P K P^T."""
    indptr, indices, data = gram
    n = indptr.shape[0] - 1
    inv = np.empty(n, dtype=np.int64)
    inv[perm] = np.arange(n, dtype=np.int64)

    counts = (indptr[perm + 1] - indptr[perm]).astype(np.int64)
    new_indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=new_indptr[1:])

    starts = indptr[perm]
    gather = np.repeat(starts - new_indptr[:-1], counts) + np.arange(new_indptr[-1])
    new_cols = inv[indices[gather]]
    new_data = data[gather]
    new_rows = np.repeat(np.arange(n, dtype=np.int64), counts)

    o = np.lexsort((new_cols, new_rows))
    return (new_indptr, np.ascontiguousarray(new_cols[o], dtype=indices.dtype),
            np.ascontiguousarray(new_data[o], dtype=data.dtype))


def admissible_offsets(Z, ix, iy, iz, lengthscale, radius):
    """Grid offsets whose *geometric* contribution alone already reaches r >= 1.

    r^2 = sum_f (dZ_f / l_f)^2 and the geometric columns are exact functions of
    the grid offset, so their partial sum is a lower bound on r^2 for every pair
    at that offset. If the bound reaches 1 the Wendland kernel is zero for all
    of them and the offset can be dropped before any block is touched.

    At radius_z=2 this kills every dz=+-2 offset outright, because `level`
    duplicates `z` and the two together consume 2*(dz/r_z)^2 of the budget.
    """
    Z = np.asarray(Z, dtype=float)
    nf = Z.shape[1]
    ell = np.asarray(lengthscale, dtype=float)
    ell = np.full(nf, float(ell)) if ell.ndim == 0 else ell
    ngrid = [int(ix.max()) + 1, int(iy.max()) + 1, int(iz.max()) + 1]

    # normalised distance covered by one grid step, per geometric column
    span = [(Z[:, c].max() - Z[:, c].min()) for c in range(min(4, nf))]
    stepn = [span[c] / max(ngrid[min(c, 2)] - 1, 1) for c in range(len(span))]

    rx, ry, rz = (int(v) for v in radius)
    offs = []
    for dx in range(-rx, rx + 1):
        for dy in range(-ry, ry + 1):
            for dz in range(-rz, rz + 1):
                d = [dx, dy, dz, -dz][:len(stepn)]
                r2 = sum((d[c] * stepn[c] / ell[c]) ** 2 for c in range(len(stepn)))
                if r2 < 1.0:
                    offs.append((dx, dy, dz))
    return np.asarray(offs, dtype=np.int64).reshape(-1, 3)


def wendland_c0_sparse_gram(Z, ix, iy, iz, radius=(4, 4, 1),
                            lengthscale=50.0, sigma2=1.0, jitter=1e-6,
                            fast=True, dtype=np.float32, index_dtype=np.int32):
    """Sparse Gram matrix of the Wendland C0 kernel over grid neighbours.

        phi(r) = sigma2 * (1 - r)_+^5,     r = ||(z_i - z_j) / l||

    Askey's theorem makes (1-r)_+^L positive definite on R^d for
    L >= floor(d/2)+1; with L=5 that covers d<=8, i.e. every feature column
    build_features produces. So this is a genuine metric, not a truncated
    Matern -- truncating Matern would break positive definiteness and with it
    Dykstra's convergence.

    Like Matern-1/2 it is C0 at the origin, so it does not smooth across
    ore/waste or bench discontinuities.

    Support is compact: entries with r >= 1 are exactly zero, not merely small.
    The pattern is NOT guaranteed exact, though: the neighbourhood is defined on
    grid offsets while r runs over every feature column, and score/value/tonnage
    are not monotone in grid offset, so a far-on-the-grid pair can still be close
    in feature space. `boundary_max` reports the largest kernel value on the
    boundary shell, which is a necessary but not sufficient check -- it cannot
    see past the shell. Call sparse_gram_masking_error for the real bound.

    Parameters
    ----------
    Z : (n, nf) normalised features -- minmax_normalize(build_features(...))
    ix, iy, iz : (n,) int grid indices
    radius : (rx, ry, rz) neighbourhood half-widths, in grid steps
    lengthscale : scalar or (nf,) ARD lengthscales

    Returns
    -------
    (indptr, indices, data), info
    """
    Z = np.ascontiguousarray(Z, dtype=np.float64)
    n, nf = Z.shape
    ell = np.asarray(lengthscale, dtype=np.float64)
    ell = np.full(nf, float(ell)) if ell.ndim == 0 else np.ascontiguousarray(ell)
    Zs = Z / ell

    ix = np.asarray(ix, dtype=np.int64)
    iy = np.asarray(iy, dtype=np.int64)
    iz = np.asarray(iz, dtype=np.int64)
    nx, ny, nz = int(ix.max()) + 1, int(iy.max()) + 1, int(iz.max()) + 1

    # dense grid -> block id lookup (-1 where the grid has no block)
    lut = np.full(nx * ny * nz, -1, dtype=np.int64)
    lut[(ix * ny + iy) * nz + iz] = np.arange(n, dtype=np.int64)

    if fast and HAVE_NUMBA:
        offs = admissible_offsets(Z, ix, iy, iz, ell, radius)
        counts = np.zeros(n, dtype=np.int64)
        _gram_count_fast(Zs, lut, ix, iy, iz, nx, ny, nz, offs, counts)
        indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(counts, out=indptr[1:])
        nnz = int(indptr[-1])
        indices = np.zeros(nnz, dtype=np.int64)
        data = np.zeros(nnz, dtype=np.float64)
        _gram_fill_fast(Zs, lut, ix, iy, iz, nx, ny, nz, offs, indptr, indices,
                        data, float(sigma2), float(jitter))
        # narrow after filling: data+indices dominate streamed memory traffic in
        # the inner loop, and the kernel values only need ~1e-8 of accuracy
        indices = np.ascontiguousarray(indices, dtype=index_dtype)
        data = np.ascontiguousarray(data, dtype=dtype)
        info = {
            "nnz": nnz, "nnz_per_row": float(nnz / n),
            "mem_MB": float((data.nbytes + indices.nbytes + indptr.nbytes) / 1e6),
            "dense_mem_GB": float(n * n * 8 / 1e9),
            "boundary_max": 0.0,
            "offsets_kept": int(offs.shape[0]),
            "offsets_total": int(np.prod([2 * int(v) + 1 for v in radius])),
        }
        return (indptr, indices, data), info

    rx, ry, rz = (int(v) for v in radius)
    rows, cols, vals = [], [], []
    boundary_max = 0.0

    for ox in range(-rx, rx + 1):
        for oy in range(-ry, ry + 1):
            for oz in range(-rz, rz + 1):
                jx, jy, jz = ix + ox, iy + oy, iz + oz
                ok = ((jx >= 0) & (jx < nx) & (jy >= 0) & (jy < ny)
                      & (jz >= 0) & (jz < nz))
                if not ok.any():
                    continue
                src = np.flatnonzero(ok)
                dst = lut[(jx[src] * ny + jy[src]) * nz + jz[src]]
                keep = dst >= 0
                src, dst = src[keep], dst[keep]
                if src.size == 0:
                    continue
                dz = Zs[src] - Zs[dst]
                r = np.sqrt((dz * dz).sum(axis=1))
                phi = sigma2 * np.maximum(0.0, 1.0 - r) ** 5
                nzm = phi > 0.0
                if max(abs(ox), abs(oy)) == rx or abs(oz) == rz:
                    boundary_max = max(boundary_max, float(phi.max(initial=0.0)))
                rows.append(src[nzm]); cols.append(dst[nzm]); vals.append(phi[nzm])

    rows = np.concatenate(rows); cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    vals = vals + jitter * (rows == cols)

    # CSR assembly, rows sorted
    order_rc = np.lexsort((cols, rows))
    rows, cols, vals = rows[order_rc], cols[order_rc], vals[order_rc]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(rows, minlength=n), out=indptr[1:])

    info = {
        "nnz": int(vals.size),
        "nnz_per_row": float(vals.size / n),
        "mem_MB": float((vals.nbytes + cols.nbytes + indptr.nbytes) / 1e6),
        "dense_mem_GB": float(n * n * 8 / 1e9),
        "boundary_max": boundary_max,
    }
    return (indptr, np.ascontiguousarray(cols), np.ascontiguousarray(vals)), info


def project_dykstra_sparse(s0, gram, par, chi, order=None, max_sweeps=30000,
                           tol=1e-12):
    """Exact minimum-norm projection with a sparse Gram matrix.

    Identical mathematics to project_dykstra; `gram` is the (indptr, indices,
    data) CSR triple from wendland_c0_sparse_gram. Cost per edge update drops
    from O(n) to O(nnz_per_row), and memory from O(n^2) to O(nnz).
    """
    indptr, indices, data = gram
    s = np.array(s0, dtype=np.float64, copy=True)
    n = s.shape[0]
    m = par.shape[0]
    if order is None:
        order = topological_order(n, par, chi)

    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    edge_order = np.argsort(rank[chi], kind="stable").astype(np.int64)
    par = np.ascontiguousarray(par, dtype=np.int64)
    chi = np.ascontiguousarray(chi, dtype=np.int64)

    # d_e = K_pp - 2 K_pc + K_cc, read out of the sparse rows
    diag = np.zeros(n)
    for i in range(n):
        row = indices[indptr[i]:indptr[i + 1]]
        hit = np.flatnonzero(row == i)
        if hit.size:
            diag[i] = data[indptr[i] + hit[0]]
    kpc = np.zeros(m)
    for e in range(m):
        p, ch = par[e], chi[e]
        row = indices[indptr[p]:indptr[p + 1]]
        hit = np.flatnonzero(row == ch)
        if hit.size:
            kpc[e] = data[indptr[p] + hit[0]]
    d = diag[par] - 2.0 * kpc + diag[chi]
    c = np.zeros(m, dtype=np.float64)

    fn = (_dykstra_sparse_fast if HAVE_NUMBA else _dykstra_sweeps_sparse)
    sweeps = fn(s, indptr, indices, data, par, chi, d, c,
                edge_order, int(max_sweeps), float(tol))

    lam = -c
    g = slack(s, par, chi)
    info = {
        "sweeps": int(sweeps),
        "primal_residual": float(max(0.0, -g.min())) if m else 0.0,
        "complementarity": float(np.abs(lam * g).max()) if m else 0.0,
        "dual_min": float(lam.min()) if m else 0.0,
        "active_edges": int((lam > 1e-9).sum()),
        "degenerate_edges": int((d <= 0).sum()),
        "converged": int(sweeps) < int(max_sweeps),
        "numba": bool(HAVE_NUMBA),
        "sparse": True,
    }
    return s, lam, info


# --------------------------------------------------------------------------
# precedence graph
# --------------------------------------------------------------------------

def build_edges(predecessors):
    """predecessors[i] = iterable of blocks that must be mined before i.

    Returns (parent, child) index arrays -- one entry per precedence edge.
    """
    par, chi = [], []
    for child, preds in enumerate(predecessors):
        for p in preds:
            par.append(int(p))
            chi.append(child)
    return np.asarray(par, dtype=np.int64), np.asarray(chi, dtype=np.int64)


def topological_order(n, par, chi):
    """Kahn's algorithm on parent -> child. Parents come first, so a single
    forward pass can enforce s[parent] >= s[child]."""
    indeg = np.bincount(chi, minlength=n)
    children = [[] for _ in range(n)]
    for p, c in zip(par, chi):
        children[p].append(c)

    stack = [i for i in range(n) if indeg[i] == 0]
    order = []
    indeg = indeg.copy()
    while stack:
        u = stack.pop()
        order.append(u)
        for v in children[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                stack.append(v)
    if len(order) != n:
        raise ValueError("precedence graph contains a cycle")
    return np.asarray(order, dtype=np.int64)


def slack(s, par, chi):
    """B s -- one entry per edge, negative where precedence is violated."""
    return s[par] - s[chi]


def n_violations(s, par, chi, tol=1e-9):
    return int((slack(s, par, chi) < -tol).sum())


# --------------------------------------------------------------------------
# projections
# --------------------------------------------------------------------------

def project_hard_clamp(s0, par, chi, order=None):
    """K = I baseline: one topological pass of s[child] <- min(s[child], parents).

    Always feasible in a single sweep, O(n + m), but it only ever moves scores
    down, so it is systematically biased toward the roots of the DAG.
    """
    s = np.array(s0, dtype=float, copy=True)
    n = s.shape[0]
    if order is None:
        order = topological_order(n, par, chi)

    parents = [[] for _ in range(n)]
    for p, c in zip(par, chi):
        parents[c].append(p)

    for node in order:
        ps = parents[node]
        if ps:
            m = min(s[p] for p in ps)
            if s[node] > m:
                s[node] = m
    return s


def project_sequential(s0, K, par, chi, order=None, max_sweeps=50, tol=1e-9):
    """The 'fuzzy rule, applied in an ordered way' -- kernel-smoothed.

    For each violated edge, take the exact projection onto that single
    halfspace in the K^-1 metric:

        if b's < 0:   s <- s - (b's / b'Kb) K b,      b = e_parent - e_child

    so the correction is K[:, parent] - K[:, child], a kernel-weighted message
    rather than an edit of two entries. Sweeps run in topological order and
    repeat because each update perturbs blocks outside the edge.

    Cyclic projection converges to a point in the intersection but not to the
    nearest one -- that needs Dykstra's corrections. Use project_qp for the
    minimum-norm point.
    """
    s = np.array(s0, dtype=float, copy=True)
    n = s.shape[0]
    if order is None:
        order = topological_order(n, par, chi)

    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    edge_order = np.argsort(rank[chi], kind="stable")

    sweeps = 0
    for sweeps in range(1, max_sweeps + 1):
        moved = 0
        for e in edge_order:
            p, c = par[e], chi[e]
            v = s[p] - s[c]
            if v < -tol:
                denom = K[p, p] - 2.0 * K[p, c] + K[c, c]
                if denom <= 0:
                    continue
                s += (-v / denom) * (K[:, p] - K[:, c])
                moved += 1
        if moved == 0:
            break

    info = {"sweeps": sweeps, "converged": n_violations(s, par, chi, tol) == 0}
    return s, info


def project_dykstra(s0, K, par, chi, order=None, max_sweeps=2000, tol=1e-12,
                    use_numba=True):
    """Exact minimum-norm projection by Dykstra-corrected cyclic projection.

    Same per-edge operator as project_sequential, plus one stored increment per
    halfspace:

        y  <- s + z_e
        s  <- proj_{H_e}(y)
        z_e <- y - s

    which is what turns cyclic projection (a feasibility method, landing on an
    order-dependent point) into best approximation (landing on Pi_K(s0)).

    The increment is always parallel to K b_e, so z_e = c_e * K b_e needs only a
    scalar per edge rather than an n-vector -- m floats instead of m*n. In that
    form this is Hildreth's algorithm, i.e. dual coordinate descent on the same
    QP that project_qp attacks with FISTA, and it returns the same multipliers
    lam = -c >= 0, so s = s0 + K B' lam and the KKT residuals are checkable.
    """
    K = np.ascontiguousarray(K, dtype=np.float64)
    s = np.array(s0, dtype=np.float64, copy=True)
    n = s.shape[0]
    m = par.shape[0]
    if order is None:
        order = topological_order(n, par, chi)

    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    edge_order = np.argsort(rank[chi], kind="stable").astype(np.int64)

    par = np.ascontiguousarray(par, dtype=np.int64)
    chi = np.ascontiguousarray(chi, dtype=np.int64)
    d = K[par, par] - 2.0 * K[par, chi] + K[chi, chi]   # b_e' K b_e
    c = np.zeros(m, dtype=np.float64)                    # z_e = c_e * K b_e, c_e <= 0

    sweep_fn = (_dykstra_sweeps_fast if (use_numba and HAVE_NUMBA)
                else _dykstra_sweeps_numpy)
    sweeps = sweep_fn(s, K, par, chi, d, c, edge_order, int(max_sweeps), float(tol))

    lam = -c
    g = slack(s, par, chi)
    info = {
        "sweeps": int(sweeps),
        "primal_residual": float(max(0.0, -g.min())) if m else 0.0,
        "complementarity": float(np.abs(lam * g).max()) if m else 0.0,
        "dual_min": float(lam.min()) if m else 0.0,
        "active_edges": int((lam > 1e-9).sum()),
        "numba": bool(use_numba and HAVE_NUMBA),
    }
    return s, lam, info


def project_pocs_sparse(s0, gram, par, chi, order=None, max_sweeps=5000,
                        tol=1e-10, omega=0.2, engine="flat"):
    """Feasibility-only sibling of project_dykstra_sparse.

    Same per-edge operator, no stored increments. Returns a point of the cone
    but NOT the nearest one: cyclic projection recovers best approximation only
    for affine sets, and halfspaces are not affine. There is no meaningful
    multiplier and no KKT certificate -- only feasibility, which is exact.

    In exchange the sweep count does not grow with n, so the cost is linear
    rather than roughly quadratic.

    Defaults are the fastest configuration measured (see the notes in
    kernel_projection's module docstring):
      engine='flat'  materialised edge arrays, 1.2-1.3x over 'indexed'
      omega=0.2      under-relaxation; costs ~2.6% NPV against a full Dykstra
                     projection instead of the ~7.3% that omega=1.0 costs
      max_sweeps     5000, enough for the real block model's ~1240 at omega=0.2
                     (the old default of 500 truncated silently)
    """
    indptr, indices, data = gram
    s = np.array(s0, dtype=np.float64, copy=True)
    n = s.shape[0]
    m = par.shape[0]
    if order is None:
        order = topological_order(n, par, chi)

    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    edge_order = np.argsort(rank[chi], kind="stable").astype(np.int64)
    par = np.ascontiguousarray(par, dtype=np.int64)
    chi = np.ascontiguousarray(chi, dtype=np.int64)

    diag = np.zeros(n)
    for i in range(n):
        row = indices[indptr[i]:indptr[i + 1]]
        hit = np.flatnonzero(row == i)
        if hit.size:
            diag[i] = data[indptr[i] + hit[0]]
    kpc = np.zeros(m)
    for e in range(m):
        p, ch = par[e], chi[e]
        row = indices[indptr[p]:indptr[p + 1]]
        hit = np.flatnonzero(row == ch)
        if hit.size:
            kpc[e] = data[indptr[p] + hit[0]]
    d = diag[par] - 2.0 * kpc + diag[chi]

    if engine == "indexed" or not HAVE_NUMBA:
        fn = (_pocs_sparse_fast if HAVE_NUMBA else _pocs_sweeps_sparse)
        sweeps = fn(s, indptr, indices, data, par, chi, d, edge_order,
                    int(max_sweeps), float(tol), float(omega))
    else:
        # materialise par/chi/d in sweep order: the inner loop then reads three
        # contiguous arrays instead of chasing edge_order[k]
        pf = np.ascontiguousarray(par[edge_order])
        cf = np.ascontiguousarray(chi[edge_order])
        dfl = np.ascontiguousarray(d[edge_order])
        if engine == "flat":
            sweeps = _pocs_flat_fast(s, indptr, indices, data, pf, cf, dfl,
                                     int(max_sweeps), float(tol), float(omega))
        elif engine == "worklist":
            # block -> incident edges (as positions in the materialised order)
            deg = np.bincount(pf, minlength=n) + np.bincount(cf, minlength=n)
            inc_ptr = np.zeros(n + 1, dtype=np.int64)
            np.cumsum(deg, out=inc_ptr[1:])
            inc_idx = np.empty(int(inc_ptr[-1]), dtype=np.int64)
            # stable counting sort: each edge appears once under each endpoint
            ends = np.concatenate([pf, cf])
            eids = np.concatenate([np.arange(m), np.arange(m)]).astype(np.int64)
            o = np.argsort(ends, kind="stable")
            inc_idx[:] = eids[o]
            active = np.arange(m, dtype=np.int64)
            touched = np.full(n, -1, dtype=np.int64)
            mark = np.full(m, -1, dtype=np.int64)
            sweeps = _pocs_worklist_fast(s, indptr, indices, data, pf, cf, dfl,
                                         inc_ptr, inc_idx, active, m, touched,
                                         mark, int(max_sweeps), float(tol),
                                         float(omega))
        else:
            raise ValueError("engine must be 'flat', 'worklist' or 'indexed'")

    g = slack(s, par, chi)
    return s, {"sweeps": int(sweeps),
               "primal_residual": float(max(0.0, -g.min())) if m else 0.0,
               "converged": int(sweeps) < int(max_sweeps),
               "numba": bool(HAVE_NUMBA), "min_norm": False, "engine": engine}


def project_qp(s0, K, par, chi, max_iter=5000, tol=1e-10):
    """Exact minimum-norm projection onto {B s >= 0} in the K^-1 metric.

    Primal   min (s - s0)' K^-1 (s - s0)   s.t.  B s >= 0
    KKT      s* = s0 + K B' lam,  lam >= 0,  B s* >= 0,  lam'(B s*) = 0
    Dual     min_{lam >= 0} 1/2 lam' (B K B') lam + lam' (B s0)

    Solved by FISTA on the dual. B K B' is never formed -- only matvecs, so the
    cost is O(n^2) per iteration rather than O(m^2) storage.
    """
    s0 = np.asarray(s0, dtype=float)
    n = s0.shape[0]
    m = par.shape[0]

    def Bt(lam):
        return (np.bincount(par, weights=lam, minlength=n)
                - np.bincount(chi, weights=lam, minlength=n))

    def Qmv(lam):
        t = K @ Bt(lam)
        return t[par] - t[chi]

    # Lipschitz constant of the dual gradient = spectral norm of B K B'
    v = np.random.default_rng(0).standard_normal(m)
    v /= np.linalg.norm(v)
    L = 1.0
    for _ in range(100):
        w = Qmv(v)
        nw = np.linalg.norm(w)
        if nw <= 0:
            break
        v = w / nw
        L = nw
    L = max(L, 1e-12)

    g0 = slack(s0, par, chi)
    lam = np.zeros(m)
    yk = lam.copy()
    t = 1.0
    iters = max_iter

    for it in range(1, max_iter + 1):
        grad = Qmv(yk) + g0
        lam_new = np.maximum(yk - grad / L, 0.0)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        yk = lam_new + ((t - 1.0) / t_new) * (lam_new - lam)
        step = np.linalg.norm(lam_new - lam)
        lam, t = lam_new, t_new
        if step <= tol * max(1.0, np.linalg.norm(lam)):
            iters = it
            break

    s = s0 + K @ Bt(lam)
    g = slack(s, par, chi)
    info = {
        "iterations": iters,
        "lipschitz": L,
        "primal_residual": float(max(0.0, -g.min())) if m else 0.0,
        "complementarity": float(np.abs(lam * g).max()) if m else 0.0,
        "dual_min": float(lam.min()) if m else 0.0,
        "objective": float(lam @ (Qmv(lam))),   # equals (s-s0)' K^-1 (s-s0)
    }
    return s, lam, info


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------

def schedule_from_scores(s, par, chi, order=None, return_snap=False):
    """Mining sequence (highest score first), feasible by construction.

    Two things stand between B s >= 0 and a feasible permutation, and both are
    handled here:

    1. Any iterative solver leaves residual slacks of order its own tolerance.
       argsort reads the sign of a -1e-12 slack literally and emits the child
       first, so a projection that is feasible to 1e-12 still yields hundreds of
       sequence violations. A final topological clamp pass snaps those to exact
       feasibility; the perturbation it applies equals the solver residual.
    2. Genuine ties (s[parent] == s[child], which the projection creates in bulk
       on active faces) leave the order undetermined. Breaking them by
       topological rank resolves it in the only feasible direction.

    Set return_snap=True to also get max |s_snapped - s|, i.e. how much the
    safeguard actually moved -- worth checking rather than assuming.
    """
    s = np.asarray(s, dtype=float)
    n = s.shape[0]
    if order is None:
        order = topological_order(n, par, chi)

    s_snap = project_hard_clamp(s, par, chi, order=order)
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    seq = np.lexsort((rank, -s_snap))

    if return_snap:
        return seq, float(np.abs(s_snap - s).max())
    return seq


def sequence_violations(seq, par, chi):
    """Precedence edges whose parent is mined after its child in `seq`."""
    pos = np.empty(seq.shape[0], dtype=np.int64)
    pos[seq] = np.arange(seq.shape[0])
    return int((pos[par] > pos[chi]).sum())


def rkhs_distance(s, s0, K):
    """(s - s0)' K^-1 (s - s0) -- the quantity project_qp minimizes."""
    d = np.asarray(s, dtype=float) - np.asarray(s0, dtype=float)
    return float(d @ np.linalg.solve(K, d))


def kendall_tau(a, b):
    """Rank correlation between two score vectors. O(n^2), fine at this size."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = a.shape[0]
    da = a[:, None] - a[None, :]
    db = b[:, None] - b[None, :]
    iu = np.triu_indices(n, k=1)
    sa, sb = np.sign(da[iu]), np.sign(db[iu])
    valid = (sa != 0) & (sb != 0)
    if not valid.any():
        return float("nan")
    return float((sa[valid] * sb[valid]).mean())


def count_ties(s, tol=1e-9):
    """Blocks sharing a score with at least one other block. Ties make the
    argsort order arbitrary among them, so they are a real remaining degree of
    freedom rather than a rounding artifact."""
    u, counts = np.unique(np.round(np.asarray(s, dtype=float) / tol) * tol,
                          return_counts=True)
    return int(counts[counts > 1].sum())
