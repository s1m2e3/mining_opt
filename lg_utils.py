
import numpy as np
import pandas as pd

from pathlib import Path
from itertools import chain
def LG_to_NPVLG(data):
    data['blocks'] = {k: v for k, v in data['blocks'].items() if v['solution']==1}
    return data
def block_model_to_LG_data(
    block_model_filename,
    x_col, y_col, z_col,
    x_inc=10.0, y_inc=10.0, z_inc=10.0,
    return_indices=False
):
    """
    Vectorized binning of a block model into a 3D grid.
    Produces integer voxel codes (ix,iy,iz) and voxel *centers* (x_c,y_c,z_c),
    plus a mapping {(ix,iy,iz) -> rows} for fast neighborhood queries.

    Parameters
    ----------
    block_model_filename : str | Path | pd.DataFrame
    x_col, y_col, z_col : str
    x_inc, y_inc, z_inc : float
    return_indices : bool
        If True, voxel map values are row-index arrays; else DataFrames.

    Returns
    -------
    df : DataFrame
        Original data + columns: ix,iy,iz, x_c,y_c,z_c
    voxmap : dict
        {(ix,iy,iz): np.ndarray-of-row-indices or sub-DataFrame}
    meta : dict
        Grid metadata with edges, centers and increments.
    """
    # ---------- Load ----------
    if isinstance(block_model_filename, (str, Path)):
        p = Path(block_model_filename)
        if p.suffix.lower() == ".csv":
            df = pd.read_csv(p)
        elif p.suffix.lower() == ".json":
            try:
                df = pd.read_json(p)
            except ValueError:
                df = pd.read_json(p, lines=True)
        else:
            raise ValueError("Expected a .csv or .json path")
    elif isinstance(block_model_filename, pd.DataFrame):
        df = block_model_filename.copy()
    else:
        raise TypeError("block_model_filename must be path or DataFrame")

    # Ensure required columns
    for c in (x_col, y_col, z_col):
        if c not in df.columns:
            raise KeyError(f"Missing coordinate column: {c}")

    # ---------- Build edges ----------
    def edges_for(col, inc):
        cmin = df[col].min()
        cmax = df[col].max()
        start = np.floor(cmin / inc) * inc
        stop  = np.ceil((cmax + np.finfo(float).eps) / inc) * inc
        return np.arange(start, stop + inc, inc)

    x_edges = edges_for(x_col, x_inc)
    y_edges = edges_for(y_col, y_inc)
    z_edges = edges_for(z_col, z_inc)

    # Centers (for reporting/plotting)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    # ---------- Bin to integer voxel codes ----------
    x_vals = df[x_col].to_numpy()
    y_vals = df[y_col].to_numpy()
    z_vals = df[z_col].to_numpy()

    ix = np.searchsorted(x_edges, x_vals, side="right") - 1
    iy = np.searchsorted(y_edges, y_vals, side="right") - 1
    iz = np.searchsorted(z_edges, z_vals, side="right") - 1

    ix = np.clip(ix, 0, len(x_edges) - 2)
    iy = np.clip(iy, 0, len(y_edges) - 2)
    iz = np.clip(iz, 0, len(z_edges) - 2)

    # Attach voxel codes and *centers* (not lower edges)
    df["ix"] = ix
    df["iy"] = iy
    df["iz"] = iz
    df["x_c"] = x_centers[ix]
    df["y_c"] = y_centers[iy]
    df["z_c"] = z_centers[iz]

    # ---------- Build voxel → rows map ----------
    groups = df.groupby(["ix", "iy", "iz"], sort=False, observed=True)

    voxmap = {}
    if return_indices:
        for key, g in groups:
            voxmap[key] = g.index.to_numpy()
    else:
        for key, g in groups:
            voxmap[key] = g

    meta = {
        "x_edges": x_edges, "y_edges": y_edges, "z_edges": z_edges,
        "x_centers": x_centers, "y_centers": y_centers, "z_centers": z_centers,
        "x_inc": float(x_inc), "y_inc": float(y_inc), "z_inc": float(z_inc),
        "shape_ix": len(x_centers),
        "shape_iy": len(y_centers),
        "shape_iz": len(z_centers),
    }
    return df, voxmap, meta

def square_pyramid_predecessors(
    df: pd.DataFrame,
    x_col: str = "x_c",
    y_col: str = "y_c",
    z_col: str = "z_c",
    x_inc: float | None = None,
    y_inc: float | None = None,
    z_inc: float | None = None,
    slope_h_per_v: float = 1.5,   # s = horizontal meters per vertical meter
    id_col: str | None = None
) -> pd.Series:
    """
    For each row/block, return the list of *immediate* predecessors: blocks in the
    inverted square footprint one level above (i.e., iz' = iz + 1). Uses only
    the center columns (x_c, y_c, z_c). Grid increments are inferred if not given.

    Footprint (one lift, t=1):
        max(|dx|*x_inc, |dy|*y_inc) <= s * (1 * z_inc)

    Returns
    -------
    predecessors : pd.Series (aligned to df.index)
        Each entry is a sorted list of predecessor ids (if id_col provided)
        or row indices (default).
    """
    # --- checks ---
    for c in (x_col, y_col, z_col):
        if c not in df.columns:
            raise KeyError(f"Missing column '{c}' in DataFrame.")
    if slope_h_per_v < 0:
        raise ValueError("slope_h_per_v must be >= 0")
    use_ids = id_col is not None and id_col in df.columns

    # --- infer grid increments if needed ---
    def infer_inc(vals: np.ndarray) -> float:
        u = np.unique(vals)
        if len(u) < 2:
            raise ValueError("Not enough distinct coordinates to infer increment.")
        diffs = np.diff(np.sort(u))
        diffs = diffs[diffs > (np.min(diffs) * 1e-6)]  # drop jitter
        return float(np.median(diffs))

    x_vals = df[x_col].to_numpy(float)
    y_vals = df[y_col].to_numpy(float)
    z_vals = df[z_col].to_numpy(float)

    if x_inc is None: x_inc = infer_inc(x_vals)
    if y_inc is None: y_inc = infer_inc(y_vals)
    if z_inc is None: z_inc = infer_inc(z_vals)

    # --- snap to grid & integer codes ---
    x0, y0, z0 = float(x_vals.min()), float(y_vals.min()), float(z_vals.min())
    ix = np.rint((x_vals - x0) / x_inc).astype(int)
    iy = np.rint((y_vals - y0) / y_inc).astype(int)
    iz = np.rint((z_vals - z0) / z_inc).astype(int)

    df_idxed = df.copy()
    df_idxed["_ix"] = ix
    df_idxed["_iy"] = iy
    df_idxed["_iz"] = iz

    max_ix, max_iy, max_iz = ix.max(), iy.max(), iz.max()

    # --- build voxel -> row-index map ---
    vox_to_idx = {
        key: g.index.to_numpy()
        for key, g in df_idxed.groupby(["_ix", "_iy", "_iz"], sort=False, observed=True)
    }

    # --- one-lift offsets (Chebyshev square at t=1) ---
    L = slope_h_per_v * 1.0 * z_inc
    rx = int(np.floor(L / x_inc))
    ry = int(np.floor(L / y_inc))
    eps = 1e-12
    one_lift_offsets = [
        (dx, dy)
        for dx in range(-rx, rx + 1)
        for dy in range(-ry, ry + 1)
        if max(abs(dx) * x_inc, abs(dy) * y_inc) <= L + eps
    ]

    # --- collect immediate parents for each row (only iz+1) ---
    preds_per_row: list[list] = []
    ix_arr = df_idxed["_ix"].to_numpy(int)
    iy_arr = df_idxed["_iy"].to_numpy(int)
    iz_arr = df_idxed["_iz"].to_numpy(int)

    for r in range(len(df_idxed)):
        i, j, k = ix_arr[r], iy_arr[r], iz_arr[r]
        kz = k + 1
        if kz > max_iz:
            preds_per_row.append([])
            continue

        plist = []
        for dx, dy in one_lift_offsets:
            ii, jj = i + dx, j + dy
            if 0 <= ii <= max_ix and 0 <= jj <= max_iy:
                idxs = vox_to_idx.get((ii, jj, kz))
                if idxs is not None and len(idxs):
                    if use_ids:
                        plist.extend(df_idxed.loc[idxs, id_col].tolist())
                    else:
                        plist.extend(idxs.tolist())

        preds_per_row.append(sorted(set(plist)) if plist else [])

    return pd.Series(preds_per_row, index=df_idxed.index, name="predecessors")


def prepare_grid_and_predecessors(
    block_model_filename,
    x_col, y_col, z_col,
    x_inc=10.0, y_inc=10.0, z_inc=10.0,
    slope_h_per_v=1.5,
    t_max=1,
    id_col=None
):
    """
    Runs binning, then computes square-pyramid predecessors for every row.
    Returns the enriched DataFrame (with 'predecessors' column) and meta.
    """
    df, voxmap, meta = block_model_to_LG_data(
        block_model_filename, x_col, y_col, z_col,
        x_inc=x_inc, y_inc=y_inc, z_inc=z_inc,
        return_indices=True
    )
    preds = square_pyramid_predecessors(
        df, voxmap, meta,
        slope_h_per_v=slope_h_per_v, t_max=t_max, id_col=id_col
    )
    df = df.copy()
    df["predecessors"] = preds
    return df, meta

def concat_unique(s):
    seen = set()
    out = []
    for x in chain.from_iterable(s):
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def kmeanspp_centroids_3d(
    x_min, x_max,
    y_min, y_max,
    z_min, z_max,
    n_centroids,
    n_samples=20000,
    seed=None,
):
    """
    Generate k centroids in a 3D axis-aligned box using k-means++ (D^2) sampling.

    Parameters
    ----------
    x_min, x_max, y_min, y_max, z_min, z_max : float
        Box bounds.
    n_centroids : int
        Number of centroids (k).
    n_samples : int, default=20000
        Number of random points sampled uniformly from the box to run k-means++ on.
        Larger -> better coverage, slower.
    seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    centroids : (n_centroids, 3) np.ndarray
        Chosen centroids [x, y, z].
    """
    if n_centroids <= 0:
        raise ValueError("n_centroids must be >= 1")
    if n_samples < n_centroids:
        raise ValueError("n_samples must be >= n_centroids")
    if not (x_min < x_max and y_min < y_max and z_min < z_max):
        raise ValueError("Each min must be < its corresponding max")

    rng = np.random.default_rng(seed)

    # 1) Sample candidate points uniformly in the box
    pts = np.column_stack([
        rng.uniform(x_min, x_max, size=n_samples),
        rng.uniform(y_min, y_max, size=n_samples),
        rng.uniform(z_min, z_max, size=n_samples),
    ])  # shape: (n_samples, 3)

    # 2) k-means++ initialization (D^2 sampling)
    centroids = np.empty((n_centroids, 3), dtype=float)

    # Choose the first centroid uniformly at random from sampled points
    first_idx = rng.integers(n_samples)
    centroids[0] = pts[first_idx]

    # Track squared distance to nearest chosen centroid for each point
    d2 = np.sum((pts - centroids[0]) ** 2, axis=1)

    for i in range(1, n_centroids):
        total = d2.sum()
        if not np.isfinite(total) or total <= 0:
            # Degenerate case: all points identical or numerical issue
            # Fall back to random remaining points
            centroids[i:] = pts[rng.choice(n_samples, size=n_centroids - i, replace=False)]
            break

        probs = d2 / total
        idx = rng.choice(n_samples, p=probs)
        centroids[i] = pts[idx]

        # Update nearest-centroid squared distances
        new_d2 = np.sum((pts - centroids[i]) ** 2, axis=1)
        d2 = np.minimum(d2, new_d2)

    return centroids