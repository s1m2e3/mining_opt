
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

def compute_block_precedence(blocks_csv, block_ids, x_step=None, y_step=None, z_step=None):
    """
    True per-block precedence over the raw grid-block model (blocks.csv):
    predecessors of a block at (x,y,z) are the block directly above it at
    (x,y,z+z_step), plus its 4 lateral neighbors at that same upper level -
    (x+x_step,y,z+z_step), (x-x_step,y,z+z_step), (x,y+y_step,z+z_step),
    (x,y-y_step,z+z_step). All must be mined before the block itself is
    eligible.

    Restricted to `block_ids`: a geometric predecessor that isn't in that
    set (i.e. outside the candidate pushbacks' combined footprint) is
    dropped rather than treated as a constraint - it's out of scope for
    whatever reduced dataset block_ids represents.

    Returns {block_id: [predecessor_block_id, ...]}.
    """
    df = pd.read_csv(blocks_csv)
    df['z'] = -df['z']  # match the sign convention pushbacks.csv/process_pushbacks.py uses
    if x_step is None:
        x_step = float(df['x_step'].iloc[0])
    if y_step is None:
        y_step = float(df['y_step'].iloc[0])
    if z_step is None:
        z_step = float(df['z_step'].iloc[0])

    wanted = set(block_ids)

    # vectorized: build plain-dict lookups once, instead of repeated
    # pandas .loc scalar access per block (which is slow at this scale)
    ids = df['index'].to_numpy()
    xs = df['x'].round(6).to_numpy()
    ys = df['y'].round(6).to_numpy()
    zs = df['z'].round(6).to_numpy()

    id_to_xyz = {int(i): (float(x), float(y), float(z)) for i, x, y, z in zip(ids, xs, ys, zs) if int(i) in wanted}
    loc_to_id = {(x, y, z): int(i) for i, (x, y, z) in id_to_xyz.items()}

    predecessors = {}
    for bid, (x, y, z) in id_to_xyz.items():
        candidates = [
            (x, y, round(z + z_step, 6)),
            (round(x + x_step, 6), y, round(z + z_step, 6)),
            (round(x - x_step, 6), y, round(z + z_step, 6)),
            (x, round(y + y_step, 6), round(z + z_step, 6)),
            (x, round(y - y_step, 6), round(z + z_step, 6)),
        ]
        preds = {loc_to_id[c] for c in candidates if c in loc_to_id}
        preds.discard(bid)
        predecessors[bid] = sorted(preds)
    return predecessors


def load_pushback_schedule_data(
    pushbacks_csv,
    num_periods,
    period_size,
    discount_rate,
    columns=None,
    max_level=None,
    tonnage_capacity_per_period=None,
    blocks_csv='inputs/blocks.csv',
):
    """
    Build the input dict consumed by problems.PushbackScheduleMIP from the
    pushback-level dataset produced by process_pushbacks.py.

    Precedence is derived from the true, per-raw-block rule (see
    compute_block_precedence): a pushback p's predecessors are every OTHER
    pushback (excluding alternate depth levels at p's own row, which are
    mutually exclusive with p, not predecessors of it) that owns a true
    geometric predecessor of any block p covers. Any one of them
    completing is enough (an OR condition, same as before) - since the
    predecessor rule reaches into laterally-adjacent columns (direct above
    + 4 neighbors above), this is not confined to p's own (x,y) column the
    way an earlier, coarser version of this model assumed.

    Overlap: pushbacks at different rows/columns can share the same
    underlying grid-block (their footprints overlap). Since a block can
    physically only be mined once, any grid-block referenced by more than
    one pushback gets a "mined at most once, across all pushbacks that
    contain it" constraint. Note a deeper level's footprint fans out wide
    at its shallow end (up to ~100+ units radius), so a single deep-level
    pushback can overlap dozens of neighboring columns - `max_level` lets
    you restrict to shallow/local pushbacks when testing a small patch.

    Tonnage: process_blocks.py drops per-block tonnage during aggregation,
    so blocks.csv's 'tonnage' column is recovered separately (see the
    tonnage-recovery step that sums true per-raw-block tonnage - volume *
    the same density factor used to compute 'income' - into each grid
    cell, matching process_blocks.py's own spatial binning exactly). Each
    pushback's tonnage is the sum of its constituent blocks' tonnage.
    `tonnage_capacity_per_period`, if given, is a true tons/period cap
    (e.g. daily_mining_rate * 365 * period_size-in-years, matching the
    convention already used by data_classes.NPVLGData).
    """
    df = pd.read_csv(pushbacks_csv)
    if columns is not None:
        wanted = set(columns)
        df = df[df.apply(lambda r: (r['x'], r['y']) in wanted, axis=1)]
    if max_level is not None:
        df = df[df['level'] <= max_level]
    df = df.reset_index(drop=True)
    df['pid'] = df.index

    block_tonnage = pd.read_csv(blocks_csv).set_index('index')['tonnage'].to_dict()

    pushbacks = {}
    for row in df.itertuples():
        blocks = [int(b) for b in row.blocks.strip('[]').split(',')]
        pushbacks[row.pid] = {
            'x': row.x, 'y': row.y, 'z': row.z, 'level': row.level,
            'income': row.income, 'cost': row.cost,
            'blocks': blocks,
            'tonnage': sum(block_tonnage.get(b, 0.0) for b in blocks),
        }

    # group pushback ids by their exact (x,y,z) row - these are mutually
    # exclusive depth-level alternatives, not predecessors of each other
    row_groups_map = {}
    row_key_of = {}
    for pid, v in pushbacks.items():
        key = (v['x'], v['y'], v['z'])
        row_groups_map.setdefault(key, []).append(pid)
        row_key_of[pid] = key
    row_groups = list(row_groups_map.values())

    # full reverse index: raw grid-block id -> every pushback that covers it
    block_to_pids = {}
    for pid, v in pushbacks.items():
        for b in v['blocks']:
            block_to_pids.setdefault(b, []).append(pid)

    all_block_ids = set(block_to_pids.keys())
    block_precedence = compute_block_precedence(blocks_csv, all_block_ids)

    # per-pushback predecessor requirements. A pushback's own blocks
    # already have internally-consistent (top-down) ordering, so only
    # EXTERNAL predecessor blocks (not already covered by p itself) impose
    # a real requirement. Each *distinct* external predecessor block is
    # its own independent requirement (AND across distinct blocks) - only
    # the choice of *which pushback ends up supplying that one block* is
    # an OR (any of its owners, since block-exclusivity guarantees at most
    # one of them is ever actually selected). Pooling every external
    # predecessor block into one flat "any single one suffices" OR (as an
    # earlier version of this function did) is too permissive once a
    # pushback's footprint spans several physically distinct predecessor
    # locations - each of those still needs its own support.
    pushback_predecessor_groups = {}
    for pid, v in pushbacks.items():
        own_blocks = set(v['blocks'])
        groups_by_block = {}
        for b in v['blocks']:
            for b_pred in block_precedence.get(b, []):
                if b_pred in own_blocks:
                    continue  # internal to p's own footprint - already ordered
                if b_pred in groups_by_block:
                    continue
                owners = [q for q in block_to_pids.get(b_pred, []) if row_key_of[q] != row_key_of[pid]]
                if owners:
                    groups_by_block[b_pred] = sorted(set(owners))
        pushback_predecessor_groups[pid] = list(groups_by_block.values()) if groups_by_block else None

    # reverse index restricted to blocks shared by more than one pushback -
    # only those actually need a hard-exclusivity constraint
    block_owner = {b: ids for b, ids in block_to_pids.items() if len(ids) > 1}

    return {
        'pushbacks': pushbacks,
        'row_groups': row_groups,
        'pushback_predecessor_groups': pushback_predecessor_groups,
        'block_owner': block_owner,
        'num_periods': num_periods,
        'period_size': period_size,
        'discount_rate': discount_rate,
        'tonnage_capacity_per_period': tonnage_capacity_per_period,
    }


# ---------------------------------------------------------------------------
# The Pushback Sequencing Problem
#
# This is the single, canonical problem definition that both
# problems.PushbackScheduleMIP (exact) and beam_search_pushback_schedule
# (heuristic, below) solve. Any other solver for this data should conform to
# the same definition to be comparable.
#
# Given: a set of candidate pushbacks, each covering a set of underlying
# grid-blocks, each with a fixed (income, cost) - cost stored as an
# already-negative quantity - and organized into rows (a distinct (x,y,z))
# and columns (a distinct (x,y), containing multiple rows at decreasing z).
#
# Decide: for each pushback p, whether it is ever selected (z[p], binary -
# "this is the depth level committed to for its row"), and if so, what
# fraction of it gets completed in each period k in {0, ..., T-1} where
# T = num_periods // period_size (x[p][k] in [0,1], continuous - income,
# cost, and grid-block count all scale linearly with completed fraction).
# A pushback need not finish within a single period: if it doesn't fit in
# one period's remaining capacity, the remainder carries into a later
# period, same as it would in the raw-block NPVLG_Indexed model.
#
# Subject to:
#   1. A pushback can be completed at most once in total across all
#      periods (sum_k x[p][k] <= 1), and can only make progress
#      (x[p][k] > 0) in a period if it has been selected (x[p][k] <= z[p]).
#   2. At most one depth level is selected per row (sum z[p] over a row's
#      levels <= 1) - a deeper level's footprint already contains the
#      shallower ones at that row, so they are mutually exclusive
#      alternatives, not additive.
#   3. Precedence: a raw grid-block at (x,y,z) requires the block directly
#      above it (x,y,z+z_step) AND all 4 of its lateral neighbors one level
#      up (x+-x_step,y,z+z_step), (x,y+-y_step,z+z_step) to already be
#      mined - a plus/cross-shaped footprint that reaches into laterally
#      adjacent columns, not just straight up the same column. A
#      pushback's own blocks are already internally top-down ordered, so
#      only *external* predecessor blocks (not covered by the pushback
#      itself) impose a real requirement, and each distinct external
#      predecessor block is its own independent requirement: pushback p's
#      cumulative completed fraction through period k cannot exceed, for
#      EVERY distinct external predecessor block, the cumulative completed
#      fraction (through period k-1) of whichever pushback(s) could supply
#      that specific block (any one of them suffices there, since
#      block-exclusivity guarantees at most one is ever actually selected -
#      see compute_block_precedence/pushback_predecessor_groups in
#      load_pushback_schedule_data). A pushback with no external
#      predecessors has no precedence requirement at all.
#   4. Hard block-exclusivity: a grid-block can be claimed by at most one
#      *selected* pushback (sum z[p] over a block's owners <= 1) - pushback
#      footprints overlap, and once a pushback is committed to, its
#      footprint is claimed regardless of how much of it is completed yet.
#   5. Per-period capacity (optional): the total tonnage committed in a
#      period, weighted by completed fraction that period, cannot exceed a
#      cap (e.g. daily_mining_rate * 365 * period_size-in-years - true
#      per-block tonnage, recovered from the raw block model since
#      process_blocks.py drops it during pushback-level aggregation).
#
# Maximize: sum over (pushback, period) of
#   x[p][k] * beta**k * (income + cost),  where beta = 1 / (1 + discount_rate)
# ---------------------------------------------------------------------------

def beam_search_pushback_schedule(data, beam_width=4, candidate_pool_size=4, max_steps=None):
    """
    Heuristic solver for the Pushback Sequencing Problem defined above -
    the same problem problems.PushbackScheduleMIP solves exactly, including
    multi-period completion (a pushback too big for one period's remaining
    capacity spills into a later one instead of being dropped). Operates
    directly on the `data` dict from load_pushback_schedule_data, so the
    block-exclusivity/capacity/row-exclusivity rules are identical to the
    MIP's by construction, not by parallel maintenance.

    One deliberate simplification versus the MIP: a row only becomes
    eligible once its predecessor row is *fully* completed (fraction 1.0),
    not merely "far enough ahead" the way the MIP's fractional precedence
    constraint allows. This is a sufficient (if occasionally more
    conservative) condition for the MIP's actual constraint, chosen
    because it keeps a simple greedy "start it, then fill it" procedure
    always MIP-feasible - every schedule this returns is a valid, just
    not necessarily optimal, solution to the exact same problem.

    A genuine (multi-path) beam search: at each step, every state in the
    beam is expanded by its top `candidate_pool_size` next moves (which
    pushback to START next, ranked by discounted value), the chosen
    pushback is then greedily filled across as many periods as it takes
    to complete (or as far as remaining capacity allows), all resulting
    children are pooled, and pruned back to the best `beam_width` by
    cumulative value. Stops once no state has any positive-value move left
    - "leave it unmined" is always an available (and often optimal) choice.

    Returns {'chosen': {pid: {...,'schedule':{k:frac},'first_period':k}}, 'order': [pid,...], 'objective_value': float}
    """
    pushbacks = data['pushbacks']
    row_groups = data['row_groups']
    predecessor_groups = data['pushback_predecessor_groups']
    T = int(data['num_periods'] // data['period_size'])
    beta = 1.0 / (1.0 + data['discount_rate'])
    capacity = data.get('tonnage_capacity_per_period')

    pid_to_row_idx = {}
    for ridx, pids in enumerate(row_groups):
        for pid in pids:
            pid_to_row_idx[pid] = ridx

    def fill_schedule(capacity_used, tonnage, start_period):
        """Greedily allocate fraction across periods from start_period on,
        respecting remaining capacity each period. Returns
        (allocations=[(k,frac),...], completed_period or None, updated capacity_used)."""
        remaining = 1.0
        allocations = []
        capacity_used = list(capacity_used)
        for k in range(start_period, T):
            if remaining <= 1e-9:
                break
            if capacity is None:
                add = remaining
            else:
                free = capacity - capacity_used[k]
                if free <= 1e-9:
                    continue
                add = min(remaining, free / tonnage) if tonnage > 0 else remaining
            if add <= 1e-9:
                continue
            allocations.append((k, add))
            capacity_used[k] += add * tonnage
            remaining -= add
        completed_period = allocations[-1][0] if (allocations and remaining <= 1e-9) else None
        return allocations, completed_period, capacity_used

    def candidates_for_state(state):
        cands = []
        for ridx, pids_in_row in enumerate(row_groups):
            if ridx in state['row_selected_pid']:
                continue
            for pid in pids_in_row:
                groups = predecessor_groups.get(pid)
                if not groups:
                    start_period = 0
                else:
                    # every distinct required block (group) must be
                    # satisfied (AND); within a group, any one supplier
                    # completing is enough (OR) - the binding constraint
                    # is whichever group is satisfied last
                    group_ready_periods = []
                    ready = True
                    for group in groups:
                        done = [state['completed_period'][q] for q in group if state['completed_period'].get(q) is not None]
                        if not done:
                            ready = False
                            break
                        group_ready_periods.append(min(done))
                    if not ready:
                        continue  # at least one required predecessor block unsatisfied
                    start_period = max(group_ready_periods) + 1
                if start_period >= T:
                    continue
                pb = pushbacks[pid]
                if any(b in state['claimed_blocks'] for b in pb['blocks']):
                    continue
                net = pb['income'] + pb['cost']
                if net <= 0:
                    continue
                score = (beta ** start_period) * net  # optimistic ranking score
                cands.append((score, ridx, pid, start_period))
        return cands

    def apply_candidate(state, ridx, pid, start_period):
        pb = pushbacks[pid]
        allocations, completed_period, new_capacity_used = fill_schedule(
            state['capacity_used'], pb['tonnage'], start_period
        )
        if not allocations:
            return None
        net = pb['income'] + pb['cost']
        gain = sum(frac * (beta ** k) * net for k, frac in allocations)
        new_state = {
            'row_selected_pid': dict(state['row_selected_pid']),
            'completed_period': dict(state['completed_period']),
            'schedule': dict(state['schedule']),
            'claimed_blocks': set(state['claimed_blocks']),
            'capacity_used': new_capacity_used,
            'value': state['value'] + gain,
        }
        new_state['row_selected_pid'][ridx] = pid
        new_state['schedule'][pid] = allocations
        new_state['claimed_blocks'].update(pb['blocks'])
        if completed_period is not None:
            new_state['completed_period'][pid] = completed_period
        return new_state

    beam = [{
        'row_selected_pid': {}, 'completed_period': {}, 'schedule': {},
        'claimed_blocks': set(), 'capacity_used': [0.0] * T, 'value': 0.0,
    }]
    steps = 0
    while max_steps is None or steps < max_steps:
        steps += 1
        children = []
        any_expanded = False
        for state in beam:
            cands = sorted(candidates_for_state(state), key=lambda c: c[0], reverse=True)
            applied = False
            for score, ridx, pid, start_period in cands[:candidate_pool_size]:
                child = apply_candidate(state, ridx, pid, start_period)
                if child is not None:
                    children.append(child)
                    applied = True
            if not applied:
                children.append(state)
            else:
                any_expanded = True
        if not any_expanded:
            break
        children.sort(key=lambda s: s['value'], reverse=True)
        beam = children[:beam_width]

    best = max(beam, key=lambda s: s['value'])

    chosen = {}
    for pid, allocations in best['schedule'].items():
        entry = dict(pushbacks[pid])
        entry['schedule'] = {k: frac for k, frac in allocations}
        entry['first_period'] = min(k for k, _ in allocations)
        entry['completed_fraction'] = sum(frac for _, frac in allocations)
        chosen[pid] = entry
    order = sorted(
        chosen.keys(),
        key=lambda p: (chosen[p]['first_period'], -(chosen[p]['income'] + chosen[p]['cost']))
    )

    return {'chosen': chosen, 'order': order, 'objective_value': best['value']}


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