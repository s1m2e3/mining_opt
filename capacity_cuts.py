"""
Cut a precedence-feasible mining sequence into periods under tonnage capacity.

Given the permutation pi produced by kernel_projection.schedule_from_scores, the
period boundaries are not decisions -- they are determined by inverting the
cumulative tonnage curve at the cumulative capacity levels:

    W_k   = sum_{j<=k} w_{pi(j)}          cumulative tonnage along the sequence
    C^_t  = sum_{u<=t} C_u                 cumulative capacity
    kappa_t = W^{-1}(C^_t)                 period-t boundary, a real position

Splitting the straddling block makes kappa_t continuous and wastes no capacity.
Each block pi(j) occupies the tonnage interval [W_{j-1}, W_j]; period t occupies
[C^_{t-1}, C^_t]; the block's effective discount is the tonnage-weighted overlap

    delta_j = (1/w_j) * sum_t gamma^(t-1) * |[W_{j-1}, W_j] ^ [C^_{t-1}, C^_t]|
    NPV     = sum_j v_{pi(j)} * delta_j

A block whose interval extends past C^_T is partly or wholly unmined, which
falls out of the same overlap arithmetic with no special case.

With a second resource (processing capacity, charged only to ore) there are two
tonnage axes and a period ends wherever the first binds. Capacity is consumed
per period and does not bank forward.

The result is piecewise linear in the capacities and piecewise *constant* in the
scores -- it depends on s only through the permutation, so no gradient survives
this step.
"""

import numpy as np


def make_resource(name, weight, capacity):
    """One capacity axis: a per-block consumption vector and a per-period cap.

    mining      weight = tonnage                      (everything moved)
    processing  weight = tonnage * is_ore             (mill feed)
    stripping   weight = tonnage * ~is_ore            (waste handling)

    Note mining = processing + stripping, so only two of those three are
    independent; a separate stripping cap only bites if waste handling is
    limited independently of total movement.
    """
    return {"name": str(name),
            "weight": np.asarray(weight, dtype=float),
            "capacity": np.asarray(capacity, dtype=float)}


def evaluate_cuts_multi(seq, resources, value=None, discount=1.0):
    """Cut a sequence into periods under an arbitrary number of capacity axes.

    A period ends where the *first* resource runs out:

        kappa_t = min_r (W^r)^{-1}( W^r(kappa_{t-1}) + C^r_t )

    Capacity is consumed per period and does not bank forward. Blocks straddling
    a boundary are split pro rata, so no capacity is wasted and a block may span
    several periods.

    Parameters
    ----------
    seq : (n,) int      mining order, from schedule_from_scores
    resources : list    of make_resource(...) dicts, all with the same T
    value : (n,) float  undiscounted per-block value, for the npv field
    discount : float    gamma; period 1 is undiscounted

    Returns dict: cuts, start_period, discount_factor, mined_fraction,
    per_resource {name: (T,) consumption}, binding {name: (T,) bool}, npv,
    n_unmined, n_partial, n_split.
    """
    seq = np.asarray(seq, dtype=np.int64)
    n = seq.shape[0]
    if not resources:
        raise ValueError("at least one resource is required")
    T = int(resources[0]["capacity"].shape[0])
    for r in resources:
        if r["capacity"].shape[0] != T:
            raise ValueError("all resources must share the same number of periods")
        if r["weight"].shape[0] != n:
            raise ValueError("resource weights must be length n")

    R = len(resources)
    Wt = np.stack([r["weight"] for r in resources])        # (R, n)
    Cap = np.stack([r["capacity"] for r in resources])     # (R, T)

    start_period = np.full(n, -1, dtype=np.int64)
    delta = np.zeros(n)
    mined = np.zeros(n)
    used = np.zeros((R, T))
    cuts = np.zeros(T)

    t = 0
    left = Cap[:, 0].copy() if T else np.zeros(R)
    pos = 0.0
    n_split = 0
    eps = 1e-12

    for j in range(n):
        b = seq[j]
        wb = Wt[:, b]
        if not np.any(wb > 0):                 # consumes nothing: free
            if t < T:
                start_period[b] = t
                delta[b] = discount ** t
                mined[b] = 1.0
            pos += 1.0
            continue

        remaining = 1.0
        pieces = 0
        while remaining > eps and t < T:
            f = remaining
            for r in range(R):
                if wb[r] > 0:
                    f = min(f, left[r] / wb[r])
            if f <= eps:                       # a resource is exhausted
                cuts[t] = pos
                t += 1
                if t < T:
                    left = Cap[:, t].copy()
                continue
            if start_period[b] < 0:
                start_period[b] = t
            delta[b] += f * (discount ** t)
            mined[b] += f
            used[:, t] += f * wb
            left -= f * wb
            remaining -= f
            pos += f
            pieces += 1
        if pieces > 1:
            n_split += 1
        if remaining > eps:                    # horizon exhausted mid-block
            break

    for u in range(t, T):
        cuts[u] = pos

    per_res = {r["name"]: used[k] for k, r in enumerate(resources)}
    binding = {r["name"]: (used[k] >= r["capacity"] * (1.0 - 1e-9))
               for k, r in enumerate(resources)}
    return {
        "cuts": cuts,
        "start_period": start_period,
        "discount_factor": delta,
        "mined_fraction": mined,
        "per_resource": per_res,
        "capacity": {r["name"]: r["capacity"] for r in resources},
        "binding": binding,
        "n_unmined": int((mined <= 1e-12).sum()),
        "n_partial": int(((mined > 1e-12) & (mined < 1.0 - 1e-12)).sum()),
        "n_split": int(n_split),
        "npv": (float(np.dot(np.asarray(value, dtype=float), delta))
                if value is not None else float("nan")),
    }


def evaluate_cuts(seq, tonnage, capacities, value=None, discount=1.0,
                  ore_mask=None, proc_capacities=None):
    """Assign a mining sequence to periods under one or two capacity limits.

    Parameters
    ----------
    seq : (n,) int
        Mining order, highest priority first -- from schedule_from_scores.
    tonnage : (n,) float
        Per-block tonnage, indexed by block id (not by position in seq).
    capacities : (T,) float
        Mining capacity per period, in tonnes. Charged to every block.
    value : (n,) float, optional
        Undiscounted per-block value. Required for the npv field.
    discount : float
        Per-period discount gamma; period 1 is undiscounted (gamma^0).
    ore_mask : (n,) bool, optional
        True for blocks that consume processing capacity.
    proc_capacities : (T,) float, optional
        Processing capacity per period. Requires ore_mask.

    Returns
    -------
    dict with
        cuts            (T,) float   period boundaries as continuous positions
        start_period    (n,) int     first period each block is mined in, -1 if never
        discount_factor (n,) float   delta_j, the tonnage-weighted discount
        mined_fraction  (n,) float   fraction of each block mined within the horizon
        tons_per_period (T,) float
        ore_tons_per_period (T,) float
        npv             float
        n_unmined, n_partial, n_split   ints
    """
    seq = np.asarray(seq, dtype=np.int64)
    w = np.asarray(tonnage, dtype=float)
    n = seq.shape[0]
    caps = np.asarray(capacities, dtype=float)
    T = caps.shape[0]

    if proc_capacities is not None and ore_mask is None:
        raise ValueError("proc_capacities requires ore_mask")
    ore = (np.asarray(ore_mask, dtype=bool) if ore_mask is not None
           else np.zeros(n, dtype=bool))
    pcaps = (np.asarray(proc_capacities, dtype=float)
             if proc_capacities is not None else None)
    if pcaps is not None and pcaps.shape[0] != T:
        raise ValueError("proc_capacities must have the same length as capacities")

    start_period = np.full(n, -1, dtype=np.int64)
    delta = np.zeros(n)
    mined = np.zeros(n)
    tons_t = np.zeros(T)
    ore_t = np.zeros(T)
    cuts = np.zeros(T)

    t = 0
    cap_left = caps[0] if T else 0.0
    pcap_left = pcaps[0] if (pcaps is not None and T) else np.inf
    pos = 0.0
    n_split = 0

    for j in range(n):
        b = seq[j]
        wb = w[b]
        if wb <= 0:                       # zero-tonnage block: free, no capacity
            if t < T:
                start_period[b] = t
                delta[b] = discount ** t
                mined[b] = 1.0
            pos += 1.0
            continue

        ore_rate = 1.0 if ore[b] else 0.0
        remaining = 1.0                   # fraction of this block still to place
        pieces = 0

        while remaining > 1e-15 and t < T:
            f_mine = cap_left / wb
            f_proc = (pcap_left / (wb * ore_rate)) if ore_rate > 0 else np.inf
            f = min(remaining, f_mine, f_proc)

            if f <= 1e-15:                # this period is full; advance
                cuts[t] = pos
                t += 1
                if t < T:
                    cap_left = caps[t]
                    pcap_left = pcaps[t] if pcaps is not None else np.inf
                continue

            if start_period[b] < 0:
                start_period[b] = t
            delta[b] += f * (discount ** t)
            mined[b] += f
            tons_t[t] += f * wb
            ore_t[t] += f * wb * ore_rate
            cap_left -= f * wb
            pcap_left -= f * wb * ore_rate
            remaining -= f
            pos += f
            pieces += 1

        if pieces > 1:
            n_split += 1
        if remaining > 1e-15:             # horizon exhausted mid-block
            pos += 0.0
            break

    # any period never reached ends where the sequence stopped
    for u in range(t, T):
        cuts[u] = pos

    out = {
        "cuts": cuts,
        "start_period": start_period,
        "discount_factor": delta,
        "mined_fraction": mined,
        "tons_per_period": tons_t,
        "ore_tons_per_period": ore_t,
        "n_unmined": int((mined <= 1e-12).sum()),
        "n_partial": int(((mined > 1e-12) & (mined < 1.0 - 1e-12)).sum()),
        "n_split": int(n_split),
        "npv": float(np.dot(np.asarray(value, dtype=float), delta))
               if value is not None else float("nan"),
    }
    return out


def capacity_report(res, capacities, proc_capacities=None):
    """Per-period utilisation table as (period, tons, cap, util%, ore, ocap, outil%)."""
    caps = np.asarray(capacities, dtype=float)
    rows = []
    for t in range(caps.shape[0]):
        u = 100.0 * res["tons_per_period"][t] / caps[t] if caps[t] > 0 else 0.0
        if proc_capacities is not None:
            pc = float(np.asarray(proc_capacities, dtype=float)[t])
            ou = 100.0 * res["ore_tons_per_period"][t] / pc if pc > 0 else 0.0
            rows.append((t + 1, res["tons_per_period"][t], caps[t], u,
                         res["ore_tons_per_period"][t], pc, ou))
        else:
            rows.append((t + 1, res["tons_per_period"][t], caps[t], u,
                         np.nan, np.nan, np.nan))
    return rows
