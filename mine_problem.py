"""
The knob -> schedule -> NPV map, in one place.

Everything downstream of this module (the sensitivity sweep, the decoder
comparison, and the Bayesian-optimisation loop that follows) calls exactly one
function, `evaluate`, and differs only in how it chooses the knobs. Keeping the
map single-sourced is what makes the sweep's conclusions transferable to the
outer loop: if the sweep says a knob is inert, it is inert for BO too.

Pipeline, all of it inside `evaluate`:

    knobs -> priority  reprice by shadow prices, add cone look-ahead bonuses
          -> transform 'raw' | 'smooth' (kernel interpolation) | 'pocs' (kernel
                       projection onto the precedence cone)
          -> decode    'kahn' (priority-driven ready set) | 'sort' (clamp+argsort)
          -> cuts      evaluate_cuts_multi, NPV against the TRUE value

The priority may carry look-ahead bonuses; NPV never scores them. That split is
inherited from reprice.reprice_loop and it matters -- a look-ahead bonus is a
ranking device, not income, and counting it in the objective would flatter
every result by exactly the amount of the bonus.

Cost, measured on the real model (n=12,213, m=96,250): gram 0.06s, POCS ~1.6s,
decode ~0.1s, cuts ~0.05s. So ~1.8s per evaluation on the 'pocs' path and
~0.2s on 'raw'/'smooth'. A 200-point BO run is minutes, not hours.
"""

import numpy as np
import pandas as pd

from block_lookahead import cone_sums
from capacity_cuts import evaluate_cuts_multi
from decoders import children_csr, decode
from kernel_projection import (ard_lengthscales, build_edges, build_features,
                               kernel_interpolate, minmax_normalize,
                               project_hard_clamp, project_pocs_sparse,
                               schedule_from_scores, sequence_violations,
                               topological_order, wendland_c0_sparse_gram)
from lg_utils import square_pyramid_predecessors

T_PERIODS = 10
DISCOUNT = 0.90
SLOPE_H_PER_V = 1.5

# Cost deck, $ per tonne. process_blocks.py emits physical quantities only, so
# every economic assumption is here and can be changed without re-aggregating.
#   MINING     charged on all material moved
#   HAULAGE    per tonne per bench of depth -- the only term that makes depth
#              cost anything, and therefore the only reason a deep block is
#              worse than a shallow one of identical grade
#   PROCESSING charged on mill feed only
MINING_COST_PER_T = 3.0
HAULAGE_COST_PER_T_BENCH = 0.15
PROCESSING_COST_PER_T = 18.0

# POCS convergence tolerance. Loosened from project_pocs_sparse's own 1e-10
# because with the kahn decoder the projection is no longer load-bearing for
# FEASIBILITY -- the ready-set construction enforces precedence whatever the
# scores look like, so a residual slack of 1e-6 cannot produce a violation. It
# only affects how thoroughly the score field is smoothed, which makes the
# tolerance a quality/cost dial rather than a correctness requirement, and the
# sweep count is roughly logarithmic in it.
POCS_TOL = 1e-6

# name -> (lo, hi, default, kind). This is the BO search space, declared once.
# 'active' records which transform each knob can possibly influence; a knob
# outside the active transform is a constant and the sweep reports it as such
# rather than pretending to have measured it.
#
# Ranges were widened after the first BO run on the crop-14 instance put its
# best point on the boundary in 5 of 11 dimensions -- the optimum was outside
# the declared box, so BO and random search were both pinned against the same
# wall and tied at +7.5%. A boundary hit is not a result, it is a statement
# that the box was drawn too small.
#
# omega spans both sides of 1.0 deliberately: the full model preferred 0.05
# (heavy under-relaxation) while the crop preferred 1.0, so the useful setting
# is instance-dependent and the range has to contain both. Values above 1.0 are
# over-relaxation and may leave POCS short of the cone, which is harmless here
# because the kahn decoder enforces precedence itself.
KNOBS = {
    "mu_mining":       (-0.5, 1.2, 0.0, "float", ("raw", "smooth", "pocs")),
    "mu_processing":   (-0.5, 1.2, 0.0, "float", ("raw", "smooth", "pocs")),
    "mu_stripping":    (-0.5, 1.2, 0.0, "float", ("raw", "smooth", "pocs")),
    "w_below_value":   (-6.0, 6.0, 0.0, "float", ("raw", "smooth", "pocs")),
    "w_below_income":  (-6.0, 6.0, 0.0, "float", ("raw", "smooth", "pocs")),
    "w_below_tonnage": (-6.0, 6.0, 0.0, "float", ("raw", "smooth", "pocs")),
    "w_above_value":   (-6.0, 6.0, 0.0, "float", ("raw", "smooth", "pocs")),
    # intrinsic position -- see intrinsic_geometry(). These four are what
    # replaces absolute x/y/z as a *predictor*; all zero recovers the previous
    # parameterisation exactly, so the smaller space is nested in this one and
    # BO can always fall back to it.
    "w_bench_frac":    (-6.0, 6.0, 0.0, "float", ("raw", "smooth", "pocs")),
    "w_r_centroid":    (-6.0, 6.0, 0.0, "float", ("raw", "smooth", "pocs")),
    "w_d_ore":         (-6.0, 6.0, 0.0, "float", ("raw", "smooth", "pocs")),
    "w_anc_frac":      (-6.0, 6.0, 0.0, "float", ("raw", "smooth", "pocs")),
    "cone_levels":     (1, 12, 5, "int", ("raw", "smooth", "pocs")),
    # capped at 6, down from 10. nnz_per_row of the Gram scales as
    # (2*radius_xy+1)^2 * (2*radius_z+1), and POCS reads two Gram rows per edge
    # per sweep, so the projection cost -- 90% of an evaluation -- is roughly
    # linear in it. The sweep put the optimum near 3.3 and measured only 3.06%
    # span across the whole 2-10 range, so the upper half was buying nothing
    # and costing ~3x.
    # defaults dropped from 6.0/2.0. Narrowing the support to 3/1 measured 1.9x
    # faster (nnz/row 37.6 -> 10.4) and +0.52% NPV on crop-14, and the
    # sensitivity sweep put the optimum near 3.3 on the full model. The old
    # default was buying density nothing wanted.
    "radius_xy":       (2.0, 6.0, 3.5, "float", ("smooth", "pocs")),
    "radius_z":        (1.0, 4.0, 1.5, "float", ("smooth", "pocs")),
    "alpha_smooth":    (0.0, 1.0, 1.0, "float", ("smooth",)),
    "omega":           (0.02, 1.5, 0.2, "float", ("pocs",)),
}


def default_knobs():
    return {k: v[2] for k, v in KNOBS.items()}


def knob_active(name, transform):
    return transform in KNOBS[name][4]


def _z(a):
    """Standardise so the weight knobs are commensurate. Cone aggregates run
    six orders of magnitude above per-block value; without this the w_* range
    [-1, 1] would mean 'ignore value entirely' at one end and nothing at the
    other."""
    a = np.asarray(a, dtype=float)
    sd = a.std()
    return (a - a.mean()) / (sd if sd > 0 else 1.0)


def crop_slab(blocks, width, step):
    """Keep a contiguous slab of x-columns. A smaller instance for fast work.

    `width` is either an int -- a slab of that many columns centred on the
    income centroid -- or an explicit (lo, hi) window in grid-column units,
    which is how a family of genuinely different instances is cut from one
    block model.

    Cropping in x only, keeping every y and z, is the one subsample that leaves
    the problem well posed: the square-pyramid precedence cone is local in x/y
    and full-depth in z, so a contiguous slab keeps every cone intact except at
    the two cut faces -- which is exactly what the boundary of a smaller pit
    looks like anyway. Randomly sampling blocks instead would delete blocks
    from the middle of cones and produce a precedence graph no pit could have.

    Centring on the income centroid rather than the grid centre keeps the
    high-grade material in frame; a slab of barren rock would have a
    degenerate schedule and would tell us nothing.
    """
    ix = np.rint((blocks["x"].to_numpy(float)
                  - blocks["x"].min()) / step).astype(int)
    if isinstance(width, (tuple, list)):
        lo, hi = int(width[0]), int(width[1])
    else:
        inc = blocks["income"].to_numpy(float)
        centre = int(round(float((ix * inc).sum() / max(inc.sum(), 1e-12))))
        half = int(width) // 2
        lo = max(0, min(centre - half, int(ix.max()) + 1 - int(width)))
        hi = lo + int(width)
    keep = (ix >= lo) & (ix < hi)
    return blocks.loc[keep].reset_index(drop=True), (lo, hi)


def load_static(path="inputs/blocks.csv", t_periods=T_PERIODS, crop=None):
    """Everything that does not depend on the knobs: geometry, economics, the
    precedence graph, the topological order and the CSR adjacency.

    Reads the physical columns process_blocks.py writes -- tonnage (all
    material, host rock included), ore_tonnage (mill feed), income (gross
    revenue), and the grades -- and applies the cost deck above. Nothing is
    synthesised here: an earlier version of blocks.csv had a constant income
    and zero tonnage on every uncovered cell, which forced callers to invent
    tonnage from a random fill. That is fixed upstream, so the instance is now
    deterministic.

    `is_ore` is the economic definition -- material whose revenue clears its
    own processing cost -- rather than a geometric flag, so the processing
    capacity is charged against exactly the tonnes a mill would accept.
    """
    blocks = pd.read_csv(path)
    step = float(blocks["x_step"].iloc[0])

    required = {"tonnage", "ore_tonnage", "income"}
    missing = required - set(blocks.columns)
    if missing:
        raise ValueError(
            f"{path} is missing {sorted(missing)} -- it predates the "
            "process_blocks.py aggregation fix. Re-run `python process_blocks.py`.")

    crop_window = None
    if crop:
        blocks, crop_window = crop_slab(blocks, crop, step)
    n = len(blocks)

    x = blocks["x"].to_numpy(float)
    y = blocks["y"].to_numpy(float)
    z_elev = -blocks["z"].to_numpy(float)
    income = blocks["income"].to_numpy(float)
    tonnage = blocks["tonnage"].to_numpy(float)
    ore_tonnage = blocks["ore_tonnage"].to_numpy(float)

    ix = np.rint((x - x.min()) / step).astype(int)
    iy = np.rint((y - y.min()) / step).astype(int)
    iz = np.rint((z_elev - z_elev.min()) / step).astype(int)
    bench = iz.max() - iz

    is_ore = income > PROCESSING_COST_PER_T * ore_tonnage
    mill_feed = np.where(is_ore, ore_tonnage, 0.0)
    value = (income
             - MINING_COST_PER_T * tonnage
             - HAULAGE_COST_PER_T_BENCH * bench * tonnage
             - PROCESSING_COST_PER_T * mill_feed)

    par, chi = build_edges(square_pyramid_predecessors(
        pd.DataFrame({"x_c": x, "y_c": y, "z_c": z_elev}),
        slope_h_per_v=SLOPE_H_PER_V).tolist())
    order = topological_order(n, par, chi)
    csr = children_csr(n, par, chi)

    topo_rank = np.empty(n, dtype=np.float64)
    topo_rank[order] = np.arange(n)

    # mining = processing + stripping by construction, so the stripping cap
    # only bites where waste handling is limited independently of total movement
    waste = tonnage - mill_feed
    w = np.stack([tonnage, mill_feed, waste])
    # 0.55 of the deposit fits inside the horizon, so capacity is genuinely
    # scarce and the sequence -- not just the pit outline -- decides the NPV
    caps = np.stack([np.full(t_periods, .55 * tonnage.sum() / t_periods),
                     np.full(t_periods, .55 * mill_feed.sum() / t_periods),
                     np.full(t_periods, .60 * waste.sum() / t_periods)])
    resources = [{"name": nm, "weight": w[k], "capacity": caps[k]}
                 for k, nm in enumerate(("mining", "processing", "stripping"))]

    return {
        "n": n, "x": x, "y": y, "z": z_elev, "ix": ix, "iy": iy, "iz": iz,
        "bench": bench, "income": income, "tonnage": tonnage,
        "ore_tonnage": ore_tonnage, "mill_feed": mill_feed, "is_ore": is_ore,
        "au": blocks["au"].to_numpy(float) if "au" in blocks else np.zeros(n),
        "cu": blocks["cu"].to_numpy(float) if "cu" in blocks else np.zeros(n),
        "value": value, "par": par, "chi": chi, "order": order, "csr": csr,
        "topo_rank": topo_rank, "weights": w, "resources": resources,
        "nx": int(ix.max()) + 1, "ny": int(iy.max()) + 1, "nz": int(iz.max()) + 1,
        "path": path, "crop": crop, "crop_window": crop_window,
        "_cone": {},
    }


def cone_cache(static, levels):
    """cone_sums for a given depth, memoised -- `cone_levels` is a sweep knob
    and recomputing 285 gathers per evaluation is pure waste."""
    levels = int(levels)
    if levels not in static["_cone"]:
        qty = {"value": static["value"], "income": static["income"],
               "tonnage": static["tonnage"]}
        static["_cone"][levels] = {
            "below": cone_sums(static["ix"], static["iy"], static["iz"], qty,
                               levels=levels, direction="below"),
            "above": cone_sums(static["ix"], static["iy"], static["iz"], qty,
                               levels=levels, direction="above"),
        }
    return static["_cone"][levels]


def priority_from_knobs(static, knobs):
    """Repriced value plus cone look-ahead bonuses, standardised.

    Repricing is the reprice.py channel: a scarce resource makes the blocks
    that consume it look dearer, which promotes high value-per-tonne material.
    The look-ahead terms are the block-level analogue of the pushback
    income_look_ahead, summed over the precedence cone so the geometry matches
    the constraint. Signs are left free -- a rich ancestor cone can mean either
    'expensive to reach' or 'the stripping pays for itself', and which one wins
    is an empirical question the sweep is there to answer.

    Every term is standardised before it is weighted, mu included. Repricing on
    the raw scale, the way reprice.py does it, silently dies on this instance:
    block value spans $25M while mu * tonnage tops out near $34k, so the whole
    mu channel is three orders of magnitude too small to reorder anything and
    the first sensitivity sweep measured it as exactly inert. Standardising
    puts mu in units of 'standard deviations of value per standard deviation of
    resource consumed', which is commensurate with the w_* weights, independent
    of the price deck, and keeps the priority a plain linear combination of
    standardised features -- the form BO wants, and the form a frozen-encoder
    linear head would take later.

    Subtracting a column's mean shifts every score equally and cannot change a
    ranking, so standardising the weights is a pure rescale of the mu channel.
    """
    k = {**default_knobs(), **knobs}
    prio = _z(static["value"])
    for name, r in zip(("mu_mining", "mu_processing", "mu_stripping"),
                       static["resources"]):
        mu = float(k[name])
        if mu != 0.0:
            prio = prio - mu * _z(r["weight"])

    cone = cone_cache(static, k["cone_levels"])
    terms = (("w_below_value", cone["below"]["value"]),
             ("w_below_income", cone["below"]["income"]),
             ("w_below_tonnage", cone["below"]["tonnage"]),
             ("w_above_value", cone["above"]["value"]))
    for name, q in terms:
        wq = float(k[name])
        if wq != 0.0:
            prio = prio + wq * _z(q)

    # intrinsic position. Note these enter the PRIORITY, where absolute x/y/z
    # never did -- x/y/z appear only in the Gram, where they act as the metric
    # that makes the kernel local, not as a predictor of what to mine first.
    geom_terms = (("w_bench_frac", "bench_frac"), ("w_r_centroid", "r_centroid"),
                  ("w_d_ore", "d_ore"), ("w_anc_frac", "anc_frac"))
    if any(float(k[n]) != 0.0 for n, _ in geom_terms):
        g = intrinsic_geometry(static)
        for name, key in geom_terms:
            wq = float(k[name])
            if wq != 0.0:
                prio = prio + wq * _z(g[key])
    return _z(prio)


def intrinsic_geometry(static):
    """Position described by the deposit's own structure, not by a coordinate.

    Absolute x/y/z cannot transfer between instances -- the ore sits somewhere
    else in each one. Min-max normalising them, which is what the current
    feature path does, fixes the units but not the reference frame: the frame
    is the bounding box, and the bounding box is an artifact of where the data
    happens to have been cut. Two slabs of the same deposit give the same
    normalised coordinate to blocks in completely different geological
    positions.

    These four are intrinsic -- computable from the instance alone, meaning the
    same thing in every instance:

      bench_frac   depth below the top bench, as a fraction of total depth.
                   Depth is the one absolute direction that IS meaningful:
                   gravity picks it out, and everything above a block must come
                   off before it can be mined.
      r_centroid   distance from the income-weighted centroid, in block widths.
                   'How far into the fringe am I' -- the coordinate the pit
                   shell actually cares about.
      d_ore        distance to the nearest ore block, in block widths, which
                   separates barren rock that is on the way to something from
                   barren rock that is not.
      anc_frac     ancestors / (ancestors + descendants) from the cone counts.
                   Pure graph position: 0 at the surface, 1 at the pit bottom,
                   and it needs no geometry at all.

    Cached on the static dict; none of it depends on the knobs.
    """
    if "_geom" in static:
        return static["_geom"]

    ix, iy, iz = static["ix"], static["iy"], static["iz"]
    inc = static["income"]
    tot = max(float(inc.sum()), 1e-12)
    cx = float((ix * inc).sum() / tot)
    cy = float((iy * inc).sum() / tot)
    cz = float((iz * inc).sum() / tot)
    r_centroid = np.sqrt((ix - cx) ** 2 + (iy - cy) ** 2 + (iz - cz) ** 2)

    bench = static["bench"].astype(float)
    bench_frac = bench / max(float(bench.max()), 1.0)

    # distance to nearest ore, by multi-source BFS over the 6-neighbourhood on
    # the block grid -- O(n) and exact in grid steps, no scipy needed
    ore = np.flatnonzero(static["is_ore"])
    nx, ny, nz = static["nx"], static["ny"], static["nz"]
    lut = np.full(nx * ny * nz, -1, dtype=np.int64)
    lin = (ix * ny + iy) * nz + iz
    lut[lin] = np.arange(static["n"])
    d_ore = np.full(static["n"], -1.0)
    if ore.size:
        d_ore[ore] = 0.0
        frontier = ore
        step = 0
        while frontier.size:
            step += 1
            nbrs = []
            for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                               (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                jx, jy, jz = ix[frontier] + dx, iy[frontier] + dy, iz[frontier] + dz
                ok = ((jx >= 0) & (jx < nx) & (jy >= 0) & (jy < ny)
                      & (jz >= 0) & (jz < nz))
                if not ok.any():
                    continue
                cand = lut[(jx[ok] * ny + jy[ok]) * nz + jz[ok]]
                nbrs.append(cand[cand >= 0])
            if not nbrs:
                break
            cand = np.unique(np.concatenate(nbrs))
            cand = cand[d_ore[cand] < 0]
            if cand.size == 0:
                break
            d_ore[cand] = step
            frontier = cand
        d_ore[d_ore < 0] = d_ore.max() + 1.0   # unreachable: past the far edge
    else:
        d_ore[:] = 0.0

    cone = cone_cache(static, 5)
    anc = cone["above"]["count"]
    des = cone["below"]["count"]
    anc_frac = anc / np.maximum(anc + des, 1.0)

    static["_geom"] = {"bench_frac": bench_frac, "r_centroid": r_centroid,
                       "d_ore": d_ore, "anc_frac": anc_frac}
    return static["_geom"]


def build_gram(static, s_raw, knobs, features="full"):
    """Sparse Wendland Gram. `features` selects the kernel argument.

    'full'       x, y, z, bench, score, value, cone aggregates, tonnage -- the
                 original, and still the default. The metric is
                 economics-aware: two blocks alike in geometry but not in value
                 are far apart.
    'geometric'  x, y, z, bench only. Purely spatial.

    'geometric' looked like a clear win at radius_xy=6 -- faster AND +4.4% NPV
    on crop-14 -- but almost all of that was the old wide support hurting
    'full' more than it hurt a purely spatial metric. Re-measured at the new
    3.5/1.5 default the two are close and the ordering is not consistent:
    geometric wins NPV on all four slabs (+0.03% to +3.1%) while losing time on
    three of them, and on the full model it is 1.35x faster but 0.35% worse.
    Mixed, so it stays opt-in rather than becoming the default; changing the
    metric would re-baseline every gain measured so far for an ambiguous gain.

    The reason to keep it available is not the speed, it is that 'geometric' is
    translation-invariant on the grid: every interior row has the same weights
    up to a shift. That makes a stencil formulation possible -- one shared
    weight vector resident in L1 instead of n rows streamed from memory --
    which is the real prize and is not implemented here. 'full' can never be
    stencilled, because its score and value columns differ per block.

    Either way the support radius is tied to the lengthscale rather than held
    at a hard-coded (6,6,2): the Wendland kernel is exactly zero past one
    lengthscale, so a pattern narrower than `radius_*` would silently truncate
    nonzeros and a wider one would only store zeros.
    """
    k = {**default_knobs(), **knobs}
    if features == "geometric":
        Z = minmax_normalize(np.column_stack([
            static["x"], static["y"], static["z"],
            static["bench"].astype(float)]))
    elif features == "full":
        cone = cone_cache(static, k["cone_levels"])
        lb, la = cone["below"], cone["above"]
        Z = minmax_normalize(build_features(
            static["x"], static["y"], static["z"], static["bench"], s_raw,
            static["value"], lb["value"], static["tonnage"],
            extra=[lb["income"], lb["tonnage"], la["value"], la["tonnage"]]))
    else:
        raise ValueError("features must be 'geometric' or 'full'")
    ell = ard_lengthscales(Z.shape[1], static["nx"], static["ny"], static["nz"],
                           float(k["radius_xy"]), float(k["radius_z"]))
    radius = (int(np.ceil(k["radius_xy"])), int(np.ceil(k["radius_xy"])),
              int(np.ceil(k["radius_z"])))
    return wendland_c0_sparse_gram(Z, static["ix"], static["iy"], static["iz"],
                                   radius, ell)


def transform_scores(static, s_raw, knobs, transform, max_sweeps=5000,
                     pocs_tol=POCS_TOL):
    """'raw' | 'smooth' (kernel interpolation) | 'pocs' (kernel projection).

    Both kernel paths finish with project_hard_clamp, which is what actually
    makes the returned score a point of the cone.

    This is not belt-and-braces. POCS terminates when its largest per-sweep
    shift falls under `tol`, which leaves residual slacks of that order --
    measured 2,071 edges violated at 4.5e-6 on the crop-14 instance at
    tol=1e-6, and the same thing happens at 1e-10, just four orders smaller.
    schedule_from_scores has always known this and clamped internally for
    exactly this reason; the kahn decoder does not need it, so nothing
    downstream was enforcing it any more and the projection was quietly
    returning a near-feasible point. The clamp costs 0.010s against POCS's
    0.34s, moves the score by at most 2.5e-05, and costs 0.16% NPV. Cheap
    enough that there is no reason to run without it.
    """
    k = {**default_knobs(), **knobs}
    if transform == "raw":
        return s_raw, {}
    gram, ginfo = build_gram(static, s_raw, knobs)
    par, chi, order = static["par"], static["chi"], static["order"]
    if transform == "smooth":
        s = kernel_interpolate(s_raw, gram, alpha=float(k["alpha_smooth"]))
        return project_hard_clamp(s, par, chi, order), \
            {"nnz_per_row": ginfo["nnz_per_row"]}
    if transform == "pocs":
        s, pinfo = project_pocs_sparse(s_raw, gram, par, chi, order,
                                       max_sweeps=max_sweeps,
                                       tol=pocs_tol, omega=float(k["omega"]))
        s = project_hard_clamp(s, par, chi, order)
        return s, {"nnz_per_row": ginfo["nnz_per_row"], "sweeps": pinfo["sweeps"],
                   "converged": pinfo["converged"],
                   "primal_residual": pinfo["primal_residual"]}
    raise ValueError("transform must be 'raw', 'smooth' or 'pocs'")


def evaluate(static, knobs=None, transform="pocs", decoder="sort",
             tiebreak=None, noise=None, discount=DISCOUNT, check=True,
             max_sweeps=5000, pocs_tol=POCS_TOL, s_raw=None):
    """One knob setting -> one hard NPV. The objective an outer loop optimises.

    `tiebreak` is passed through to the kahn decoder; None means topological
    rank, which is the deterministic convention. Returning `pos` and
    `start_period` lets callers measure how far two schedules actually are
    apart, which is the quantity that decides whether a search space is worth
    searching.

    `noise` is added to the priority BEFORE the transform, which is the only
    placement that measures anything useful: perturbing afterwards would step
    around the contraction the transform applies rather than through it.
    """
    knobs = {} if knobs is None else knobs
    # `s_raw` bypasses the knob parameterisation entirely, which is how a
    # learned scorer enters: the network emits a score field directly and the
    # rest of the pipeline -- projection, decoder, cuts -- is unchanged. The
    # knobs still describe the projection, so they are not ignored.
    if s_raw is None:
        s_raw = priority_from_knobs(static, knobs)
    else:
        s_raw = _z(np.asarray(s_raw, dtype=float))
    if noise is not None:
        s_raw = s_raw + np.asarray(noise, dtype=float)
    s_used, tinfo = transform_scores(static, s_raw, knobs, transform,
                                     max_sweeps=max_sweeps, pocs_tol=pocs_tol)

    tb = static["topo_rank"] if tiebreak is None else tiebreak
    seq = decode(s_used, decoder, par=static["par"], chi=static["chi"],
                 order=static["order"], csr=static["csr"], tiebreak=tb)

    res = evaluate_cuts_multi(seq, static["resources"], value=static["value"],
                              discount=discount)
    pos = np.empty(static["n"], dtype=np.int64)
    pos[seq] = np.arange(static["n"])

    out = {"npv": float(res["npv"]), "seq": seq, "pos": pos,
           "start_period": res["start_period"], "n_unmined": res["n_unmined"],
           "mined_fraction": res["mined_fraction"], "s_raw": s_raw,
           "s_used": s_used, "transform": transform, "decoder": decoder,
           **tinfo}
    if check:
        out["violations"] = sequence_violations(seq, static["par"], static["chi"])
    return out
