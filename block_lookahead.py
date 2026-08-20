"""
Per-block cone aggregates: what a block unlocks below it, and what must be
stripped above it.

process_pushbacks.py gives each pushback an `income_look_ahead` /
`cost_look_ahead` summed over a box beneath its footprint. This is the same
idea at block level, but summed over the *precedence cone* rather than a box,
so the geometry matches the constraint the projection actually enforces.

With the square-pyramid rule (one lateral step per bench), the descendants of a
block at (ix, iy, iz) truncated at L benches are

    { (ix+dx, iy+dy, iz-l) : l = 1..L,  max(|dx|,|dy|) <= l }

an inverted pyramid widening downward -- exactly the set of blocks that require
this one to be mined first. Reversing the sign of l gives the ancestor cone:
the material that must come off before this block is reachable.

Enumerating grid offsets means each cell is counted once, so there is no
double-counting across overlapping descendant sets (which is what makes the
naive "sum over children's sums" recursion wrong).

Cost: sum_{l=1..L} (2l+1)^2 gathers, each vectorised over n. L=5 is 285 passes.
"""

import numpy as np


def cone_sums(ix, iy, iz, quantities, levels=5, direction="below",
              slope_steps_per_bench=1):
    """Sum each quantity over the truncated precedence cone of every block.

    Parameters
    ----------
    ix, iy, iz : (n,) int   grid indices; iz increases upward (elevation)
    quantities : dict name -> (n,) float
    levels : int            how many benches deep to look
    direction : 'below'     descendants -- what this block unlocks
                'above'     ancestors  -- what must be stripped to reach it
    slope_steps_per_bench : lateral growth per bench; 1 matches
                            square_pyramid_predecessors at equal x/y/z steps

    Returns
    -------
    dict name -> (n,) float, plus 'count' -- blocks in the cone (boundary
    truncation makes this vary, so per-block means are meaningful)
    """
    if direction not in ("below", "above"):
        raise ValueError("direction must be 'below' or 'above'")
    sign = -1 if direction == "below" else +1

    ix = np.asarray(ix, dtype=np.int64)
    iy = np.asarray(iy, dtype=np.int64)
    iz = np.asarray(iz, dtype=np.int64)
    n = ix.shape[0]
    nx, ny, nz = int(ix.max()) + 1, int(iy.max()) + 1, int(iz.max()) + 1

    lut = np.full(nx * ny * nz, -1, dtype=np.int64)
    lut[(ix * ny + iy) * nz + iz] = np.arange(n, dtype=np.int64)

    out = {k: np.zeros(n) for k in quantities}
    count = np.zeros(n)

    for l in range(1, int(levels) + 1):
        r = l * int(slope_steps_per_bench)
        jz = iz + sign * l
        okz = (jz >= 0) & (jz < nz)
        if not okz.any():
            continue
        for dx in range(-r, r + 1):
            jx = ix + dx
            okx = okz & (jx >= 0) & (jx < nx)
            if not okx.any():
                continue
            for dy in range(-r, r + 1):
                jy = iy + dy
                ok = okx & (jy >= 0) & (jy < ny)
                if not ok.any():
                    continue
                src = np.flatnonzero(ok)
                dst = lut[(jx[src] * ny + jy[src]) * nz + jz[src]]
                keep = dst >= 0
                src, dst = src[keep], dst[keep]
                if src.size == 0:
                    continue
                for k, q in quantities.items():
                    np.add.at(out[k], src, np.asarray(q, dtype=float)[dst])
                np.add.at(count, src, 1.0)

    out["count"] = count
    return out
