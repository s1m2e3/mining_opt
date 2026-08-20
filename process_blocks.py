"""
Regularise the sub-block model 'MB UG.csv' onto a 20 m cubic grid.

The source is a variable-sized sub-block model of a high-grade underground
envelope: 264,742 sub-blocks, volumes 0.016-1000 m3, carrying CU, AU and a real
density SG. It covers only ~1,759 cells' worth of volume, so on a 12,213-cell
20 m grid the great majority of cells contain no source data at all -- they are
host rock outside the modelled envelope, not zero-tonnage blocks.

Two things this fixes relative to the previous version:

1. Aggregation. The old inner loop computed the containment mask into `df_int`
   and then wrote `df['income'].mean()` -- the mean over the WHOLE dataframe,
   not over the cell. Every non-empty cell therefore received the identical
   global mean, 137,131.65, which is why the resulting income column had
   exactly two distinct values and every block tied at the raw score. The
   source has 185,335 distinct sub-block incomes; all of that variation was
   being discarded.

   Income and tonnage are extensive, so a cell takes the SUM of what it
   contains, not a mean. Grades are intensive and are recovered as
   tonnage-weighted means.

2. Uncovered volume. A cell may be empty or only partly covered. The remainder
   is host rock, and on an open-pit precedence problem it still has to be
   moved, so it is charged its real tonnage at the mean envelope density.
   Leaving it at zero made those blocks free of every capacity constraint,
   which is what forced downstream scripts to synthesise tonnage from a random
   fill.

Density jitter (`sg_noise`) breaks the exact ties among cells that are pure
host rock and would otherwise be numerically identical. It is seeded and
defaults to 4%, which is inside the real within-lithology spread of specific
gravity; a constant waste density is the modelling artifact, not the jitter.
Set sg_noise=0.0 for the strictly deterministic version.

Outputs physical quantities only -- tonnage, gross income, grades, coverage.
The cost model and the resulting block value live in mine_problem.load_static,
so that price and cost assumptions can be changed without re-running this.
"""

import os

import numpy as np
import pandas as pd

STEP = 20.0
# price deck, unchanged from the original: gold at 4000 $/oz over 32 g/oz, and
# copper at 8000 $/t of metal against a grade in percent
AU_PRICE_PER_G = 4000.0 / 32.0
CU_PRICE_PER_PCT_T = 0.01 * 8000.0


def _numeric(col):
    """The source encodes missing values as '-', which makes the whole column
    object dtype. Coerce and treat non-numeric as absent."""
    return pd.to_numeric(col, errors="coerce")


def aggregate_to_grid(df, step=STEP, sg_noise=0.04, seed=0):
    """Sum sub-blocks into 20 m cells; fill uncovered volume with host rock.

    Returns a DataFrame in the same meshgrid row order the previous version
    produced (y-major, then x, then z), so `index` stays aligned with every
    artifact already derived from blocks.csv.
    """
    xc = df["XC"].to_numpy(float)
    yc = df["YC"].to_numpy(float)
    zc = df["ZC"].to_numpy(float)

    vol = (df["XINC"].to_numpy(float) * df["YINC"].to_numpy(float)
           * df["ZINC"].to_numpy(float))
    sg = _numeric(df["SG"]).to_numpy(float)
    sg_default = float(np.nanmedian(sg))
    sg = np.where(np.isfinite(sg), sg, sg_default)

    au = _numeric(df["AU"]).to_numpy(float)
    cu = _numeric(df["CU"]).to_numpy(float)
    au = np.where(np.isfinite(au), au, 0.0)
    cu = np.where(np.isfinite(cu), cu, 0.0)

    ton = vol * sg
    income = ton * (au * AU_PRICE_PER_G + cu * CU_PRICE_PER_PCT_T)

    # same grid the original built: arange from the integer floor of the min
    x0, y0, z0 = int(xc.min()), int(yc.min()), int(zc.min())
    x_grid = np.arange(x0, int(xc.max()), step=step)
    y_grid = np.arange(y0, int(yc.max()), step=step)
    z_grid = np.arange(z0, int(zc.max()), step=step)
    nx, ny, nz = len(x_grid), len(y_grid), len(z_grid)

    ix = np.floor((xc - x0) / step).astype(np.int64)
    iy = np.floor((yc - y0) / step).astype(np.int64)
    iz = np.floor((zc - z0) / step).astype(np.int64)
    inside = ((ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
              & (iz >= 0) & (iz < nz))
    dropped = int((~inside).sum())

    # meshgrid(x, y, z) with default 'xy' indexing has shape (ny, nx, nz), and
    # .flatten() walks it y-major. Reproduce that exactly.
    lin = (iy[inside] * nx + ix[inside]) * nz + iz[inside]
    ncell = nx * ny * nz

    cell_ton = np.zeros(ncell)
    cell_income = np.zeros(ncell)
    cell_vol = np.zeros(ncell)
    cell_au_t = np.zeros(ncell)
    cell_cu_t = np.zeros(ncell)
    np.add.at(cell_ton, lin, ton[inside])
    np.add.at(cell_income, lin, income[inside])
    np.add.at(cell_vol, lin, vol[inside])
    np.add.at(cell_au_t, lin, au[inside] * ton[inside])
    np.add.at(cell_cu_t, lin, cu[inside] * ton[inside])

    cell_volume = step ** 3
    covered = np.minimum(cell_vol, cell_volume)
    uncovered = cell_volume - covered

    rng = np.random.default_rng(seed)
    dens = np.full(ncell, sg_default)
    if sg_noise > 0:
        dens = dens * rng.normal(1.0, float(sg_noise), ncell).clip(0.6, 1.4)
    waste_ton = uncovered * dens

    with np.errstate(invalid="ignore", divide="ignore"):
        au_mean = np.where(cell_ton > 0, cell_au_t / np.maximum(cell_ton, 1e-12), 0.0)
        cu_mean = np.where(cell_ton > 0, cell_cu_t / np.maximum(cell_ton, 1e-12), 0.0)

    gx, gy, gz = np.meshgrid(x_grid, y_grid, z_grid)
    out = pd.DataFrame({
        "x": gx.flatten(), "y": gy.flatten(), "z": gz.flatten(),
        "x_step": step, "y_step": step, "z_step": step,
        "income": cell_income,
        "tonnage": cell_ton + waste_ton,
        "ore_tonnage": cell_ton,
        "waste_tonnage": waste_ton,
        "au": au_mean, "cu": cu_mean,
        "coverage": covered / cell_volume,
    }).reset_index(drop=False)

    info = {"n_cells": ncell, "grid": (nx, ny, nz), "dropped_subblocks": dropped,
            "n_covered": int((cell_vol > 0).sum()), "sg_default": sg_default}
    return out, info


def main():
    inputs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs")
    src = os.path.join(inputs_dir, "MB UG.csv")
    df = pd.read_csv(src, low_memory=False)

    out, info = aggregate_to_grid(df)
    print(f"grid {info['grid']} -> {info['n_cells']} cells "
          f"({info['n_covered']} with source coverage), "
          f"{info['dropped_subblocks']} sub-blocks outside the grid")
    print(f"income  nunique {out.income.nunique():>6}  "
          f"max {out.income.max():,.0f}  nonzero {int((out.income > 0).sum())}")
    print(f"tonnage nunique {out.tonnage.nunique():>6}  "
          f"min {out.tonnage.min():,.0f}  max {out.tonnage.max():,.0f}")

    dest = os.path.join(inputs_dir, "blocks.csv")
    prev = pd.read_csv(dest) if os.path.exists(dest) else None
    if prev is not None and len(prev) == len(out):
        for c in ("x", "y", "z"):
            if not np.allclose(prev[c].to_numpy(float), out[c].to_numpy(float)):
                raise SystemExit(
                    f"row order changed on column '{c}' -- refusing to overwrite "
                    "blocks.csv, downstream artifacts are indexed by position")
        print("row order matches the existing blocks.csv; index alignment preserved")
    out.to_csv(dest, index=False)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
