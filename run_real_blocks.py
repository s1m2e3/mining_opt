"""
End-to-end run on the real block model: inputs/blocks.csv (12,213 blocks).

    raw score -> sparse Wendland projection onto the precedence cone
              -> feasible mining sequence
              -> capacity cuts into periods
              -> NPV

Uses ARD lengthscales so the smoothing radius is fixed in block-widths rather
than as a fraction of the deposit, which is what keeps nnz/row (and memory)
from growing with the model size.

Note on orientation: blocks.csv stores z as DEPTH (positive down, 20..460), so
it is negated before building precedence -- square_pyramid_predecessors expects
predecessors to sit at iz+1, i.e. higher z means shallower.
"""

import time

import numpy as np
import pandas as pd

from lg_utils import square_pyramid_predecessors
from kernel_projection import (build_features, minmax_normalize, ard_lengthscales,
                               wendland_c0_sparse_gram, sparse_gram_masking_error,
                               build_edges, topological_order, n_violations,
                               project_dykstra_sparse, project_hard_clamp,
                               schedule_from_scores, sequence_violations,
                               kendall_tau, count_ties)
from capacity_cuts import evaluate_cuts, capacity_report

RADIUS_XY = 6.0          # smoothing radius in block widths, lateral
RADIUS_Z = 2.0           # and vertical (benches are discrete; keep it tighter)
MAX_SWEEPS = 30000       # Dykstra needs far more sweeps here than on synthetic data
T = 10                   # periods
DISCOUNT = 0.90
COST_PER_BLOCK = 1.0e3
COST_PER_BENCH = 2.0e3   # haulage penalty per bench of depth

# blocks.csv carries tonnage only for ore blocks; every waste block is 0 t, which
# makes stripping free and leaves mining capacity permanently slack. Rebuild
# tonnage from block volume and density instead.
ORE_DENSITY = 4.5        # t/m3
WASTE_DENSITY = 3.0      # t/m3

t_all = time.perf_counter()
blocks = pd.read_csv("inputs/blocks.csv")
n = len(blocks)
x = blocks["x"].to_numpy(float)
y = blocks["y"].to_numpy(float)
z_depth = blocks["z"].to_numpy(float)
income = blocks["income"].to_numpy(float)
tonnage_raw = blocks["tonnage"].to_numpy(float)
step = float(blocks["x_step"].iloc[0])

volume = (float(blocks["x_step"].iloc[0]) * float(blocks["y_step"].iloc[0])
          * float(blocks["z_step"].iloc[0]))
is_ore_flag = income > 0
tonnage = np.where(is_ore_flag, ORE_DENSITY, WASTE_DENSITY) * volume
_implied = tonnage_raw[is_ore_flag] / volume
print(f"tonnage rebuilt      block volume {volume:,.0f} m3;  ore {ORE_DENSITY} t/m3, "
      f"waste {WASTE_DENSITY} t/m3")
print(f"  original file      ore density implied by 'tonnage' column: "
      f"mean {_implied.mean():.2f}, max {_implied.max():.2f} t/m3 "
      f"(waste was 0 throughout)")
print(f"  totals             {tonnage.sum():,.0f} t  vs  {tonnage_raw.sum():,.0f} t before; "
      f"strip ratio {tonnage[~is_ore_flag].sum()/tonnage[is_ore_flag].sum():.2f} : 1")

z_elev = -z_depth                                  # positive up
ix = np.rint((x - x.min()) / step).astype(int)
iy = np.rint((y - y.min()) / step).astype(int)
iz = np.rint((z_elev - z_elev.min()) / step).astype(int)
bench = iz.max() - iz                               # 0 at surface, deeper = larger

value_now = income - COST_PER_BLOCK - COST_PER_BENCH * bench
below = {}
for i in range(n):
    below[(ix[i], iy[i], iz[i])] = i
value_future = np.array([
    0.7 * income[below[(ix[i], iy[i], iz[i] - 1)]] if (ix[i], iy[i], iz[i] - 1) in below else 0.0
    for i in range(n)])
is_ore = income > 0

print(f"blocks              {n}   grid {ix.max()+1} x {iy.max()+1} x {iz.max()+1}")
print(f"tonnage             {tonnage.sum():,.0f} t   ore blocks {int(is_ore.sum())} "
      f"({100*is_ore.mean():.1f}%)   zero-tonnage blocks {int((tonnage<=0).sum())}")
print(f"value_now           min {value_now.min():,.0f}  max {value_now.max():,.0f}  "
      f"positive {int((value_now>0).sum())}")

t = time.perf_counter()
df = pd.DataFrame({"x_c": x, "y_c": y, "z_c": z_elev})
par, chi = build_edges(square_pyramid_predecessors(df, slope_h_per_v=1.5).tolist())
order = topological_order(n, par, chi)
t_prec = time.perf_counter() - t
print(f"precedence          {par.size:,} edges in {t_prec:.2f}s "
      f"({par.size/n:.1f} per block)")

rng = np.random.default_rng(0)
s_raw = (value_now - value_now.mean()) / value_now.std() + 0.25 * rng.standard_normal(n)
print(f"raw-score violations {n_violations(s_raw, par, chi):,} / {par.size:,}")

Z = minmax_normalize(build_features(x, y, z_elev, bench, s_raw,
                                    value_now, value_future, tonnage))
ell = ard_lengthscales(Z.shape[1], ix.max()+1, iy.max()+1, iz.max()+1,
                       radius_xy=RADIUS_XY, radius_z=RADIUS_Z)
print(f"\nARD lengthscales     {np.array2string(ell, precision=1)}")

t = time.perf_counter()
gram, gi = wendland_c0_sparse_gram(Z, ix, iy, iz,
                                   radius=(int(np.ceil(RADIUS_XY)),
                                           int(np.ceil(RADIUS_XY)),
                                           int(np.ceil(RADIUS_Z))),
                                   lengthscale=ell)
t_gram = time.perf_counter() - t
drop = sparse_gram_masking_error(Z, gram, ell, n_sample=64)
print(f"sparse Gram          {gi['nnz']:,} nnz  ({gi['nnz_per_row']:.1f}/row, "
      f"{100*gi['nnz_per_row']/n:.2f}% dense)  {gi['mem_MB']:.1f} MB "
      f"vs {gi['dense_mem_GB']:.2f} GB dense   built in {t_gram:.2f}s")
print(f"masking error        {drop:.2e}  (largest discarded kernel value; "
      f"Weyl bound on eigenvalues {n*drop:.2e})")

t = time.perf_counter()
s_proj, lam, info = project_dykstra_sparse(s_raw, gram, par, chi, order,
                                           max_sweeps=MAX_SWEEPS)
t_proj = time.perf_counter() - t
converged = info["sweeps"] < MAX_SWEEPS
print(f"\nprojection           {t_proj:.2f}s  {info['sweeps']:,} sweeps  "
      f"({1000*t_proj/info['sweeps']:.1f} ms/sweep)  "
      f"active {info['active_edges']:,}/{par.size:,}")
print(f"  converged          {converged}"
      f"{'' if converged else '  <-- hit the sweep cap, NOT the exact projection'}")
print(f"  KKT primal resid   {info['primal_residual']:.2e}")
print(f"  KKT complementarity{info['complementarity']:.2e}")
print(f"  min lambda         {info['dual_min']:.2e}   degenerate edges "
      f"{info['degenerate_edges']}")
print(f"  violations >1e-8   {n_violations(s_proj, par, chi, 1e-8)}  "
      f"(snapped away by schedule_from_scores)")

s_clamp = project_hard_clamp(s_raw, par, chi, order)

mine_cap = np.full(T, 0.85 * tonnage.sum() / T)
proc_cap = np.full(T, 0.50 * tonnage[is_ore].sum() / T)
print(f"\ncapacity             T={T}, mining {mine_cap[0]:,.0f} t/period, "
      f"processing {proc_cap[0]:,.0f} t/period")

print(f"\n{'method':<22} {'tau':>7} {'ties':>7} {'seq viol':>9} {'NPV':>18} {'unmined':>8}")
print("-" * 76)
results = {}
for name, s in (("hard clamp (K=I)", s_clamp), ("wendland projection", s_proj)):
    seq = schedule_from_scores(s, par, chi, order)
    res = evaluate_cuts(seq, tonnage, mine_cap, value=value_now, discount=DISCOUNT,
                        ore_mask=is_ore, proc_capacities=proc_cap)
    results[name] = res
    print(f"{name:<22} {kendall_tau(s, s_raw):>7.3f} {count_ties(s, 1e-7):>7} "
          f"{sequence_violations(seq, par, chi):>9} {res['npv']:>18,.0f} "
          f"{res['n_unmined']:>8}")

gain = results["wendland projection"]["npv"] - results["hard clamp (K=I)"]["npv"]
print(f"\nprojection vs clamp  {gain:+,.0f} NPV")

res = results["wendland projection"]
print(f"\nper-period utilisation (projection):")
print(f"{'t':>3} {'cut pos':>10} {'tons':>14} {'util%':>7} {'ore t':>13} {'util%':>7}")
print("-" * 60)
for (t_, tons, cap, util, ot, ocap, outil) in capacity_report(res, mine_cap, proc_cap):
    print(f"{t_:>3} {res['cuts'][t_-1]:>10.1f} {tons:>14,.0f} {util:>7.1f} "
          f"{ot:>13,.0f} {outil:>7.1f}")

print(f"\ntotal wall time      {time.perf_counter()-t_all:.1f}s")
