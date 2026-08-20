"""
Full block model: all 12,213 blocks, three capacity axes, repricing loop.

Unlike run_subset_schedule.py this is not a lateral slab, so no precedence cone
is truncated and the NPV is a real number for this deposit rather than for a
fictitious easier one. Cost: the projection is ~160 s and runs once per
repricing iteration.

Outputs
    outputs/full_schedule.csv    per-block result
    outputs/full_schedule.json   run summary
    outputs/full_schedule.html   interactive 3D animation
"""

import json
import os
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from lg_utils import square_pyramid_predecessors
from kernel_projection import (build_features, minmax_normalize, ard_lengthscales,
                               wendland_c0_sparse_gram, build_edges,
                               topological_order, n_violations,
                               schedule_from_scores, sequence_violations)
from capacity_cuts import make_resource, evaluate_cuts_multi
from block_lookahead import cone_sums
from reprice import reprice_loop

T = 10
DISCOUNT = 0.90
ORE_DENSITY, WASTE_DENSITY = 4.5, 3.0
COST_PER_BLOCK, COST_PER_BENCH = 1.0e3, 2.0e3
RADIUS_XY, RADIUS_Z = 6.0, 2.0
LOOKAHEAD_LEVELS = 5
N_ITERS = 6
# Capacity as a fraction of the deposit per period. These are placeholders --
# a real plan sets them from fleet and mill rates (t/day x 365 x years/period).
MINE_FRAC, PROC_FRAC, STRIP_FRAC = 0.55, 0.55, 0.60

os.makedirs("outputs", exist_ok=True)
t_all = time.perf_counter()

blocks = pd.read_csv("inputs/blocks.csv")
step = float(blocks["x_step"].iloc[0])
volume = step ** 3
n = len(blocks)
x = blocks["x"].to_numpy(float)
y = blocks["y"].to_numpy(float)
z_elev = -blocks["z"].to_numpy(float)
income = blocks["income"].to_numpy(float)
is_ore = income > 0
tonnage = np.where(is_ore, ORE_DENSITY, WASTE_DENSITY) * volume

ix = np.rint((x - x.min()) / step).astype(int)
iy = np.rint((y - y.min()) / step).astype(int)
iz = np.rint((z_elev - z_elev.min()) / step).astype(int)
bench = iz.max() - iz
value = income - COST_PER_BLOCK - COST_PER_BENCH * bench

print(f"FULL model           {n} blocks, grid {ix.max()+1} x {iy.max()+1} x {iz.max()+1}"
      f"  (no lateral truncation -> no broken cones)", flush=True)
print(f"tonnage              {tonnage.sum()/1e6:,.1f} Mt   ore {int(is_ore.sum())} blocks"
      f"   strip ratio {tonnage[~is_ore].sum()/tonnage[is_ore].sum():.2f} : 1", flush=True)

df = pd.DataFrame({"x_c": x, "y_c": y, "z_c": z_elev})
par, chi = build_edges(square_pyramid_predecessors(df, slope_h_per_v=1.5).tolist())
order = topological_order(n, par, chi)
print(f"precedence           {par.size:,} edges", flush=True)

cost = COST_PER_BLOCK + COST_PER_BENCH * bench
qty = {"income": income, "cost": cost, "tonnage": tonnage, "value": value}
lo_b = cone_sums(ix, iy, iz, qty, levels=LOOKAHEAD_LEVELS, direction="below")
lo_a = cone_sums(ix, iy, iz, qty, levels=LOOKAHEAD_LEVELS, direction="above")
print(f"cone look-ahead      below {lo_b['count'].mean():.0f} blk/cone, "
      f"above {lo_a['count'].mean():.0f} blk/cone", flush=True)

s_seed = (value - value.mean()) / value.std()
Z = minmax_normalize(build_features(
    x, y, z_elev, bench, s_seed, value, lo_b["value"], tonnage,
    extra=[lo_b["income"], lo_b["cost"], lo_b["tonnage"],
           lo_a["value"], lo_a["tonnage"]]))
ell = ard_lengthscales(Z.shape[1], ix.max()+1, iy.max()+1, iz.max()+1,
                       radius_xy=RADIUS_XY, radius_z=RADIUS_Z)
gram, gi = wendland_c0_sparse_gram(Z, ix, iy, iz,
                                   radius=(int(np.ceil(RADIUS_XY)),
                                           int(np.ceil(RADIUS_XY)),
                                           int(np.ceil(RADIUS_Z))),
                                   lengthscale=ell)
print(f"sparse Gram          {gi['nnz']:,} nnz ({gi['nnz_per_row']:.1f}/row, "
      f"{gi['mem_MB']:.1f} MB vs {gi['dense_mem_GB']:.2f} GB dense)", flush=True)

resources = [
    make_resource("mining",     tonnage,           np.full(T, MINE_FRAC*tonnage.sum()/T)),
    make_resource("processing", tonnage*is_ore,    np.full(T, PROC_FRAC*tonnage[is_ore].sum()/T)),
    make_resource("stripping",  tonnage*(~is_ore), np.full(T, STRIP_FRAC*tonnage[~is_ore].sum()/T)),
]
print(f"capacities/period    mining {resources[0]['capacity'][0]/1e6:.2f} Mt, "
      f"processing {resources[1]['capacity'][0]/1e6:.2f} Mt, "
      f"stripping {resources[2]['capacity'][0]/1e6:.2f} Mt", flush=True)

print(f"\nrepricing loop ({N_ITERS} iterations, under-relaxed POCS):", flush=True)
best, history = reprice_loop(value, resources, gram, par, chi, order, tonnage,
                             discount=DISCOUNT, n_iters=N_ITERS, damping=0.5,
                             noise=np.zeros(n), true_value=value)

s_star, seq, res = best["score"], best["seq"], best["result"]
print(f"\nbest at iter {best['iter']}   NPV {best['npv']:,.0f}", flush=True)
print(f"precedence violations in score    {n_violations(s_star, par, chi, 1e-8)}")
print(f"precedence violations in sequence {sequence_violations(seq, par, chi)}")
print(f"split {res['n_split']}  partial {res['n_partial']}  unmined {res['n_unmined']}"
      f" / {n}")

print(f"\n{'t':>3} " + " ".join(f"{r['name'][:10]:>13}" for r in resources) + "   binding")
for t in range(T):
    cells, binds = [], []
    for r in resources:
        u, c = res["per_resource"][r["name"]][t], r["capacity"][t]
        cells.append(f"{100*u/c if c > 0 else 0:>12.1f}%")
        if res["binding"][r["name"]][t]:
            binds.append(r["name"])
    print(f"{t+1:>3} " + " ".join(cells) + "   " + (", ".join(binds) or "-"))

period = res["start_period"]
mining_position = np.empty(n, dtype=np.int64)
mining_position[seq] = np.arange(1, n + 1)
out = pd.DataFrame({
    "block": blocks["index"].to_numpy(), "x": x, "y": y,
    "z_depth": blocks["z"].to_numpy(), "z_elev": z_elev, "bench": bench,
    "ix": ix, "iy": iy, "iz": iz, "is_ore": is_ore, "tonnage": tonnage,
    "income": income, "value": value, "score": s_star,
    "period": np.where(period >= 0, period + 1, -1),
    "mined_fraction": res["mined_fraction"],
    "discount_factor": res["discount_factor"],
    "mining_position": mining_position,
}).sort_values("mining_position").reset_index(drop=True)
out.to_csv("outputs/full_schedule.csv", index=False)

with open("outputs/full_schedule.json", "w") as f:
    json.dump({"n_blocks": int(n), "n_edges": int(par.size), "periods": T,
               "discount": DISCOUNT, "npv": best["npv"], "best_iter": best["iter"],
               "multipliers": best["mu"], "cuts": res["cuts"].tolist(),
               "n_unmined": res["n_unmined"], "n_split": res["n_split"],
               "per_resource": {k: v.tolist() for k, v in res["per_resource"].items()},
               "capacity": {k: v.tolist() for k, v in res["capacity"].items()},
               "history": history,
               "tonnage_Mt": float(tonnage.sum()/1e6),
               "strip_ratio": float(tonnage[~is_ore].sum()/tonnage[is_ore].sum())},
              f, indent=2)

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
           "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"]
frames, steps = [], []
for t in range(1, T + 1):
    traces = []
    for p in range(1, t + 1):
        m = out["period"] == p
        traces.append(go.Scatter3d(
            x=out.loc[m, "x"], y=out.loc[m, "y"], z=out.loc[m, "z_elev"],
            mode="markers", name=f"period {p}",
            marker=dict(size=2.5, color=PALETTE[(p-1) % len(PALETTE)], opacity=0.85),
            hovertemplate=("period %{customdata[0]}<br>bench %{customdata[1]}"
                           "<br>%{customdata[2]}<br>%{customdata[3]:,.0f} t"
                           "<br>value %{customdata[4]:,.0f}<extra></extra>"),
            customdata=np.stack([out.loc[m, "period"], out.loc[m, "bench"],
                                 np.where(out.loc[m, "is_ore"], "ore", "waste"),
                                 out.loc[m, "tonnage"], out.loc[m, "value"]], axis=-1)))
    mu_ = out["period"] == -1
    traces.append(go.Scatter3d(
        x=out.loc[mu_, "x"], y=out.loc[mu_, "y"], z=out.loc[mu_, "z_elev"],
        mode="markers", name="never mined",
        marker=dict(size=1.5, color="#d9d9d9", opacity=0.18), hoverinfo="skip"))
    frames.append(go.Frame(data=traces, name=str(t), traces=list(range(len(traces)))))
    steps.append(dict(method="animate", label=f"{t}",
                      args=[[str(t)], dict(mode="immediate",
                                           frame=dict(duration=0, redraw=True),
                                           transition=dict(duration=0))]))
fig = go.Figure(data=frames[-1].data, frames=frames)
fig.update_layout(
    title=(f"Full pit development, {n:,} blocks, {T} periods &mdash; "
           f"NPV {best['npv']:,.0f}"),
    scene=dict(xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="elevation (m)",
               aspectmode="data"),
    updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=1.05, buttons=[
        dict(label="Play", method="animate",
             args=[None, dict(frame=dict(duration=900, redraw=True), fromcurrent=True)]),
        dict(label="Pause", method="animate",
             args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))])])],
    sliders=[dict(active=T-1, currentvalue=dict(prefix="mined through period "),
                  pad=dict(t=40), steps=steps)],
    legend=dict(itemsizing="constant"), margin=dict(l=0, r=0, t=60, b=0))
fig.write_html("outputs/full_schedule.html", include_plotlyjs="cdn")

print(f"\nwrote outputs/full_schedule.csv, .json, .html")
print(f"total wall time      {time.perf_counter()-t_all:.1f}s")
