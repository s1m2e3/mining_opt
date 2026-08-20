"""
Small end-to-end run (~1k blocks) with three capacity axes and the repricing
loop, then a Plotly animation of the pit developing period by period.

Outputs
    outputs/subset_schedule.csv    per-block result
    outputs/subset_schedule.json   run summary
    outputs/subset_schedule.html   interactive 3D animation
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

N_X_COLUMNS = 5          # x-slab width -> 5 * 9 * 23 = 1035 blocks
T = 8
DISCOUNT = 0.90
ORE_DENSITY, WASTE_DENSITY = 4.5, 3.0
COST_PER_BLOCK, COST_PER_BENCH = 1.0e3, 2.0e3
RADIUS_XY, RADIUS_Z = 6.0, 2.0
MINE_FRAC, PROC_FRAC, STRIP_FRAC = 0.55, 0.55, 0.60   # of total / T
LOOKAHEAD_LEVELS = 5     # benches of precedence cone summed above and below
BETA_BELOW = 0.60        # weight on unlocked value in the priority score

os.makedirs("outputs", exist_ok=True)
t_all = time.perf_counter()

blocks = pd.read_csv("inputs/blocks.csv")
step = float(blocks["x_step"].iloc[0])
volume = step ** 3
x_keep = np.sort(blocks["x"].unique())[:N_X_COLUMNS]
blocks = blocks[blocks["x"].isin(x_keep)].reset_index(drop=True)

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

print(f"subset               {n} blocks, grid {ix.max()+1} x {iy.max()+1} x {iz.max()+1}"
      f"   (x-slab of the full 59-column model)")
print(f"tonnage              {tonnage.sum():,.0f} t   ore {int(is_ore.sum())} blocks"
      f"   strip ratio {tonnage[~is_ore].sum()/tonnage[is_ore].sum():.2f} : 1")

df = pd.DataFrame({"x_c": x, "y_c": y, "z_c": z_elev})
par, chi = build_edges(square_pyramid_predecessors(df, slope_h_per_v=1.5).tolist())
order = topological_order(n, par, chi)
print(f"precedence           {par.size:,} edges")

cost = COST_PER_BLOCK + COST_PER_BENCH * bench          # positive magnitude
qty = {"income": income, "cost": cost, "tonnage": tonnage, "value": value}
lo_below = cone_sums(ix, iy, iz, qty, levels=LOOKAHEAD_LEVELS, direction="below")
lo_above = cone_sums(ix, iy, iz, qty, levels=LOOKAHEAD_LEVELS, direction="above")

income_below, cost_below = lo_below["income"], lo_below["cost"]
value_below, tons_below = lo_below["value"], lo_below["tonnage"]
value_above, tons_above = lo_above["value"], lo_above["tonnage"]
print(f"cone look-ahead      {LOOKAHEAD_LEVELS} benches;  below: "
      f"{lo_below['count'].mean():.0f} blocks/cone avg, income "
      f"{income_below.mean():,.0f}, cost {cost_below.mean():,.0f}")
print(f"                     above: {lo_above['count'].mean():.0f} blocks/cone avg, "
      f"stripping burden {(-value_above).mean():,.0f} avg")

# Priority = own value plus a discounted claim on what this block unlocks.
# This is the piece the pure-value score was missing: a block is worth mining
# partly for what sits beneath it, which is exactly what makes stripping
# rational. BETA_BELOW discounts it for being realised later.
value_eff = value + BETA_BELOW * value_below

rng = np.random.default_rng(0)
noise = np.zeros(n)          # was 0.25*randn, a leftover from simulating a model
s_seed = (value_eff - value_eff.mean()) / value_eff.std() + noise

Z = minmax_normalize(build_features(
    x, y, z_elev, bench, s_seed, value, value_below, tonnage,
    extra=[income_below, cost_below, tons_below, value_above, tons_above]))
ell = ard_lengthscales(Z.shape[1], ix.max()+1, iy.max()+1, iz.max()+1,
                       radius_xy=RADIUS_XY, radius_z=RADIUS_Z)
gram, gi = wendland_c0_sparse_gram(Z, ix, iy, iz,
                                   radius=(int(np.ceil(RADIUS_XY)),
                                           int(np.ceil(RADIUS_XY)),
                                           int(np.ceil(RADIUS_Z))),
                                   lengthscale=ell)
print(f"sparse Gram          {gi['nnz']:,} nnz ({gi['nnz_per_row']:.1f}/row, "
      f"{gi['mem_MB']:.1f} MB)")

resources = [
    make_resource("mining",     tonnage,             np.full(T, MINE_FRAC*tonnage.sum()/T)),
    make_resource("processing", tonnage*is_ore,      np.full(T, PROC_FRAC*tonnage[is_ore].sum()/T)),
    make_resource("stripping",  tonnage*(~is_ore),   np.full(T, STRIP_FRAC*tonnage[~is_ore].sum()/T)),
]
print(f"capacities/period    mining {resources[0]['capacity'][0]:,.0f} t, "
      f"processing {resources[1]['capacity'][0]:,.0f} t, "
      f"stripping {resources[2]['capacity'][0]:,.0f} t")

def zs(a):
    a = np.asarray(a, dtype=float)
    sd = a.std()
    return (a - a.mean()) / (sd if sd > 0 else 1.0)


# value_below averages ~3.0e6 against a per-block value of ~1e5, so adding them
# raw lets the cone term swamp the block's own economics. Standardise first,
# then beta actually means "relative weight".
variants = {
    "value only": value,
    "value / tonnage": value / tonnage,
    "z(value) + 0.25 z(below)": zs(value) + 0.25 * zs(value_below),
    "z(value) + 0.50 z(below)": zs(value) + 0.50 * zs(value_below),
    "z(value) + 1.00 z(below)": zs(value) + 1.00 * zs(value_below),
    "cone value density": (value + value_below) / (tonnage + tons_below),
    "raw value + 0.6*below": value + BETA_BELOW * value_below,
}

print("\npriority-score variants (NPV always scored on true block value):")
ab = {}
for label, prio in variants.items():
    b, _ = reprice_loop(prio, resources, gram, par, chi, order, tonnage,
                        discount=DISCOUNT, n_iters=8, damping=0.5,
                        max_sweeps=20000, noise=noise, verbose=False,
                        true_value=value)
    ab[label] = b
    print(f"  {label:<26} NPV {b['npv']:>15,.0f}   unmined {b['result']['n_unmined']:>5}")
base = ab["value only"]["npv"]
for label in variants:
    print(f"  {label:<26} vs value-only {100*(ab[label]['npv']/base - 1):+7.2f}%")

best_label = max(ab, key=lambda k: ab[k]["npv"])
print(f"\nwinner: {best_label}")

print("\nrepricing loop (winning priority):")
best, history = reprice_loop(variants[best_label], resources, gram, par, chi,
                             order, tonnage, discount=DISCOUNT, n_iters=8,
                             damping=0.5, max_sweeps=20000, noise=noise,
                             true_value=value)

s_star, seq, res = best["score"], best["seq"], best["result"]
print(f"\nbest at iter {best['iter']}   NPV {best['npv']:,.0f}")
print(f"precedence violations in score    {n_violations(s_star, par, chi, 1e-8)}")
print(f"precedence violations in sequence {sequence_violations(seq, par, chi)}")
print(f"blocks split across periods       {res['n_split']}   partial {res['n_partial']}"
      f"   unmined {res['n_unmined']}")

print(f"\n{'t':>3} " + " ".join(f"{r['name'][:10]:>13}" for r in resources)
      + "   binding")
for t in range(T):
    cells = []
    binds = []
    for r in resources:
        u = res["per_resource"][r["name"]][t]
        c = r["capacity"][t]
        cells.append(f"{100*u/c if c>0 else 0:>12.1f}%")
        if res["binding"][r["name"]][t]:
            binds.append(r["name"])
    print(f"{t+1:>3} " + " ".join(cells) + "   " + (", ".join(binds) or "-"))

# --------------------------------------------------------------- persist
period = res["start_period"]
# position must come from the emitted sequence, not from re-sorting on score:
# the projection creates many exact ties and schedule_from_scores breaks them
# topologically, which a plain score sort does not reproduce.
mining_position = np.empty(n, dtype=np.int64)
mining_position[seq] = np.arange(1, n + 1)
out = pd.DataFrame({
    "block": blocks["index"].to_numpy(), "x": x, "y": y, "z_depth": blocks["z"].to_numpy(),
    "z_elev": z_elev, "bench": bench, "ix": ix, "iy": iy, "iz": iz,
    "is_ore": is_ore, "tonnage": tonnage, "income": income, "value": value,
    "score": s_star, "period": np.where(period >= 0, period + 1, -1),
    "mined_fraction": res["mined_fraction"],
    "discount_factor": res["discount_factor"],
    "mining_position": mining_position,
})
out = out.sort_values("mining_position").reset_index(drop=True)
out.to_csv("outputs/subset_schedule.csv", index=False)

summary = {
    "n_blocks": int(n), "n_edges": int(par.size), "periods": T,
    "discount": DISCOUNT, "npv": best["npv"], "best_iter": best["iter"],
    "multipliers": best["mu"], "cuts": res["cuts"].tolist(),
    "n_unmined": res["n_unmined"], "n_split": res["n_split"],
    "per_resource": {k: v.tolist() for k, v in res["per_resource"].items()},
    "capacity": {k: v.tolist() for k, v in res["capacity"].items()},
    "history": [{k: (v if not isinstance(v, dict) else v) for k, v in h.items()}
                for h in history],
    "strip_ratio": float(tonnage[~is_ore].sum()/tonnage[is_ore].sum()),
}
with open("outputs/subset_schedule.json", "w") as f:
    json.dump(summary, f, indent=2)

# --------------------------------------------------------------- plotly
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
           "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"]

frames, slider_steps = [], []
for t in range(1, T + 1):
    traces = []
    for p in range(1, t + 1):
        m = out["period"] == p
        traces.append(go.Scatter3d(
            x=out.loc[m, "x"], y=out.loc[m, "y"], z=out.loc[m, "z_elev"],
            mode="markers", name=f"period {p}",
            marker=dict(size=4, color=PALETTE[(p-1) % len(PALETTE)], opacity=0.9),
            hovertemplate=("period %{customdata[0]}<br>bench %{customdata[1]}"
                           "<br>%{customdata[2]}<br>%{customdata[3]:,.0f} t"
                           "<br>value %{customdata[4]:,.0f}<extra></extra>"),
            customdata=np.stack([out.loc[m, "period"], out.loc[m, "bench"],
                                 np.where(out.loc[m, "is_ore"], "ore", "waste"),
                                 out.loc[m, "tonnage"], out.loc[m, "value"]], axis=-1),
        ))
    mu_ = out["period"] == -1
    traces.append(go.Scatter3d(
        x=out.loc[mu_, "x"], y=out.loc[mu_, "y"], z=out.loc[mu_, "z_elev"],
        mode="markers", name="never mined",
        marker=dict(size=2, color="#d9d9d9", opacity=0.25),
        hoverinfo="skip"))
    frames.append(go.Frame(data=traces, name=str(t),
                           traces=list(range(len(traces)))))
    slider_steps.append(dict(method="animate", label=f"{t}",
                             args=[[str(t)], dict(mode="immediate",
                                                  frame=dict(duration=0, redraw=True),
                                                  transition=dict(duration=0))]))

fig = go.Figure(data=frames[-1].data, frames=frames)
fig.update_layout(
    title=(f"Pit development, {n} blocks, {T} periods &mdash; "
           f"NPV {best['npv']:,.0f}  (cumulative: colour = period mined)"),
    scene=dict(xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="elevation (m)",
               aspectmode="data"),
    updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=1.05,
                      buttons=[
                          dict(label="Play", method="animate",
                               args=[None, dict(frame=dict(duration=900, redraw=True),
                                                fromcurrent=True)]),
                          dict(label="Pause", method="animate",
                               args=[[None], dict(mode="immediate",
                                                  frame=dict(duration=0, redraw=False))])]),],
    sliders=[dict(active=T-1, currentvalue=dict(prefix="mined through period "),
                  pad=dict(t=40), steps=slider_steps)],
    legend=dict(itemsizing="constant"), margin=dict(l=0, r=0, t=60, b=0),
)
fig.write_html("outputs/subset_schedule.html", include_plotlyjs="cdn")

print(f"\nwrote outputs/subset_schedule.csv, .json, .html")
print(f"total wall time      {time.perf_counter()-t_all:.1f}s")
