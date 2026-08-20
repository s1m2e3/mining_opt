"""Train the transformer to schedule blocks, end to end on continuous NPV.

    x  ->  SimpleTransformer  ->  s_raw
       ->  anchor interpolation  ->  s*   precedence-FEASIBLE score
       ->  continuous_time       ->  sigma   start time in periods
       ->  npv_soft              ->  objective

Every arrow carries gradient. Verified against central finite differences taken
on the transformer weights themselves (see `continuous_time.__main__`), not
merely checked for being finite -- an earlier version produced finite,
non-zero, wrong gradients and only that test caught it.

WHY THE PROJECTION IS IN THE CHAIN. Measured on an 84-block cone of this
deposit, training on NPV alone drives the schedule INTO infeasibility: 52 ->
190 of 315 precedence edges violated, because the value sits at depth and NPV
wants it first while precedence demands the waste above it first. That run
reaches 68.0M, but 68.0M is not a mining schedule. Projecting to the precedence
cone first gives 41.0M against a 29.0M topological baseline -- +41%, entirely
feasible, about a third of the gap to the unreachable unconstrained optimum.

WHY THE OBJECTIVE IS SCALED. NPV here is 1e7-1e9, and an unscaled loss hands
the optimiser gradients of the same order, growing ~60x over 200 steps with the
projection in the chain. `npv_scale` divides by sum |v_j|, making the loss the
fraction of available value captured -- dimensionless, in [-1, 1] -- so one
learning rate and one clipping threshold transfer across instances. Combined
with grad clipping this is what keeps the long runs from wandering.

TIE-BREAK. The projection returns a field with zero score violations but a
handful of exact parent/child ties (it creates them on active faces). Breaking
those by value puts children ahead of parents; breaking them topologically
costs 0.46% of NPV and gives exact feasibility, so `tie_break="topo"` is the
default whenever the projection is on.

The previous pushback-level pipeline (Simulator, income_loss,
constraint_sequencing) is preserved in git history at 5246e26.

Run: python train.py [steps] [lr] [full]
"""

import json
import os
import sys
import time

import numpy as np
import torch

import continuous_time as ct
from anchor_interpolation import interpolate_precedence_torch
from kernel_projection import (ard_lengthscales, minmax_normalize,
                               topological_order, wendland_c0_sparse_gram)
from block_lookahead import cone_sums
from mine_problem import load_static
from model import SimpleTransformer

# ---------------------------------------------------------------- config
STEPS = 300
LR = 1e-3
CLIP = 1.0                 # meaningful because the objective is dimensionless
DISCOUNT = 0.90
T_PERIODS = 10
# crop-14 is the standard fast instance (n ~ 2,900). The full model runs but is
# heavy: the projection rebuilds a 1.96M-nonzero gram every inner iteration, so
# a step costs ~18 s and peak RAM is enough to OOM a 16 GiB box mid-run. Set to
# None for the full model once you are ready to pay for it.
CROP = 14
PROJECT = True             # precedence projection between the net and the time map
SEED = 0

D_MODEL = 128
N_HEAD = 8
N_LAYERS = 4
FF = 512
DROPOUT = 0.1

FEATURES = ("x", "y", "z", "bench", "tonnage", "ore_tonnage", "income",
            "au", "cu")

# Cone aggregates. The single quantity that decides whether to defer a valuable
# block is how much waste sits ABOVE it and how much value sits BELOW -- and
# neither was an input, so the network had to infer the slope-cone structure
# from raw coordinates through self-attention. block_lookahead.cone_sums
# computes them directly.
#
# Note this does relax the "value is not an input" stance, but only for
# AGGREGATES: per-block value is still withheld, and the aggregation is the
# part that is genuinely hard to learn rather than the arithmetic.
CONE_FEATURES = True
CONE_LEVELS = 5


def block_features(static):
    """Physical and geological columns only, z-scored.

    Value is deliberately not an input. It is a linear function of income,
    tonnage, bench and mill feed, so the network can recover it -- that is the
    intended difficulty, and handing it over directly would make the task
    a sort rather than a learning problem.
    """
    cols = [np.asarray(static[c], dtype=np.float64) for c in FEATURES]
    if CONE_FEATURES:
        cols.extend(cone_columns(static, CONE_LEVELS))
    F = np.stack(cols, axis=1)
    mu, sd = F.mean(0, keepdims=True), F.std(0, keepdims=True)
    return (F - mu) / np.where(sd > 0, sd, 1.0)


def cone_columns(static, levels=CONE_LEVELS):
    """Value and tonnage summed over the cone above and below each block.

    `above` is the overburden that must be moved to reach the block -- the
    reason to defer it. `below` is what becomes reachable once it is gone --
    the reason to dig here. Cached on the static dict, since it depends only on
    geometry and the value field.
    """
    key = f"_cone_cols_{levels}"
    if key in static:
        return static[key]
    qty = {"value": static["value"], "tonnage": static["tonnage"]}
    up = cone_sums(static["ix"], static["iy"], static["iz"], qty,
                   levels=levels, direction="above")
    dn = cone_sums(static["ix"], static["iy"], static["iz"], qty,
                   levels=levels, direction="below")
    out = [up["value"], up["tonnage"], dn["value"], dn["tonnage"]]
    static[key] = out
    return out


def build_gram(static):
    """Wendland gram over GEOMETRIC features only (x, y, z, bench).

    Not mine_problem's 'full' feature set: that one carries the score itself,
    which would make the kernel -- and therefore the projection operator --
    change on every forward pass. Geometry is constant, so the gram is built
    once and the layer is a fixed operator with a stable gradient.
    """
    Z = minmax_normalize(np.column_stack([
        static["x"], static["y"], static["z"],
        static["bench"].astype(float)]))
    ell = ard_lengthscales(Z.shape[1], static["nx"], static["ny"], static["nz"],
                           6.0, 2.0)
    gram, _ = wendland_c0_sparse_gram(Z, static["ix"], static["iy"],
                                      static["iz"], (6, 6, 2), ell)
    return gram


def main(steps=STEPS, lr=LR, project=PROJECT, crop=CROP, out="outputs"):
    torch.manual_seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    static = load_static(t_periods=T_PERIODS, crop=crop)
    n = static["n"]
    tonnage, value = static["tonnage"], static["value"]
    par, chi = np.asarray(static["par"]), np.asarray(static["chi"])

    # one resource: mining / strip capacity. Processing is out of scope here.
    capacity = tonnage.sum() / T_PERIODS
    pr = ct.prepare(tonnage, capacity, value, discount=DISCOUNT, device=dev,
                    dtype=torch.float32)

    order = topological_order(n, par, chi)
    topo_rank = np.empty(n, dtype=np.int64)
    topo_rank[order] = np.arange(n)
    tie = dict(tie_break="topo", topo_rank=topo_rank) if project else {}

    x = torch.tensor(block_features(static), dtype=torch.float32,
                     device=dev).unsqueeze(0)
    gram = build_gram(static) if project else None

    net = SimpleTransformer(in_dim=x.shape[-1], d_model=D_MODEL, nhead=N_HEAD,
                            num_layers=N_LAYERS, dim_feedforward=FF,
                            dropout=DROPOUT, out_dim=1,
                            use_posenc=False).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    tau_np = pr["tau"].cpu().numpy()
    base = ct.npv(ct.start_times(-topo_rank.astype(float), tau_np, **tie),
                  tau_np, value, discount=DISCOUNT, scale=pr["scale"])
    ceil_ = ct.npv(ct.start_times(value / tau_np, tau_np, value=value), tau_np,
                   value, discount=DISCOUNT, scale=pr["scale"])

    print(f"blocks {n}   edges {len(par)}   horizon {pr['horizon']:.2f} periods"
          f"   window k={pr['window']} (blur {pr['blur']:.4f} p)   {dev}")
    print(f"projection {'ON' if project else 'OFF'}   objective scaled by"
          f" sum|v| = {pr['scale']:,.0f}   lr={lr}  clip={CLIP}")
    print(f"reference  topological (feasible) {base:+.4f}"
          f"    value-density (infeasible ceiling) {ceil_:+.4f}\n")
    print(f"{'step':>5} {'loss':>10} {'soft NPV':>10} {'hard NPV':>10}"
          f" {'|grad|':>10} {'viol':>10} {'s/step':>8}")

    hist = []
    for i in range(steps):
        t0 = time.perf_counter()
        opt.zero_grad()
        s_raw = net(x, pool=None).squeeze(-1)
        if project:
            s = interpolate_precedence_torch(s_raw.squeeze(0), gram, par,
                                             chi).unsqueeze(0)
        else:
            s = s_raw
        sigma = ct.start_times_soft(s, pr["tau"], value=pr["value"],
                                    window=pr["window"])
        soft = ct.npv_soft(sigma, pr["tau"], pr["value"], psi=pr["psi"],
                           scale=pr["scale"]).sum()
        loss = -soft
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP))
        opt.step()

        with torch.no_grad():
            sn = s.detach().reshape(-1).cpu().numpy().astype(np.float64)
        sg = ct.start_times(sn, tau_np, value=value, **tie)
        hard = ct.npv(sg, tau_np, value, discount=DISCOUNT, scale=pr["scale"])
        viol = ct.time_violations(sg, par, chi)
        dt = time.perf_counter() - t0
        hist.append({"step": i, "loss": loss.item(), "soft": soft.item(),
                     "hard": float(hard), "grad_norm": gnorm,
                     "violations": viol, "seconds": dt})
        if i % max(1, steps // 20) == 0 or i == steps - 1:
            print(f"{i:>5} {loss.item():>10.4f} {soft.item():>10.4f}"
                  f" {hard:>10.4f} {gnorm:>10.3e} {viol:>6}/{len(par):<4}"
                  f" {dt:>8.2f}")

    best = max(hist, key=lambda h: h["hard"])
    print(f"\nbest hard NPV {best['hard']:+.4f} of available value at step"
          f" {best['step']}, {best['violations']} violations")
    print(f"  topological baseline {base:+.4f}   ->  "
          f"{(best['hard'] - base) / abs(base) * 100:+.1f}%")
    print(f"  steps with a zero gradient: "
          f"{sum(h['grad_norm'] == 0 for h in hist)}/{len(hist)}")

    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "train_continuous_npv.json")
    with open(path, "w") as f:
        json.dump({"config": {"steps": steps, "lr": lr, "clip": CLIP,
                              "discount": DISCOUNT, "t_periods": T_PERIODS,
                              "project": project, "crop": crop, "n": n,
                              "scale": pr["scale"], "window": pr["window"],
                              "blur": pr["blur"]},
                   "reference": {"topological": float(base),
                                 "value_density": float(ceil_)},
                   "history": hist}, f, indent=2)
    print(f"  wrote {path}")
    return hist


if __name__ == "__main__":
    a = sys.argv[1:]
    main(steps=int(a[0]) if len(a) > 0 else STEPS,
         lr=float(a[1]) if len(a) > 1 else LR,
         crop=(None if len(a) > 2 and a[2] == "full" else CROP))
