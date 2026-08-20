"""
Supervised imitation of the heuristic scheduler.

Expert : value score -> POCS projection -> schedule -> capacity cuts -> reprice
Student: SimpleTransformer(block features) -> s_hat -> [projection] -> schedule

Loss is ListMLE, the Plackett-Luce negative log-likelihood of the expert's
mining order:

    L(s) = - sum_t [ s_{pi_t} - logsumexp(s_{pi_t}, ..., s_{pi_n}) ]

Ranking-aware and shift-invariant, which matters because the schedule depends
only on the ORDER of the scores -- MSE against the expert's score values would
over-constrain.

Two variants:
  'raw'  loss on s_hat directly; project only at inference
  'proj' loss on Pi(s_hat), projection inside the graph via PrecedenceProjection

Reported metric is held-out NPV from the real hard pipeline, against the
untrained value heuristic as baseline.

Memory: instances are built one at a time and everything not needed later is
freed immediately -- an earlier version held all 32 at once alongside the
autograd graph and was OOM-killed on a 7.2 GB machine.

    python train_imitation.py small     # ~1 min smoke run
    python train_imitation.py full
"""

import gc
import sys
import time

import numpy as np
import pandas as pd
import torch

from lg_utils import square_pyramid_predecessors
from kernel_projection import (build_features, minmax_normalize, ard_lengthscales,
                               wendland_c0_sparse_gram, build_edges,
                               topological_order, project_pocs_sparse,
                               schedule_from_scores, sequence_violations)
from capacity_cuts import make_resource, evaluate_cuts_multi
from block_lookahead import cone_sums
from reprice import reprice_loop
from model import SimpleTransformer
from torch_projection import project_torch

STEP = 10.0
T, DISCOUNT = 8, 0.90

CONFIGS = {
    "small": dict(nx=11, nz=4, n_train=12, n_test=4, epochs=30,
                  d_model=64, layers=2, nhead=4, ff=128, reprice_iters=4),
    "full":  dict(nx=16, nz=4, n_train=24, n_test=8, epochs=60,
                  d_model=128, layers=4, nhead=4, ff=256, reprice_iters=6),
}

# Tonnage variance matters more than it looks. With tonnage = density x constant
# volume it takes only two values (ore / waste), so the repricing term
# v_i - sum_r mu_r w_r,i is a constant shift per group and can only slide the
# ore/waste boundary -- it cannot reorder within a group. Giving blocks a
# partial-fill fraction restores the continuous spread that makes shadow prices
# discriminative at all.
FILL_MIN, FILL_MAX = 0.45, 1.0


def P(*a):
    print(*a, flush=True)


def make_instance(seed, nx, nz, keep_gram):
    """One synthetic deposit. Geometry fixed, economics and capacities vary.

    `keep_gram=False` drops the Gram matrix and precedence arrays once the
    expert has been solved -- training on the 'raw' variant never needs them.
    """
    rng = np.random.default_rng(seed)
    gx, gy, gz = np.meshgrid(np.arange(nx)*STEP, np.arange(nx)*STEP,
                             np.arange(nz)*STEP, indexing="ij")
    x, y, z = gx.ravel(), gy.ravel(), gz.ravel()
    n = x.size
    ix = (x/STEP).astype(int); iy = (y/STEP).astype(int); iz = (z/STEP).astype(int)
    bench = iz.max() - iz

    cx, cy = rng.uniform(0.25, 0.75, 2) * (nx-1) * STEP
    spread = rng.uniform(0.15, 0.35) * nx * STEP
    depth = rng.uniform(0.3, 0.9) * max(bench.max(), 1)
    grade = rng.uniform(0.8, 1.6)
    ore = (np.exp(-((x-cx)**2 + (y-cy)**2)/(2*spread**2))
           * np.exp(-((bench-depth)**2)/(2*rng.uniform(1.5, 3.5)**2)))
    income = np.where(ore > 0.3, grade * 1.4e5 * ore, 0.0)
    is_ore = income > 0
    if is_ore.sum() < 10:
        return None
    fill = rng.uniform(FILL_MIN, FILL_MAX, n)          # partial blocks
    tonnage = np.where(is_ore, 4.5, 3.0) * STEP**3 * fill
    income = income * fill                              # grade scales with mass
    value = income - 1e3 - rng.uniform(1.5e3, 2.5e3) * bench

    par, chi = build_edges(square_pyramid_predecessors(
        pd.DataFrame({"x_c": x, "y_c": y, "z_c": z}), slope_h_per_v=1.5).tolist())
    order = topological_order(n, par, chi)

    qty = {"income": income, "cost": 1e3 + 2e3*bench,
           "tonnage": tonnage, "value": value}
    lb = cone_sums(ix, iy, iz, qty, levels=5, direction="below")
    s_val = (value - value.mean()) / value.std()
    Z = minmax_normalize(build_features(x, y, z, bench, s_val, value,
                                        lb["value"], tonnage))
    ell = ard_lengthscales(Z.shape[1], ix.max()+1, iy.max()+1, iz.max()+1, 6.0, 2.0)
    gram, _ = wendland_c0_sparse_gram(Z, ix, iy, iz, (6, 6, 2), ell)

    # tighter than before: with generous capacity the horizon never bites and
    # every reasonable order scores about the same
    mf, pf, sf = rng.uniform(.30, .50), rng.uniform(.30, .50), rng.uniform(.35, .60)
    resources = [
        make_resource("mining", tonnage, np.full(T, mf*tonnage.sum()/T)),
        make_resource("processing", tonnage*is_ore,
                      np.full(T, pf*tonnage[is_ore].sum()/T)),
        make_resource("stripping", tonnage*(~is_ore),
                      np.full(T, sf*tonnage[~is_ore].sum()/T))]

    feat = np.stack([x, y, z, bench, income, tonnage, value, lb["value"]], axis=1)
    feat = (feat - feat.mean(0)) / (feat.std(0) + 1e-9)

    D = dict(n=n, par=par, chi=chi, order=order, gram=gram, value=value,
             tonnage=tonnage, is_ore=is_ore, resources=resources,
             feat=feat.astype(np.float32), s_val=s_val)
    del Z, ell, lb, qty, gx, gy, gz
    return D, keep_gram


def hard_npv(s_np, D):
    sp, _ = project_pocs_sparse(s_np, D["gram"], D["par"], D["chi"], D["order"])
    seq = schedule_from_scores(sp, D["par"], D["chi"], D["order"])
    r = evaluate_cuts_multi(seq, D["resources"], value=D["value"], discount=DISCOUNT)
    return r["npv"], seq, sequence_violations(seq, D["par"], D["chi"])


def solve_expert(D, iters):
    best, _ = reprice_loop(D["value"], D["resources"], D["gram"], D["par"],
                           D["chi"], D["order"], D["tonnage"], discount=DISCOUNT,
                           n_iters=iters, damping=0.5, noise=np.zeros(D["n"]),
                           true_value=D["value"], verbose=False)
    return best["npv"], best["seq"]


def listmle(scores, target_seq):
    sp = scores[target_seq]
    lse = torch.flip(torch.logcumsumexp(torch.flip(sp, [0]), 0), [0])
    return -(sp - lse).mean()


def main(cfg_name="small"):
    cfg = CONFIGS[cfg_name]
    torch.set_num_threads(4)
    t0 = time.perf_counter()

    P(f"config '{cfg_name}': {cfg['nx']}x{cfg['nx']}x{cfg['nz']} blocks, "
      f"{cfg['n_train']} train / {cfg['n_test']} test, {cfg['epochs']} epochs")
    P("building instances and expert solutions (one at a time)...")

    train, test, seed = [], [], 0
    need = cfg["n_train"] + cfg["n_test"]
    lifts = []
    while len(train) + len(test) < need:
        made = make_instance(seed, cfg["nx"], cfg["nz"], keep_gram=True)
        seed += 1
        if made is None:
            continue
        D, _ = made
        D["expert_npv"], D["expert_seq"] = solve_expert(D, cfg["reprice_iters"])
        D["base_npv"] = hard_npv(D["s_val"], D)[0]
        lifts.append(D["expert_npv"] / D["base_npv"] - 1.0)
        if len(train) < cfg["n_train"]:
            train.append(D)
        else:
            test.append(D)
        gc.collect()

    n_blocks = train[0]["n"]
    P(f"  {need} instances of {n_blocks} blocks in {time.perf_counter()-t0:.0f}s")
    P(f"  expert vs value heuristic: {100*np.mean(lifts):+.2f}% mean, "
      f"{100*np.max(lifts):+.2f}% best, {100*np.min(lifts):+.2f}% worst")
    if np.mean(lifts) < 0.005:
        P("  NOTE: the expert is barely better than the baseline on these")
        P("        instances, so there is little for a student to learn.")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    feat_batch = torch.from_numpy(np.stack([D["feat"] for D in train])).to(dev)
    tgt_batch = [torch.from_numpy(D["expert_seq"].copy()).to(dev) for D in train]
    P(f"  batched input {tuple(feat_batch.shape)} on {dev}")

    results = {}
    for variant in ("raw", "proj"):
        torch.manual_seed(0)
        net = SimpleTransformer(in_dim=8, d_model=cfg["d_model"], nhead=cfg["nhead"],
                                num_layers=cfg["layers"], dim_feedforward=cfg["ff"],
                                out_dim=1, use_posenc=False)
        net = net.to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        P(f"\ntraining '{variant}' on {dev}")
        te = time.perf_counter()
        for ep in range(cfg["epochs"]):
            opt.zero_grad()
            if variant == "raw":
                # all instances share n, so one batched forward replaces
                # len(train) separate ones -- the whole point of fixing nx
                s = net(feat_batch).squeeze(-1)                  # (B, n)
                loss = torch.stack([listmle(s[b], tgt_batch[b])
                                    for b in range(s.shape[0])]).mean()
            else:
                s = net(feat_batch).squeeze(-1)
                parts = []
                for b, D in enumerate(train):
                    sp = project_torch(s[b].double().cpu(), D["gram"], D["par"],
                                       D["chi"], D["order"], projector="pocs")
                    parts.append(listmle(sp, tgt_batch[b].cpu()))
                loss = torch.stack(parts).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            if ep % max(cfg["epochs"]//4, 1) == 0 or ep == cfg["epochs"]-1:
                P(f"  epoch {ep:>3}  ListMLE {float(loss.detach()):.4f}  "
                  f"({time.perf_counter()-te:.1f}s)")
            del s, loss

        net.eval()
        rows = []
        with torch.no_grad():
            for D in test:
                x_in = torch.from_numpy(D["feat"]).unsqueeze(0).to(dev)
                s = net(x_in).squeeze().cpu().numpy()
                npv, _, viol = hard_npv(s.astype(np.float64), D)
                rows.append((npv, D["base_npv"], D["expert_npv"], viol))
        results[variant] = rows
        del net, opt
        gc.collect()

    P(f"\nheld-out results ({cfg['n_test']} unseen instances)")
    P(f"{'variant':<8} {'vs value heuristic':>20} {'vs expert':>12} {'wins':>8} {'viol':>6}")
    P("-"*58)
    for variant, rows in results.items():
        g_base = np.mean([r[0]/r[1]-1 for r in rows])
        g_exp = np.mean([r[0]/r[2]-1 for r in rows])
        wins = sum(1 for r in rows if r[0] > r[1])
        P(f"{variant:<8} {100*g_base:>19.2f}% {100*g_exp:>11.2f}% "
          f"{wins:>5}/{len(rows)} {sum(r[3] for r in rows):>6}")
    P(f"\ntotal {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "small")
