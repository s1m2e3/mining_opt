"""
Stage 3b: a transformer proposes the candidates, BO screens them, and the
schedules BO liked are regressed back into the transformer.

    transformer  ->  score field  ->  gumbel samples  ->  phi_score
                                                             |
                    true evaluation  <-  top q by logEI  <-  GP
                            |
                            +-> best schedule so far -> Plackett-Luce loss -> transformer

Three design decisions worth stating, because each one is what keeps this from
being strictly worse than the Stage 3a control it has to beat.

RESIDUAL, NOT REPLACEMENT. The network emits s = s_base + f(features), with the
output layer zero-initialised, so at step 0 it reproduces the knob-based scorer
exactly. An untrained transformer emitting scores from scratch would propose
noise for the first several rounds and lose to 3a on sample budget alone. This
way the learned part can only add.

MIXED POOL. Each round's candidate pool is half Sobol knob draws (exactly 3a's
proposal) and half transformer samples. The GP screens the union, so the pool
is a superset of 3a's and the method cannot be beaten by its own control on
proposal diversity. The fraction of picks that came from the network is
reported every round -- that number, rising or not, is the honest measure of
whether the transformer is contributing anything.

PLACKETT-LUCE, NOT L2. The loss is the NLL of the target ORDER under the
network's scores, not the distance between score vectors. Two score fields
differing wildly in L2 can decode to the identical schedule -- any monotone
rescaling does -- so an L2 loss would spend most of its gradient on differences
the objective cannot see. Computed in O(n log n) with a reverse cumulative
logsumexp rather than the naive O(n^2) sequential form, and truncated to the
first `top_k` positions because discounting makes the early schedule carry
almost all of the NPV.

    python policy_bo.py --crop 14 --rounds 12 --q 4 --seeds 3
"""

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from botorch.acquisition.analytic import LogExpectedImprovement
from torch.quasirandom import SobolEngine

from bo_search import DTYPE, KnobSpace, fit_gp
from cluster_profile import fit_clusters, profile, score_profile
from mine_problem import (KNOBS, _z, cone_cache, default_knobs, evaluate,
                          intrinsic_geometry, knob_active, load_static,
                          priority_from_knobs)
from model import MultiViewSparseMHA_3a

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def P(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------
# per-block features and attention neighbourhoods
# --------------------------------------------------------------------------

def block_features(static):
    """Standardised per-block inputs. Deliberately the same information the
    knob scorer sees, so any gain is attributable to the network combining it
    non-linearly and with context, not to it being handed extra data."""
    cone = cone_cache(static, 5)
    g = intrinsic_geometry(static)
    cols = [static["value"], static["tonnage"], static["mill_feed"],
            static["au"], static["cu"], static["bench"].astype(float),
            cone["below"]["value"], cone["below"]["income"],
            cone["below"]["tonnage"], cone["above"]["value"],
            cone["above"]["tonnage"], g["bench_frac"], g["r_centroid"],
            g["d_ore"], g["anc_frac"]]
    return np.column_stack([_z(c) for c in cols])


def cone_offsets(levels, direction, cross=True):
    """(dx, dy, dz) offsets of the square-pyramid cone, `levels` benches deep.

    `direction` is +1 for the ANCESTOR cone (up: what must be stripped before
    this block is reachable) and -1 for the DESCENDANT cone (down: the inverted
    pyramid this block unlocks).

    The half-width is `l`, not slope_h_per_v * l. The precedence DAG is built
    from IMMEDIATE predecessors only -- the 3x3 footprint one bench up, since
    max(|dx|,|dy|) <= 1.5 admits exactly |dx|,|dy| <= 1 on a cubic grid -- so
    its transitive closure moves at most one lateral step per bench and the true
    l-level cone is max(|dx|,|dy|) <= l. Using the raw 1.5 slope here would
    reach blocks that are not actually related by precedence.

    `cross` takes only the dy=0 and dx=0 slices, i.e. the neighbours along x and
    along y, giving 4l+1 offsets per level instead of (2l+1)^2. At levels=5 that
    is 65 against 285, and the difference is not cosmetic: attention gathers a
    (B, heads, n, K, dh) key tensor and another for the values, so at n=12,213
    the full cone costs ~446 MB per tensor per view against ~103 MB for the
    cross, and autograd holds both for every view until backward. The cross
    keeps both axes and both signs of each axis; what it drops are the diagonal
    corners, which the second attention layer can still reach by composing an
    x-step with a y-step.
    """
    offs = []
    for l in range(1, levels + 1):
        dz = direction * l
        for d in range(-l, l + 1):
            offs.append((d, 0, dz))
            if not cross:
                offs.extend((d, e, dz) for e in range(-l, l + 1) if e != 0)
            elif d != 0:
                offs.append((0, d, dz))     # d == 0 is the spine, already added
    return offs


def gather_offsets(static, offs):
    """Offsets -> (idx, mask), both (n, 1 + len(offs)).

    Slot 0 is the node itself and is always valid, so no row is ever entirely
    masked -- a surface block with no ancestors attends to itself alone rather
    than producing an undefined softmax. Every other slot is a genuine
    neighbour or is masked off; padding never competes for attention mass.
    """
    n = static["n"]
    ix, iy, iz = static["ix"], static["iy"], static["iz"]
    nx, ny, nz = static["nx"], static["ny"], static["nz"]
    lut = np.full(nx * ny * nz, -1, dtype=np.int64)
    lut[(ix * ny + iy) * nz + iz] = np.arange(n)

    K = len(offs) + 1
    idx = np.repeat(np.arange(n, dtype=np.int64)[:, None], K, axis=1)
    mask = np.zeros((n, K), dtype=bool)
    mask[:, 0] = True

    for c, (dx, dy, dz) in enumerate(offs, start=1):
        jx, jy, jz = ix + dx, iy + dy, iz + dz
        ok = ((jx >= 0) & (jx < nx) & (jy >= 0) & (jy < ny)
              & (jz >= 0) & (jz < nz))
        cand = np.full(n, -1, dtype=np.int64)
        cand[ok] = lut[(jx[ok] * ny + jy[ok]) * nz + jz[ok]]
        hit = cand >= 0
        idx[hit, c] = cand[hit]
        mask[hit, c] = True
    return idx, mask


def build_views(static, levels=5, cross=True):
    """Three neighbourhoods, each a different notion of 'related':

      ancestors    the cone ABOVE, `levels` benches deep -- everything that
                   must come off before this block is reachable
      descendants  the inverted pyramid BELOW -- what this block unlocks
      column       the full column straight up to surface, however deep

    The two cones carry the lateral structure: at l benches away the cone is l
    blocks wide in x and in y, so a block sees the material whose removal it
    depends on and the material whose availability depends on it, at the exact
    geometry the precedence constraint uses.

    The column is separate and uncapped because stripping cost is a column
    property. Everything directly overhead has to be moved for this block to be
    mined, and truncating that at `levels` would hide most of it -- the model is
    23 benches deep, so a bottom block's true overburden is four times deeper
    than the cone reaches.

    Returns (idxs, masks), each a list of three (1, n, K_v) tensors.
    """
    views = [cone_offsets(levels, +1, cross),          # ancestors, up
             cone_offsets(levels, -1, cross),          # descendants, down
             [(0, 0, dz) for dz in range(1, static["nz"])]]   # column, up

    idxs, masks = [], []
    for offs in views:
        idx, mask = gather_offsets(static, offs)
        idxs.append(torch.tensor(idx, dtype=torch.long, device=DEV).unsqueeze(0))
        masks.append(torch.tensor(mask, dtype=torch.bool, device=DEV).unsqueeze(0))
    return idxs, masks


class BlockScorer(nn.Module):
    """Two sparse multi-view attention layers and a linear head.

    The head is zero-initialised on purpose -- see the module docstring. The
    output is a residual on the knob score, so an untrained network is exactly
    the Stage 3a baseline rather than a source of noise.
    """

    def __init__(self, d_in, d_model=96, n_heads=6, n_views=3, dropout=0.0):
        super().__init__()
        self.l1 = MultiViewSparseMHA_3a(d_in=d_in, d_model=d_model,
                                        n_heads=n_heads, n_views=n_views,
                                        d_out=d_model, dropout=dropout)
        self.l2 = MultiViewSparseMHA_3a(d_in=d_model, d_model=d_model,
                                        n_heads=n_heads, n_views=n_views,
                                        d_out=d_model, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, idxs, masks=None):
        h = self.l1(x, idxs, masks)
        h = h + self.l2(torch.relu(h), idxs, masks)
        return self.head(self.norm(h)).squeeze(-1).squeeze(0)


def pl_nll(scores, order, top_k=None, temperature=1.0):
    """Plackett-Luce negative log-likelihood of `order` under `scores`.

        -sum_t [ s_{pi(t)} - logsumexp( s_{pi(t)}, ..., s_{pi(n)} ) ]

    The suffix logsumexps are a reverse cumulative logsumexp, so this is
    O(n log n) rather than the O(n^2) of evaluating each denominator directly.
    Truncating to `top_k` focuses the gradient on the early schedule, which at
    a 0.90 discount is where essentially all the NPV is decided.
    """
    s = scores[order] / temperature
    denom = torch.flip(torch.logcumsumexp(torch.flip(s, [0]), 0), [0])
    ll = s - denom
    if top_k is not None:
        ll = ll[:top_k]
    return -ll.mean()


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------

def knob_candidates(static, space, size, seed, labels, k):
    U = SobolEngine(space.dim, scramble=True, seed=seed).draw(size).numpy()
    out = []
    for u in U:
        kn = space.from_unit(u)
        s = priority_from_knobs(static, kn)
        out.append({"s": s, "phi": score_profile(s, labels, k), "src": "knob"})
    return out


def net_candidates(base, resid, size, tau, seed, labels, k, cluster_sigma=1.0):
    """Diversify at the CLUSTER scale, not per block.

    Per-block iid Gumbel is the wrong noise for this descriptor. phi_score is a
    mean over ~90 blocks per cluster, so iid perturbations average out at rate
    1/sqrt(90) and every sample lands on essentially the same descriptor. The
    GP then sees one already-visited point with near-zero expected improvement
    and never picks any of them -- measured: 0 of 48 picks on two seeds out of
    three, while structurally-varying knob candidates took every slot.

    Per-cluster offsets move phi_score directly, which is what the screening
    GP can actually see. The per-block Gumbel is kept on top because it still
    breaks ties within a cluster, where the ordering is otherwise fixed.
    """
    rng = np.random.default_rng(seed)
    s0 = base + resid
    out = []
    for _ in range(size):
        offs = rng.normal(0.0, cluster_sigma, k)[labels]
        s = s0 + offs + tau * rng.gumbel(size=s0.shape[0])
        out.append({"s": s, "phi": score_profile(s, labels, k), "src": "net"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", type=int, default=14)
    ap.add_argument("--transform", default="pocs",
                    choices=("raw", "smooth", "pocs"),
                    help="ablation: does the projection smooth away what the "
                         "network adds before the decoder can use it?")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--levels", type=int, default=5,
                    help="cone depth in benches for the ancestor/descendant "
                         "attention views; K grows as 2*levels^2 + 3*levels")
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--pool", type=int, default=400, help="per source")
    ap.add_argument("--q", type=int, default=4)
    ap.add_argument("--n-init", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tau", type=float, default=0.3)
    ap.add_argument("--cluster-sigma", type=float, default=1.0,
                    help="per-cluster score offset sd -- the diversity phi_score can see")
    ap.add_argument("--train-steps", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--top-k", type=int, default=1000)
    ap.add_argument("--out", default="outputs/policy_bo.json")
    args = ap.parse_args()
    crop = args.crop if args.crop > 0 else None

    t_all = time.perf_counter()
    S = load_static(crop=crop)
    names = [kk for kk in KNOBS if knob_active(kk, args.transform)]
    space = KnobSpace(names)
    labels = fit_clusters(S, k=args.k)["labels"]
    X_feat = torch.tensor(block_features(S), dtype=torch.float32,
                          device=DEV).unsqueeze(0)
    idxs, masks = build_views(S, levels=args.levels)
    P(f"n={S['n']:,}  features={X_feat.shape[-1]}  k={args.k}  device={DEV}")
    P("views  anc/des/column  K = "
      + "/".join(str(i.shape[-1]) for i in idxs)
      + f"   valid slots {'/'.join(f'{m.float().mean():.0%}' for m in masks)}")
    P(f"pool={args.pool}/source  q={args.q}  rounds={args.rounds}  "
      f"-> {args.rounds*args.q} true evaluations per arm\n")

    report = {"crop": args.crop, "k": args.k, "rounds": args.rounds,
              "q": args.q, "pool": args.pool, "runs": []}
    gaps = []

    for seed in range(args.seeds):
        torch.manual_seed(seed)
        init = knob_candidates(S, space, args.n_init, 1000 + seed, labels, args.k)
        init_res = [evaluate(S, {}, transform=args.transform, decoder="kahn",
                             check=False, s_raw=c["s"]) for c in init]
        init_y = [r["npv"] for r in init_res]
        b0 = int(np.argmax(init_y))
        s_base = init[b0]["s"].copy()

        arms = {}
        for arm in ("policy", "knob_only"):
            net = BlockScorer(d_in=X_feat.shape[-1]).to(DEV)
            opt = torch.optim.Adam(net.parameters(), lr=args.lr)
            Xg = [c["phi"] for c in init]
            Yg = list(init_y)
            best_seq = init_res[b0]["seq"]
            best_y = max(init_y)
            n_net_picks = 0
            t0 = time.perf_counter()

            for rnd in range(args.rounds):
                pool = knob_candidates(S, space, args.pool,
                                       seed * 100 + rnd, labels, args.k)
                if arm == "policy":
                    with torch.no_grad():
                        resid = net(X_feat, idxs, masks).cpu().numpy().astype(float)
                    pool += net_candidates(s_base, resid, args.pool, args.tau,
                                           seed * 100 + rnd, labels, args.k,
                                           cluster_sigma=args.cluster_sigma)

                gp, _, Yt = fit_gp(Xg, Yg)
                acq = LogExpectedImprovement(model=gp, best_f=Yt.max())
                phis = np.array([c["phi"] for c in pool])
                with torch.no_grad():
                    vals = acq(torch.tensor(phis, dtype=DTYPE
                                            ).unsqueeze(1)).numpy()
                picks = np.argsort(-vals)[:args.q]
                bo_target = None
                for rank, i in enumerate(picks):
                    c = pool[i]
                    r = evaluate(S, {}, transform=args.transform, decoder="kahn",
                                 check=False, s_raw=c["s"])
                    Xg.append(c["phi"])
                    Yg.append(r["npv"])
                    if c["src"] == "net":
                        n_net_picks += 1
                    if rank == 0:
                        bo_target = r["seq"]          # the acquisition argmax
                    if r["npv"] > best_y:
                        best_y, best_seq = r["npv"], r["seq"]

                if arm == "policy" and args.train_steps and bo_target is not None:
                    # Regress onto what BO says to try NEXT, not onto the best
                    # schedule seen so far. Those are different targets and the
                    # distinction decides whether this can work at all:
                    #
                    #   best-so-far  is the incumbent. The network learns to
                    #                reproduce a schedule the proposal pool
                    #                already contains, so its samples duplicate
                    #                candidates that were available anyway.
                    #                Measured: +0.00% on every seed.
                    #   acquisition  is where the GP expects improvement, which
                    #     argmax     combines the posterior mean with its
                    #                uncertainty. It points at regions nothing
                    #                has evaluated yet, so the target carries
                    #                information the pool does not already have.
                    #
                    # Over rounds this amortises the acquisition: the network
                    # learns to emit what BO would have had to screen 800
                    # candidates to find.
                    tgt = torch.tensor(bo_target.copy(), dtype=torch.long,
                                       device=DEV)
                    base_t = torch.tensor(s_base, dtype=torch.float32,
                                          device=DEV)
                    for _ in range(args.train_steps):
                        opt.zero_grad()
                        loss = pl_nll(base_t + net(X_feat, idxs, masks), tgt,
                                      top_k=args.top_k)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                        opt.step()

            arms[arm] = {"best": float(best_y), "net_picks": n_net_picks,
                         "seconds": time.perf_counter() - t0}

        gap = (arms["policy"]["best"] / arms["knob_only"]["best"] - 1.0) * 100
        gaps.append(gap)
        tot = args.rounds * args.q
        report["runs"].append({"seed": seed, "init_best": float(max(init_y)),
                               "policy": arms["policy"]["best"],
                               "knob_only": arms["knob_only"]["best"],
                               "gap_pct": gap,
                               "net_picks": arms["policy"]["net_picks"],
                               "net_pick_frac": arms["policy"]["net_picks"] / tot})
        P(f"seed {seed}:  init {max(init_y):>15,.0f}   "
          f"policy {arms['policy']['best']:>15,.0f}   "
          f"knob-only {arms['knob_only']['best']:>15,.0f}   "
          f"gap {gap:>+6.2f}%   net picks "
          f"{arms['policy']['net_picks']}/{tot}   "
          f"({arms['policy']['seconds']:.0f}s/{arms['knob_only']['seconds']:.0f}s)")

    gaps = np.asarray(gaps)
    frac = float(np.mean([r["net_pick_frac"] for r in report["runs"]]))
    report["summary"] = {
        "mean_gap_pct": float(gaps.mean()),
        "sd_gap_pct": float(gaps.std(ddof=1)) if len(gaps) > 1 else 0.0,
        "wins": int((gaps > 0).sum()), "n": int(gaps.size),
        "mean_net_pick_frac": frac}
    P(f"\npolicy - knob_only over {gaps.size} seeds: mean {gaps.mean():+.2f}%  "
      f"sd {gaps.std(ddof=1) if len(gaps) > 1 else 0:.2f}%  "
      f"wins {int((gaps > 0).sum())}/{gaps.size}")
    P(f"transformer candidates chosen by the GP: {frac*100:.0f}% of picks")

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    P(f"wrote {args.out}   total {time.perf_counter()-t_all:.0f}s")


if __name__ == "__main__":
    main()
