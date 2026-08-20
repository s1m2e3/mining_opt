"""
Stage 3c: BO proposes a target profile, the transformer learns to realise it.

    transformer -> raw scores -> projection (feasible) -> cluster profile
                                                              |
                                          GP over (profile, NPV) from history
                                                              |
                            psi* = argmax logEI  subject to the profile polytope
                                                              |
              loss = || phi_soft(transformer scores) - psi* ||^2  -> transformer

The difference from Stage 3b, which measured nothing: there the teacher was a
SELECTOR -- the acquisition argmax over a candidate pool -- so the target was
always something the pool already contained and the network could only learn to
duplicate it. Here the teacher GENERATES: psi* is a point on the constrained
polytope that no candidate achieved, obtained by extrapolating the GP.

Why the target is a rank profile and not a period profile. NPV is
sum_c V_c gamma^(period_c), monotone decreasing in every period, so maximising
an acquisition over period profiles returns 'every valuable cluster in period
1' for every deposit that has ever existed -- infeasible, identical across
instances, and knowable without a GP. That the GP fitted period profiles at
R^2=0.99 was the tell: it had rediscovered the discounting formula, not learned
anything about the mine.

Rank profiles cannot degenerate that way. Ranks are a permutation, so

    sum_c |c| * psi_c  =  n/2      exactly, always

and promoting one cluster necessarily demotes another. The budget is a property
of the representation rather than a trust region bolted on afterwards. The GP
still has to be told about it -- see solve_target -- but the constraint is
exact rather than a guess.

Why the transformer is the right actuator: the 15-knob family's image in k-dim
profile space is a 15-dimensional manifold, so almost every psi* is unreachable
by ANY knob setting. A per-block score field has n degrees of freedom.

    python target_bo.py --crop 14 --rounds 10 --q 3 --seeds 3
"""

import argparse
import json
import time

import numpy as np
import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from torch.quasirandom import SobolEngine

from bo_search import DTYPE, KnobSpace, fit_gp
from cluster_profile import fit_clusters, score_profile
from mine_problem import KNOBS, evaluate, knob_active, load_static, priority_from_knobs
from policy_bo import DEV, BlockScorer, block_features, build_views

# --------------------------------------------------------------------------
# the profile polytope
# --------------------------------------------------------------------------


def cluster_precedence(static, labels, k, min_edges=20, dominance=8.0):
    """Cluster-level precedence: A before B if block edges A->B dominate B->A.

    This is what keeps psi* inside the REALISABLE region rather than merely the
    arithmetically valid one. The budget constraint alone permits profiles that
    no schedule can produce -- you cannot rank a deep cluster early without
    ranking everything above it earlier still -- and the GP has no way to know
    that. Reading the constraint off the block DAG costs nothing and is not a
    tuned trust region.

    `dominance` guards against reading a constraint out of a handful of
    boundary edges: the forward count must beat the reverse count by this
    factor. Any cycles that survive are broken by dropping the weakest edge,
    since a cyclic constraint set is infeasible by construction.
    """
    par, chi = static["par"], static["chi"]
    a, b = labels[par], labels[chi]
    keep = a != b
    a, b = a[keep], b[keep]
    cnt = np.zeros((k, k))
    np.add.at(cnt, (a, b), 1.0)

    pairs = []
    for i in range(k):
        for j in range(k):
            if i == j or cnt[i, j] < min_edges:
                continue
            if cnt[i, j] > dominance * max(cnt[j, i], 1.0):
                pairs.append((i, j, cnt[i, j]))

    # break cycles: repeatedly drop the weakest edge on any cycle found
    while True:
        adj = [[] for _ in range(k)]
        for i, j, _ in pairs:
            adj[i].append(j)
        colour = [0] * k
        cyc = []

        def dfs(u, stack):
            colour[u] = 1
            stack.append(u)
            for v in adj[u]:
                if colour[v] == 1:
                    cyc.append(stack[stack.index(v):] + [v])
                    return True
                if colour[v] == 0 and dfs(v, stack):
                    return True
            stack.pop()
            colour[u] = 2
            return False

        found = any(colour[u] == 0 and dfs(u, []) for u in range(k))
        if not found:
            break
        ring = cyc[0]
        edges = [(ring[t], ring[t + 1]) for t in range(len(ring) - 1)]
        weakest = min(edges, key=lambda e: cnt[e[0], e[1]])
        pairs = [p for p in pairs if (p[0], p[1]) != weakest]
    return [(i, j) for i, j, _ in pairs]


def project_polytope(psi, w, rhs, iters=40):
    """Alternating projection onto {sum w_c psi_c = rhs} and the unit box."""
    p = np.clip(np.asarray(psi, dtype=float), 0.0, 1.0)
    ww = float(w @ w)
    for _ in range(iters):
        p = p - w * ((w @ p - rhs) / ww)
        p = np.clip(p, 0.0, 1.0)
    return p


def solve_target(gp, best_f, w, rhs, prec_pairs, k, n_restarts=8, steps=200,
                 lr=0.05, penalty=50.0, seed=0, init_from=None):
    """psi* = argmax logEI on the profile polytope, by projected gradient.

    Hand-rolled rather than optimize_acqf's constrained path: the equality
    constraint has an exact closed-form projection, the precedence constraints
    are cheap as a hinge penalty, and this avoids depending on how BoTorch
    seeds feasible initial conditions under equality constraints.
    """
    rng = np.random.default_rng(seed)
    starts = []
    for r in range(n_restarts):
        base = (init_from[rng.integers(len(init_from))] if init_from is not None
                and len(init_from) else rng.random(k))
        starts.append(project_polytope(base + rng.normal(0, 0.15, k), w, rhs))

    ii = torch.tensor([p[0] for p in prec_pairs], dtype=torch.long, device="cpu")
    jj = torch.tensor([p[1] for p in prec_pairs], dtype=torch.long, device="cpu")
    best, best_val = None, -np.inf
    for s0 in starts:
        x = torch.tensor(s0, dtype=DTYPE, requires_grad=True)
        opt = torch.optim.Adam([x], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            ei = gp_logei(gp, best_f, x)
            pen = (torch.clamp(x[ii] - x[jj], min=0.0) ** 2).sum() if len(prec_pairs) else 0.0
            (-(ei) + penalty * pen).backward()
            opt.step()
            with torch.no_grad():
                x.copy_(torch.tensor(project_polytope(x.detach().numpy(), w, rhs),
                                     dtype=DTYPE))
        with torch.no_grad():
            v = float(gp_logei(gp, best_f, x))
        if v > best_val:
            best_val, best = v, x.detach().numpy().copy()
    return best, best_val


def gp_logei(gp, best_f, x):
    acq = LogExpectedImprovement(model=gp, best_f=best_f)
    return acq(x.unsqueeze(0).unsqueeze(0)).squeeze()


def random_target(w, rhs, k, observed, seed):
    """Control target: a random point on the same polytope. If GP-chosen
    targets are no better than these, the acquisition contributes nothing and
    the loop is an elaborate random walk."""
    rng = np.random.default_rng(seed)
    base = (observed[rng.integers(len(observed))] if len(observed)
            else rng.random(k))
    return project_polytope(base + rng.normal(0, 0.25, k), w, rhs)


# --------------------------------------------------------------------------
# differentiable profile
# --------------------------------------------------------------------------

def soft_profile(s, onehot, counts, tau=0.05, chunk=512):
    """Cluster-mean SOFT rank, differentiable in s.

        r_soft(i) = sum_j sigmoid( (s_j - s_i) / tau )  / (n - 1)

    The hard rank in score_profile is an argsort and stops the gradient. This
    is the same relaxation soft_rank.py uses, asked to do far less: it only has
    to be monotonically related to the true rank, not to predict NPV, so its
    bias is not load-bearing.

    Chunked over rows -- the pairwise term is n x n and only the row sums are
    needed, so it never has to exist all at once.
    """
    n = s.shape[0]
    tot = torch.zeros(counts.shape[0], dtype=s.dtype, device=s.device)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        m = torch.sigmoid((s.unsqueeze(0) - s[a:b].unsqueeze(1)) / tau)
        tot = tot + onehot[a:b].T @ m.sum(dim=1)
    return tot / (counts * (n - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", type=int, default=14)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--levels", type=int, default=5,
                    help="cone depth in benches for the ancestor/descendant "
                         "attention views; K grows as 2*levels^2 + 3*levels")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--q", type=int, default=3)
    ap.add_argument("--n-init", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--train-steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--cluster-sigma", type=float, default=0.6)
    ap.add_argument("--transform", default="pocs")
    ap.add_argument("--out", default="outputs/target_bo.json")
    args = ap.parse_args()
    crop = args.crop if args.crop > 0 else None

    t_all = time.perf_counter()
    S = load_static(crop=crop)
    n, k = S["n"], args.k
    cl = fit_clusters(S, k=k)
    labels, counts = cl["labels"], cl["counts"].astype(float)
    w = counts.copy()
    rhs = n / 2.0
    prec = cluster_precedence(S, labels, k)
    P = print
    P(f"n={n:,}  k={k}  clusters {counts.min():.0f}/{np.median(counts):.0f}/"
      f"{counts.max():.0f}  precedence constraints {len(prec)}")

    X_feat = torch.tensor(block_features(S), dtype=torch.float32,
                          device=DEV).unsqueeze(0)
    idxs, masks = build_views(S, levels=args.levels)
    onehot = torch.zeros(n, k, dtype=torch.float32, device=DEV)
    onehot[torch.arange(n), torch.tensor(labels, device=DEV)] = 1.0
    counts_t = torch.tensor(counts, dtype=torch.float32, device=DEV)

    names = [kk for kk in KNOBS if knob_active(kk, args.transform)]
    space = KnobSpace(names)
    budget = args.rounds * args.q
    P(f"budget {budget} true evaluations per arm, {args.seeds} seeds\n")

    report = {"crop": args.crop, "k": k, "rounds": args.rounds, "q": args.q,
              "n_prec": len(prec), "runs": []}
    gaps = []

    for seed in range(args.seeds):
        torch.manual_seed(seed)
        U = SobolEngine(space.dim, scramble=True, seed=1000 + seed).draw(args.n_init).numpy()
        init = []
        for u in U:
            kn = space.from_unit(u)
            r = evaluate(S, kn, transform=args.transform, decoder="kahn", check=False)
            init.append((score_profile(r["s_used"], labels, k), r["npv"], r["s_raw"]))
        b0 = int(np.argmax([y for _, y, _ in init]))
        s_base = init[b0][2].copy()
        base_t = torch.tensor(s_base, dtype=torch.float32, device=DEV)

        arms = {}
        for arm in ("gp", "random"):
            net = BlockScorer(d_in=X_feat.shape[-1]).to(DEV)
            opt = torch.optim.Adam(net.parameters(), lr=args.lr)
            Xg = [p for p, _, _ in init]
            Yg = [y for _, y, _ in init]
            miss = []
            t0 = time.perf_counter()

            for rnd in range(args.rounds):
                gp, _, Yt = fit_gp(Xg, Yg)
                if arm == "gp":
                    psi, _ = solve_target(gp, Yt.max(), w, rhs, prec, k,
                                          seed=seed * 100 + rnd,
                                          init_from=np.array(Xg))
                else:
                    psi = random_target(w, rhs, k, np.array(Xg),
                                        seed=seed * 100 + rnd)
                psi_t = torch.tensor(psi, dtype=torch.float32, device=DEV)

                for _ in range(args.train_steps):
                    opt.zero_grad()
                    s = base_t + net(X_feat, idxs, masks)
                    loss = ((soft_profile(s, onehot, counts_t, args.tau)
                             - psi_t) ** 2).mean()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                    opt.step()

                with torch.no_grad():
                    s_np = (base_t + net(X_feat, idxs, masks)).cpu().numpy().astype(float)
                rng = np.random.default_rng(seed * 977 + rnd)
                for c in range(args.q):
                    sc = (s_np if c == 0 else
                          s_np + rng.normal(0, args.cluster_sigma, k)[labels])
                    r = evaluate(S, {}, transform=args.transform, decoder="kahn",
                                 check=False, s_raw=sc)
                    phi = score_profile(r["s_used"], labels, k)
                    Xg.append(phi)
                    Yg.append(r["npv"])
                    if c == 0:
                        miss.append(float(np.linalg.norm(phi - psi) / np.sqrt(k)))

            arms[arm] = {"best": float(max(Yg)), "miss": miss,
                         "seconds": time.perf_counter() - t0}

        gap = (arms["gp"]["best"] / arms["random"]["best"] - 1.0) * 100
        gaps.append(gap)
        m = arms["gp"]["miss"]
        report["runs"].append({"seed": seed, "gp": arms["gp"]["best"],
                               "random": arms["random"]["best"], "gap_pct": gap,
                               "miss_first": m[0], "miss_last": m[-1],
                               "miss": m})
        P(f"seed {seed}:  GP-target {arms['gp']['best']:>15,.0f}   "
          f"random-target {arms['random']['best']:>15,.0f}   gap {gap:>+6.2f}%"
          f"   |phi-psi*| {m[0]:.4f} -> {m[-1]:.4f}"
          f"   ({arms['gp']['seconds']:.0f}s/{arms['random']['seconds']:.0f}s)")

    gaps = np.asarray(gaps)
    report["summary"] = {"mean_gap_pct": float(gaps.mean()),
                         "sd_gap_pct": float(gaps.std(ddof=1)) if len(gaps) > 1 else 0.0,
                         "wins": int((gaps > 0).sum()), "n": int(gaps.size)}
    P(f"\nGP-target - random-target over {gaps.size} seeds: "
      f"mean {gaps.mean():+.2f}%  "
      f"sd {gaps.std(ddof=1) if len(gaps) > 1 else 0:.2f}%  "
      f"wins {int((gaps > 0).sum())}/{gaps.size}")
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    P(f"wrote {args.out}   total {time.perf_counter()-t_all:.0f}s")


if __name__ == "__main__":
    main()
