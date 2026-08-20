"""
Stage 1: Bayesian optimisation over the physical knobs. No transformer yet.

    knobs -> priority -> kernel projection -> priority-Kahn -> hard NPV
                                                                  |
             GP surrogate + LogEI  <---------------------------- NPV

BO sees nothing but the pair (theta, NPV). No gradient, no soft-rank surrogate,
no relaxation -- which is the entire point, since the objective is piecewise
constant in the scores and the projection is a contraction, so neither a
gradient nor a relaxation of this map is trustworthy.

Why this stage exists before the transformer: it is the control. Any later
claim that a learned representation helps has to beat a plain GP over eleven
hand-named knobs, and the sensitivity sweep already established what the bar
is. Greedy coordinate descent over the same knobs reached +2.83%, while the
sum of the one-factor-at-a-time gains was +12.91% -- an additive bound that is
provably not achievable, since setting every knob to its individually best
value returns only +2.35%. The knobs interact antagonistically. That gap
between +2.83% and the loose +12.91% is what BO is being asked to close.

FAIRNESS. BO and the random-search control are given *identical* warm-start
data -- every point the sensitivity sweep already evaluated -- and then the
same number of additional evaluations. Only the rule for choosing the next
point differs. Without that control the result is uninterpretable: with MIP
and beam search off the table there is no optimality bound in this pipeline,
so a plain 'BO found +X%' claim has nothing to stand against.

MULTI-INSTANCE. The objective is the mean *relative* gain over a list of
instances, not raw NPV. Deposits differ by orders of magnitude in absolute
value, so averaging raw NPV would let the largest instance silently own the
objective. Worst-case gain across instances is recorded alongside the mean,
which is what to optimise instead if the knobs turn out to be instance-specific.

    python bo_search.py --budget 100
    python bo_search.py --instances inputs/blocks.csv,inputs/blocks_b.csv
"""

import argparse
import json
import os
import time
import warnings

import numpy as np
import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.exceptions.errors import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.optim import optimize_acqf
from botorch.optim.fit import fit_gpytorch_mll_torch
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.constraints import GreaterThan
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.utils.errors import NotPSDError
from torch.quasirandom import SobolEngine

from mine_problem import KNOBS, default_knobs, evaluate, knob_active, load_static
from parallel_eval import ParallelObjective

DTYPE = torch.double


def P(*a):
    print(*a, flush=True)


class KnobSpace:
    """Maps the declared knob ranges to and from the unit cube BO searches.

    Integer knobs are searched continuously and rounded on the way out. That
    is the standard treatment and it is sound here because the objective is
    piecewise constant in them anyway -- the GP sees a step function it can
    model, rather than a categorical it cannot.
    """

    def __init__(self, names):
        self.names = list(names)
        self.lo = np.array([KNOBS[n][0] for n in self.names], dtype=float)
        self.hi = np.array([KNOBS[n][1] for n in self.names], dtype=float)
        self.kind = [KNOBS[n][3] for n in self.names]

    @property
    def dim(self):
        return len(self.names)

    def to_unit(self, knobs):
        v = np.array([float(knobs[n]) for n in self.names], dtype=float)
        return np.clip((v - self.lo) / (self.hi - self.lo), 0.0, 1.0)

    def from_unit(self, u):
        u = np.clip(np.asarray(u, dtype=float).ravel(), 0.0, 1.0)
        v = self.lo + u * (self.hi - self.lo)
        out = dict(default_knobs())
        for i, n in enumerate(self.names):
            out[n] = int(round(v[i])) if self.kind[i] == "int" else float(v[i])
        return out


class Objective:
    """theta -> mean relative NPV gain across instances."""

    def __init__(self, instances, transform, decoder):
        self.instances = instances          # list of (name, static)
        self.transform = transform
        self.decoder = decoder
        self.baseline = {}
        for name, S in instances:
            r = evaluate(S, {}, transform=transform, decoder=decoder)
            if r["violations"] != 0:
                raise SystemExit(f"{name}: baseline has {r['violations']} violations")
            self.baseline[name] = r["npv"]
        self.n_calls = 0

    def __call__(self, knobs):
        self.n_calls += 1
        gains, npvs = [], {}
        for name, S in self.instances:
            r = evaluate(S, knobs, transform=self.transform,
                         decoder=self.decoder, check=False)
            npvs[name] = r["npv"]
            gains.append(r["npv"] / self.baseline[name] - 1.0)
        gains = np.asarray(gains)
        return float(gains.mean()), {"npv": npvs, "gain_mean": float(gains.mean()),
                                     "gain_worst": float(gains.min())}


def problem_id(paths, crop):
    """Identity of the objective, so warm-start data from a different problem
    cannot be silently mixed in. A cropped slab and the full model share a
    filename but are different functions with different baselines."""
    return {"instances": list(paths), "crop": crop}


def load_warm_start(path, space, obj, transform, decoder, pid):
    """Reuse the sensitivity sweep's evaluations as initial GP data.

    The sweep moved one knob at a time from the defaults, so each row is a
    complete point in the space -- 77 free observations that already cover
    every axis end to end, which is better coverage than a Sobol design of the
    same size would give on the axes that matter.

    Only valid when the sweep ran the same transform/decoder on the same single
    instance; otherwise the recorded NPVs are for a different objective.
    """
    if not os.path.exists(path):
        return [], []
    rep = json.load(open(path))
    if rep.get("transform") != transform or rep.get("decoder") != decoder:
        P(f"warm start skipped: {path} is {rep.get('transform')}/{rep.get('decoder')}")
        return [], []
    if len(obj.instances) != 1:
        P("warm start skipped: sweep is single-instance, objective is not")
        return [], []
    # sweeps written before problem_id existed were all full inputs/blocks.csv
    rep_pid = rep.get("problem", {"instances": ["inputs/blocks.csv"], "crop": None})
    if rep_pid != pid:
        P(f"warm start skipped: {path} is for {rep_pid}, objective is {pid}")
        return [], []

    base = rep["baseline_npv"]
    X, Y = [], []
    for name, row in rep["knobs"].items():
        for v, npv in zip(row["values"], row["npv"]):
            k = dict(default_knobs())
            k[name] = v
            X.append(space.to_unit(k))
            Y.append(npv / base - 1.0)
    # the all-defaults point appears once per knob sweep; keep a single copy
    X = np.array(X)
    Y = np.array(Y)
    _, keep = np.unique(np.round(X, 12), axis=0, return_index=True)
    return list(X[np.sort(keep)]), list(Y[np.sort(keep)])


def _new_gp(X, Y):
    Xt = torch.tensor(np.asarray(X), dtype=DTYPE)
    Yt = torch.tensor(np.asarray(Y), dtype=DTYPE).unsqueeze(-1)
    covar = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=Xt.shape[-1]))
    likelihood = GaussianLikelihood(noise_constraint=GreaterThan(1e-6))
    gp = SingleTaskGP(Xt, Yt, likelihood=likelihood, covar_module=covar,
                      outcome_transform=Standardize(m=1))
    return gp, Xt, Yt


def fit_gp(X, Y, reuse_from=None):
    """Matern-5/2 ARD GP, fitted with an L-BFGS attempt and an Adam fallback.

    `reuse_from` skips hyperparameter optimisation and inherits the kernel,
    mean and noise from an earlier model, conditioning on the new data without
    re-solving the MLL. One added observation moves the fitted lengthscales
    very little, while the optimiser costs more than the objective evaluation
    it is choosing -- so refitting every iteration is the wrong default.

    Only the three hyperparameter modules are copied. The outcome transform is
    left as freshly constructed, because its stored mean and scale must reflect
    the CURRENT Y; inheriting stale standardisation would quietly shift every
    posterior the acquisition sees.

    Two things make the default fit fail on this problem, both worth naming
    because they are properties of the data rather than bad luck:

      * the warm-start design is 69 points lying on eleven axis-aligned lines
        through one centre, so whole regions of the cube carry no information
        and the ARD marginal likelihood is close to flat in those directions;
      * the objective is deterministic, so the MLE wants zero observation
        noise and drives the likelihood's noise term toward its floor, which is
        exactly where the Cholesky of K + sigma^2 I stops being stable.

    A noise floor of 1e-6 on a standardised objective fixes the second, and
    falling back to Adam -- which cannot fail the way L-BFGS reports ABNORMAL
    -- covers the first.
    """
    # Standardize: the objective is a relative gain of order 1e-2, and an
    # unstandardised GP on that scale fits the noise floor instead of the signal
    gp, Xt, Yt = _new_gp(X, Y)
    if reuse_from is not None:
        for mod in ("covar_module", "likelihood", "mean_module"):
            getattr(gp, mod).load_state_dict(getattr(reuse_from, mod).state_dict())
        gp.eval()
        return gp, Xt, Yt

    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_gpytorch_mll(mll)
    except (ModelFittingError, NotPSDError, RuntimeError):
        gp, Xt, Yt = _new_gp(X, Y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_gpytorch_mll_torch(mll, step_limit=300)
    return gp, Xt, Yt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default="inputs/blocks.csv")
    ap.add_argument("--crop", type=int, default=None,
                    help="keep a slab of this many x-columns (smaller, faster instance)")
    ap.add_argument("--transform", default="pocs", choices=("raw", "smooth", "pocs"))
    ap.add_argument("--decoder", default="kahn", choices=("sort", "kahn"))
    ap.add_argument("--budget", type=int, default=100,
                    help="adaptive evaluations for BO, and the same number for the control")
    ap.add_argument("--n-init", type=int, default=16,
                    help="Sobol points added when there is no warm-start data")
    ap.add_argument("--restarts", type=int, default=10)
    ap.add_argument("--raw-samples", type=int, default=512)
    ap.add_argument("--q", type=int, default=4,
                    help="candidates proposed and evaluated per GP fit")
    ap.add_argument("--workers", type=int, default=None,
                    help="evaluation processes; default = cpu count, 1 = serial")
    ap.add_argument("--refit-every", type=int, default=3,
                    help="re-optimise GP hyperparameters every k rounds")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warm-start", default=None,
                    help="sensitivity sweep JSON; default matches transform/decoder")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tag = f"{args.transform}_{args.decoder}" + (f"_crop{args.crop}" if args.crop else "")
    warm_path = args.warm_start or f"outputs/sensitivity_{tag}.json"
    out_path = args.out or f"outputs/bo_stage1_{tag}.json"
    torch.manual_seed(args.seed)

    t_all = time.perf_counter()
    paths = [p.strip() for p in args.instances.split(",") if p.strip()]
    pid = problem_id(paths, args.crop)
    specs = [(os.path.basename(p), p, args.crop) for p in paths]

    names = [k for k in KNOBS if knob_active(k, args.transform)]
    space = KnobSpace(names)
    obj = ParallelObjective(specs, args.transform, args.decoder,
                            workers=args.workers)
    instances = obj.instances
    for (nm, S), p in zip(instances, paths):
        P(f"instance {p}: n={S['n']:,} edges={S['par'].size:,}"
          + (f"  crop x[{S['crop_window'][0]}:{S['crop_window'][1]}]"
             if S["crop_window"] else ""))
    P(f"\nq={args.q}  workers={obj.workers}  refit_every={args.refit_every}")
    P(f"transform={args.transform} decoder={args.decoder}  D={space.dim}")
    P(f"knobs: {', '.join(space.names)}")
    for nm, b in obj.baseline.items():
        P(f"baseline {nm}: {b:,.0f}")

    Xw, Yw = load_warm_start(warm_path, space, obj, args.transform, args.decoder, pid)
    if Xw:
        P(f"warm start: {len(Xw)} points from {warm_path} "
          f"(best {max(Yw)*100:+.2f}%)")
    else:
        sob = SobolEngine(space.dim, scramble=True, seed=args.seed)
        u = sob.draw(args.n_init).numpy()
        # the initial design has no sequential dependence at all, so it is one
        # batch across every worker rather than n_init round trips
        res = obj.batch([space.from_unit(row) for row in u])
        Xw = [row for row in u]
        Yw = [y for y, _ in res]
        P(f"  init {args.n_init} Sobol points, best {max(Yw)*100:+.3f}%")

    bounds = torch.stack([torch.zeros(space.dim, dtype=DTYPE),
                          torch.ones(space.dim, dtype=DTYPE)])
    report = {"transform": args.transform, "decoder": args.decoder,
              "problem": pid, "instances": paths, "knobs": space.names,
              "baseline": obj.baseline, "budget": args.budget,
              "n_warm": len(Xw), "warm_best": float(max(Yw)),
              "bo": [], "random": []}

    def flush():
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)

    # ---- BO ---------------------------------------------------------------
    q = max(1, args.q)
    n_rounds = (args.budget + q - 1) // q
    P(f"\n--- BO ({args.budget} adaptive evaluations, {n_rounds} rounds of q={q}) ---")
    X, Y = list(Xw), list(Yw)
    best = max(Y)
    prev_gp = None
    done = 0
    for rnd in range(n_rounds):
        t0 = time.perf_counter()
        refit = (rnd % max(1, args.refit_every) == 0)
        gp, Xt, Yt = fit_gp(X, Y, reuse_from=None if refit else prev_gp)
        prev_gp = gp
        qq = min(q, args.budget - done)
        if qq == 1:
            acq = LogExpectedImprovement(model=gp, best_f=Yt.max())
        else:
            # qLogEI scores a SET jointly, so the q points are diverse by
            # construction. Proposing the top-q of the analytic LogEI instead
            # would return q near-copies of the same maximiser.
            acq = qLogExpectedImprovement(
                model=gp, best_f=Yt.max(),
                sampler=SobolQMCNormalSampler(sample_shape=torch.Size([128])))
        cand, _ = optimize_acqf(acq, bounds=bounds, q=qq,
                                num_restarts=args.restarts,
                                raw_samples=args.raw_samples)
        t_fit = time.perf_counter() - t0

        U = cand.detach().numpy().reshape(qq, -1)
        knobs_list = [space.from_unit(u) for u in U]
        t1 = time.perf_counter()
        results = obj.batch(knobs_list)
        t_eval = time.perf_counter() - t1

        gains = []
        for u, knobs, (y, detail) in zip(U, knobs_list, results):
            X.append(u); Y.append(y)
            best = max(best, y)
            gains.append(y)
            report["bo"].append({"round": rnd, "u": u.tolist(), "knobs": knobs,
                                 "gain": y, "best": best, "detail": detail,
                                 "refit": refit, "fit_s": t_fit,
                                 "eval_s": t_eval})
        done += qq
        flush()
        P(f"  bo round {rnd+1:>3}/{n_rounds} ({done:>3}/{args.budget})  "
          f"batch best {max(gains)*100:+7.3f}%  best {best*100:+7.3f}%   "
          f"(fit {t_fit:4.1f}s{'' if refit else ' reused'} "
          f"eval {t_eval:4.1f}s for {qq})")

    bo_best = best
    bo_best_i = int(np.argmax(Y))
    bo_best_knobs = space.from_unit(X[bo_best_i])

    # ---- random-search control, same warm start, same budget --------------
    # Random search has no sequential dependence, so the whole control is a
    # single batch. It gets the same warm start and the same budget as BO.
    P(f"\n--- random control ({args.budget} evaluations, identical warm start) ---")
    sob = SobolEngine(space.dim, scramble=True, seed=args.seed + 1)
    u_rand = sob.draw(args.budget).numpy()
    t1 = time.perf_counter()
    res_rand = obj.batch([space.from_unit(row) for row in u_rand])
    best_r = max(Yw)
    for it, (row, (y, detail)) in enumerate(zip(u_rand, res_rand)):
        best_r = max(best_r, y)
        report["random"].append({"iter": it, "u": row.tolist(),
                                 "knobs": space.from_unit(row),
                                 "gain": y, "best": best_r, "detail": detail})
    flush()
    P(f"  {args.budget} evaluations in {time.perf_counter()-t1:.1f}s   "
      f"best {best_r*100:+7.3f}%")

    # ---- verdict ----------------------------------------------------------
    final = evaluate(instances[0][1], bo_best_knobs, transform=args.transform,
                     decoder=args.decoder)
    report["result"] = {
        "warm_best_pct": float(max(Yw) * 100),
        "bo_best_pct": float(bo_best * 100),
        "random_best_pct": float(best_r * 100),
        "bo_minus_random_pct": float((bo_best - best_r) * 100),
        "best_knobs": bo_best_knobs,
        "best_npv": final["npv"],
        "best_violations": final["violations"],
        "seconds": time.perf_counter() - t_all,
    }
    flush()

    P(f"\n{'':<22}{'best gain':>12}")
    P(f"{'warm start only':<22}{max(Yw)*100:>11.3f}%")
    P(f"{'random control':<22}{best_r*100:>11.3f}%")
    P(f"{'BO':<22}{bo_best*100:>11.3f}%")
    P(f"{'BO - random':<22}{(bo_best-best_r)*100:>11.3f}%")
    P(f"\nbest NPV {final['npv']:,.0f}   violations {final['violations']}")
    P("best knobs: " + "  ".join(
        f"{k}={bo_best_knobs[k]:g}" if isinstance(bo_best_knobs[k], float)
        else f"{k}={bo_best_knobs[k]}" for k in space.names))
    secs = time.perf_counter() - t_all
    P(f"\nwrote {out_path}   total {secs:.0f}s "
      f"({obj.n_calls} objective calls, {obj.n_calls/max(secs,1e-9)*60:.0f}/min)")
    obj.close()


if __name__ == "__main__":
    main()
