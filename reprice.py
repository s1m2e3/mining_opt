"""
Step 7: make the mining ORDER capacity-aware by repricing block values.

The cut rule never violates a capacity -- it just binds, and binding costs NPV
by pushing value later or past the horizon. So the feedback signal is not a
constraint violation but a *shadow price*: how much NPV one extra tonne of a
resource would be worth.

    mu_r  =  d NPV / d C_r        estimated by finite difference on the cuts
    v_i(mu)  =  v_i  -  sum_r mu_r * w^r_i

Repricing enters the SCORE, which is the only channel by which capacity can
influence the order at all (Steps 1-5 are capacity-blind by construction). A
scarce resource makes the blocks that consume it look dearer, which promotes
high value-per-tonne material -- the ratio rule, arrived at economically.

This is a heuristic, not a certified Lagrangian dual: the inner order is
produced by a projection rather than re-optimised exactly, so there is no
duality bound. It monotonically tracks the best schedule seen, so it can only
improve on the un-repriced starting point.
"""

import numpy as np

from kernel_projection import (project_dykstra_sparse, project_pocs_sparse,
                               schedule_from_scores)
from capacity_cuts import evaluate_cuts_multi


def shadow_prices(seq, resources, value, discount, bump=0.05):
    """d NPV / d(total capacity of r), per unit, by central-ish difference."""
    base = evaluate_cuts_multi(seq, resources, value=value, discount=discount)
    mus = {}
    for k, r in enumerate(resources):
        up = [dict(q) for q in resources]
        up[k] = dict(r, capacity=r["capacity"] * (1.0 + bump))
        hi = evaluate_cuts_multi(seq, up, value=value, discount=discount)
        denom = bump * float(r["capacity"].sum())
        mus[r["name"]] = (hi["npv"] - base["npv"]) / denom if denom > 0 else 0.0
    return mus, base


def reprice_loop(priority, resources, gram, par, chi, order, tonnage,
                 discount=0.90, n_iters=8, damping=0.5, max_sweeps=5000,
                 noise=None, verbose=True, true_value=None,
                 projector="pocs", omega=0.2):
    """Iterate  reprice -> rescore -> reproject -> recut,  keeping the best.

    `priority` drives the ORDER and may carry look-ahead bonuses; `true_value`
    is what NPV is scored against and defaults to `priority`. Keeping them
    separate matters: a look-ahead bonus is a ranking device, not income, and
    counting it in the objective would flatter the result.

    Returns (best, history) where best carries the winning score, sequence,
    cut result and multipliers.
    """
    priority = np.asarray(priority, dtype=float)
    value = priority if true_value is None else np.asarray(true_value, dtype=float)
    n = priority.shape[0]
    if noise is None:
        noise = np.zeros(n)

    mu = {r["name"]: 0.0 for r in resources}
    best = None
    history = []

    for it in range(n_iters):
        v_eff = priority - sum(mu[r["name"]] * r["weight"] for r in resources)
        sd = v_eff.std()
        s_raw = (v_eff - v_eff.mean()) / (sd if sd > 0 else 1.0) + noise

        if projector == "pocs":
            s_proj, pinfo = project_pocs_sparse(s_raw, gram, par, chi, order,
                                                max_sweeps=max_sweeps, omega=omega)
        elif projector == "dykstra":
            s_proj, _, pinfo = project_dykstra_sparse(s_raw, gram, par, chi, order,
                                                      max_sweeps=max_sweeps)
        else:
            raise ValueError("projector must be 'pocs' or 'dykstra'")
        seq = schedule_from_scores(s_proj, par, chi, order)
        mu_new, res = shadow_prices(seq, resources, value, discount)

        row = {"iter": it, "npv": res["npv"], "mu": dict(mu),
               "unmined": res["n_unmined"], "sweeps": pinfo["sweeps"],
               "converged": bool(pinfo.get("converged",
                                           pinfo["sweeps"] < max_sweeps))}
        history.append(row)
        if verbose:
            mus = "  ".join(f"{k} {v:8.4f}" for k, v in mu.items())
            print(f"  iter {it}  NPV {res['npv']:>16,.0f}  unmined {res['n_unmined']:>5}  "
                  f"sweeps {pinfo['sweeps']:>6}  mu[{mus}]")

        if best is None or res["npv"] > best["npv"]:
            best = {"npv": res["npv"], "score": s_proj, "seq": seq, "result": res,
                    "mu": dict(mu), "iter": it}

        for k in mu:
            mu[k] = (1.0 - damping) * mu[k] + damping * max(0.0, mu_new[k])

    return best, history
