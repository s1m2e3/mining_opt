from typing import Any, Dict, List, Optional, Tuple, Hashable
import pandas as pd
import torch
import copy
import numpy as np

# # ------------------------------
# # Core marginal value (row-level)
# # ------------------------------
# def marginal_value(
#     row: pd.Series,
#     mined_blocks: set,
#     discount_factor: float,
#     steps_to_ahead: int,
#     t: int,
#     prob_weight: Optional[float] = None,
#     apply_fraction_to_lookahead: bool = True,
#     alpha_prob: float = 0.5
# ) -> float:
#     """
#     Discounted marginal NPV contribution of a single candidate row at position t.
#     Required columns: 'income','cost','income_look_ahead','cost_look_ahead','blocks'
#     """
#     blocks = row["blocks"]
#     if not isinstance(blocks, set):
#         blocks = set(blocks)
#     if len(blocks) == 0:
#         uncovered_fraction = 1.0
#     else:
#         intersect_ct = len(blocks & mined_blocks)
#         uncovered_fraction = 1.0 - (intersect_ct / max(1, len(blocks)))

#     # Current net value
#     net_now = (row["income"] - row["cost"]) * uncovered_fraction

#     # Look-ahead net value (further discounted by delta^steps_to_ahead)
#     look = (row["income_look_ahead"] - row["cost_look_ahead"])
#     if apply_fraction_to_lookahead:
#         look *= uncovered_fraction
#     net_future = (discount_factor ** steps_to_ahead) * look

#     # Position discount
#     base = (net_now + net_future) * (discount_factor ** t)

#     # Optional probability weighting (p^alpha)
#     if prob_weight is not None:
#         base *= (float(prob_weight) ** alpha_prob)

#     return float(base)


# # -------------------------------------------------
# # Helpers for precedence using (x, y, z) as location key
# # -------------------------------------------------
# def _loc_key(row: pd.Series, x_col: str, y_col: str, z_col: str) -> Tuple[Hashable, Hashable, Hashable]:
#     """
#     Build a hashable key for the location from coordinate columns.
#     Consider pre-rounding if floats (e.g., df[x_col] = df[x_col].round(6)).
#     """
#     return (row[x_col], row[y_col], row[z_col])


# # -------------------------------------------------
# # Greedy maximization for a single batch (one DataFrame)
# # with precedence: shallower level must precede deeper at same (x,y,z),
# # and guaranteed full ordering output (length == len(df))
# # -------------------------------------------------
# def _greedy_max_npv_for_batch(
#     df: pd.DataFrame,
#     probs_row: Optional[torch.Tensor],
#     discount_factor: float,
#     steps_to_ahead: int,
#     apply_fraction_to_lookahead: bool,
#     alpha_prob: float,
#     x_col: str = "x",
#     y_col: str = "y",
#     z_col: str = "z",
#     level_col: str = "level",
#     enforce_level_by_xyz: bool = True
# ) -> Dict[str, Any]:
#     """
#     Internal: chooses an execution order maximizing discounted NPV greedily,
#     subject to precedence: for any (x,y,z), you may mine level L only after
#     level L-1 at that same (x,y,z) has been mined (L=0 is always eligible).

#     If strict precedence leaves no eligible rows (e.g., bad input where lowest
#     available level at a location > 0), we soft-relax by allowing the minimum
#     remaining level at each (x,y,z) to be eligible so progress is guaranteed.

#     Columns required in df:
#       - 'income','cost','income_look_ahead','cost_look_ahead','blocks'
#       - If enforce_level_by_xyz=True: also x_col, y_col, z_col, level_col
#     """
#     df = df.reset_index(drop=True).copy()

#     # Normalize 'blocks' to sets
#     df["blocks"] = [
#         set(b) if isinstance(b, (list, set, tuple)) else set([b]) for b in df["blocks"]
#     ]

#     if enforce_level_by_xyz:
#         for c in (x_col, y_col, z_col, level_col):
#             if c not in df.columns:
#                 raise ValueError(
#                     f"Precedence enforcement requires coordinate columns '{x_col}', '{y_col}', '{z_col}' and level column '{level_col}'. Missing: {c}"
#                 )

#     N = len(df)

#     # Tracks which micro-blocks have been mined (for overlap discount)
#     mined_blocks: set = set()

#     # Tracks deepest mined level per (x,y,z) for precedence eligibility
#     mined_levels_by_xyz: Dict[Tuple[Hashable, Hashable, Hashable], int] = {}

#     remaining = list(df.index)
#     order: List[int] = []
#     marginals: List[float] = []
#     total_value: float = 0.0
#     t = 0

#     prob_np = None
#     if probs_row is not None and probs_row.ndim == 1 and probs_row.shape[0] == N:
#         prob_np = probs_row.detach().cpu().numpy()

#     def _eligible_strict(idx: int) -> bool:
#         if not enforce_level_by_xyz:
#             return True
#         row = df.loc[idx]
#         xyz = _loc_key(row, x_col, y_col, z_col)
#         lvl = int(row[level_col])
#         deepest = mined_levels_by_xyz.get(xyz, -1)
#         return deepest >= (lvl - 1)

#     while remaining:
#         # 1) Strictly eligible rows
#         strict_eligible = [j for j in remaining if _eligible_strict(j)]

#         candidate_pool = strict_eligible

#         # 2) If none strictly eligible, soft-relax precedence:
#         #    allow the minimum remaining level at each (x,y,z) to become eligible
#         if not candidate_pool and enforce_level_by_xyz:
#             # compute min level among remaining per (x,y,z)
#             min_level_by_xyz: Dict[Tuple[Hashable, Hashable, Hashable], int] = {}
#             idxs_by_xyz_min: Dict[Tuple[Hashable, Hashable, Hashable], List[int]] = {}

#             for j in remaining:
#                 row = df.loc[j]
#                 xyz = _loc_key(row, x_col, y_col, z_col)
#                 lvl = int(row[level_col])
#                 if (xyz not in min_level_by_xyz) or (lvl < min_level_by_xyz[xyz]):
#                     min_level_by_xyz[xyz] = lvl

#             # collect all rows that match the min level per location
#             for j in remaining:
#                 row = df.loc[j]
#                 xyz = _loc_key(row, x_col, y_col, z_col)
#                 lvl = int(row[level_col])
#                 if lvl == min_level_by_xyz[xyz]:
#                     idxs_by_xyz_min.setdefault(xyz, []).append(j)

#             # pool is the union of these minimal-level rows
#             candidate_pool = [j for lst in idxs_by_xyz_min.values() for j in lst]

#         # 3) If still somehow empty (shouldn't happen), fall back to all remaining
#         if not candidate_pool:
#             candidate_pool = list(remaining)

#         # Evaluate marginal values and pick argmax (even if negative)
#         best_idx, best_gain = None, None
#         for j in candidate_pool:
#             p = prob_np[j] if prob_np is not None else None
#             gain = marginal_value(
#                 row=df.loc[j],
#                 mined_blocks=mined_blocks,
#                 discount_factor=discount_factor,
#                 steps_to_ahead=steps_to_ahead,
#                 t=t,
#                 prob_weight=p,
#                 apply_fraction_to_lookahead=apply_fraction_to_lookahead,
#                 alpha_prob=alpha_prob
#             )
#             if (best_gain is None) or (gain > best_gain):
#                 best_gain, best_idx = gain, j

#         # Commit selection
#         order.append(best_idx)
#         marginals.append(best_gain if best_gain is not None else 0.0)
#         total_value += (best_gain if best_gain is not None else 0.0)

#         # Update mined micro-block coverage (for overlap discounting)
#         mined_blocks |= df.loc[best_idx, "blocks"]

#         # Update precedence frontier at this (x,y,z)
#         if enforce_level_by_xyz:
#             row = df.loc[best_idx]
#             xyz = _loc_key(row, x_col, y_col, z_col)
#             lvl = int(row[level_col])
#             mined_levels_by_xyz[xyz] = max(mined_levels_by_xyz.get(xyz, -1), lvl)

#         remaining.remove(best_idx)
#         t += 1

#     assert len(order) == N, "Internal error: order should include all rows."

#     return {
#         "order": order,                        # list of row indices (full permutation)
#         "value": total_value,                  # discounted NPV of chosen sequence
#         "marginals": marginals,                # stepwise contributions (may be ≤ 0)
#         "covered_blocks": mined_blocks,        # final coverage set (diagnostics)
#         "deepest_levels": mined_levels_by_xyz  # per-(x,y,z) deepest mined level
#     }


# # -------------------------------------------------
# # Public API matching your income_* signature
# # -------------------------------------------------
# def income(
#     probs: torch.Tensor,
#     dfs: List[pd.DataFrame],
#     discount_factor: float = 0.99,
#     steps_to_ahead: int = 8,
#     apply_fraction_to_lookahead: bool = True,
#     alpha_prob: float = 0.5,
#     x_col: str = "x",
#     y_col: str = "y",
#     z_col: str = "z",
#     level_col: str = "level",
#     enforce_level_by_xyz: bool = True
# ) -> Dict[str, Any]:
#     """
#     Wrapper that:
#       - Applies a greedy search with precedence (x,y,z, level).
#       - Always returns a full permutation for each batch (length equals df length).
#       - Uses your original marginal_value scoring.

#     Returns:
#       {
#         "orders":   List[torch.LongTensor]  # per-batch order (length N_i each)
#         "values":   torch.FloatTensor       # shape [B], discounted value per batch
#         "details":  List[Dict[str,Any]]     # raw dicts with marginals, coverage, etc.
#       }
#     """
#     device = probs.device if isinstance(probs, torch.Tensor) else torch.device("cpu")

#     orders: List[torch.LongTensor] = []
#     values_list: List[float] = []
#     details: List[Dict[str, Any]] = []

#     B = len(dfs)
#     has_prob_rows = (isinstance(probs, torch.Tensor) and probs.ndim == 2 and probs.shape[0] == B)

#     for b in range(B):
#         df = dfs[b]
#         pr = probs[b] if has_prob_rows and probs.shape[1] == len(df) else None  # disable weighting if not aligned

#         res = _greedy_max_npv_for_batch(
#             df=df,
#             probs_row=pr,
#             discount_factor=discount_factor,
#             steps_to_ahead=steps_to_ahead,
#             apply_fraction_to_lookahead=apply_fraction_to_lookahead,
#             alpha_prob=alpha_prob,
#             x_col=x_col,
#             y_col=y_col,
#             z_col=z_col,
#             level_col=level_col,
#             enforce_level_by_xyz=enforce_level_by_xyz
#         )

#         # Full-length order tensor for this batch
#         orders.append(torch.tensor(res["order"], dtype=torch.long, device=device))
#         values_list.append(res["value"])
#         details.append(res)
    
#     # Return values per-batch (not mean), since order now covers all rows
#     values = torch.tensor(values_list, dtype=torch.float32, device=device)
#     orders = torch.stack(orders)
#     return {"orders": orders, "values": values, "details": details}



# ------------------------------
# Core marginal value (row-level)
# (unchanged)
# ------------------------------
def marginal_value(
    row: pd.Series,
    mined_blocks: set,
    discount_factor: float,
    steps_to_ahead: int,
    t: int,
    prob_weight: Optional[float] = None,
    apply_fraction_to_lookahead: bool = True,
    alpha_prob: float = 0.5
) -> float:
    """
    Discounted marginal NPV contribution of a single candidate row at position t.
    Required columns: 'income','cost','income_look_ahead','cost_look_ahead','blocks'
    """
    blocks = row["blocks"]
    if not isinstance(blocks, set):
        blocks = set(blocks)
    if len(blocks) == 0:
        uncovered_fraction = 1.0
    else:
        intersect_ct = len(blocks & mined_blocks)
        uncovered_fraction = 1.0 - (intersect_ct / max(1, len(blocks)))

    # Current net value ('cost' is stored as an already-negative quantity,
    # so combine with addition, not subtraction)
    net_now = (row["income"] + row["cost"]) * uncovered_fraction

    # Look-ahead net value (further discounted by delta^steps_to_ahead)
    look = (row["income_look_ahead"] + row["cost_look_ahead"])
    if apply_fraction_to_lookahead:
        look *= uncovered_fraction
    net_future = (discount_factor ** steps_to_ahead) * look

    # Position discount
    base = (net_now + net_future) * (discount_factor ** t)

    # Optional probability weighting (p^alpha)
    if prob_weight is not None:
        base *= (float(prob_weight) ** alpha_prob)

    return float(base)


# -------------------------------------------------
# Helpers for precedence using (x, y, z) as location key
# (unchanged)
# -------------------------------------------------
def _loc_key(row: pd.Series, x_col: str, y_col: str, z_col: str) -> Tuple[Hashable, Hashable, Hashable]:
    """
    Build a hashable key for the location from coordinate columns.
    Consider pre-rounding if floats (e.g., df[x_col] = df[x_col].round(6)).
    """
    return (row[x_col], row[y_col], row[z_col])


# -------------------------------------------------
# Beam/Rollout maximization for a single batch
# -------------------------------------------------
def _beam_max_npv_for_batch(
    df: pd.DataFrame,
    probs_row: Optional[torch.Tensor],
    discount_factor: float,
    steps_to_ahead: int,
    apply_fraction_to_lookahead: bool,
    alpha_prob: float,
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
    level_col: str = "level",
    enforce_level_by_xyz: bool = True,
    # --- Beam parameters ---
    beam_width: int = 5,           # k (k=1 ≈ greedy)
    rollout_horizon: int = 10,      # H (H=1 ≈ greedy)
    candidate_pool_size: int = 4  # evaluate top-M eligibles by myopic score before rollouts
) -> Dict[str, Any]:
    """
    Chooses an execution order maximizing discounted NPV via a beam/rollout policy.
    - At each step t:
        1) Build eligible candidate pool (with soft-relax fallback identical to original).
        2) Rank by immediate marginal_value and keep top-M (candidate_pool_size).
        3) For each of those candidates, simulate H-1 steps ahead using the same
           myopic rule (respecting precedence) and compute the *cumulative* discounted gain
           from the simulated mini-trajectory starting with that candidate.
        4) Pick the candidate with best rollout value. (If beam_width>1, maintain
           multiple partial trajectories; see below.)
    - Beam variant:
        We keep up to 'beam_width' partial states and expand each by the top-M candidates,
        then retain the best 'beam_width' resulting states by their total value so far.

    Notes:
      * Setting beam_width=1 reduces to single-trajectory rollout (policy rollout).
      * Setting rollout_horizon=1 reduces to pure greedy (original behavior).
    """
    df = df.reset_index(drop=True).copy()

    # Normalize 'blocks' to sets
    df["blocks"] = [
        set(b) if isinstance(b, (list, set, tuple)) else set([b]) for b in df["blocks"]
    ]

    if enforce_level_by_xyz:
        for c in (x_col, y_col, z_col, level_col):
            if c not in df.columns:
                raise ValueError(
                    f"Precedence enforcement requires coordinate columns '{x_col}', '{y_col}', '{z_col}' and level column '{level_col}'. Missing: {c}"
                )

    N = len(df)

    prob_np = None
    if probs_row is not None and probs_row.ndim == 1 and probs_row.shape[0] == N:
        prob_np = probs_row.detach().cpu().numpy()

    # ---------- Helpers ----------
    def eligible_set(remaining: List[int], mined_levels_by_xyz: Dict[Tuple[Hashable, Hashable, Hashable], int]) -> List[int]:
        """Strictly eligible indices given current mined levels; if none, soft-relax."""
        def _eligible_strict(idx: int) -> bool:
            if not enforce_level_by_xyz:
                return True
            row = df.loc[idx]
            xyz = _loc_key(row, x_col, y_col, z_col)
            lvl = int(row[level_col])
            deepest = mined_levels_by_xyz.get(xyz, -1)
            return deepest >= (lvl - 1)

        strict_eligible = [j for j in remaining if _eligible_strict(j)]
        if strict_eligible or (not enforce_level_by_xyz):
            return strict_eligible

        # soft-relax: allow min level per (x,y,z)
        min_level_by_xyz: Dict[Tuple[Hashable, Hashable, Hashable], int] = {}
        idxs_by_xyz_min: Dict[Tuple[Hashable, Hashable, Hashable], List[int]] = {}
        for j in remaining:
            row = df.loc[j]
            xyz = _loc_key(row, x_col, y_col, z_col)
            lvl = int(row[level_col])
            if (xyz not in min_level_by_xyz) or (lvl < min_level_by_xyz[xyz]):
                min_level_by_xyz[xyz] = lvl
        for j in remaining:
            row = df.loc[j]
            xyz = _loc_key(row, x_col, y_col, z_col)
            lvl = int(row[level_col])
            if lvl == min_level_by_xyz[xyz]:
                idxs_by_xyz_min.setdefault(xyz, []).append(j)
        pool = [j for lst in idxs_by_xyz_min.values() for j in lst]
        return pool if pool else list(remaining)

    def immediate_score(idx: int, mined_blocks: set, t: int) -> float:
        p = prob_np[idx] if prob_np is not None else None
        return marginal_value(
            row=df.loc[idx],
            mined_blocks=mined_blocks,
            discount_factor=discount_factor,
            steps_to_ahead=steps_to_ahead,
            t=t,
            prob_weight=p,
            apply_fraction_to_lookahead=apply_fraction_to_lookahead,
            alpha_prob=alpha_prob
        )

    def commit_state_update(idx: int,
                            state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply choosing 'idx' to a copy of state; return new state and the marginal gain."""
        new_state = {
            "mined_blocks": set(state["mined_blocks"]),
            "mined_levels_by_xyz": dict(state["mined_levels_by_xyz"]),
            "remaining": list(state["remaining"]),
            "order": list(state["order"]),
            "marginals": list(state["marginals"]),
            "total_value": float(state["total_value"]),
            "t": int(state["t"])
        }
        gain = immediate_score(idx, new_state["mined_blocks"], new_state["t"])
        new_state["order"].append(idx)
        new_state["marginals"].append(gain)
        new_state["total_value"] += gain
        new_state["mined_blocks"] |= df.loc[idx, "blocks"]

        if enforce_level_by_xyz:
            row_i = df.loc[idx]
            xyz = _loc_key(row_i, x_col, y_col, z_col)
            lvl = int(row_i[level_col])
            new_state["mined_levels_by_xyz"][xyz] = max(new_state["mined_levels_by_xyz"].get(xyz, -1), lvl)

        new_state["remaining"].remove(idx)
        new_state["t"] += 1
        return new_state, gain

    def greedy_simulate_H_minus_1(state: Dict[str, Any], H: int) -> float:
        """Rollout: from 'state' simulate H-1 steps by greedy myopic selection; return added value."""
        if H <= 1:
            return 0.0
        added = 0.0
        sim_state = {
            "mined_blocks": set(state["mined_blocks"]),
            "mined_levels_by_xyz": dict(state["mined_levels_by_xyz"]),
            "remaining": list(state["remaining"]),
            "order": list(state["order"]),
            "marginals": list(state["marginals"]),
            "total_value": float(state["total_value"]),
            "t": int(state["t"])
        }
        steps = min(H - 1, len(sim_state["remaining"]))
        for _ in range(steps):
            pool = eligible_set(sim_state["remaining"], sim_state["mined_levels_by_xyz"])
            if not pool:
                pool = list(sim_state["remaining"])
            # purely myopic choice during rollout
            best_idx, best_gain = None, None
            for j in pool:
                g = immediate_score(j, sim_state["mined_blocks"], sim_state["t"])
                if (best_gain is None) or (g > best_gain):
                    best_gain, best_idx = g, j
            sim_state, g = commit_state_update(best_idx, sim_state)
            added += g
        return added

    # ---------- Initialize beam ----------
    init_state = {
        "mined_blocks": set(),
        "mined_levels_by_xyz": {},
        "remaining": list(df.index),
        "order": [],
        "marginals": [],
        "total_value": 0.0,
        "t": 0
    }
    beam: List[Dict[str, Any]] = [init_state]

    # ---------- Beam expansion loop ----------
    while True:
        # All done?
        if all(len(s["remaining"]) == 0 for s in beam):
            break

        candidates_next_states: List[Dict[str, Any]] = []

        for state in beam:
            if not state["remaining"]:
                # already complete; carry forward as-is
                candidates_next_states.append(state)
                continue

            pool = eligible_set(state["remaining"], state["mined_levels_by_xyz"])
            if not pool:
                pool = list(state["remaining"])

            # rank by immediate myopic score; keep top-M
            scored = [(j, immediate_score(j, state["mined_blocks"], state["t"])) for j in pool]
            scored.sort(key=lambda x: x[1], reverse=True)
            top_M = [j for j, _ in scored[:max(1, candidate_pool_size)]]

            # rollout for each top candidate
            best_local = []
            for j in top_M:
                # First commit j
                st_j, gain_j = commit_state_update(j, state)
                # Then greedy simulate H-1 steps
                added = greedy_simulate_H_minus_1(st_j, rollout_horizon)
                # Store a lightweight key for tie-breaking: prefer higher immediate gain
                total_preview = st_j["total_value"] + added
                best_local.append( (total_preview, gain_j, st_j) )

            # Keep the single best continuation from this parent (classic rollout),
            # but we still allow multiple parents via beam_width.
            if best_local:
                best_local.sort(key=lambda x: (x[0], x[1]), reverse=True)
                candidates_next_states.append(best_local[0][2])
            else:
                # If no local expansions (shouldn't happen), carry forward state
                candidates_next_states.append(state)
            print("len(candidates_next_states)", len(candidates_next_states))
            print([len(s["remaining"]) for s in beam])
        # Prune to beam_width by total_value (so far)
        candidates_next_states.sort(key=lambda s: s["total_value"], reverse=True)
        beam = candidates_next_states[:max(1, beam_width)]

        # Optional early exit: if the best state is complete and dominates
        if len(beam[0]["remaining"]) == 0:
            # You could also check if all states complete; we break at top loop anyway
            if all(len(s["remaining"]) == 0 for s in beam):
                break

    # Finalize with the best state
    best_state = max(beam, key=lambda s: s["total_value"])
    assert len(best_state["order"]) == N, "Internal error: order should include all rows."

    return {
        "order": best_state["order"],
        "value": best_state["total_value"],
        "marginals": best_state["marginals"],
        "covered_blocks": set(best_state["mined_blocks"]),
        "deepest_levels": dict(best_state["mined_levels_by_xyz"])
    }


# -------------------------------------------------
# Public API matching your income_* signature
# (switched to beam search; add new knobs)
# -------------------------------------------------
def income(
    probs: torch.Tensor,
    dfs: List[pd.DataFrame],
    discount_factor: float = 0.99,
    steps_to_ahead: int = 8,
    apply_fraction_to_lookahead: bool = True,
    alpha_prob: float = 0.5,
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
    level_col: str = "level",
    enforce_level_by_xyz: bool = True,
    # --- Beam knobs (safe defaults approximate greedy but with some lookahead) ---
    beam_width: int = 4,
    rollout_horizon: int = 10,
    candidate_pool_size: int = 4
) -> Dict[str, Any]:
    """
    Wrapper that:
      - Applies a beam/rollout search with precedence (x,y,z, level).
      - Returns a full permutation per batch.
      - Uses the same marginal_value scoring within rollouts.

    Returns:
      {
        "orders":   torch.LongTensor [B, N_i]
        "values":   torch.FloatTensor [B]
        "details":  List[Dict[str,Any]]
      }
    """
    device = probs.device if isinstance(probs, torch.Tensor) else torch.device("cpu")

    orders: List[torch.LongTensor] = []
    values_list: List[float] = []
    details: List[Dict[str, Any]] = []

    B = len(dfs)
    has_prob_rows = (isinstance(probs, torch.Tensor) and probs.ndim == 2 and probs.shape[0] == B)

    for b in range(B):
        df = dfs[b]
        pr = probs[b] if has_prob_rows and probs.shape[1] == len(df) else None  # disable weighting if not aligned

        res = _beam_max_npv_for_batch(
            df=df,
            probs_row=pr,
            discount_factor=discount_factor,
            steps_to_ahead=steps_to_ahead,
            apply_fraction_to_lookahead=apply_fraction_to_lookahead,
            alpha_prob=alpha_prob,
            x_col=x_col,
            y_col=y_col,
            z_col=z_col,
            level_col=level_col,
            enforce_level_by_xyz=enforce_level_by_xyz,
            beam_width=beam_width,
            rollout_horizon=rollout_horizon,
            candidate_pool_size=candidate_pool_size
        )

        orders.append(torch.tensor(res["order"], dtype=torch.long, device=device))
        values_list.append(res["value"])
        details.append(res)

    values = torch.tensor(values_list, dtype=torch.float32, device=device)
    orders = torch.stack(orders)
    return {"orders": orders, "values": values, "details": details}


def npv(x,discount_factor=0.99,steps_to_ahead=8):
    # columns: 0=x,1=y,2=z,3=cost,4=cost_look_ahead,5=income,6=income_look_ahead
    # 'cost'/'cost_look_ahead' are already-negative quantities, so add rather than subtract
    x = ((x[:,:,5]+x[:,:,3])+(x[:,:,6]+x[:,:,4])*discount_factor**steps_to_ahead)*(discount_factor**torch.arange(0,x.shape[1]).to(x.device))
    return torch.sum(x,dim=1).mean()
def income_loss(probs,dfs,discount_factor=0.99,steps_to_ahead=8):
    rewards = []
    probs_sorted,probs_index = torch.sort(probs,descending=True)
    for batch in range(probs.shape[0]):
        blocks_already_mined = set()
        dfs[batch]=dfs[batch].reset_index(drop=True)
        for idx_ in probs_index[batch]:
            idx = idx_.item()
            row = dfs[batch].iloc[idx]
            intersect =  len(set(row['blocks']).intersection(blocks_already_mined))
            total_count = len(row['blocks'])
            if intersect == 0:
                blocks_already_mined.update(row['blocks'])
            else:
                proportion_remaining = 1.0 - (intersect / max(1, total_count))
                dfs[batch].at[idx, 'income'] *= proportion_remaining
                dfs[batch].at[idx, 'cost'] *= proportion_remaining
                
            blocks_already_mined.update(row['blocks'])
        rewards.append(torch.from_numpy(((dfs[batch]['income']+dfs[batch]['cost'])+(dfs[batch]['income_look_ahead']+dfs[batch]['cost_look_ahead'])*discount_factor**steps_to_ahead).to_numpy().astype('float32')))
    rewards = torch.stack(rewards).to(probs.device)
    rewards_gathered = rewards.gather(dim=1,index=probs_index)
    probs_avg_return = torch.sum(probs_sorted**(1/2)*rewards_gathered*(discount_factor**torch.arange(0,probs_sorted.shape[1])).to(probs_sorted.device).unsqueeze(0),dim=1)
    npv_return = torch.sum(rewards_gathered*(discount_factor**torch.arange(0,probs_sorted.shape[1])).to(probs_sorted.device).unsqueeze(0),dim=1)
    
    return probs_avg_return.mean(),npv_return.mean() 

def entropy_loss(probs,eps):
    return -torch.sum(probs*torch.log(probs+eps),dim=1).mean()

def constraint_sequencing(initial_pred, x_unnormalized, num_steps=100, step_size=10.0, early_stop_tol=1e-5):
    pred = initial_pred.clone() # Work on a clone to keep the graph

    B = pred.shape[0]
    
    # Pre-compute groups for each batch item to avoid expensive lookups in the loop
    groups_by_batch = []
    for i in range(B):
        coords = x_unnormalized[i, :, :3].cpu().numpy().round(2) # Use rounded coords for stable hashing
        loc_to_indices = {}
        for j in range(coords.shape[0]):
            loc_tuple = tuple(coords[j])
            if loc_tuple not in loc_to_indices:
                loc_to_indices[loc_tuple] = []
            loc_to_indices[loc_tuple].append(j)
        
        # For each location, sort indices by level (assuming level is at a specific column, e.g., 7)
        # And only keep groups with more than one level
        stacks = []
        level_col_idx = 7 # Assuming 'level' is the 8th column (index 7)
        for loc, indices in loc_to_indices.items():
            if len(indices) > 1:
                # Sort indices based on the 'level' value in x_unnormalized
                sorted_indices = sorted(indices, key=lambda k: x_unnormalized[i, k, level_col_idx])
                stacks.append(torch.tensor(sorted_indices, device=pred.device))
        groups_by_batch.append(stacks)

    for _ in range(num_steps):
        total_constraint_violation = 0
        probs = torch.softmax(pred, dim=-1)
        for i in range(B):
            if not groups_by_batch[i]:
                continue
            for stack_indices in groups_by_batch[i]:
                # Probabilities for the current stack, sorted by level
                stack_probs = probs[i, stack_indices]
                # Violation is when a deeper level has higher probability: P(k+1) > P(k)
                # We want to minimize max(0, P(k+1) - P(k))
                violations = torch.relu(stack_probs[1:] - stack_probs[:-1])
                total_constraint_violation += violations.sum()
        if total_constraint_violation.item() < early_stop_tol:
            print(total_constraint_violation.item())
            print("early stop")
            break
        grad = torch.autograd.grad(total_constraint_violation, pred, retain_graph=True, create_graph=True)[0]
        pred = pred - step_size * grad
        print("updated step")
    return torch.softmax(pred, dim=-1)


def get_idx(df,len_per_view =100, num_views=4):
    total_indices = []
    for idx,row in df.iterrows():
        indices = []
        distance = np.sqrt((row['x']-df['x'])**2 + (row['y']-df['y'])**2 + (row['z']-df['z'])**2)
        for i in range(num_views):
            idxs = np.argsort(distance.to_numpy())[::i+1][:len_per_view]
            idxs = torch.from_numpy(idxs)
            indices.append(idxs)
        indices = torch.stack(indices)
        total_indices.append(indices)
    total_indices = torch.stack(total_indices)
    idx_all = total_indices.long()              # must be int64
    idx_all = idx_all.unsqueeze(0)        # (1, 10001, 4, 100) add batch dim
    idxs = [idx_all[:, :, v, :] for v in range(num_views)]  # list of 4 tensors, each (1,10001,100)
    return idxs