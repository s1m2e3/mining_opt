from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict
# Google OR-Tools
import math, os
from ortools.math_opt.python import mathopt
import pandas as pd
from lg_utils import kmeanspp_centroids_3d
import numpy as np
import torch
import torch.nn.functional as F
import random

def all_dependencies(blocks, index):
    """
    Return a list of all blocks that `index` depends on (transitively),
    with duplicates removed. Raises on cycles.
    """
    visited = set()
    visiting = set()  # recursion stack for cycle detection

    def dfs(node):
        if node in visiting:
            raise ValueError(f"Cycle detected involving block {node}")
        if node in visited:
            return
        visiting.add(node)
        for dep in blocks[node].get('depends', []):
            dfs(dep)
            visited.add(dep)  # add dependency itself
        visiting.remove(node)

    dfs(index)
    return list(visited)


def available_cores() -> int:
    # Prefer affinity (respects cgroups/containers), fall back to os.cpu_count()
    try:
        if hasattr(os, "sched_getaffinity"):
            return len(os.sched_getaffinity(0))
    except Exception:
        pass
    return os.cpu_count() or 1


def pick_threads(ratio: float = 0.9, min_threads: int = 1) -> int:
    cores = max(available_cores(), 1)
    return max(min(int(math.floor(ratio * cores)), cores), min_threads)

def make_solver_params(
    solver_type: mathopt.SolverType,
    ratio: float = 0.9,
    deterministic: bool = True,
) -> mathopt.SolveParameters:
    """
    Build MathOpt SolveParameters with sensible thread configuration.
    Notes:
      - MathOpt's generic params.threads does NOT apply to HiGHS in MathOpt.
        For HiGHS, set highs option 'threads' (and optionally 'parallel').
      - For other solvers, params.threads is typically honored.
    """
    n = pick_threads(ratio)  # assumes you already have this function

    params = mathopt.SolveParameters()

    # Generic (solver-independent) threads: use for non-HiGHS backends.
    if solver_type != mathopt.SolverType.HIGHS:
        params.threads = n

    # HiGHS-specific options
    if solver_type == mathopt.SolverType.HIGHS:
        # HiGHS option keys are strings; threads is an integer option.
        params.highs.int_options["threads"] = n
        # Enable parallel mode where applicable ("on" / "off" / "choose")
        params.highs.string_options["parallel"] = "on"
    # Optional: if you want determinism, you generally need to control
    # solver-specific random seeds; HiGHS has options like "random_seed"
    # but exact behavior depends on algorithm. Left as a placeholder.
    _ = deterministic

    return params


@dataclass
class Problem:
    data: Dict[str, Any] = field(default_factory=dict)

    # Choose backend once (PDLP, HIGHS, GLOP, SCIP, etc.)
    solver_type: mathopt.SolverType = mathopt.SolverType.HIGHS
    
    # Model and storage
    model: mathopt.Model = field(default_factory=mathopt.Model, init=False)
    x: Dict[str, Any] = field(default_factory=dict, init=False)
    ctrs: Dict[Any, mathopt.LinearConstraint] = field(default_factory=dict, init=False)

    # Solver params
    solve_params: mathopt.SolveParameters = field(init=False)

    def __post_init__(self) -> None:
        # Build solver parameters AFTER solver_type is set on the instance.
        self.solve_params = make_solver_params(self.solver_type)

    def writeModel(self) -> None:
        """Subclasses build the model here: vars, constraints, objective."""
        raise NotImplementedError

    def solve(self) -> mathopt.SolveResult:
        """Build (if needed) and solve."""
        # If you want to allow multiple solves with modifications,
        # you can guard against rebuilding here.
        self.writeModel()

        result = mathopt.solve(
            self.model,
            solver_type=self.solver_type,
            params=self.solve_params,
        )
        return result


@dataclass
class LGData:
    blocks: Dict[str, Any] = field(default_factory=dict)
@dataclass
class NPVLGData:
    blocks: Dict[str, Any] = field(default_factory=dict)
    discount_rate: float = 0.1 # Defined per years
    num_periods: int = 10 # Defined in years
    daily_mining_rate: float = 100 # Defined in tons/day time defined in num_periods
    time_penalty: float = 1 # Per period
    ramp: float = 0.9 # Defined in fraction of tonnage
    value_scaling: float = 1.0
    tonnage_scaling: float = 1.0
    discount_rate: float = 0.1
    period_size: int = 5
    binary_penalty: float = 1e-1
    window_scale: float = 3.0
    ub_threshold: float = 0.7
    lb_threshold: float = 0.4
    distance_penalty: float = 1e-3
    centroids: int = 10
    def find_neighbours(self, batch_size=1000, num_elems=100):
        df = pd.DataFrame.from_dict(self.blocks, orient='index')
        centroids = kmeanspp_centroids_3d(x_min=df['x_c'].min(), x_max=df['x_c'].max(),
                                            y_min=df['y_c'].min(), y_max=df['y_c'].max(),
                                            z_min=df['z_c'].min(), z_max=df['z_c'].max(),
                                            n_centroids=self.centroids, n_samples=int(1e3))
        locations = df[['x_c', 'y_c', 'z_c']].to_numpy()
        nodes_index = list(df.index)
        dist = np.sqrt(np.sum((locations[:,None,:]-centroids[None])**2,axis=2))
        normalized_dist = dist / np.max(dist,axis=0,keepdims=True)
        # Torch tensor: (n_blocks, n_centroids)
        D = torch.from_numpy(normalized_dist).to(torch.float32)
        # 3 nearest centroids per block (indices): (n_blocks, 3)
        # largest=False => smallest distances
        _, top3 = torch.topk(D, k=3, dim=1, largest=False, sorted=False)
        bin_top3 = F.one_hot(top3,num_classes = self.centroids).sum(1)
        top3_list = top3.tolist()
        top3_list = [tuple(set(top3_list[i])) for i in range(len(top3_list))]
        centroids = {}
        for i in range(len(top3_list)):
            if top3_list[i] not in centroids:
                centroids[top3_list[i]] = []
            centroids[top3_list[i]].append(nodes_index[i])
        
        rows, cols = [], []
        N = bin_top3.shape[0]
        neighbours = {}
        print("Finding neighbours")
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            sub = bin_top3[start:end]                 # (B, C)

            row, col = torch.where((bin_top3 @ sub.T) > 0)  # row global, col in [0, B-1]
            col = col + start                               # now col global

            # optional sampling (GPU-friendly)
            k = min(num_elems, row.numel())
            if k > 0:
                idx = torch.randperm(row.numel(), device=row.device)[:k]
                row, col = row[idx], col[idx]

            rows.append(row)
            cols.append(col)
        edge_index = torch.stack([torch.cat(rows), torch.cat(cols)], dim=0).T
        
        for i in range(edge_index.shape[0]):
            src_id = nodes_index[int(edge_index[i,0])]
            dst_id = nodes_index[int(edge_index[i,1])]

            neighbours.setdefault(src_id, [])
            # optional: skip self-neighbors
            if src_id == dst_id: continue

            # optional: undirected duplicate check
            if dst_id in neighbours and src_id in neighbours.get(dst_id, []):
                continue
            neighbours[src_id].append(dst_id)
        print("Found neighbours")
        return neighbours,centroids
    def __post_init__(self):
        if self.blocks:
            self.blocks = {int(k): v for k, v in self.blocks.items()}
            values = [abs(b['value']) for b in self.blocks.values() if 'value' in b]
            tonnages = [b['tonnage'] for b in self.blocks.values() if 'tonnage' in b]
            neighbours,centroids = self.find_neighbours()
            self.neighbours = neighbours
            self.centroids = centroids
            if values:
                self.value_scaling = max(values)
            if tonnages:
                self.tonnage_scaling = max(tonnages)
            self.mining_rate = int(self.daily_mining_rate*365/(1/self.period_size))
            self.discount_rate = (1+self.discount_rate)**(1/self.period_size)-1
            for block in self.blocks:
                if self.blocks[block]['depends'] != []:
                    depends = all_dependencies(self.blocks, block)
                    tonnage = 0
                    for dep in depends:
                        tonnage += self.blocks[dep]['tonnage']
                    min_time = math.floor(tonnage/self.mining_rate)
                    self.blocks[block]['min_time'] = min_time
                else:
                    self.blocks[block]['min_time'] = 0