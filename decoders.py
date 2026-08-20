"""
Score -> precedence-feasible mining sequence. Two decoders, and they differ in
how much of the score they destroy on the way.

    sort   the current path. project_hard_clamp pushes every child down to
           min(parents), then lexsort(-s, topo_rank). The clamp *edits the
           scores*: a child scoring above its parent is dragged down to the
           parent's value, which merges the two into an exact tie and hands the
           ordering decision to the topological tiebreak instead of to s.
           Whole subtrees collapse onto one value this way.

    kahn   maintain the ready set -- blocks whose predecessors are all mined --
           and repeatedly pop its highest-priority member. The scores are never
           touched; precedence acts as a *filter on what is available*, not as
           an edit to the objective. Feasible by construction, same O(n log n +
           m), and strictly more expressive: two score vectors that the clamp
           maps to the same sequence generally give different sequences here.

The distinction is the whole reason this module exists. Under `sort`, a search
over scores is really a search over the much smaller set of post-clamp scores,
and the projection upstream has already contracted things once. Under `kahn`
every score vector in R^n reaches a distinct feasible schedule, which is what a
derivative-free outer loop needs to have anything to move along.

Both decoders return a permutation with zero precedence violations; that is
asserted by the callers via kernel_projection.sequence_violations.
"""

import heapq

import numpy as np

from kernel_projection import (project_hard_clamp, schedule_from_scores,
                               topological_order)


def children_csr(n, par, chi):
    """Adjacency in CSR form plus in-degrees, built once and reused.

    The decoders are called hundreds of times inside a sweep or an outer search
    loop while the graph never changes, so the O(m) rebuild belongs outside the
    hot path. Returns a dict with plain Python lists as well: the Kahn loop is
    interpreted, and list indexing beats numpy scalar indexing by ~10x there.
    """
    par = np.asarray(par, dtype=np.int64)
    chi = np.asarray(chi, dtype=np.int64)
    m = par.shape[0]

    deg = np.bincount(par, minlength=n)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(deg, out=indptr[1:])
    o = np.argsort(par, kind="stable")
    indices = chi[o].astype(np.int64)

    indeg = np.bincount(chi, minlength=n).astype(np.int64)
    return {
        "n": int(n), "m": int(m),
        "indptr": indptr, "indices": indices, "indeg": indeg,
        "indptr_l": indptr.tolist(), "indices_l": indices.tolist(),
        "indeg_l": indeg.tolist(),
    }


def schedule_priority_kahn(s, csr, tiebreak=None):
    """Highest-priority ready block first. Scores are read, never written.

    `tiebreak` breaks equal scores, smaller first, and matters more than it
    looks: this block model ties every one of its blocks at the raw score (a
    two-valued income column times an integer bench cost), so the tiebreak is
    doing real ordering work, not cleaning up rounding. Pass topological rank
    for the deterministic convention that matches schedule_from_scores, or a
    random key to sample the tie freedom -- it is free diversity, and the
    resulting schedules are all exactly feasible.
    """
    n = csr["n"]
    s = np.asarray(s, dtype=np.float64)
    if s.shape[0] != n:
        raise ValueError(f"s has length {s.shape[0]}, graph has n={n}")

    if tiebreak is None:
        tiebreak = np.arange(n, dtype=np.float64)
    tiebreak = np.asarray(tiebreak, dtype=np.float64)
    if tiebreak.shape[0] != n:
        raise ValueError("tiebreak must have length n")

    indptr = csr["indptr_l"]
    indices = csr["indices_l"]
    indeg = list(csr["indeg_l"])
    neg_s = (-s).tolist()
    tb = tiebreak.tolist()

    heap = [(neg_s[i], tb[i], i) for i in range(n) if indeg[i] == 0]
    if not heap:
        raise ValueError("no source blocks: precedence graph contains a cycle")
    heapq.heapify(heap)

    seq = np.empty(n, dtype=np.int64)
    k = 0
    pop = heapq.heappop
    push = heapq.heappush
    while heap:
        u = pop(heap)[2]
        seq[k] = u
        k += 1
        for j in range(indptr[u], indptr[u + 1]):
            v = indices[j]
            d = indeg[v] - 1
            indeg[v] = d
            if d == 0:
                push(heap, (neg_s[v], tb[v], v))
    if k != n:
        raise ValueError(f"only {k} of {n} blocks scheduled: graph has a cycle")
    return seq


def schedule_sort_clamp(s, par, chi, order=None):
    """The existing decoder, wrapped for a uniform signature."""
    return schedule_from_scores(s, par, chi, order=order)


def decode(s, kind, par=None, chi=None, order=None, csr=None, tiebreak=None):
    """Dispatch on decoder name. `kind` is 'kahn' or 'sort'."""
    if kind == "kahn":
        if csr is None:
            raise ValueError("decoder 'kahn' needs csr=children_csr(...)")
        return schedule_priority_kahn(s, csr, tiebreak=tiebreak)
    if kind == "sort":
        if par is None or chi is None:
            raise ValueError("decoder 'sort' needs par/chi")
        return schedule_sort_clamp(s, par, chi, order=order)
    raise ValueError("kind must be 'kahn' or 'sort'")


def clamp_collapse(s, par, chi, order=None, tol=1e-12):
    """How much of the score the `sort` decoder's hard clamp destroys.

    Returns the number of blocks the clamp moved, the mean absolute move, and
    the number of parent/child pairs it merged into an exact tie -- pairs whose
    relative order s had an opinion about and the clamp discarded.
    """
    s = np.asarray(s, dtype=float)
    n = s.shape[0]
    if order is None:
        order = topological_order(n, par, chi)
    sc = project_hard_clamp(s, par, chi, order=order)
    moved = np.abs(sc - s) > tol
    merged = np.abs(sc[par] - sc[chi]) <= tol
    was_distinct = np.abs(s[par] - s[chi]) > tol
    return {
        "n_moved": int(moved.sum()),
        "frac_moved": float(moved.mean()),
        "mean_abs_move": float(np.abs(sc - s)[moved].mean()) if moved.any() else 0.0,
        "max_abs_move": float(np.abs(sc - s).max()),
        "edges_merged": int(merged.sum()),
        "edges_merged_from_distinct": int((merged & was_distinct).sum()),
        "frac_edges_merged": float(merged.mean()),
    }
