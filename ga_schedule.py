"""Combinatorial search over mining sequences.

Step 1 of the search-then-imitate plan: find out how much a stochastic search
beats the trained transformer before building any imitation loop.

THE GENOME IS THE SEQUENCE, NOT A SCORE. Measured on crop-14, a score-space
genome would need the precedence projection on every individual at 2963 ms
each, so a 512 x 100 GA would take 42 hours; on permutations, where evaluating
a schedule is a gather and a cumsum, the same GA spends under a second on
evaluation. A factor of ~630,000 decides the representation outright.

FEASIBILITY IS STRUCTURAL. Every operator produces a precedence-feasible
sequence by construction, so no individual is ever repaired or rejected:

  shift        move one block inside its FEASIBLE WINDOW -- after its last
               predecessor, before its first successor.
  segment      relocate a contiguous run intact. Precedence WITHIN the run is
               preserved automatically, so only the run's outside relations
               bind; and because the permutation is already feasible, every
               outside predecessor lies before the run and every outside
               successor at or after it, which reduces both bounds to one max
               and one min.
  crossover    uniform crossover on the two parents' POSITION vectors, decoded
               through a ready-set (Kahn) loop -- the same construction
               decoders.py already relies on. Precedence is enforced by the
               ready set whatever the mixed keys look like. No score vector is
               ever optimised; the keys are read off parent permutations and
               discarded after the decode.

`sequence_violations` is asserted on the result rather than assumed.

DIVERSITY IS MAINTAINED, NOT ASSUMED. A first version with a fat elite, mild
mutation and hard selection collapsed to a monoculture by generation 50 -- best
and mean fitness sat within 2e-4 of each other for the remaining 200
generations, so it was a hill climb wearing a GA's clothes. The settings below
(thin elite, heavy-tailed mutation, segment moves, low selection pressure,
random immigrants, duplicate rejection) hold the population apart, and
`diversity` reports whether they are working rather than leaving it to trust.

Calibration for reading pos_std: a population of INDEPENDENT random feasible
schedules scores about 0.064 on this instance, not the 0.289 of unconstrained
permutations, because precedence removes most of the freedom before any search
starts. 0.064 is the ceiling to judge against.

SPEED. Nothing here needs a gradient, so the hot loops are numba-compiled
rather than vectorised. That is the right tool: the mutation loop is inherently
sequential -- each move reads the permutation the previous one wrote -- which
is exactly the case numpy cannot help with. Falls back to pure Python if numba
is missing: correct, far slower.

Run: python ga_schedule.py [generations] [population]
"""

import sys
import time

import numpy as np

import continuous_time as ct
from kernel_projection import sequence_violations, topological_order

DISCOUNT = 0.90
T_PERIODS = 10
CROP = 14
SEED = 0

GENERATIONS = 120
POPULATION = 256

ELITE = 0.04               # a thin elite; a fat one floods the pool
P_CROSSOVER = 0.70
MUT_MOVES = 40             # MEDIAN moves per child; the count is heavy-tailed
MUT_HEAVY_TAIL = 1.6       # Pareto exponent on the move count
P_SEGMENT = 0.35           # chance a move relocates a whole run
SEG_MAX = 0.05             # longest run, as a fraction of n
TOURNAMENT = 2             # low pressure keeps weak lineages alive
IMMIGRANT = 0.08           # fraction replaced by fresh random schedules
NO_DUPLICATES = True
# Sensitivity weighting is OFF by default because it was measured to HURT.
# Isolated on crop-5 at a 5 s budget: uniform 0.58046, weighting only 0.57771,
# repair only 0.58821, both 0.58368. The likely reason is that moving a
# high-|dNPV/dsigma| block earlier requires displacing whatever sits there, so
# concentrating picks on the valuable blocks starves the moves that make room
# for them. Kept as a knob because it may pay with a different mixture.
P_WEIGHTED = 0.0
# Local dominance repair is ON: +1.3% on the raw GA result and inversions
# 459 -> 84, for 23% fewer evaluations. Every individual is near
# dominance-optimal, so the search runs over local optima.
REPAIR_SWEEPS = 3
WEIGHT_REFRESH = 5         # generations between sensitivity-weight rebuilds


# --------------------------------------------------------------------------
# precedence structure
# --------------------------------------------------------------------------

def _csr(n, src, dst):
    counts = np.bincount(dst, minlength=n)
    indptr = np.zeros(n + 1, np.int64)
    np.cumsum(counts, out=indptr[1:])
    o = np.argsort(dst, kind="stable")
    return indptr, np.ascontiguousarray(src[o].astype(np.int64))


def adjacency(n, par, chi):
    """Predecessors, successors and in-degrees, in CSR form.

    CSR rather than ragged lists because the segment operators walk every edge
    of a whole run of blocks -- a tight indexed loop in compiled code, and a
    Python loop otherwise.
    """
    par, chi = np.asarray(par, np.int64), np.asarray(chi, np.int64)
    pi, pd = _csr(n, par, chi)          # predecessors of each block
    si, sd = _csr(n, chi, par)          # successors of each block
    return {"pi": pi, "pd": pd, "si": si, "sd": sd,
            "indeg": np.ascontiguousarray(pi[1:] - pi[:-1]), "n": n}


# --------------------------------------------------------------------------
# compiled kernels -- plain functions here, jitted at the end of the section
# --------------------------------------------------------------------------

def _feasible_window(pos, pi, pd, si, sd, b, n):
    lo = 0
    for k in range(pi[b], pi[b + 1]):
        q = pos[pd[k]] + 1
        if q > lo:
            lo = q
    hi = n - 1
    for k in range(si[b], si[b + 1]):
        q = pos[sd[k]] - 1
        if q < hi:
            hi = q
    return lo, hi


def _segment_window(perm, pos, pi, pd, si, sd, a, b, n):
    L = b - a
    lo = 0
    hi = n - L
    for t in range(a, b):
        blk = perm[t]
        for k in range(pi[blk], pi[blk + 1]):
            q = pos[pd[k]]
            if q < a and q + 1 > lo:      # q >= b impossible: perm is feasible
                lo = q + 1
        for k in range(si[blk], si[blk + 1]):
            q = pos[sd[k]]
            if q >= b and q - L < hi:
                hi = q - L
    return lo, hi


def _shift(perm, pos, b, dst):
    src = pos[b]
    if src == dst:
        return
    if dst < src:
        for i in range(src, dst, -1):
            perm[i] = perm[i - 1]
            pos[perm[i]] = i
    else:
        for i in range(src, dst):
            perm[i] = perm[i + 1]
            pos[perm[i]] = i
    perm[dst] = b
    pos[b] = dst


def _move_segment(perm, pos, a, b, dst, buf):
    L = b - a
    if dst == a:
        return
    for i in range(L):
        buf[i] = perm[a + i]
    if dst < a:
        for i in range(a - 1, dst - 1, -1):
            perm[i + L] = perm[i]
            pos[perm[i + L]] = i + L
    else:
        for i in range(b, dst + L):
            perm[i - L] = perm[i]
            pos[perm[i - L]] = i - L
    for i in range(L):
        perm[dst + i] = buf[i]
        pos[buf[i]] = dst + i


def _is_edge(keys, key):
    i = np.searchsorted(keys, key)
    return i < keys.size and keys[i] == key


def _repair_span(perm, pos, keys, dens, n, lo, hi, sweeps):
    """Restore dominance order among adjacent free pairs, in a WINDOW only.

    A mutation can only create new inversions inside the span it disturbed, so
    repairing that span keeps every individual dominance-optimal for O(span)
    instead of the O(n) a full sweep costs. That is what makes a memetic GA
    affordable here: a full sweep per child would be ~3 ms x 7,680 children.
    """
    a = lo - 1 if lo > 0 else 0
    b = hi + 1 if hi + 1 < n else n - 1
    for _ in range(sweeps):
        moved = 0
        for p in range(a, b):
            i = perm[p]
            j = perm[p + 1]
            if dens[i] < dens[j] - 1e-12 and not _is_edge(keys, i * n + j):
                perm[p] = j
                perm[p + 1] = i
                pos[j] = p
                pos[i] = p + 1
                moved += 1
        if moved == 0:
            break


def _draw_block(cdf, n, p_weighted):
    """A block to move: sensitivity-weighted with probability p_weighted.

    Uniform choice spends half the mutation budget on blocks carrying 2% of the
    total |dNPV/dsigma| -- measured. Weighting concentrates the same number of
    moves where NPV actually lives. It stays a MIXTURE because pure weighting is
    an exploitation bias, and this population collapsed to a monoculture once
    already; `diversity` is the check on whether it has gone too far.
    """
    if np.random.random() < p_weighted:
        return np.searchsorted(cdf, np.random.random() * cdf[n - 1])
    return np.random.randint(0, n)


def _mutate(perm, pos, pi, pd, si, sd, n, moves, seg_max, p_segment, buf,
            cdf, p_weighted, keys, dens, repair_sweeps):
    for _ in range(moves):
        if np.random.random() < p_segment:
            a = np.random.randint(0, n - 1)
            b = a + np.random.randint(2, seg_max + 1)
            if b > n:
                b = n
            lo, hi = _segment_window(perm, pos, pi, pd, si, sd, a, b, n)
            if hi > lo:
                dst = np.random.randint(lo, hi + 1)
                _move_segment(perm, pos, a, b, dst, buf)
                if repair_sweeps > 0:
                    u = dst if dst < a else a
                    v = b if dst < a else dst + (b - a)
                    _repair_span(perm, pos, keys, dens, n, u, v, repair_sweeps)
        else:
            blk = _draw_block(cdf, n, p_weighted)
            lo, hi = _feasible_window(pos, pi, pd, si, sd, blk, n)
            if hi > lo:
                src = pos[blk]
                dst = np.random.randint(lo, hi + 1)
                _shift(perm, pos, blk, dst)
                if repair_sweeps > 0:
                    u = dst if dst < src else src
                    v = src if dst < src else dst
                    _repair_span(perm, pos, keys, dens, n, u, v, repair_sweeps)


def _mutate_pop(pop, posm, pi, pd, si, sd, n, moves, seg_max, p_segment,
                cdf, p_weighted, keys, dens, repair_sweeps):
    """Mutate the WHOLE population inside one compiled call.

    84% of the GA's wall clock was operator overhead, not fitness -- one Python
    call into numba per child. Looping over the population inside the kernel
    removes that boundary entirely.
    """
    buf = np.empty(n, np.int64)
    for r in range(pop.shape[0]):
        _mutate(pop[r], posm[r], pi, pd, si, sd, n, moves[r], seg_max,
                p_segment, buf, cdf, p_weighted, keys, dens, repair_sweeps)


def _kahn(key, si, sd, indeg0, n, out):
    """Ready-set decode: repeatedly take the available block of lowest key.

    A hand-rolled binary heap rather than heapq so the whole decode compiles.
    Feasible by construction for ANY key vector -- precedence acts as a filter
    on what is available, never as an edit to the key.
    """
    indeg = indeg0.copy()
    hk = np.empty(n, np.float64)
    hv = np.empty(n, np.int64)
    hn = 0
    for i in range(n):
        if indeg[i] == 0:
            hk[hn] = key[i]
            hv[hn] = i
            j = hn
            hn += 1
            while j > 0:
                p = (j - 1) // 2
                if hk[p] <= hk[j]:
                    break
                hk[p], hk[j] = hk[j], hk[p]
                hv[p], hv[j] = hv[j], hv[p]
                j = p
    k = 0
    while hn > 0:
        u = hv[0]
        out[k] = u
        k += 1
        hn -= 1
        hk[0] = hk[hn]
        hv[0] = hv[hn]
        j = 0
        while True:
            l = 2 * j + 1
            r = l + 1
            m = j
            if l < hn and hk[l] < hk[m]:
                m = l
            if r < hn and hk[r] < hk[m]:
                m = r
            if m == j:
                break
            hk[m], hk[j] = hk[j], hk[m]
            hv[m], hv[j] = hv[j], hv[m]
            j = m
        for e in range(si[u], si[u + 1]):
            v = sd[e]
            indeg[v] -= 1
            if indeg[v] == 0:
                hk[hn] = key[v]
                hv[hn] = v
                j = hn
                hn += 1
                while j > 0:
                    p = (j - 1) // 2
                    if hk[p] <= hk[j]:
                        break
                    hk[p], hk[j] = hk[j], hk[p]
                    hv[p], hv[j] = hv[j], hv[p]
                    j = p
    return k


try:                                                # pragma: no cover
    from numba import njit
    _feasible_window = njit(cache=True)(_feasible_window)
    _segment_window = njit(cache=True)(_segment_window)
    _shift = njit(cache=True)(_shift)
    _move_segment = njit(cache=True)(_move_segment)
    _is_edge = njit(cache=True)(_is_edge)
    _repair_span = njit(cache=True)(_repair_span)
    _draw_block = njit(cache=True)(_draw_block)
    _mutate = njit(cache=True)(_mutate)
    _mutate_pop = njit(cache=True)(_mutate_pop)
    _kahn = njit(cache=True)(_kahn)
    HAVE_NUMBA = True
except Exception:                                   # pragma: no cover
    HAVE_NUMBA = False


# --------------------------------------------------------------------------
# python-side operators
# --------------------------------------------------------------------------

def kahn(key, adj):
    """Feasible sequence from a key vector, lowest key mined first."""
    n = adj["n"]
    out = np.empty(n, np.int64)
    k = _kahn(np.ascontiguousarray(key, dtype=np.float64), adj["si"],
              adj["sd"], adj["indeg"], n, out)
    if k != n:
        raise ValueError("precedence graph contains a cycle")
    return out


def random_feasible(adj, rng):
    return kahn(rng.random(adj["n"]), adj)


def mutation_strength(rng, median=MUT_MOVES):
    """Heavy-tailed move count.

    A fixed count gives every child the same step size and the search settles
    into that one scale. A Pareto tail makes most children small local edits
    and a few large structural rearrangements; the occasional big jump is what
    unsticks a converged population, and it is nearly free because evaluating a
    schedule is a cumsum.
    """
    return int(median * (1.0 + rng.pareto(MUT_HEAVY_TAIL)))


UNIFORM_CDF = {}


def sensitivity_cdf(value, tau, sigma, discount=DISCOUNT, floor=1e-3):
    """Cumulative weights proportional to |dNPV/dsigma| = |d v psi exp(-d s)|.

    Recomputed once per generation from the current best schedule rather than
    per move: the ranking is stable enough, and per-move recomputation would
    cost more than the move. `floor` keeps a small uniform component so no
    block is ever unreachable.
    """
    d = ct.delta_from_discount(discount)
    psi = ct.within_block_shape(tau, d)
    w = np.abs(d * np.asarray(value) * psi * np.exp(-d * np.asarray(sigma)))
    w = w / max(w.max(), 1e-300) + floor
    return np.ascontiguousarray(np.cumsum(w))


def mutate(perm, pos, adj, n, rng, moves, buf, cdf=None, p_weighted=0.0,
           keys=None, dens=None, repair_sweeps=0):
    if cdf is None:
        cdf = np.cumsum(np.ones(n))
    if keys is None:
        keys = np.zeros(1, np.int64)
    if dens is None:
        dens = np.zeros(n)
    _mutate(perm, pos, adj["pi"], adj["pd"], adj["si"], adj["sd"], n,
            int(moves), max(2, int(SEG_MAX * n)), P_SEGMENT, buf,
            cdf, float(p_weighted), keys, dens, int(repair_sweeps))


def crossover(pa, pb, adj, rng, n):
    """Uniform crossover on position keys, in contiguous runs.

    Per-block coin flips shred useful sub-sequences; inheriting in runs keeps
    them intact. The child is decoded by the ready set, so it is feasible
    whatever the mixed keys look like.
    """
    ka = np.empty(n, np.float64)
    kb = np.empty(n, np.float64)
    ka[pa] = np.arange(n)
    kb[pb] = np.arange(n)
    runs = max(1, n // 64)
    seg = np.repeat(rng.random(runs) < 0.5, int(np.ceil(n / runs)))[:n]
    return kahn(np.where(seg, ka, kb), adj)


# --------------------------------------------------------------------------

def _key(row):
    """64-bit identity for a schedule, for the duplicate and novelty sets.

    These sets used to hold the full permutation as bytes -- 8n bytes each. On
    crop-14 at 400 generations x 256 population that is ~98,000 schedules x 23
    KB = about 2 GiB, for two bookkeeping structures that only ever answer
    "have I seen this before". A 64-bit digest answers the same question in 8
    bytes, and the cost is O(n) either way.

    Collisions are the obvious worry and they are not a real one: at 1e5
    distinct schedules the birthday probability is ~3e-10, and both uses
    degrade harmlessly anyway -- a collision in `seen` undercounts the
    diagnostic by one, and a collision in `live` declares a false duplicate and
    mutates that child again.
    """
    return hash(row.tobytes())


def edge_keys(par, chi, n):
    """Sorted i*n+j keys, for O(log m) "is this pair a precedence edge" tests."""
    return np.sort(np.asarray(par, np.int64) * n + np.asarray(chi, np.int64))


def dominance_sweep(seq, value, tau, keys, n, max_sweeps=200):
    """Bubble adjacent precedence-free pairs into value-density order.

    An exchange argument settles the order of two blocks that are ADJACENT in
    the sequence and unrelated by precedence: swapping them is always feasible
    and moves only their two start times, and working it through, psi cancels
    exactly, leaving `i before j` optimal iff v_i/tau_i > v_j/tau_j. So any
    schedule holding such a pair the other way round is provably improvable.

    Adjacency is what makes the test cheap: a precedence PATH between two
    neighbouring positions would need an intermediate block between them, and
    there is no room, so only a DIRECT edge can relate them.

    Odd-even transposition so each sweep is vectorised and its swaps are
    non-overlapping by construction. Measured on crop-6, this lifts a
    200-generation GA result by 1.15% -- 452 of 531 flagged swaps improve NPV
    individually -- which the GA had not found on its own.
    """
    dens = np.asarray(value, dtype=float) / np.maximum(np.asarray(tau), 1e-300)
    q = np.asarray(seq, np.int64).copy()
    last = keys.size - 1
    for _ in range(max_sweeps):
        moved = 0
        for parity in (0, 1):
            p = np.arange(parity, n - 1, 2)
            if p.size == 0:
                continue
            a, b = q[p], q[p + 1]
            k = a * n + b
            idx = np.clip(np.searchsorted(keys, k), 0, last)
            bad = (keys[idx] != k) & (dens[a] < dens[b] - 1e-12)
            if bad.any():
                pp = p[bad]
                tmp = q[pp].copy()
                q[pp] = q[pp + 1]
                q[pp + 1] = tmp
                moved += int(bad.sum())
        if moved == 0:
            break
    return q


def count_inversions(seq, value, tau, keys, n):
    """How many adjacent precedence-free pairs are out of value-density order."""
    dens = np.asarray(value, dtype=float) / np.maximum(np.asarray(tau), 1e-300)
    a, b = seq[:-1], seq[1:]
    k = a * n + b
    idx = np.clip(np.searchsorted(keys, k), 0, keys.size - 1)
    free = keys[idx] != k
    return int((free & (dens[a] < dens[b] - 1e-12)).sum()), int(free.sum())


def diversity(pop, n):
    """How different are these schedules, really?

    unique    exact distinct permutations; 1 in a monoculture.
    pos_std   mean over blocks of the population std of that block's POSITION,
              over n. 0 means every member puts every block in the same place.
              Independent random FEASIBLE schedules sit near 0.064 here -- that,
              not 0.289, is the ceiling.

    Fitness spread can look healthy while pos_std is already flat, because many
    distinct permutations differ only among blocks whose order does not price.
    Equal fitness is not evidence of sameness; near-zero pos_std is.
    """
    uniq = np.unique(pop, axis=0).shape[0]
    pos = np.empty_like(pop)
    rows = np.arange(pop.shape[0])[:, None]
    pos[rows, pop] = np.arange(n)[None, :]
    return uniq, float(pos.std(axis=0).mean()) / n


def evaluate(pop, tau, value, scale):
    sigma = ct.start_times_from_order(pop, tau)
    return ct.npv(sigma, tau, value, discount=DISCOUNT, scale=scale)


def run_ga(seed_perms, tau, value, scale, adj, n, rng,
           generations=GENERATIONS, population=POPULATION, label="", every=10,
           seconds=None, quiet=False, keys=None, p_weighted=P_WEIGHTED,
           repair_sweeps=REPAIR_SWEEPS, init_pop=None):
    """(mu + lambda) GA. Returns the best sequence, its fitness and a report."""
    if init_pop is not None:
        # continue an existing population, so an outer loop can advance the GA
        # in slices without throwing away everything it has found
        pop = np.array(init_pop, np.int64, copy=True)
        population = pop.shape[0]
        k = min(len(seed_perms), population)
        if k:
            pop[population - k:] = np.asarray(seed_perms[:k])
        seed_perms = []
    else:
        pop = np.empty((population, n), np.int64)
        k = min(len(seed_perms), population)
        pop[:k] = np.asarray(seed_perms[:k])
    # the rest are INDEPENDENT random feasible schedules, never perturbed
    # copies of the seeds -- filling from the seeds collapses diversity, and
    # made an earlier warm start look worse than a cold one purely as an
    # artefact of its own initialisation
    if init_pop is None:
        for i in range(k, population):
            pop[i] = random_feasible(adj, rng)

    fit = evaluate(pop, tau, value, scale)
    n_elite = max(1, int(ELITE * population))
    n_imm = int(IMMIGRANT * population)
    buf = np.empty(n, np.int64)
    seg_max = max(2, int(SEG_MAX * n))
    keys_arr = (keys if keys is not None else edge_keys(
        np.zeros(1, np.int64), np.zeros(1, np.int64), n))
    dens_arr = np.ascontiguousarray(
        np.asarray(value, float) / np.maximum(np.asarray(tau), 1e-300))
    cdf = np.ascontiguousarray(np.cumsum(np.ones(n)))
    seen = set(map(_key, pop))
    trace = [float(fit.max())]
    t0 = time.perf_counter()

    def report(g):
        if quiet:
            return
        u, d = diversity(pop, n)
        print(f"    {label} gen {g:>4}  best {fit.max():+.5f}"
              f"  mean {fit.mean():+.5f}  sd {fit.std():.2e}"
              f" | uniq {u:>4}/{population}  pos_std {d:.4f}"
              f"  seen {len(seen):>7}  ({time.perf_counter() - t0:6.1f}s)")

    report(0)
    for g in range(1, generations + 1):
        if seconds is not None and time.perf_counter() - t0 > seconds:
            break
        elite_idx = np.argpartition(-fit, n_elite - 1)[:n_elite]
        new_pop = np.empty_like(pop)
        new_pop[:n_elite] = pop[elite_idx]
        i = n_elite

        # a standing trickle of unrelated feasible schedules, so the pool can
        # never become the descendants of one lucky individual
        while i < min(n_elite + n_imm, population):
            new_pop[i] = random_feasible(adj, rng)
            i += 1
        n_fixed = i

        # build every child first, then mutate the whole block in ONE compiled
        # call. Doing it per child crossed the Python/numba boundary 128 times a
        # generation, which was most of the 84% of wall clock that went to
        # operators rather than fitness.
        for j in range(i, population):
            cand = rng.integers(0, population, TOURNAMENT)
            a = cand[np.argmax(fit[cand])]
            if rng.random() < P_CROSSOVER:
                cand = rng.integers(0, population, TOURNAMENT)
                b = cand[np.argmax(fit[cand])]
                new_pop[j] = crossover(pop[a], pop[b], adj, rng, n)
            else:
                new_pop[j] = pop[a]
        kids = new_pop[n_fixed:]
        if kids.shape[0]:
            posm = np.empty_like(kids)
            rows = np.arange(kids.shape[0])[:, None]
            posm[rows, kids] = np.arange(n)[None, :]
            moves = np.array([mutation_strength(rng) for _ in range(kids.shape[0])],
                             dtype=np.int64)
            _mutate_pop(kids, posm, adj["pi"], adj["pd"], adj["si"], adj["sd"],
                        n, moves, seg_max, P_SEGMENT, cdf, p_weighted,
                        keys_arr, dens_arr, repair_sweeps)
            if NO_DUPLICATES:
                # one extra batched pass over just the repeats, rather than a
                # per-child retry loop
                k = np.array([_key(r) for r in kids])
                _, first = np.unique(k, return_index=True)
                dup = np.setdiff1d(np.arange(k.size), first)
                if dup.size:
                    sub = kids[dup]
                    sp = np.empty_like(sub)
                    r2 = np.arange(sub.shape[0])[:, None]
                    sp[r2, sub] = np.arange(n)[None, :]
                    _mutate_pop(sub, sp, adj["pi"], adj["pd"], adj["si"],
                                adj["sd"], n, moves[dup] * 2, seg_max,
                                P_SEGMENT, cdf, p_weighted, keys_arr,
                                dens_arr, repair_sweeps)
                    kids[dup] = sub

        pop = new_pop
        fit = evaluate(pop, tau, value, scale)
        if p_weighted > 0.0 and g % WEIGHT_REFRESH == 0:
            cdf = sensitivity_cdf(value, tau, ct.start_times_from_order(
                pop[int(np.argmax(fit))], tau))
        trace.append(float(fit.max()))
        seen.update(map(_key, pop))
        if g % every == 0 or g == generations:
            report(g)

    u, d = diversity(pop, n)
    return pop[int(np.argmax(fit))], float(fit.max()), {
        "trace": trace, "distinct": len(seen), "unique": u, "pos_std": d,
        "evaluations": population * (generations + 1), "pop": pop,
        "seconds": time.perf_counter() - t0}


# --------------------------------------------------------------------------

def load_instance(crop=CROP, t_periods=T_PERIODS):
    from mine_problem import load_static
    static = load_static(t_periods=t_periods, crop=crop)
    n = static["n"]
    par, chi = np.asarray(static["par"]), np.asarray(static["chi"])
    tau = ct.occupancy(static["tonnage"], static["tonnage"].sum() / t_periods)
    return {"static": static, "n": n, "par": par, "chi": chi, "tau": tau,
            "value": static["value"], "scale": ct.npv_scale(static["value"]),
            "adj": adjacency(n, par, chi),
            "order": topological_order(n, par, chi)}


def main(generations=GENERATIONS, population=POPULATION, crop=CROP):
    rng = np.random.default_rng(SEED)
    P = load_instance(crop=crop)
    n, tau, value, scale, adj = (P["n"], P["tau"], P["value"], P["scale"],
                                 P["adj"])

    def sc(seq):
        return float(ct.npv(ct.start_times_from_order(seq, tau), tau, value,
                            discount=DISCOUNT, scale=scale))

    base = sc(P["order"])
    ceil_ = float(ct.npv(ct.start_times(value / tau, tau, value=value), tau,
                         value, discount=DISCOUNT, scale=scale))
    print(f"crop-{crop}: n={n} blocks, {len(P['par'])} edges, horizon "
          f"{tau.sum():.2f} periods, scaled by sum|v|={scale:,.0f}")
    print(f"  numba {'ON' if HAVE_NUMBA else 'OFF'}")
    print(f"  topological baseline       {base:+.5f}")
    print(f"  value-density upper bound  {ceil_:+.5f}   (relaxes precedence)\n")

    ref = np.stack([random_feasible(adj, rng) for _ in range(population)])
    u, d = diversity(ref, n)
    print(f"  reference, {population} independent random feasible schedules:"
          f" uniq {u}/{population}  pos_std {d:.4f}\n")

    best, fit, info = run_ga([P["order"]], tau, value, scale, adj, n, rng,
                             generations, population, "ga")
    assert sequence_violations(best, P["par"], P["chi"]) == 0
    print(f"\n  {info['distinct']:,} distinct of {info['evaluations']:,} "
          f"evaluations ({info['distinct'] / info['evaluations'] * 100:.1f}% new)")
    print(f"  {info['seconds'] / generations * 1000:.0f} ms per generation")
    print(f"  best {fit:+.5f}  =  {(fit - base) / abs(base) * 100:+.1f}% over "
          f"topological,  {(ceil_ - fit) / abs(ceil_) * 100:.1f}% below the bound")
    print("  0 precedence violations (asserted)")
    return best, fit, info


if __name__ == "__main__":
    a = sys.argv[1:]
    main(generations=int(a[0]) if len(a) > 0 else GENERATIONS,
         population=int(a[1]) if len(a) > 1 else POPULATION)
