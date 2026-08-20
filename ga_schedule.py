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


def _mutate(perm, pos, pi, pd, si, sd, n, moves, seg_max, p_segment, buf):
    for _ in range(moves):
        if np.random.random() < p_segment:
            a = np.random.randint(0, n - 1)
            b = a + np.random.randint(2, seg_max + 1)
            if b > n:
                b = n
            lo, hi = _segment_window(perm, pos, pi, pd, si, sd, a, b, n)
            if hi > lo:
                _move_segment(perm, pos, a, b,
                              np.random.randint(lo, hi + 1), buf)
        else:
            blk = np.random.randint(0, n)
            lo, hi = _feasible_window(pos, pi, pd, si, sd, blk, n)
            if hi > lo:
                _shift(perm, pos, blk, np.random.randint(lo, hi + 1))


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
    _mutate = njit(cache=True)(_mutate)
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


def mutate(perm, pos, adj, n, rng, moves, buf):
    _mutate(perm, pos, adj["pi"], adj["pd"], adj["si"], adj["sd"], n,
            int(moves), max(2, int(SEG_MAX * n)), P_SEGMENT, buf)


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
           generations=GENERATIONS, population=POPULATION, label="", every=10):
    """(mu + lambda) GA. Returns the best sequence, its fitness and a report."""
    pop = np.empty((population, n), np.int64)
    k = min(len(seed_perms), population)
    pop[:k] = np.asarray(seed_perms[:k])
    # the rest are INDEPENDENT random feasible schedules, never perturbed
    # copies of the seeds -- filling from the seeds collapses diversity, and
    # made an earlier warm start look worse than a cold one purely as an
    # artefact of its own initialisation
    for i in range(k, population):
        pop[i] = random_feasible(adj, rng)

    fit = evaluate(pop, tau, value, scale)
    n_elite = max(1, int(ELITE * population))
    n_imm = int(IMMIGRANT * population)
    buf = np.empty(n, np.int64)
    seen = set(map(bytes, pop))
    trace = [float(fit.max())]
    t0 = time.perf_counter()

    def report(g):
        u, d = diversity(pop, n)
        print(f"    {label} gen {g:>4}  best {fit.max():+.5f}"
              f"  mean {fit.mean():+.5f}  sd {fit.std():.2e}"
              f" | uniq {u:>4}/{population}  pos_std {d:.4f}"
              f"  seen {len(seen):>7}  ({time.perf_counter() - t0:6.1f}s)")

    report(0)
    for g in range(1, generations + 1):
        elite_idx = np.argpartition(-fit, n_elite - 1)[:n_elite]
        new_pop = np.empty_like(pop)
        new_pop[:n_elite] = pop[elite_idx]
        live = set(map(bytes, new_pop[:n_elite]))
        i = n_elite

        # a standing trickle of unrelated feasible schedules, so the pool can
        # never become the descendants of one lucky individual
        while i < min(n_elite + n_imm, population):
            new_pop[i] = random_feasible(adj, rng)
            live.add(bytes(new_pop[i]))
            i += 1

        while i < population:
            cand = rng.integers(0, population, TOURNAMENT)
            a = cand[np.argmax(fit[cand])]
            if rng.random() < P_CROSSOVER:
                cand = rng.integers(0, population, TOURNAMENT)
                b = cand[np.argmax(fit[cand])]
                child = crossover(pop[a], pop[b], adj, rng, n)
            else:
                child = pop[a].copy()
            q = np.empty(n, np.int64)
            q[child] = np.arange(n)
            mutate(child, q, adj, n, rng, mutation_strength(rng), buf)
            if NO_DUPLICATES:
                # an exact repeat carries no information and eats a slot
                for _ in range(4):
                    if bytes(child) not in live:
                        break
                    mutate(child, q, adj, n, rng,
                           mutation_strength(rng) * 2, buf)
            live.add(bytes(child))
            new_pop[i] = child
            i += 1

        pop = new_pop
        fit = evaluate(pop, tau, value, scale)
        trace.append(float(fit.max()))
        seen.update(map(bytes, pop))
        if g % every == 0 or g == generations:
            report(g)

    u, d = diversity(pop, n)
    return pop[int(np.argmax(fit))], float(fit.max()), {
        "trace": trace, "distinct": len(seen), "unique": u, "pos_std": d,
        "evaluations": population * (generations + 1),
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
