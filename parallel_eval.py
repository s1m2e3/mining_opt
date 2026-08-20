"""
Process-pool evaluation of the knob -> NPV objective.

Processes, not threads: the POCS sweep is the 90% of an evaluation that matters
and its numba kernels are compiled with cache=True but without nogil=True, so
they hold the GIL for their whole run. Threads would serialise exactly the part
worth parallelising. cache=True does pay off here though -- each worker loads
the compiled kernel from disk instead of recompiling it.

The instances are built once per worker by the pool initialiser and kept in a
module global. That matters more than it looks: `static` carries the precedence
edges, the topological order, the CSR adjacency and the cone caches, which is
tens of megabytes, and shipping it with every task would cost more in pickling
than the evaluation itself. Tasks send a knob dict and receive a handful of
floats.
"""

import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import numpy as np

from mine_problem import POCS_TOL, evaluate, load_static

_STATE = None

# One thread per worker. Without this each worker's BLAS and numba runtime
# start their own thread pools, so N workers ask for N x cores threads and
# spend their time context-switching: measured 4 workers at 1.3x rather than
# the ~4x the work is worth. Set in the parent before the pool is created so
# spawned children inherit it -- setting it inside the child is too late, the
# runtimes read it at import.
_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS")


def _init_worker(specs, transform, decoder, pocs_tol):
    """Build every instance once in this worker and cache the baselines."""
    global _STATE
    insts = []
    for name, path, crop in specs:
        S = load_static(path, crop=crop)
        base = evaluate(S, {}, transform=transform, decoder=decoder,
                        pocs_tol=pocs_tol)["npv"]
        insts.append((name, S, base))
    _STATE = (insts, transform, decoder, pocs_tol)


def _eval_one(knobs):
    insts, transform, decoder, pocs_tol = _STATE
    out = {}
    for name, S, base in insts:
        r = evaluate(S, knobs, transform=transform, decoder=decoder,
                     check=False, pocs_tol=pocs_tol)
        out[name] = (r["npv"], r["npv"] / base - 1.0)
    return out


class ParallelObjective:
    """Same contract as bo_search.Objective, plus `batch`.

    Falls back to in-process evaluation when workers <= 1, so the serial path
    stays available for debugging without a second code path to keep in sync.
    """

    def __init__(self, specs, transform="pocs", decoder="kahn",
                 pocs_tol=POCS_TOL, workers=None):
        self.specs = list(specs)
        self.transform = transform
        self.decoder = decoder
        self.pocs_tol = pocs_tol
        self.n_calls = 0

        # the parent needs the instances too, for baselines and for scoring the
        # final winner outside the pool
        self.instances = []
        self.baseline = {}
        for name, path, crop in self.specs:
            S = load_static(path, crop=crop)
            r = evaluate(S, {}, transform=transform, decoder=decoder,
                         pocs_tol=pocs_tol)
            if r["violations"] != 0:
                raise SystemExit(f"{name}: baseline has {r['violations']} violations")
            self.instances.append((name, S))
            self.baseline[name] = r["npv"]

        # Four by default, regardless of core count. POCS streams Gram rows and
        # is memory-bandwidth bound, not compute bound, so throughput peaks
        # early and then falls over. Measured evals/min on 12 cores:
        #
        #   workers        1     2     4     6
        #   crop14       114   226   197   165
        #   full n=12213  12    19    26    20
        #
        # Past the peak the workers contend for bandwidth, and every worker
        # also holds its own edges, CSR, cone caches and Gram -- all 12 died
        # with MemoryError on the full model. Override with --workers.
        w = 4 if workers is None else int(workers)
        self.workers = max(1, min(w, 8))
        self.pool = None
        if self.workers > 1:
            for v in _THREAD_VARS:
                os.environ.setdefault(v, "1")
            self.pool = ProcessPoolExecutor(
                max_workers=self.workers, initializer=_init_worker,
                initargs=(self.specs, transform, decoder, pocs_tol))

    def _score(self, per_instance):
        gains = np.array([g for _, g in per_instance.values()], dtype=float)
        return float(gains.mean()), {
            "npv": {k: v[0] for k, v in per_instance.items()},
            "gain_mean": float(gains.mean()), "gain_worst": float(gains.min())}

    def batch(self, knobs_list):
        """Evaluate q knob settings at once. The whole point of q > 1: one GP
        fit and one acquisition solve are amortised over q evaluations, and the
        q evaluations themselves overlap across cores."""
        self.n_calls += len(knobs_list)
        if self.pool is None:
            results = []
            for knobs in knobs_list:
                per = {}
                for name, S in self.instances:
                    r = evaluate(S, knobs, transform=self.transform,
                                 decoder=self.decoder, check=False,
                                 pocs_tol=self.pocs_tol)
                    per[name] = (r["npv"], r["npv"] / self.baseline[name] - 1.0)
                results.append(per)
        else:
            try:
                results = list(self.pool.map(_eval_one, knobs_list))
            except (BrokenProcessPool, MemoryError, OSError) as e:
                # A worker died -- almost always memory. Losing a multi-hour
                # search to one dead child is worse than finishing it slowly,
                # so drop to serial and carry on rather than raising.
                print(f"  [pool failed: {type(e).__name__}: {e} -- "
                      f"falling back to serial evaluation]", flush=True)
                self.close()
                self.n_calls -= len(knobs_list)
                return self.batch(knobs_list)
        return [self._score(p) for p in results]

    def __call__(self, knobs):
        return self.batch([knobs])[0]

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True)
            self.pool = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
