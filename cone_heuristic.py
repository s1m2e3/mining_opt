"""A DYNAMIC cone priority: the look-ahead is recomputed as the pit empties.

The static descendant-cone-efficiency key is strong -- it beat a 5-second GA on
three of five crops -- but it cannot express one thing, and the thing it cannot
express is timing. It scores a block identically whether that block is mined in
period 1 or period 8, when unlocking the same value at period 8 is worth
gamma^8 ~ 43% as much.

Two static attempts to patch that both made it WORSE on all five crops
(discounting by access time, and charging access time in the denominator). The
reason is that by the moment the decoder actually considers a block, all of its
ancestors are already mined, so the access cost is SUNK -- charging for it
reintroduces exactly the error that made the ancestor-side key worse than plain
v/tau. The sunk cost and the timing effect pull opposite ways, and no per-block
scalar holds both.

What does hold both is a key evaluated DURING the decode, when the clock is
known and the descendant set has already shrunk:

    score_j = [ v_j psi(tau_j) + alpha V_D psi(T_D) exp(-delta tau_j) ]
              / (tau_j + T_D)

with V_D, T_D the value and time of j's still-UNMINED descendants. It is value
per period for the whole package -- the block plus what mining it unlocks --
measured from now. With no descendants left it collapses to v_j / tau_j, the
exact exchange-argument optimum, so the two regimes agree at the boundary.

Note the discount factor common to the whole ready set cancels out of any
within-step comparison; what survives, and what makes this dynamic rather than
static, is that each candidate defers its own look-ahead by its own tau_j and
carries its own shrinking V_D.

Cost is O(n^2): the descendant totals are maintained by decrementing every
ancestor of each mined block, using a dense boolean reachability matrix. 1 MB
and milliseconds at n=1035; 149 MB and seconds at n=12213.
"""

import numpy as np

import continuous_time as ct

DISCOUNT = 0.90
T_PERIODS = 10


def reachability(n, par, chi, order):
    """Dense boolean ancestor and descendant matrices, excluding the diagonal."""
    par, chi = np.asarray(par, np.int64), np.asarray(chi, np.int64)
    pre = [[] for _ in range(n)]
    suc = [[] for _ in range(n)]
    for a, b in zip(par.tolist(), chi.tolist()):
        pre[b].append(a)
        suc[a].append(b)
    anc = np.zeros((n, n), dtype=bool)
    for j in order:
        for a in pre[j]:
            anc[j] |= anc[a]
            anc[j, a] = True
    des = np.zeros((n, n), dtype=bool)
    for j in order[::-1]:
        for b in suc[j]:
            des[j] |= des[b]
            des[j, b] = True
    return anc, des, pre, suc


def dynamic_cone_schedule(n, par, chi, order, tonnage, value, capacity,
                          discount=DISCOUNT, alpha=1.0, anc=None, des=None,
                          suc=None):
    """Greedy decode under the dynamic cone rate. Feasible by construction.

    Only ready blocks are ever candidates -- the ready set is what enforces
    precedence, exactly as in decoders.schedule_priority_kahn -- so the result
    needs no repair and no projection.
    """
    w = np.asarray(tonnage, float)
    v = np.asarray(value, float)
    tau = w / float(capacity)
    d = ct.delta_from_discount(discount)
    psi = ct.within_block_shape(tau, d)

    if anc is None or des is None or suc is None:
        anc, des, _, suc = reachability(n, par, chi, order)

    # still-unmined descendant totals, maintained incrementally
    VD = des @ v
    WD = des @ w
    indeg = np.zeros(n, np.int64)
    for a, b in zip(np.asarray(par).tolist(), np.asarray(chi).tolist()):
        indeg[b] += 1
    ready = indeg == 0
    mined = np.zeros(n, bool)
    out = np.empty(n, np.int64)

    for k in range(n):
        cand = np.flatnonzero(ready & ~mined)
        if cand.size == 0:
            raise ValueError("precedence graph contains a cycle")
        TD = WD[cand] / float(capacity)
        fut = VD[cand] * _psi(TD, d) * np.exp(-d * tau[cand])
        score = (v[cand] * psi[cand] + alpha * fut) / np.maximum(
            tau[cand] + TD, 1e-12)
        b = cand[int(np.argmax(score))]
        out[k] = b
        mined[b] = True
        # every ancestor of b loses b from its remaining descendant cone
        VD[anc[b]] -= v[b]
        WD[anc[b]] -= w[b]
        for c in suc[b]:
            indeg[c] -= 1
            if indeg[c] == 0:
                ready[c] = True
    return out


def _psi(t, d):
    """(1 - exp(-d t)) / (d t), the within-interval discount shape, safe at 0."""
    x = d * np.asarray(t, float)
    return np.where(x > 1e-12, -np.expm1(-np.minimum(x, 700.0))
                    / np.maximum(x, 1e-12), 1.0)


if __name__ == "__main__":
    import time

    import ga_schedule as G
    from decoders import children_csr, schedule_priority_kahn
    from kernel_projection import sequence_violations
    import train as T

    print(f"{'crop':>9}{'topo':>9}{'v/tau':>9}{'static':>9}"
          + "".join(f"{'dyn a=' + str(a):>11}" for a in (0.0, 0.5, 1.0, 2.0))
          + f"{'GA 5s':>9}{'sec':>7}")
    for wnd in [(0, 5), (8, 13), (16, 21), (24, 29), (32, 37)]:
        P = G.load_instance(crop=wnd)
        n, tau, value, scale = P["n"], P["tau"], P["value"], P["scale"]
        par, chi = np.asarray(P["par"]), np.asarray(P["chi"])
        w = P["static"]["tonnage"]
        cap = w.sum() / T_PERIODS
        csr = children_csr(n, par, chi)
        tr = np.empty(n, np.int64)
        tr[P["order"]] = np.arange(n)
        keys = G.edge_keys(par, chi, n)

        def sc(q):
            assert sequence_violations(q, par, chi) == 0
            return float(ct.npv(ct.start_times_from_order(q, tau), tau, value,
                                discount=DISCOUNT, scale=scale))

        def dec(k):
            return sc(schedule_priority_kahn(np.asarray(k, float), csr,
                                             tiebreak=tr.astype(float)))

        anc, des, _, suc = reachability(n, par, chi, P["order"])
        row = []
        t0 = time.perf_counter()
        for a in (0.0, 0.5, 1.0, 2.0):
            row.append(sc(dynamic_cone_schedule(
                n, par, chi, P["order"], w, value, cap, alpha=a,
                anc=anc, des=des, suc=suc)))
        dt = (time.perf_counter() - t0) / 4
        rng = np.random.default_rng(0)
        _, fit, _ = G.run_ga([P["order"]], tau, value, scale, P["adj"], n, rng,
                             generations=10 ** 9, population=128, label="",
                             every=10 ** 9, seconds=5.0, quiet=True, keys=keys)
        st = T.efficiency_columns(P["static"])[3]
        print(f"{str(wnd):>9}{dec(-tr.astype(float)):>9.4f}"
              f"{dec(value / tau):>9.4f}{dec(st):>9.4f}"
              + "".join(f"{r:>11.4f}" for r in row)
              + f"{fit:>9.4f}{dt:>7.2f}")
