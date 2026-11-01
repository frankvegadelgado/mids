# Version: v0.0.8
# Created on 31/10/2026
# Author: Frank Vega

"""
Repair, completion and budgeted local search for independent dominating sets.

An independent dominating set is exactly a maximal independent set, so every
routine here keeps (or produces) a maximal independent set of the index graph
``adj`` (``adj[v]`` = list of neighbours of ``v``, no self-loops).

:func:`complete_ids` -- the "prune" step applied to every candidate of the
ensemble (the analogue of Salvador's ``prune_redundant_vertices`` and of the
``maximize_solution`` pass shared by Salvador and Esperanza):

1. *Repair*: the seed is made independent; seed vertices are visited by
   decreasing number of seed neighbours (ties: smaller degree first) and a
   vertex that still has a seed neighbour is discarded.
2. *Complete*: while some vertex is undominated, the undominated vertex with
   the most undominated vertices in its closed neighbourhood is inserted
   (bucket queue; gains only decrease). An undominated vertex has no
   neighbour in the set, so every insertion keeps the set independent.

Both phases cost ``O(n + m)``.

:func:`improve_ids` -- the "refine" step (the analogue of Esperanza's local
restarts and local search, turned around to *shrink* the set). A
*move* at a vertex ``u`` outside the set ``S`` inserts ``u``, evicts its
``k = |N(u) & S|`` neighbours in ``S`` and refills the vertices left
undominated (they all lie in the neighbourhoods of the evicted vertices) with
the greedy completion restricted to them. The set changes by
``1 - k + refill``; the move is committed when it shrinks the set.

* *Descent*: moves at vertices with ``k >= 2`` (a move with ``k = 1`` can never
  shrink the set) until no move improves; committed moves requeue the
  vertices around them, so improvements are chained.
* *Weighted swap search* (:mod:`siriaisa.weighted_search`): from the descent
  result, a search over sets of a fixed size ``k`` that may leave vertices
  undominated and contain conflicting edges, both weighted, with swap moves
  (remove the best of sampled set vertices, add the best vertex next to a
  random undominated vertex, under configuration checking) and weights that
  grow on persistent violations; every violation-free set found is an
  independent dominating set of size ``k``, after which ``k - 1`` is tried
  (NuMVC-style). A last descent is run on its result.

Linear-time guarantee: every adjacency scan, insertion, removal, undo step and
sort is charged to a work counter; a move is started only when its cost bound
fits in the remaining budget, and the search stops at
``max(budget * (n + m), floor)`` units of work, where ``floor`` is a fixed
constant (so small graphs get a useful search). For fixed budgets the routine
runs in ``O(n + m)`` time. It never returns a larger set than its input.
"""

from __future__ import annotations

from collections import deque
from .weighted_search import weighted_swap_search

#: Descent + weighted swap search budget, in units of (n + m), used for every candidate.
DEFAULT_REFINE_BUDGET: float = 5.0
#: Weighted swap search budget, in units of (n + m), for the final search on the best candidate.
DEFAULT_ILS_BUDGET: float = 50.0
#: Constant work floor of each search (independent of the graph size).
DEFAULT_WORK_FLOOR: int = 2_000_000
#: Seed of the weighted swap search.
DEFAULT_SEED: int = 20261031


# ---------------------------------------------------------------------------
# Repair + greedy completion
# ---------------------------------------------------------------------------


def _greedy_fill(adj: list[list[int]], in_s: list[bool], dominated: list[bool],
                 candidates: list[int]) -> list[int]:
    """Insert undominated vertices of ``candidates`` (every undominated vertex that some
    candidate can dominate must itself be a candidate) by decreasing number of undominated
    vertices in the closed neighbourhood. Updates ``in_s`` and ``dominated``; returns the
    inserted vertices. ``O(sum of degrees of the candidates)``."""
    gain: dict[int, int] = {}
    for v in candidates:
        if not dominated[v] and v not in gain:
            gain[v] = 1 + sum(1 for y in adj[v] if not dominated[y])
    if not gain:
        return []
    top = max(gain.values())
    buckets: list[list[int]] = [[] for _ in range(top + 1)]
    for v, g in gain.items():
        buckets[g].append(v)
    added: list[int] = []
    ptr = top
    while ptr > 0:
        b = buckets[ptr]
        if not b:
            ptr -= 1
            continue
        v = b.pop()
        if dominated[v] or gain[v] != ptr:
            continue
        in_s[v] = True
        added.append(v)
        newly = [v] + [y for y in adj[v] if not dominated[y]]
        for w in newly:
            dominated[w] = True
        for w in newly:
            for z in adj[w]:
                if not dominated[z]:
                    gain[z] -= 1
                    buckets[gain[z]].append(z)
    return added


def _bucket(items: list[int], key) -> list[int]:
    """Stable counting sort of ``items`` by the non-negative integer ``key``."""
    if not items:
        return []
    keys = [key(v) for v in items]
    buckets: list[list[int]] = [[] for _ in range(max(keys) + 1)]
    for v, k in zip(items, keys):
        buckets[k].append(v)
    return [v for b in buckets for v in b]


def complete_ids(adj: list[list[int]], seed, priority=None) -> list[bool]:
    """Repair ``seed`` into an independent set and complete it greedily into an
    independent dominating (maximal independent) set. Returns the membership list.
    In the repair, a seed vertex with more seed neighbours is discarded first; ties are
    broken by ``priority`` (smaller first, when given) and then by degree (smaller first).
    ``priority`` values must be non-negative integers. ``O(n + m + max priority)``."""
    n = len(adj)
    in_s = [False] * n
    for v in seed:
        in_s[v] = True
    members = [v for v in range(n) if in_s[v]]
    conf = {v: sum(1 for y in adj[v] if in_s[y]) for v in members}
    conflicted = [v for v in members if conf[v]]
    if conflicted:
        # Stable counting sorts (all keys are integers): degree ascending, then priority
        # ascending, then conflicts descending.
        order = _bucket(conflicted, lambda v: len(adj[v]))
        if priority is not None:
            order = _bucket(order, lambda v: priority[v])
        max_conf = max(conf[v] for v in conflicted)
        order = _bucket(order, lambda v: max_conf - conf[v])
        for v in order:
            if in_s[v] and conf[v] > 0:
                in_s[v] = False
                for y in adj[v]:
                    if in_s[y]:
                        conf[y] -= 1
    dominated = [False] * n
    for v in range(n):
        if in_s[v]:
            dominated[v] = True
            for y in adj[v]:
                dominated[y] = True
    _greedy_fill(adj, in_s, dominated, [v for v in range(n) if not dominated[v]])
    return in_s


# ---------------------------------------------------------------------------
# Budgeted local search
# ---------------------------------------------------------------------------


class _State:
    """Maximal independent set with tightness counters (number of neighbours in the set),
    a work counter and an undo log."""

    __slots__ = ("adj", "in_s", "tight", "size", "work", "log")

    def __init__(self, adj: list[list[int]], in_s: list[bool]):
        self.adj = adj
        self.in_s = list(in_s)
        self.tight = [0] * len(adj)
        self.size = 0
        self.work = 0
        self.log: list[tuple[bool, int]] = []
        for v in range(len(adj)):
            if self.in_s[v]:
                self.size += 1
                for y in adj[v]:
                    self.tight[y] += 1
                self.work += len(adj[v]) + 1

    def insert(self, v: int) -> None:
        self.in_s[v] = True
        self.size += 1
        nb = self.adj[v]
        self.work += len(nb) + 1
        for y in nb:
            self.tight[y] += 1
        self.log.append((True, v))

    def remove(self, v: int) -> None:
        self.in_s[v] = False
        self.size -= 1
        nb = self.adj[v]
        self.work += len(nb) + 1
        for y in nb:
            self.tight[y] -= 1
        self.log.append((False, v))

    def undo_to(self, mark: int) -> None:
        log = self.log
        while len(log) > mark:
            added, v = log.pop()
            if added:
                self.remove(v)
            else:
                self.insert(v)
            log.pop()  # drop the entry written by the inverse operation


def _move_bound(st: _State, u: int) -> int:
    """Cost bound of a move at ``u`` (evictions, refill scan, refill gains and sort, undo)."""
    adj = st.adj
    two_hop = 0
    for x in adj[u]:
        if st.in_s[x]:
            two_hop += len(adj[x])
    return 4 * (two_hop + len(adj[u]) + 1) + max((len(adj[x]) for x in adj[u]), default=0)


def _move(st: _State, u: int, stamp: list[int], clock: int) -> tuple[list[int], list[int]]:
    """Insert ``u``, evict its neighbours in the set, refill the vertices left undominated.
    Returns ``(evicted, added)``. The set stays maximal independent."""
    adj, in_s, tight = st.adj, st.in_s, st.tight
    evicted = [x for x in adj[u] if in_s[x]]
    st.work += len(adj[u])
    for x in evicted:
        st.remove(x)
    st.insert(u)
    cand: list[int] = []
    for x in evicted:
        st.work += len(adj[x])
        for w in adj[x]:
            if stamp[w] != clock and not in_s[w] and tight[w] == 0:
                stamp[w] = clock
                cand.append(w)
    if not cand:
        return evicted, []
    # Greedy refill restricted to the undominated vertices (they can only be dominated by
    # one another, since none of them has a neighbour in the set).
    gain: dict[int, int] = {}
    for w in cand:
        st.work += len(adj[w])
        gain[w] = 1 + sum(1 for y in adj[w] if stamp[y] == clock)
    # Bucket queue by gain; inside a bucket, smaller degree popped first (counting sort by
    # degree, descending, so that the last pushed -- popped first -- has the smallest degree).
    max_deg = max(len(adj[w]) for w in cand)
    st.work += len(cand) + max_deg + 1
    top = max(gain.values())
    buckets: list[list[int]] = [[] for _ in range(top + 1)]
    for w in _bucket(cand, lambda x: max_deg - len(adj[x])):
        buckets[gain[w]].append(w)
    alive = set(cand)
    added: list[int] = []
    ptr = top
    while ptr > 0:
        b = buckets[ptr]
        if not b:
            ptr -= 1
            continue
        v = b.pop()
        st.work += 1
        if v not in alive or gain[v] != ptr:
            continue
        st.insert(v)
        added.append(v)
        gone = [v] + [y for y in adj[v] if y in alive]
        for w in gone:
            alive.discard(w)
        for w in gone:
            st.work += len(adj[w])
            for z in adj[w]:
                if z in alive:
                    gain[z] -= 1
                    buckets[gain[z]].append(z)
    return evicted, added


def _descent(st: _State, queue: deque, queued: list[bool], limit: float,
             stamp: list[int], clock: list[int]) -> None:
    """Commit every move that shrinks the set, chaining from the queue, within ``limit``."""
    adj, in_s, tight = st.adj, st.in_s, st.tight
    while queue and st.work <= limit:
        u = queue.popleft()
        queued[u] = False
        st.work += 1
        if in_s[u] or tight[u] < 2:
            continue
        bound = _move_bound(st, u)
        if st.work + bound > limit:
            continue
        mark = len(st.log)
        before = st.size
        clock[0] += 1
        evicted, added = _move(st, u, stamp, clock[0])
        if st.size < before:
            for v in [u, *evicted, *added]:
                st.work += len(adj[v])
                for w in adj[v]:
                    if not in_s[w] and not queued[w] and tight[w] >= 2:
                        queued[w] = True
                        queue.append(w)
        else:
            st.undo_to(mark)


def improve_ids(adj: list[list[int]], in_s: list[bool], budget: float = DEFAULT_REFINE_BUDGET,
                seed: int = DEFAULT_SEED, floor: int = DEFAULT_WORK_FLOOR,
                ils: bool = True) -> list[bool]:
    """Budgeted descent (+ the weighted swap search when ``ils``) from the maximal independent
    set ``in_s``. Returns a maximal independent set no larger than ``in_s``.
    ``O(max(budget * (n + m), floor))``."""
    n = len(adj)
    if n == 0:
        return []
    m = sum(len(a) for a in adj) // 2
    st = _State(adj, in_s)
    start_work = st.work
    limit = st.work + max(budget * (n + m), floor)
    stamp = [0] * n
    clock = [0]
    queued = [False] * n
    queue: deque = deque()
    for v in range(n):
        if not st.in_s[v] and st.tight[v] >= 2:
            queued[v] = True
            queue.append(v)
    st.work += n
    _descent(st, queue, queued, limit, stamp, clock)
    st.log.clear()
    best = list(st.in_s)
    best_size = st.size
    st.work += n
    if not ils or m == 0:
        return best

    # Weighted swap search (siriaisa.weighted_search) from the descent result, then a last
    # descent on its result.
    rest = max(budget * (n + m), floor) - (st.work - start_work)
    found = weighted_swap_search(adj, best, max(rest, 0) / (n + m), seed, int(max(rest, 0)))
    if sum(found) < best_size:
        st2 = _State(adj, found)
        lim2 = st2.work + max(n + m, floor // 10)
        q2: deque = deque(v for v in range(n) if not st2.in_s[v] and st2.tight[v] >= 2)
        qd = [False] * n
        for v in q2:
            qd[v] = True
        _descent(st2, q2, qd, lim2, stamp, clock)
        best = list(st2.in_s)
    return best
