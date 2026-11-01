# Version: v0.0.8
# Created on 31/10/2026
# Author: Frank Vega

"""Consensus forcing for MIDS (candidate c17, Salvador's c16 adapted).

The strategies c1..c16 are different heuristics, and a vertex that all of them (or
most of them) select is likely to belong to a minimum independent dominating set.
c17 *forces* such vertices: round ``k`` takes the vertices selected by at least a
fraction ``q_k`` of the sets in the pool (``q = 1, 0.9, 0.75``), keeps an independent
subset of them (most votes first), deletes them and their neighbours from the graph
(those neighbours are dominated), and solves the residual graph -- the vertices with
no forced vertex in their closed neighbourhood -- again with the inner solver. The
forced vertices together with the residual solution form an independent dominating
set of the graph (the residual vertices are not adjacent to any forced vertex), which
joins the pool. The rounds stop when a round would force the same set again. The
smallest set is returned.

Nothing guarantees that the forced vertices lie in a minimum solution, so c17 is a
heuristic; the ensemble keeps the smallest candidate, so it never makes the result
worse. Cost: ``O(rounds * (n + m))`` plus the inner solver.
"""

from __future__ import annotations

import math

from .local_search import _bucket, complete_ids

DEFAULT_ROUNDS = 3
QUOTAS = (1.0, 0.9, 0.75)


def consensus_force_ids(adj: list[list[int]], pool: list[list[bool]], inner,
                        rounds: int = DEFAULT_ROUNDS, info: dict | None = None) -> list[bool]:
    """Smallest independent dominating set found by forcing the consensus vertices of ``pool``
    and re-solving the rest with ``inner(sub_adj) -> membership list``."""
    n = len(adj)
    pool = [list(p) for p in pool]
    best = min(pool, key=sum)
    previous: set[frozenset] = set()
    rounds_run = 0
    forced_last = 0
    for k in range(min(rounds, len(QUOTAS))):
        votes = [0] * n
        for p in pool:
            for v in range(n):
                if p[v]:
                    votes[v] += 1
        need = math.ceil(QUOTAS[k] * len(pool) - 1e-9)
        cand = [v for v in range(n) if votes[v] >= need]
        # Independent subset of the consensus vertices, most votes first (repair only, no completion).
        chosen = [False] * n
        # Most votes first, ties by larger degree (two stable counting sorts).
        max_deg = max((len(adj[v]) for v in cand), default=0)
        order = _bucket(cand, lambda x: max_deg - len(adj[x]))
        order = _bucket(order, lambda x: len(pool) - votes[x])
        for v in order:
            if not any(chosen[y] for y in adj[v]):
                chosen[v] = True
        key = frozenset(v for v in range(n) if chosen[v])
        if not key or key in previous:
            break
        previous.add(key)
        rounds_run += 1
        forced_last = len(key)
        blocked = [False] * n
        for v in key:
            blocked[v] = True
            for y in adj[v]:
                blocked[y] = True
        rest = [v for v in range(n) if not blocked[v]]
        index = {v: i for i, v in enumerate(rest)}
        sub = [[index[y] for y in adj[v] if y in index] for v in rest]
        sol = inner(sub) if rest else []
        seed = list(key) + [rest[i] for i in range(len(rest)) if sol[i]]
        result = complete_ids(adj, seed)  # already valid; the pass only checks it
        pool.append(result)
        if sum(result) < sum(best):
            best = result
    if info is not None:
        info.update({"rounds_run": rounds_run, "forced": forced_last})
    return best
