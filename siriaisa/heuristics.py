# Version: v0.0.8
# Created on 31/10/2026
# Author: Frank Vega

"""
Linear-time base heuristics of the MIDS ensemble, adapted from the Hvala/Salvador
vertex-cover strategies and from Esperanza's baseline. Every routine works on the
index graph ``adj`` and returns a seed or an independent dominating set as a
membership list; the ensemble repairs and completes every seed with
:func:`siriaisa.local_search.complete_ids`.

* :func:`matching_seed` (c2, Hvala's maximal-matching strategy): the endpoints of
  the NetworkX minimum maximal matching. Every vertex outside the matching has all
  its neighbours matched, so the endpoints (with the isolated vertices) dominate
  the graph; the repair keeps one endpoint of every matched edge.
* :func:`min_to_min_ids` (c5, Min-to-Min): the undominated vertex of smallest
  residual degree is dominated by the free vertex of its closed neighbourhood that
  dominates the most undominated vertices. Bucket queue, ``O(n + m)``.
* :func:`primal_dual_ids` (c6, primal-dual / local ratio): for the domination LP
  ``min sum x, sum_{u in N[v]} x_u >= 1``, the undominated vertices are scanned by
  ascending degree; the dual of ``v`` is raised until a free vertex of ``N[v]``
  becomes tight (its slack drops to 0), and that vertex (largest degree among the
  tight ones) is selected. ``O(n + m)``.
* :func:`caro_wei_ids` (c12, Esperanza's Caro--Wei baseline): the minimum-degree
  greedy maximal independent set, ``O(n + m)``.
"""

from __future__ import annotations

from collections import deque

import networkx as nx


def matching_seed(adj: list[list[int]]) -> list[int]:
    n = len(adj)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from((v, y) for v in range(n) for y in adj[v] if v < y)
    seed = [v for v in range(n) if not adj[v]]
    for u, v in nx.approximation.min_maximal_matching(G):
        seed += [u, v]
    return seed


def min_to_min_ids(adj: list[list[int]]) -> list[bool]:
    n = len(adj)
    in_s = [False] * n
    dominated = [False] * n
    gain = [1 + len(a) for a in adj]  # undominated vertices in the closed neighbourhood
    rdeg = [len(a) for a in adj]      # undominated neighbours (for the min-degree choice)
    max_deg = max(rdeg, default=0)
    buckets: list[deque] = [deque() for _ in range(max_deg + 1)]
    for v in range(n):
        buckets[rdeg[v]].append(v)
    ptr = 0
    while ptr <= max_deg:
        b = buckets[ptr]
        if not b:
            ptr += 1
            continue
        v = b.popleft()
        if dominated[v] or rdeg[v] != ptr:
            continue
        x = v
        for u in adj[v]:
            if not dominated[u] and gain[u] > gain[x]:
                x = u
        in_s[x] = True
        newly = [x] + [y for y in adj[x] if not dominated[y]]
        for w in newly:
            dominated[w] = True
        for w in newly:
            for z in adj[w]:
                if not dominated[z]:
                    gain[z] -= 1
                    rdeg[z] -= 1
                    buckets[rdeg[z]].append(z)
                    if rdeg[z] < ptr:
                        ptr = rdeg[z]
    return in_s


def primal_dual_ids(adj: list[list[int]]) -> list[bool]:
    n = len(adj)
    in_s = [False] * n
    dominated = [False] * n
    slack = [1.0] * n
    max_deg = max((len(a) for a in adj), default=0)
    buckets: list[list[int]] = [[] for _ in range(max_deg + 1)]
    for v in range(n):
        buckets[len(adj[v])].append(v)
    for b in buckets:
        for v in b:
            if dominated[v]:
                continue
            closed = [v] + adj[v]
            delta = min(slack[u] for u in closed if not dominated[u])
            for u in closed:
                slack[u] -= delta
            x = -1
            for u in closed:
                if not dominated[u] and slack[u] <= 1e-12 and (x == -1 or len(adj[u]) > len(adj[x])):
                    x = u
            in_s[x] = True
            dominated[x] = True
            for y in adj[x]:
                dominated[y] = True
    return in_s


def caro_wei_ids(adj: list[list[int]]) -> list[bool]:
    """Dynamic minimum-degree greedy (Esperanza's Caro--Wei baseline): take the vertex of
    smallest residual degree, delete its closed neighbourhood. ``O(n + m)``."""
    n = len(adj)
    deg = [len(a) for a in adj]
    max_deg = max(deg, default=0)
    buckets: list[deque] = [deque() for _ in range(max_deg + 1)]
    for v in range(n):
        buckets[deg[v]].append(v)
    removed = [False] * n
    in_s = [False] * n
    ptr = 0
    while ptr <= max_deg:
        b = buckets[ptr]
        if not b:
            ptr += 1
            continue
        v = b.popleft()
        if removed[v] or deg[v] != ptr:
            continue
        in_s[v] = True
        removed[v] = True
        gone = [u for u in adj[v] if not removed[u]]
        for u in gone:
            removed[u] = True
        for u in gone:
            for w in adj[u]:
                if not removed[w]:
                    deg[w] -= 1
                    buckets[deg[w]].append(w)
                    if deg[w] < ptr:
                        ptr = deg[w]
    return in_s
