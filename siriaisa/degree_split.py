# Version: v0.0.8
# Created on 31/10/2026
# Author: Frank Vega

"""Strategy c13: degree splitting at floor(sqrt(n)).

Esperanza 0.1.6 has the approximation ratio ``2 sqrt(n)`` for the maximum independent
set on every graph with maximum degree ``Delta <= sqrt(n)``. Strategy c13 builds a
*copy graph* ``H`` of ``G`` whose maximum degree is at most ``k = floor(sqrt(n))``,
solves ``H``, and maps the solution back to ``G``.

Copy graph. ``H`` starts as ``G``. The vertices of ``G`` are scanned by descending
degree (counting sort, ties by position). When the current vertex ``u`` has more
than ``k`` neighbours in ``H``, it is replaced by copies ``u_1, u_2, ...``: ``u_1`` is
joined to the first ``k`` neighbours of ``u`` (in its current neighbour list of ``H``),
``u_2`` to the next ``k``, and so on, the last copy getting the remaining at most ``k``
neighbours; the copies of ``u`` are not joined to one another. A split does not
change the degree of any other vertex, so after the scan no vertex of ``H`` has
degree larger than ``k``. Every edge of ``H`` joins two vertices whose originals are
adjacent in ``G``, and ``H`` has ``m`` edges and at most ``n + 2m / k`` vertices.

Solving ``H`` and decoding. ``H`` is solved by the linear-time inner solvers (the
greedy completion, Min-to-Min and the induced-forest layers). From each solution
the original ``u`` is added when some copy ``u_i`` (or ``u`` itself, if it was not
split) is selected. This set is in general not an independent dominating set of
``G`` (two copies of adjacent vertices may both be selected, and a vertex dominated
only through a copy in ``H`` may be dominated in ``G`` anyway or not), so it is
repaired with pruning by :func:`siriaisa.local_search.complete_ids`: conflicting
vertices are discarded, the one with more conflicts first and, among equal
conflicts, the one with fewer selected copies first, and the set is completed
greedily. The smallest repaired set is returned.

Running time: ``O(n + m)`` for ``H`` (every edge is moved at most twice) and
``O(|V(H)| + m)`` for the inner solvers and the repair, so ``O(n + m)``.
"""

from __future__ import annotations

from math import isqrt

from . import forest_layers, heuristics, local_search
from .bipartite import two_colouring


def copy_graph(adj: list[list[int]], k: int) -> tuple[list[list[int]], list[int], int]:
    """The copy graph of ``adj`` at threshold ``k``: ``(hadj, owner, split)`` with ``owner[x]`` the
    original vertex of node ``x`` of ``H`` and ``split`` the number of vertices replaced by copies."""
    n = len(adj)
    hadj: list[set[int]] = [set(a) for a in adj]
    owner = list(range(n))
    max_deg = max((len(a) for a in adj), default=0)
    buckets: list[list[int]] = [[] for _ in range(max_deg + 1)]
    for v in range(n):
        buckets[len(adj[v])].append(v)
    split = 0
    for b in reversed(buckets):
        for u in b:
            if len(hadj[u]) <= k:
                continue
            split += 1
            nbrs = list(hadj[u])
            hadj[u] = set()  # u is replaced by its copies (u keeps no edge)
            for i in range(0, len(nbrs), k):
                c = len(hadj)
                chunk = nbrs[i:i + k]
                hadj.append(set(chunk))
                owner.append(u)
                for w in chunk:
                    hadj[w].discard(u)
                    hadj[w].add(c)
    return [list(s) for s in hadj], owner, split


def degree_split_ids(adj: list[list[int]], info: dict | None = None) -> list[bool]:
    """Strategy c13 on the index graph ``adj``: an independent dominating set (membership list)."""
    n = len(adj)
    k = max(1, isqrt(n))
    hadj, owner, split = copy_graph(adj, k)
    # Nodes of H that are originals replaced by copies are isolated in H; they are removed so
    # that they are not selected (their copies stand for them).
    split_vertex = [False] * n
    for x in range(n, len(hadj)):
        split_vertex[owner[x]] = True
    keep = [x for x in range(len(hadj)) if not (x < n and split_vertex[x])]
    index = {x: i for i, x in enumerate(keep)}
    sub = [[index[y] for y in hadj[x]] for x in keep]
    comp = two_colouring(sub)[1]
    inner = {
        "greedy": local_search.complete_ids(sub, ()),
        "min_to_min": heuristics.min_to_min_ids(sub),
        "forest_layers": forest_layers.forest_layers_ids(sub, comp),
    }
    best = None
    best_key = None
    for key, sol in inner.items():
        count = [0] * n
        for i, x in enumerate(keep):
            if sol[i]:
                count[owner[x]] += 1
        seed = [v for v in range(n) if count[v]]
        repaired = local_search.complete_ids(adj, seed, priority=count)
        if best is None or sum(repaired) < sum(best):
            best, best_key = repaired, key
    if info is not None:
        info.update({"k": k, "split": split, "copies": len(hadj) - n,
                     "h_max_degree": max((len(a) for a in sub), default=0), "inner": best_key})
    return best
