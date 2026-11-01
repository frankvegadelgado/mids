# Version: v0.0.8
# Created on 31/10/2026
# Author: Frank Vega

"""
Bipartite seeds for the Minimum Independent Dominating Set (MIDS) ensemble.

The routines are the bipartite strategies of Salvador (``salvador.bipartite_peel``,
strategy c9) adapted to independent domination. They work on an index graph:
``adj[v]`` is the list of neighbours of vertex ``v`` (``0 <= v < n``, no self-loops).

* :func:`two_colouring` -- breadth-first 2-colouring in which every vertex takes
  the colour that disagrees with most of its already coloured neighbours (ties:
  the colour opposite to its parent). On a bipartite component it is a proper
  2-colouring (its bipartition); on any component at least half of the edges
  join different colours. ``O(n + m)``.
* :func:`konig_cover` -- vertex cover of the bipartite layer formed by the
  edges whose endpoints have different colours: the NetworkX minimum maximal
  matching enlarged by at most ``phases`` Hopcroft--Karp phases, then Koenig's
  construction with each colour class as the top side, the smaller cover kept.
  When no augmenting path is left it is a minimum vertex cover of the layer
  (Koenig). ``O(phases * (n + m))``.

On a connected bipartite graph with sides ``A`` and ``B`` (at least one edge),
every independent dominating set ``S`` satisfies ``S & B = B - N(S & A)``, and
each side is itself an independent dominating set. The seeds built from these
routines (a colour class, a Koenig cover, the part of a Koenig cover on one
side) are turned into independent dominating sets by
:func:`siriaisa.local_search.complete_ids`.
"""

from __future__ import annotations

from collections import deque

import networkx as nx

#: Hopcroft--Karp phases after the NetworkX minimum maximal matching.
DEFAULT_PHASES: int = 8


def two_colouring(adj: list[list[int]]) -> tuple[list[int], list[int], list[bool]]:
    """Majority 2-colouring of the index graph ``adj``.

    Returns ``(colour, comp, bipartite)``: ``colour[v]`` in {0, 1};
    ``comp[v]`` the index of the connected component of ``v``; ``bipartite[c]``
    True when component ``c`` has no edge inside a colour class (then the
    colouring of ``c`` is its bipartition). ``O(n + m)``.
    """
    n = len(adj)
    colour = [-1] * n
    comp = [-1] * n
    parent = [-1] * n
    bipartite: list[bool] = []
    for s in range(n):
        if comp[s] != -1:
            continue
        c = len(bipartite)
        comp[s] = c
        colour[s] = 0
        q = deque([s])
        while q:
            x = q.popleft()
            if x != s:
                c0 = c1 = 0
                for y in adj[x]:
                    cy = colour[y]
                    if cy == 0:
                        c0 += 1
                    elif cy == 1:
                        c1 += 1
                colour[x] = 1 if c0 > c1 else 0 if c1 > c0 else 1 - colour[parent[x]]
            for y in adj[x]:
                if comp[y] == -1:
                    comp[y] = c
                    parent[y] = x
                    q.append(y)
        bipartite.append(True)
    for v in range(n):
        cv = colour[v]
        for y in adj[v]:
            if colour[y] == cv:
                bipartite[comp[v]] = False
                break
    return colour, comp, bipartite


def _augment_phase(adj: list[list[int]], top: list[int], mate: list[int]) -> bool:
    """One Hopcroft--Karp phase on the layer ``adj`` (Salvador's routine on index lists):
    BFS layers from the free top vertices, then vertex-disjoint shortest augmenting paths
    by iterative DFS. Returns False when no augmenting path exists. ``O(n + m)``."""
    dist: dict[int, int] = {}
    q = deque()
    for x in top:
        if mate[x] == -1:
            dist[x] = 0
            q.append(x)
    found = False
    limit = None
    while q:
        x = q.popleft()
        if limit is not None and dist[x] >= limit:
            continue
        for y in adj[x]:
            z = mate[y]
            if z == -1:
                if limit is None:
                    limit = dist[x] + 1
                found = True
            elif z not in dist:
                dist[z] = dist[x] + 1
                q.append(z)
    if not found:
        return False
    it = {x: iter(adj[x]) for x in dist}
    used: set[int] = set()
    for root in top:
        if mate[root] != -1 or dist.get(root) != 0:
            continue
        stack = [root]
        path_bottom: list[int] = []
        while stack:
            x = stack[-1]
            advanced = False
            for y in it[x]:
                if y in used:
                    continue
                z = mate[y]
                if z == -1:
                    if dist[x] + 1 != limit:
                        continue
                    path_bottom.append(y)
                    for tx, by in zip(stack, path_bottom):
                        mate[tx] = by
                        mate[by] = tx
                    used.update(path_bottom)
                    stack = []
                    advanced = True
                    break
                if dist.get(z) == dist[x] + 1:
                    used.add(y)
                    path_bottom.append(y)
                    stack.append(z)
                    advanced = True
                    break
            if stack and not advanced:
                stack.pop()
                dist[x] = -1
                if path_bottom:
                    path_bottom.pop()
    return True


def konig_cover(adj: list[list[int]], colour: list[int], phases: int = DEFAULT_PHASES,
                info: dict | None = None) -> set[int]:
    """Vertex cover of the bipartite layer of ``adj`` (the edges joining different colours).

    NetworkX minimum maximal matching + at most ``phases`` Hopcroft--Karp phases +
    Koenig's construction with each colour class as the top side; the smaller cover is
    kept. It is a vertex cover of the layer for any matching, and a minimum one when no
    augmenting path is left (``info["exact"]``). ``O(phases * (n + m))``.
    """
    n = len(adj)
    layer = [[y for y in adj[v] if colour[y] != colour[v]] for v in range(n)]
    nodes = [v for v in range(n) if layer[v]]
    if not nodes:
        if info is not None:
            info.update({"exact": True, "matching": 0})
        return set()
    H = nx.Graph()
    H.add_edges_from((v, y) for v in nodes if colour[v] == 0 for y in layer[v])
    mate = [-1] * n
    for u, v in nx.approximation.min_maximal_matching(H):
        mate[u], mate[v] = v, u
    top = [v for v in nodes if colour[v] == 0]
    exact = False
    for _ in range(phases):
        if not _augment_phase(layer, top, mate):
            exact = True
            break
    best: set[int] | None = None
    for s in (0, 1):
        tops = [v for v in nodes if colour[v] == s]
        seen = [False] * n
        q = deque()
        for v in tops:
            if mate[v] == -1:
                seen[v] = True
                q.append(v)
        while q:
            x = q.popleft()
            for y in layer[x]:
                if seen[y] or mate[x] == y:
                    continue
                seen[y] = True
                z = mate[y]
                if z != -1 and not seen[z]:
                    seen[z] = True
                    q.append(z)
        cover = {v for v in tops if not seen[v]} | {v for v in nodes if colour[v] != s and seen[v]}
        if best is None or len(cover) < len(best):
            best = cover
    if info is not None:
        info.update({"exact": exact, "matching": sum(1 for v in top if mate[v] != -1)})
    return best
