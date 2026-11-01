# Version: v0.0.8
# Created on 31/10/2026
# Author: Frank Vega

"""Strategy c9: induced-forest layers, solved one after another (Salvador's c9 adapted to MIDS).

Salvador's c9 splits every connected component into bipartite layers, solves each
layer optimally (Koenig) and combines the layers in several orders. Here every
connected component is split into *induced forests*, each solved optimally by the
linear-time tree DP for the minimum independent dominating set.

Layers. The vertices of a component are scanned in ascending-degree order
(counting sort) and a vertex joins layer 0 when its neighbours already in layer 0
lie in pairwise different trees of layer 0 (union-find with union by size and
path halving), so that layer 0 stays an
induced forest; the rejected vertices are scanned the same way to build layer 1,
and so on. Every vertex lies in exactly one layer, and every layer induces a forest
of ``G``. The layering has a work budget of ``budget * (n + m)``; the vertices not
placed when it runs out are left to the greedy completion.

Solving an order. The layers are processed in an order; in layer ``t`` the vertices
already dominated (by vertices selected in earlier layers of the order) are
discarded, and the remaining vertices -- exactly the vertices of the layer that
have no neighbour in the set -- induce a forest, on which the minimum independent
dominating set is computed exactly by the tree DP and added to the set. A vertex
selected in a layer is undominated before, so it has no neighbour in the set, and
the forest is induced, so the set stays independent; every vertex of the layer is
dominated afterwards. After the last layer the vertices left undominated (only
those not placed by the layering) are completed greedily, so the result is an
independent dominating set of the component. Different orders give different sets;
every order is tried when there are at most ``max_orders`` of them (``L!`` orders
for ``L`` layers), and otherwise ``max_orders`` orders (the natural order, the
reverse order, then the rotations of both), and the smallest set is kept.

On a component that is a tree there is a single layer, and c9 is optimal on it
(reported as ``exact``). Running time ``O(budget * (n + m) + max_orders * (n + m))``,
linear for the fixed defaults.
"""

from __future__ import annotations

import itertools
from collections import deque

from .local_search import _greedy_fill

#: Work budget of the layering, in units of (n + m).
DEFAULT_LAYER_BUDGET: float = 8.0
#: Largest number of layer orders tried per component.
DEFAULT_MAX_ORDERS: int = 6

_INF = float("inf")


# ---------------------------------------------------------------------------
# Exact MIDS on an induced forest of the index graph (linear-time DP)
# ---------------------------------------------------------------------------


def forest_mids(adj: list[list[int]], verts: list[int], inside: list[int], token: int) -> list[int]:
    """Exact minimum independent dominating set of the forest induced by ``verts``
    (``inside[v] == token`` marks the members). Same three-state DP as
    :func:`siriaisa.algorithm.tree_minimum_independent_dominating_set`, on index lists.
    ``O(sum of degrees of verts)``."""
    result: list[int] = []
    parent: dict[int, int] = {}
    for root in verts:
        if root in parent:
            continue
        parent[root] = -1
        order = [root]
        q = deque([root])
        while q:
            v = q.popleft()
            for u in adj[v]:
                if inside[u] == token and u not in parent:
                    parent[u] = v
                    order.append(u)
                    q.append(u)
        children: dict[int, list[int]] = {v: [] for v in order}
        for v in order[1:]:
            children[parent[v]].append(v)
        dp: dict[int, tuple[float, float, float]] = {}
        forced: dict[int, int] = {}
        for v in reversed(order):
            ch = children[v]
            in_v, free_v, dom_v = 1.0, 0.0, 0.0
            any_in = False
            best_pen, best_c = _INF, -1
            for c in ch:
                ic, dc, fc = dp[c]
                in_v += min(dc, fc)
                free_v += dc
                if ic <= dc:
                    dom_v += ic
                    any_in = True
                else:
                    dom_v += dc
                    pen = ic - dc
                    if pen < best_pen:
                        best_pen, best_c = pen, c
            if not ch:
                dom_v = _INF
                forced[v] = -1
            elif any_in:
                forced[v] = -1
            else:
                dom_v += best_pen
                forced[v] = best_c
            dp[v] = (in_v, dom_v, free_v)
        state = {root: 0 if dp[root][0] <= dp[root][1] else 1}  # 0 IN, 1 DOM, 2 FREE
        for v in order:
            s = state[v]
            if s == 0:
                result.append(v)
                for c in children[v]:
                    _, dc, fc = dp[c]
                    state[c] = 1 if dc <= fc else 2
            elif s == 2:
                for c in children[v]:
                    state[c] = 1
            else:
                fc_forced = forced[v]
                for c in children[v]:
                    ic, dc, _ = dp[c]
                    state[c] = 0 if (c == fc_forced or ic <= dc) else 1
    return result


# ---------------------------------------------------------------------------
# Induced-forest layers of a component
# ---------------------------------------------------------------------------


def _find(parent: list[int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def forest_layers(adj: list[list[int]], comp: list[int], layer_of: list[int], uf: list[int],
                  mark: list[int], work: list[float], limit: float,
                  size: list[int] | None = None) -> tuple[list[list[int]], list[int]]:
    """Layers of the component ``comp`` (each an induced forest) and the vertices left
    unplaced when the work budget runs out. ``layer_of``, ``uf`` and ``mark`` are scratch
    arrays of length n (``layer_of`` initialised to -1, ``mark`` to 0); ``size`` (scratch, length
    n) holds the tree sizes for union by size (with path halving: ``O(alpha(n))`` per operation)."""
    if size is None:
        size = [1] * len(adj)
    max_deg = max(len(adj[v]) for v in comp)
    buckets: list[list[int]] = [[] for _ in range(max_deg + 1)]
    for v in comp:
        buckets[len(adj[v])].append(v)
    remaining = [v for b in buckets for v in b]
    work[0] += len(comp) + max_deg + 1
    layers: list[list[int]] = []
    token = 0
    while remaining:
        cost = len(remaining) + sum(len(adj[v]) for v in remaining)
        if work[0] + cost > limit:
            return layers, remaining
        work[0] += cost
        t = len(layers)
        layer: list[int] = []
        rejected: list[int] = []
        for v in remaining:
            token += 1
            roots = []
            ok = True
            for y in adj[v]:
                if layer_of[y] == t:
                    r = _find(uf, y)
                    if mark[r] == token:
                        ok = False
                        break
                    mark[r] = token
                    roots.append(r)
            if ok:
                layer_of[v] = t
                uf[v] = v
                size[v] = 1
                top = v
                for r in roots:
                    if size[r] > size[top]:
                        top = r
                for r in (v, *roots):
                    if r != top:
                        uf[r] = top
                        size[top] += size[r]
                layer.append(v)
            else:
                rejected.append(v)
        for v in rejected:
            mark[v] = 0
        layers.append(layer)
        remaining = rejected
    return layers, []


def _orders(L: int, max_orders: int) -> list[tuple[int, ...]]:
    """Salvador's order schedule: every order when L! <= max_orders, otherwise the natural and
    reverse orders and their rotations, at most max_orders of them."""
    if L <= 1:
        return [tuple(range(L))]
    total = 1
    for k in range(2, L + 1):
        total *= k
        if total > max_orders:
            break
    if total <= max_orders:
        return list(itertools.permutations(range(L)))
    base = list(range(L))
    out: list[tuple[int, ...]] = []
    for seq in (base, base[::-1]):
        for shift in range(L):
            t = tuple(seq[shift:] + seq[:shift])
            if t not in out:
                out.append(t)
            if len(out) >= max_orders:
                return out
    return out


def forest_layers_ids(adj: list[list[int]], comp_of: list[int], budget: float = DEFAULT_LAYER_BUDGET,
                      max_orders: int = DEFAULT_MAX_ORDERS, info: dict | None = None) -> list[bool]:
    """Strategy c9 on the index graph ``adj`` with component labels ``comp_of``: an independent
    dominating set (membership list). ``info`` receives ``max_layers``, ``layers`` (total over the
    components), ``layer_lists`` (layer ``t`` of every component, joined: an induced forest), ``orders`` (orders tried), ``unplaced`` (vertices left to the greedy completion)
    and ``exact`` (every component is a tree solved in one layer, so the set is minimum)."""
    n = len(adj)
    m = sum(len(a) for a in adj) // 2
    comps: dict[int, list[int]] = {}
    for v in range(n):
        comps.setdefault(comp_of[v], []).append(v)
    layer_of = [-1] * n
    uf = list(range(n))
    size = [1] * n
    mark = [0] * n
    inside = [0] * n
    work = [float(n + m)]
    limit = budget * (n + m)
    in_s = [False] * n
    dominated = [False] * n
    stats = {"max_layers": 0, "layers": 0, "orders": 0, "unplaced": 0, "exact": True}
    global_layers: list[list[int]] = []  # layer t of every component (still an induced forest)
    token = 0
    for comp in comps.values():
        if len(comp) == 1:
            in_s[comp[0]] = True
            dominated[comp[0]] = True
            continue
        layers, rest = forest_layers(adj, comp, layer_of, uf, mark, work, limit, size)
        L = len(layers)
        for t, layer in enumerate(layers):
            if t == len(global_layers):
                global_layers.append([])
            global_layers[t] += layer
        stats["max_layers"] = max(stats["max_layers"], L)
        stats["layers"] += L
        stats["unplaced"] += len(rest)
        stats["exact"] = stats["exact"] and L == 1 and not rest
        best: list[int] | None = None
        for perm in _orders(L, max_orders):
            stats["orders"] += 1
            chosen: list[int] = []
            for t in perm:
                verts = [v for v in layers[t] if not dominated[v]]
                if not verts:
                    continue
                token += 1
                for v in verts:
                    inside[v] = token
                sel = forest_mids(adj, verts, inside, token)
                for v in sel:
                    in_s[v] = True
                    dominated[v] = True
                    for y in adj[v]:
                        dominated[y] = True
                chosen += sel
            chosen += _greedy_fill(adj, in_s, dominated, [v for v in comp if not dominated[v]])
            if best is None or len(chosen) < len(best):
                best = chosen
            for v in comp:  # reset the component for the next order
                in_s[v] = False
                dominated[v] = False
        for v in best:
            in_s[v] = True
    if info is not None:
        info.update(stats)
        info["layer_lists"] = global_layers
    return in_s
