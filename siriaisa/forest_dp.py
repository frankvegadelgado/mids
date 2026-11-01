# Version: v0.0.8
# Created on 31/10/2026
# Author: Frank Vega

"""
Exact forest solvers for the MIDS ensemble: the forest counterpart of Salvador's
bipartite machinery.

Salvador solves vertex cover exactly on bipartite structures (Koenig). For the
minimum independent dominating set the structure solved exactly in linear time
is the forest, so every strategy of Siriaisa that Salvador builds on a
bipartite graph is built here on a forest.

:func:`conditioned_forest_ids` -- exact minimum-cost solution on an induced
forest ``F`` of the index graph when the rest of the graph is fixed. Every
vertex ``v`` of ``F`` carries three flags:

* ``can_in[v]`` -- ``v`` may be selected (False when ``v`` has a selected
  neighbour outside ``F``);
* ``predom[v]`` -- ``v`` is already dominated from outside ``F``;
* ``force[v]`` -- ``v`` is the only possible dominator of a vertex outside
  ``F``, so leaving ``v`` unselected costs the penalty.

A selected vertex costs 1 and an undominated vertex costs the penalty ``P``
(``P > |F|``, so a solution that dominates every vertex of ``F`` is always
preferred, as in Salvador's weighted gadgets). The three-state DP of
:func:`siriaisa.algorithm.tree_minimum_independent_dominating_set` (IN, DOM:
dominated by a child or from outside, FREE: not dominated by its children)
is extended by these flags; it runs in ``O(sum of degrees of F)``. With all
flags neutral it is the plain exact tree DP.

:func:`spanning_forest_ids` -- a spanning forest of the whole graph (BFS, DFS
or the "star" forest in which every vertex points to the vertex of largest
degree in its closed neighbourhood), solved exactly by the DP; the forest
optimum dominates the graph (every forest edge is an edge of the graph) and is
repaired into an independent dominating set by
:func:`siriaisa.local_search.complete_ids`.

:func:`forest_lns` -- large-neighbourhood search over the induced-forest
layers of :mod:`siriaisa.forest_layers`: for an independent dominating set
``S`` and a layer ``L``, everything outside ``L`` is fixed from ``S`` and the
layer is re-solved exactly by :func:`conditioned_forest_ids` (a vertex outside
``L`` whose only possible dominators lie in ``L`` forces its unique candidate,
or is re-dominated by the completion); the new set is kept when it is not
larger. Every layer of every candidate is thus solved by the exact forest
solver with the candidate as context. Each layer costs ``O(n + m)``, with a
work budget.

:func:`feedback_forest_ids` (c11, Salvador's reducing-peeling with a forest in
place of the bipartite subdivision): :func:`feedback_core` peels a feedback
vertex set (exact removal of the vertices of residual degree at most 1, the
degree-2 bypass and double-edge rules, the vertex of largest residual degree
peeled into the core otherwise), and the
forest that remains is solved exactly for several choices on the core.
"""

from __future__ import annotations

from collections import deque

from .local_search import complete_ids

_INF = float("inf")


def conditioned_forest_ids(adj: list[list[int]], verts: list[int], inside: list[int], token: int,
                           can_in, predom, force, penalty: float) -> list[int]:
    """Exact minimum of |selected| + penalty * #undominated on the forest induced by ``verts``
    (``inside[v] == token``), under the flags ``can_in``, ``predom``, ``force`` (indexable by
    vertex). Returns the selected vertices. ``O(sum of degrees of verts)``."""
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
        # dp[v] = (IN, DOM, FREE); und[v] = cost of v being FREE with a parent not selected.
        dp: dict[int, tuple[float, float, float]] = {}
        und: dict[int, float] = {}
        forced_child: dict[int, int] = {}
        for v in reversed(order):
            ch = children[v]
            in_v = 1.0 if can_in[v] else _INF
            free_v = 0.0
            base = 0.0
            any_in = False
            best_pen, best_c = _INF, -1
            for c in ch:
                ic, dc, fc = dp[c]
                if in_v < _INF:
                    in_v += min(dc, fc)
                fu = fc + und[c]
                free_v += min(dc, fu)
                alt = min(dc, fu)
                if ic <= alt:
                    base += ic
                    any_in = True
                else:
                    base += alt
                    if ic < _INF and ic - alt < best_pen:
                        best_pen, best_c = ic - alt, c
            if predom[v] or any_in:
                dom_v = base
                forced_child[v] = -1
            elif best_c != -1:
                dom_v = base + best_pen
                forced_child[v] = best_c
            else:
                dom_v = _INF
                forced_child[v] = -1
            if force[v]:  # leaving v unselected leaves a vertex outside F undominated
                dom_v += penalty
                free_v += penalty
            dp[v] = (in_v, dom_v, free_v)
            und[v] = 0.0 if predom[v] else penalty
        # Top-down reconstruction: state 0 IN, 1 DOM, 2 FREE.
        r_in, r_dom, r_free = dp[root]
        r_free += und[root]
        best = min(r_in, r_dom, r_free)
        state = {root: 0 if r_in == best else 1 if r_dom == best else 2}
        for v in order:
            s = state[v]
            if s == 0:
                result.append(v)
                for c in children[v]:
                    _, dc, fc = dp[c]
                    state[c] = 1 if dc <= fc else 2
            elif s == 2:
                for c in children[v]:
                    _, dc, fc = dp[c]
                    state[c] = 1 if dc <= fc + und[c] else 2
            else:
                fcv = forced_child[v]
                for c in children[v]:
                    ic, dc, fc = dp[c]
                    alt_state, alt = (1, dc) if dc <= fc + und[c] else (2, fc + und[c])
                    if c == fcv or ic <= alt:
                        state[c] = 0
                    else:
                        state[c] = alt_state
    return result


class _Never:
    """Indexable constant (all flags False)."""

    def __getitem__(self, _):
        return False


class _Always:
    def __getitem__(self, _):
        return True


NEVER = _Never()
ALWAYS = _Always()


# ---------------------------------------------------------------------------
# Spanning forests solved exactly (c4, c7, c8)
# ---------------------------------------------------------------------------


def _forest_adjacency(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    fadj: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        fadj[u].append(v)
        fadj[v].append(u)
    return fadj


def spanning_forest(adj: list[list[int]], kind: str) -> list[tuple[int, int]]:
    """Edges of a spanning forest of ``adj``: ``"bfs"``, ``"dfs"`` (roots by descending degree)
    or ``"star"`` (every vertex points to the vertex of largest (degree, index) in its closed
    neighbourhood; the pointers strictly increase (degree, index), so they form a forest).
    ``O(n + m)``."""
    n = len(adj)
    if kind == "star":
        edges = []
        for v in range(n):
            best = v
            for u in adj[v]:
                if (len(adj[u]), u) > (len(adj[best]), best):
                    best = u
            if best != v:
                edges.append((v, best))
        return edges
    max_deg = max((len(a) for a in adj), default=0)
    buckets: list[list[int]] = [[] for _ in range(max_deg + 1)]
    for v in range(n):
        buckets[len(adj[v])].append(v)
    roots = [v for b in reversed(buckets) for v in b]
    seen = [False] * n
    edges = []
    for r in roots:
        if seen[r]:
            continue
        seen[r] = True
        if kind == "bfs":
            q = deque([r])
            while q:
                x = q.popleft()
                for y in adj[x]:
                    if not seen[y]:
                        seen[y] = True
                        edges.append((x, y))
                        q.append(y)
        else:  # dfs
            stack = [(r, iter(adj[r]))]
            while stack:
                x, it = stack[-1]
                for y in it:
                    if not seen[y]:
                        seen[y] = True
                        edges.append((x, y))
                        stack.append((y, iter(adj[y])))
                        break
                else:
                    stack.pop()
    return edges


def spanning_forest_ids(adj: list[list[int]], kind: str) -> list[bool]:
    """Exact MIDS of the spanning forest ``kind``, repaired and completed into an independent
    dominating set of ``adj``. ``O(n + m)``."""
    n = len(adj)
    fadj = _forest_adjacency(n, spanning_forest(adj, kind))
    inside = [1] * n
    sel = conditioned_forest_ids(fadj, list(range(n)), inside, 1, ALWAYS, NEVER, NEVER, n + 1.0)
    return complete_ids(adj, sel)


# ---------------------------------------------------------------------------
# Forest large-neighbourhood search (layers solved exactly in context)
# ---------------------------------------------------------------------------


def resolve_layer(adj: list[list[int]], in_s: list[bool], layer: list[int], inside: list[int],
                  token: int, work: list[float]) -> list[bool]:
    """Re-solve the induced-forest ``layer`` exactly with the rest fixed from ``in_s``, then
    complete. Returns the new membership list (not compared with ``in_s``)."""
    n = len(adj)
    for v in layer:
        inside[v] = token
    can_in: dict[int, bool] = {}
    predom: dict[int, bool] = {}
    force: dict[int, bool] = {v: False for v in layer}
    for v in layer:
        out_sel = False
        for y in adj[v]:
            if inside[y] != token and in_s[y]:
                out_sel = True
                break
        can_in[v] = not out_sel
        predom[v] = out_sel
        work[0] += len(adj[v]) + 1
    # Outside vertices that are not selected and have no selected neighbour outside the layer
    # must be dominated from the layer: a unique candidate is forced.
    seen: set[int] = set()
    for v in layer:
        for w in adj[v]:
            if inside[w] == token or in_s[w] or w in seen:
                continue
            seen.add(w)
            work[0] += len(adj[w])
            dominated_out = False
            cand = -1
            count = 0
            for y in adj[w]:
                if inside[y] == token:
                    if can_in[y]:
                        count += 1
                        cand = y
                elif in_s[y]:
                    dominated_out = True
                    break
            if not dominated_out and count == 1:
                force[cand] = True
    sel = conditioned_forest_ids(adj, layer, inside, token, can_in, predom, force, n + 1.0)
    new = list(in_s)
    for v in layer:
        new[v] = False
    for v in sel:
        new[v] = True
    work[0] += n
    return complete_ids(adj, (v for v in range(n) if new[v]))


def forest_lns(adj: list[list[int]], layers: list[list[int]], in_s: list[bool], budget: float,
               passes: int = 2) -> list[bool]:
    """Re-solve every layer of ``layers`` exactly in the context of ``in_s`` (an independent
    dominating set), keeping a new set when it is not larger, for at most ``passes`` passes or
    ``budget * (n + m)`` work. Never returns a larger set."""
    n = len(adj)
    m = sum(len(a) for a in adj) // 2
    limit = budget * (n + m)
    work = [0.0]
    inside = [0] * n
    token = 0
    cur = list(in_s)
    size = sum(cur)
    for _ in range(passes):
        improved = False
        for layer in layers:
            if not layer:
                continue
            if work[0] + 3 * (n + m) > limit:
                return cur
            token += 1
            new = resolve_layer(adj, cur, layer, inside, token, work)
            work[0] += 2 * (n + m)  # completion
            s = sum(new)
            if s <= size:
                improved = improved or s < size
                cur, size = new, s
        if not improved:
            break
    return cur


# ---------------------------------------------------------------------------
# c11: feedback-core peeling + conditioned forest DP (Salvador's c11 with a forest)
# ---------------------------------------------------------------------------


def feedback_core(adj: list[list[int]], info: dict | None = None) -> list[int]:
    """Reducing-peeling for a feedback vertex set (Salvador's c11 reductions on a forest), on
    the residual multigraph (``mult[v][y]`` = number of parallel edges, ``deg[v]`` their sum):

    * **degree 0 or 1**: ``v`` lies on no cycle -> removed into the forest (exact);
    * **degree 2, bypass**: ``v`` with its two edges to ``a`` and ``b`` (``a != b``) -> ``v``
      is removed into the forest and an edge ``a - b`` is added (a cycle through ``v`` becomes a
      cycle through ``a - b``; a parallel edge counts as a cycle), so ``deg(a)`` and ``deg(b)``
      do not change (exact);
    * **degree 2, double edge**: both edges of ``v`` go to the same ``b`` -> ``b`` goes to the
      core (it breaks the 2-cycle and every cycle through ``b``), and ``v`` then has degree 0;
    * **peel**, when no rule applies: the vertex of largest residual degree goes to the core.

    The vertices outside the returned core induce a forest of ``G``. Every rule removes one
    vertex in ``O(its residual degree)`` dictionary operations, so ``O(n + m)``. ``info``
    receives ``deg01``, ``bypass``, ``double`` and ``peels``."""
    n = len(adj)
    mult: list[dict[int, int]] = [{} for _ in range(n)]
    for v in range(n):
        for y in adj[v]:
            mult[v][y] = mult[v].get(y, 0) + 1
    deg = [len(a) for a in adj]
    gone = [False] * n
    max_deg = max(deg, default=0)
    buckets: list[list[int]] = [[] for _ in range(max_deg + 1)]
    low: list[int] = []
    for v in range(n):
        buckets[deg[v]].append(v)
        if deg[v] <= 2:
            low.append(v)
    core: list[int] = []
    counts = {"deg01": 0, "bypass": 0, "double": 0, "peels": 0}
    ptr = max_deg
    left = n

    def touch(y: int) -> None:
        if deg[y] <= 2:
            low.append(y)
        else:
            buckets[deg[y]].append(y)

    def drop(v: int) -> None:
        """Remove v with all its edges."""
        nonlocal left
        gone[v] = True
        left -= 1
        for y, k in mult[v].items():
            del mult[y][v]
            deg[y] -= k
            touch(y)
        mult[v].clear()
        deg[v] = 0

    while left:
        while low:
            v = low.pop()
            if gone[v] or deg[v] > 2:
                continue
            if deg[v] <= 1:
                counts["deg01"] += 1
                drop(v)
                continue
            # deg[v] == 2
            items = list(mult[v].items())
            if len(items) == 1:  # double edge v = b
                b = items[0][0]
                counts["double"] += 1
                core.append(b)
                drop(b)
                low.append(v)
                continue
            (a, _), (b, _) = items
            counts["bypass"] += 1
            gone[v] = True
            left -= 1
            del mult[a][v]
            del mult[b][v]
            mult[v].clear()
            deg[v] = 0
            mult[a][b] = mult[a].get(b, 0) + 1
            mult[b][a] = mult[b].get(a, 0) + 1
            # deg(a), deg(b) unchanged; a parallel edge a = b may now enable the double rule.
            if mult[a][b] >= 2:
                touch(a)
                touch(b)
        if not left:
            break
        while ptr > 2:
            bkt = buckets[ptr]
            while bkt and (gone[bkt[-1]] or deg[bkt[-1]] != ptr):
                bkt.pop()
            if bkt:
                break
            ptr -= 1
        if ptr <= 2:
            # Every remaining vertex has degree <= 2 and is queued again.
            low.extend(v for v in range(n) if not gone[v])
            continue
        v = buckets[ptr].pop()
        counts["peels"] += 1
        core.append(v)
        drop(v)
    if info is not None:
        info.update(counts)
    return core


def feedback_forest_ids(adj: list[list[int]], reference: list[bool], info: dict | None = None) -> list[bool]:
    """Strategy c11: the forest ``F = V - core`` of :func:`feedback_core` solved exactly by
    :func:`conditioned_forest_ids` for three choices on the core -- the reference set's choice
    (``ref``), no core vertex selected (``out``), the greedy independent dominating set of the
    core (``greedy``) -- each completed; the smallest set is returned. Exact (minimum) when the
    core is empty, i.e. on forests. ``O(n + m)``."""
    n = len(adj)
    core_info: dict = {}
    core = feedback_core(adj, core_info)
    in_core = [False] * n
    for v in core:
        in_core[v] = True
    forest = [v for v in range(n) if not in_core[v]]
    contexts: dict[str, list[bool]] = {"ref": list(reference), "out": [False] * n}
    if core:
        index = {v: i for i, v in enumerate(core)}
        sub = [[index[y] for y in adj[v] if y in index] for v in core]
        g = complete_ids(sub, ())
        greedy = [False] * n
        for i, v in enumerate(core):
            greedy[v] = g[i]
        contexts["greedy"] = greedy
    inside = [0] * n
    work = [0.0]
    best, best_key = None, None
    for token, (key, ctx) in enumerate(contexts.items(), start=1):
        ctx = [ctx[v] and in_core[v] for v in range(n)]  # only the core is fixed
        sol = resolve_layer(adj, ctx, forest, inside, token, work) if forest else complete_ids(
            adj, (v for v in range(n) if ctx[v]))
        if best is None or sum(sol) < sum(best):
            best, best_key = sol, key
    if info is not None:
        info.update({"core_size": len(core), "variant": best_key, "exact": not core,
                     **{f"core_{k}": v for k, v in core_info.items()}})
    return best
