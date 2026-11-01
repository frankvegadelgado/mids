# Version: v0.0.8
# Modified on 31/10/2026
# Author: Frank Vega

"""
Ensemble heuristic for the Minimum Independent Dominating Set (MIDS) problem
(Siriaisa, version 0.0.8).

An independent dominating set is exactly a maximal independent set.
:func:`find_independent_dominating_set` runs the ensemble of Salvador 0.1.4
(seventeen strategies) adapted to independent domination, together with the
phases of Esperanza 0.1.6 and a degree-splitting strategy, eighteen strategies
numbered in the order in which they are computed: the base strategies first,
the improvers last. The key of the adaptation: where
Salvador builds a *bipartite* structure and solves vertex cover on it exactly
(Koenig), Siriaisa builds a *forest* and solves MIDS on it exactly with the
linear-time tree DP (:mod:`siriaisa.forest_dp`). Every candidate is turned into
an independent dominating set by the repair-and-complete pass
(:func:`siriaisa.local_search.complete_ids`, the counterpart of Salvador's
pruning and of the MaximizeSolution pass of Salvador and Esperanza).

Base strategies

* c1  -- spanning-tree peeling (Siriaisa 0.0.7, now linear): round by round,
  a maximum spanning forest of the residual graph (edge weight deg(u) + deg(v),
  Kruskal with the edges bucket-sorted by their integer weights) solved by the
  tree DP, an independent subset kept, it and its neighbours removed.
* c2  -- maximal matching (Hvala/Salvador c2): the endpoints of the NetworkX
  minimum maximal matching dominate the graph; repaired and completed.
* c3  -- max-gain greedy (Hvala/Salvador c3): the undominated vertex that
  dominates the most undominated vertices, bucket queue.
* c4  -- star-forest reduction (the Hallelujah degree-1 reduction, Salvador c4):
  every vertex points to the vertex of largest degree in its closed
  neighbourhood; the pointers form a spanning forest, solved exactly.
* c5  -- Min-to-Min (Salvador c5): the undominated vertex of smallest residual
  degree is dominated by its best free closed neighbour.
* c6  -- primal-dual / local ratio on the domination LP (Salvador c6).
* c7  -- spanning-forest core (Salvador c7's planar core): a BFS spanning forest
  solved exactly, repaired.
* c8  -- DFS spanning forest solved exactly, repaired (the one-pass incidence
  structure of Salvador c8, with a forest in place of the cycle gadget).
* c9  -- induced-forest layers (Salvador c9, bipartite layers -> forest layers,
  :mod:`siriaisa.forest_layers`): every component is split into induced
  forests, each solved exactly by the tree DP, in several layer orders.
* c10 -- forest core DP (Salvador c10/c12: core fixed from a reference, the rest
  exact): the first induced-forest layer is re-solved exactly with the rest of
  the graph fixed from the best of c1..c9; exact (minimum) when the graph is a
  forest.
* c11 -- feedback-core forest DP (Salvador's c11 reducing-peeling, with a
  forest in place of the bipartite subdivision,
  :func:`siriaisa.forest_dp.feedback_forest_ids`): a feedback vertex set (the
  core) is peeled -- vertices of residual degree at most 1 removed exactly,
  vertices of residual degree 2 bypassed (Salvador's degree-2 rule: v with
  neighbours a, b is removed and the edge a - b added; a double edge sends its
  end to the core), otherwise the vertex of largest residual degree peeled --
  and the remaining
  forest is solved exactly for three choices on the core (that of the best of
  c1..c10, no core vertex, the greedy solution of the core); the smallest kept.
  Exact on forests.
* c12 -- Caro--Wei baseline (Esperanza phase 1): minimum-degree greedy.
* c13 -- degree splitting at k = floor(sqrt(n)) (:mod:`siriaisa.degree_split`):
  the vertices are scanned by descending degree and every vertex u of degree
  larger than k is replaced by unconnected copies u_1, u_2, ..., each joined to
  the next k of its neighbours (the last one to the remaining at most k), so the
  copy graph has maximum degree at most k (the regime of Esperanza's 2 sqrt(n)
  theorem); the copy graph is solved, u is taken when a copy of u is selected,
  and the set is repaired with pruning and completed.

Improvers

* c14 -- union of c2..c8, re-pruned: repaired keeping the vertices chosen by
  most of them, then completed (Hvala's union pass).
* c15 -- budgeted descent + weighted swap search
  (:func:`siriaisa.local_search.improve_ids`; Esperanza's local restarts and
  local search turned around to shrink the set, with the NuMVC-style weighted
  swap moves of :mod:`siriaisa.weighted_search`) from the best of c1..c14.
* c16 -- reduce-and-search: rounds of weighted swap search alternated with the
  forest large-neighbourhood search (:func:`siriaisa.forest_dp.forest_lns`:
  every induced-forest layer re-solved exactly in the context of the current
  set), from the best of c1..c15.
* c17 -- consensus forcing (:mod:`siriaisa.consensus_force`): the vertices
  selected by all/most of c1..c16 are forced and the residual graph is solved
  again.
* c18 -- refinement of every candidate c1..c17: forest large-neighbourhood
  search and local search on each, a longer search on the best (with its own
  constant work floor ``final_floor``: on cubic graphs the nearest smaller
  solution can differ from the best one in about a third of the vertices); the
  smallest set is returned (Esperanza's final MaximizeSolution: it is maximal).

So every candidate, and every layer of every candidate (c10, c16, c18), is
solved by the exact forest solver with the ensemble as context.

Running time: no step uses a comparison sort (every sort is a counting sort
on integer keys at most n + m); c1 (Kruskal on bucket-sorted edges) and c9
(layering) use union-find with union by size and path halving, so they cost
``O((n + m) alpha(n))`` within their work budgets (``alpha`` the inverse
Ackermann function, as in Salvador's c1); every other base strategy is
``O(n + m)``; the improvers have fixed work budgets in units of ``n + m``
plus a constant floor. With the default parameters the whole ensemble runs in
``O((n + m) alpha(n))`` time, linear up to the inverse Ackermann factor
(expected time, for the hash tables of vertex labels and of c11's residual
multigraph).

MIDS is NP-hard, also on bipartite graphs, and hard to approximate in general;
on forests c1, c9, c10 and c11 (hence c18) are optimal.
"""

from __future__ import annotations

import itertools
from collections import deque
from typing import Any, Dict, Hashable, List, Optional, Set

import networkx as nx

from . import bipartite, consensus_force, degree_split, forest_dp, forest_layers, heuristics, local_search

_INF = float("inf")

# DP states for a rooted tree vertex v
_IN = 0    # v is in the set
_DOM = 1   # v is not in the set, dominated by at least one child
_FREE = 2  # v is not in the set, not dominated by children (parent must be in)

#: Spanning-tree peeling budget (c1), in units of (n + m).
DEFAULT_TREE_BUDGET: float = 40.0
#: Induced-forest layering budget (c9), in units of (n + m), and layer orders tried per component.
DEFAULT_LAYER_BUDGET: float = forest_layers.DEFAULT_LAYER_BUDGET
DEFAULT_LAYER_ORDERS: int = forest_layers.DEFAULT_MAX_ORDERS
#: Local-search budget of c15, in units of (n + m).
DEFAULT_LOCAL_SEARCH_BUDGET: float = 10.0
#: Reduce-and-search (c16): rounds and local-search budget per round, in units of (n + m).
DEFAULT_SEARCH_ROUNDS: int = 3
DEFAULT_SEARCH_BUDGET: float = 5.0
#: Forest large-neighbourhood search budget per call, in units of (n + m).
DEFAULT_LNS_BUDGET: float = 40.0
#: Consensus-forcing rounds (c17).
DEFAULT_CONSENSUS_ROUNDS: int = consensus_force.DEFAULT_ROUNDS
#: Refinement (c18): local-search budget per candidate and for the best one, in units of (n + m).
DEFAULT_REFINE_BUDGET: float = 3.0
DEFAULT_ILS_BUDGET: float = 30.0
#: Constant work floor of the final search of c18 (larger than the other searches': on cubic
#: graphs the nearest smaller solution can differ in a third of the vertices).
DEFAULT_FINAL_FLOOR: int = 24_000_000

#: Human-readable names of the strategies, in evaluation order.
CANDIDATE_NAMES: Dict[str, str] = {
    "c1": "spanning-tree peeling (tree DP), budgeted",
    "c2": "maximal matching endpoints (NetworkX min maximal matching)",
    "c3": "max-gain greedy",
    "c4": "star-forest reduction (Hallelujah) + tree DP",
    "c5": "Min-to-Min",
    "c6": "primal-dual",
    "c7": "BFS spanning-forest core + tree DP",
    "c8": "DFS spanning forest + tree DP",
    "c9": "induced-forest layers solved by the tree DP, several orders",
    "c10": "forest core DP: first forest layer exact, rest fixed from best of c1..c9",
    "c11": "feedback-core peeling + conditioned forest DP (3 core choices)",
    "c12": "Caro-Wei minimum-degree greedy (Esperanza baseline)",
    "c13": "degree splitting at floor(sqrt(n)) + repair with pruning",
    "c14": "union re-prune of c2..c8",
    "c15": "descent + weighted swap search from the best of c1..c14",
    "c16": "reduce-and-search: swap search + forest LNS rounds, from the best of c1..c15",
    "c17": "consensus forcing: vertices in all/most of c1..c16 forced, rest re-solved",
    "c18": "refinement of c1..c17 (forest LNS + local search), best kept",
}
FINAL_NAME: str = "c18"
#: The candidates whose union c14 re-prunes.
C13_INPUT_NAMES = ("c2", "c3", "c4", "c5", "c6", "c7", "c8")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean_graph(graph: nx.Graph) -> nx.Graph:
    if not isinstance(graph, nx.Graph) or graph.is_directed():
        raise ValueError("Input must be an undirected NetworkX Graph.")
    G = nx.Graph(graph)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def verify_independent_dominating_set(G: nx.Graph, S: Optional[Set[Any]]) -> bool:
    """Return True iff S is an independent dominating set of G, in O(n+m)."""
    if S is None:
        return False
    S = set(S)
    if not S.issubset(G.nodes):
        return False
    adj = G.adj
    for v in S:
        for u in adj[v]:
            if u in S:
                return False
    for v in G.nodes():
        if v in S:
            continue
        for u in adj[v]:
            if u in S:
                break
        else:
            return False
    return True


# ---------------------------------------------------------------------------
# Exact MIDS on a tree (linear-time DP)
# ---------------------------------------------------------------------------


def tree_minimum_independent_dominating_set(T: nx.Graph) -> Set[Hashable]:
    """
    Exact minimum independent dominating set of a tree (or forest) in O(n).

    Rooted DP with three states per vertex v:
      IN[v]   = 1 + sum_c min(DOM[c], FREE[c])
      FREE[v] = sum_c DOM[c]
      DOM[v]  = sum_c min(IN[c], DOM[c]) with at least one child IN
    and the root takes min(IN, DOM).
    """
    result: Set[Hashable] = set()
    seen: Set[Hashable] = set()

    for root in T.nodes():
        if root in seen:
            continue

        # Iterative BFS order + parent pointers (no recursion limits).
        parent: Dict[Hashable, Optional[Hashable]] = {root: None}
        order: List[Hashable] = [root]
        seen.add(root)
        q = deque([root])
        while q:
            v = q.popleft()
            for u in T.neighbors(v):
                if u not in seen:
                    seen.add(u)
                    parent[u] = v
                    order.append(u)
                    q.append(u)

        children: Dict[Hashable, List[Hashable]] = {v: [] for v in order}
        for v in order[1:]:
            children[parent[v]].append(v)

        dp: Dict[Hashable, List[float]] = {}
        # For DOM: the child forced into IN when no child is IN by choice.
        forced: Dict[Hashable, Optional[Hashable]] = {}

        for v in reversed(order):
            ch = children[v]
            in_v = 1.0
            free_v = 0.0
            dom_v = 0.0
            any_in = False
            best_pen = _INF
            best_c = None
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
                forced[v] = None
            elif any_in:
                forced[v] = None
            else:
                dom_v += best_pen
                forced[v] = best_c
            dp[v] = [in_v, dom_v, free_v]

        # Top-down reconstruction.
        state: Dict[Hashable, int] = {}
        r_in, r_dom, _ = dp[root]
        state[root] = _IN if r_in <= r_dom else _DOM
        for v in order:
            s = state[v]
            if s == _IN:
                result.add(v)
                for c in children[v]:
                    _, dc, fc = dp[c]
                    state[c] = _DOM if dc <= fc else _FREE
            elif s == _FREE:
                for c in children[v]:
                    state[c] = _DOM
            else:  # _DOM
                fc_forced = forced[v]
                for c in children[v]:
                    ic, dc, _ = dp[c]
                    if c == fc_forced:
                        state[c] = _IN
                    else:
                        state[c] = _IN if ic <= dc else _DOM

    return result


# ---------------------------------------------------------------------------
# c1: spanning-tree peeling (linear time)
# ---------------------------------------------------------------------------


def _find(parent: List[int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def spanning_tree_peeling(adj: List[List[int]], tree_budget: float = DEFAULT_TREE_BUDGET,
                          info: Optional[dict] = None) -> List[int]:
    """Strategy c1 on the index graph ``adj``: the spanning-tree peeling of version 0.0.7 in
    linear time.

    Round by round, on the residual graph ``R`` (initially ``G``): a maximum spanning forest
    of ``R`` (edge weight ``deg_R(u) + deg_R(v)``) is computed by Kruskal's algorithm with the
    edges bucket-sorted by weight (the weights are integers at most ``2 Delta``, so the sort
    is a counting sort) and a union-find structure with union by size and path halving; the
    forest is solved exactly by the tree DP; a maximal ``G``-independent subset of the forest
    optimum is kept, vertices of larger residual degree first (counting sort); the kept
    vertices and their neighbours are removed and the next round works on what is left. This
    is the 0.0.7 loop (one maximum spanning tree per component per round) without the
    ``O(m log m)`` comparison sort. A round costs ``O((n_R + m_R) alpha(n))``; the rounds stop
    when the total work would exceed ``tree_budget * (n + m)``, and the vertices left then
    (undominated, not adjacent to the kept set) are left to the completion pass. Worst case
    ``O((n + m) alpha(n))``. Returns the kept vertices (independent). ``info`` receives
    ``iterations`` (rounds) and ``exhausted``."""
    n = len(adj)
    m = sum(len(a) for a in adj) // 2
    limit = tree_budget * (n + m)
    work = float(n + m)
    alive = [True] * n
    rest = list(range(n))
    kept_all: List[int] = []
    kept = [False] * n
    parent = list(range(n))
    size = [1] * n
    inside = [0] * n
    rdeg = [0] * n
    fadj: List[List[int]] = [[] for _ in range(n)]
    rounds = 0
    exhausted = False
    while rest:
        cost = 4 * (len(rest) + sum(len(adj[v]) for v in rest))
        if work + cost > limit:
            exhausted = True
            break
        work += cost
        rounds += 1
        for v in rest:
            rdeg[v] = sum(1 for y in adj[v] if alive[y])
        max_w = 2 * max((rdeg[v] for v in rest), default=0)
        buckets: List[List[tuple]] = [[] for _ in range(max_w + 1)]
        for v in rest:
            for y in adj[v]:
                if v < y and alive[y]:
                    buckets[rdeg[v] + rdeg[y]].append((v, y))
        for v in rest:
            parent[v] = v
            size[v] = 1
            fadj[v] = []
        for b in reversed(buckets):
            for u, v in b:
                ru, rv = _find(parent, u), _find(parent, v)
                if ru != rv:
                    if size[ru] < size[rv]:
                        ru, rv = rv, ru
                    parent[rv] = ru
                    size[ru] += size[rv]
                    fadj[u].append(v)
                    fadj[v].append(u)
        # Exact MIDS of the forest (index lists with the forest adjacency).
        for v in rest:
            inside[v] = rounds
        sel = forest_dp.conditioned_forest_ids(fadj, rest, inside, rounds, forest_dp.ALWAYS,
                                               forest_dp.NEVER, forest_dp.NEVER, n + 1.0)
        # Maximal G-independent subset, larger residual degree first (counting sort).
        by_deg: List[List[int]] = [[] for _ in range(max_w // 2 + 1)]
        for v in sel:
            by_deg[rdeg[v]].append(v)
        newly: List[int] = []
        for b in reversed(by_deg):
            for v in b:
                if not any(kept[y] for y in adj[v]):
                    kept[v] = True
                    newly.append(v)
        kept_all += newly
        for v in newly:
            alive[v] = False
            for y in adj[v]:
                alive[y] = False
        rest = [v for v in rest if alive[v]]
    if info is not None:
        info.update({"iterations": rounds, "exhausted": exhausted})
    return kept_all


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------


def _index_graph(G: nx.Graph):
    nodes = list(G.nodes())
    index = {v: i for i, v in enumerate(nodes)}
    adj = [[index[u] for u in G.adj[v]] for v in nodes]
    return nodes, index, adj


def _members(in_s: List[bool]) -> List[int]:
    return [v for v, b in enumerate(in_s) if b]


def _best(cands: Dict[str, List[bool]], names=None) -> List[bool]:
    return min((cands[k] for k in (names or cands)), key=sum)


def _inner_solver(sub: List[List[int]]) -> List[bool]:
    """Residual solver of c17: best of greedy, Min-to-Min and the induced-forest layers."""
    if not sub:
        return []
    comp = bipartite.two_colouring(sub)[1]
    sols = [local_search.complete_ids(sub, ()), heuristics.min_to_min_ids(sub),
            forest_layers.forest_layers_ids(sub, comp)]
    return min(sols, key=sum)


def final_candidates(
    graph: nx.Graph,
    tree_budget: float = DEFAULT_TREE_BUDGET,
    layer_budget: float = DEFAULT_LAYER_BUDGET,
    layer_orders: int = DEFAULT_LAYER_ORDERS,
    local_search_budget: float = DEFAULT_LOCAL_SEARCH_BUDGET,
    search_rounds: int = DEFAULT_SEARCH_ROUNDS,
    search_budget: float = DEFAULT_SEARCH_BUDGET,
    lns_budget: float = DEFAULT_LNS_BUDGET,
    consensus_rounds: int = DEFAULT_CONSENSUS_ROUNDS,
    refine_budget: float = DEFAULT_REFINE_BUDGET,
    ils_budget: float = DEFAULT_ILS_BUDGET,
    seed: int = local_search.DEFAULT_SEED,
    work_floor: int = local_search.DEFAULT_WORK_FLOOR,
    final_floor: int = DEFAULT_FINAL_FLOOR,
    return_refined: bool = False,
    info: Optional[dict] = None,
):
    """Every strategy of Siriaisa 0.0.8 (``c1``...``c18``) by name, as sets of vertices of
    ``graph``; ``c18`` is the returned solution and ``|c18| <= min(|c1|, ..., |c17|)``.

    With ``return_refined=True`` the result is ``(candidates, refined)``, where
    ``refined[name]`` is the c18 refinement of candidate ``name``. ``info`` (optional dict)
    receives diagnostics prefixed by the strategy. Running time ``O((n + m) alpha(n))`` for the
    default parameters.
    """
    G = _clean_graph(graph)
    if G.number_of_nodes() == 0:
        empty = {name: set() for name in CANDIDATE_NAMES}
        return (empty, dict(empty)) if return_refined else empty
    nodes, _, adj = _index_graph(G)
    n = len(nodes)
    cands: Dict[str, List[bool]] = {}
    complete = local_search.complete_ids

    # ---- Base strategies ----
    tree_info: dict = {}
    cands["c1"] = complete(adj, spanning_tree_peeling(adj, tree_budget, tree_info))
    cands["c2"] = complete(adj, heuristics.matching_seed(adj))
    cands["c3"] = complete(adj, ())
    cands["c4"] = forest_dp.spanning_forest_ids(adj, "star")
    cands["c5"] = heuristics.min_to_min_ids(adj)
    cands["c6"] = heuristics.primal_dual_ids(adj)
    cands["c7"] = forest_dp.spanning_forest_ids(adj, "bfs")
    cands["c8"] = forest_dp.spanning_forest_ids(adj, "dfs")

    _, comp, is_bip = bipartite.two_colouring(adj)
    # Certified lower bound: a component with n_c vertices and maximum degree D_c needs at
    # least ceil(n_c / (D_c + 1)) vertices (each selected vertex dominates at most D_c + 1).
    comp_n: Dict[int, int] = {}
    comp_d: Dict[int, int] = {}
    for v in range(n):
        c = comp[v]
        comp_n[c] = comp_n.get(c, 0) + 1
        comp_d[c] = max(comp_d.get(c, 0), len(adj[v]))
    lower_bound = sum(-(-comp_n[c] // (comp_d[c] + 1)) for c in comp_n)
    # Size-scaled work floors: min(floor, c (n + m)^2) -- still constants, so the running time
    # stays linear, but a small graph no longer pays for millions of units of work.
    nm = n + sum(len(a) for a in adj) // 2
    work_floor = min(work_floor, 10 * nm * nm)
    final_floor = min(final_floor, 100 * nm * nm)
    layer_info: dict = {}
    cands["c9"] = forest_layers.forest_layers_ids(adj, comp, layer_budget, layer_orders, layer_info)
    layers: List[List[int]] = layer_info.pop("layer_lists")

    ref = _best(cands)
    work = [0.0]
    c10 = forest_dp.resolve_layer(adj, ref, layers[0], [0] * n, 1, work) if layers else list(ref)
    cands["c10"] = c10 if sum(c10) <= sum(ref) else list(ref)
    c10_exact = len(layers) <= 1 and layer_info["unplaced"] == 0

    ncomp = len(is_bip)
    c11_info: dict = {}
    cands["c11"] = forest_dp.feedback_forest_ids(adj, _best(cands), c11_info)
    cands["c12"] = heuristics.caro_wei_ids(adj)
    split_info: dict = {}
    cands["c13"] = degree_split.degree_split_ids(adj, split_info)

    # ---- Improvers ----
    votes = [0] * n
    for k in C13_INPUT_NAMES:
        for v in _members(cands[k]):
            votes[v] += 1
    cands["c14"] = complete(adj, (v for v in range(n) if votes[v]), priority=votes)

    best = _best(cands)
    cands["c15"] = local_search.improve_ids(adj, best, local_search_budget, seed, work_floor,
                                            target=lower_bound)

    cur = _best(cands)
    for r in range(search_rounds):
        if sum(cur) <= lower_bound:
            break
        nxt = local_search.improve_ids(adj, cur, search_budget, seed + 101 * (r + 1), work_floor // 2,
                                       target=lower_bound)
        nxt = forest_dp.forest_lns(adj, layers, nxt, lns_budget)
        if sum(nxt) <= sum(cur):
            cur = nxt
    cands["c16"] = cur

    cons_info: dict = {}
    cands["c17"] = consensus_force.consensus_force_ids(
        adj, [cands[f"c{i}"] for i in range(1, 17)], _inner_solver, consensus_rounds, cons_info)

    # ---- c18: refinement of every candidate ----
    seen: Dict[frozenset, List[bool]] = {}
    refined: Dict[str, List[bool]] = {}
    for i, (name, in_s) in enumerate(cands.items()):
        key = frozenset(_members(in_s))
        if key not in seen:
            r = forest_dp.forest_lns(adj, layers, in_s, lns_budget, passes=1)
            seen[key] = local_search.improve_ids(adj, r, refine_budget, seed + i, work_floor // 8,
                                                 target=lower_bound)
        refined[name] = seen[key]
    top = min(refined.values(), key=sum)
    final = local_search.improve_ids(adj, top, ils_budget, seed, final_floor, target=lower_bound)
    final = forest_dp.forest_lns(adj, layers, final, lns_budget)
    final = complete(adj, _members(final))  # Esperanza's final MaximizeSolution (already maximal)
    if sum(final) > sum(top):
        final = top

    def to_set(in_s: List[bool]) -> Set[Any]:
        return {nodes[v] for v in _members(in_s)}

    out = {name: to_set(in_s) for name, in_s in cands.items()}
    out[FINAL_NAME] = to_set(final)
    for name, S in out.items():
        if not verify_independent_dominating_set(G, S):
            raise RuntimeError(f"internal error: candidate {name} is not an independent dominating set")
    if info is not None:
        info.update({
            "components": ncomp,
            "lower_bound": lower_bound,
            "bipartite_components": sum(is_bip),
            "is_bipartite": all(is_bip),
            "c1_iterations": tree_info["iterations"],
            "c1_exhausted": tree_info["exhausted"],
            **{f"c9_{k}": v for k, v in layer_info.items()},
            "c10_exact": c10_exact,
            **{f"c11_{k}": v for k, v in c11_info.items()},
            **{f"c13_{k}": v for k, v in split_info.items()},
            **{f"c17_{k}": v for k, v in cons_info.items()},
        })
    if return_refined:
        ref_sets = {name: to_set(in_s) for name, in_s in refined.items()}
        ref_sets[FINAL_NAME] = out[FINAL_NAME]
        return out, ref_sets
    return out


def find_independent_dominating_set(graph: nx.Graph) -> Set[Any]:
    """
    Approximate MIDS with the Siriaisa 0.0.8 ensemble (``c18``, the smallest refined
    candidate), in ``O((n + m) alpha(n))`` time.

    Returns a set that is always independent and dominating in ``graph``.
    """
    return final_candidates(graph)[FINAL_NAME]


# ---------------------------------------------------------------------------
# Comparison baselines
# ---------------------------------------------------------------------------


def find_independent_dominating_set_approximation(graph: nx.Graph) -> Set[Any]:
    """Baseline: greedy minimum-degree maximal independent set."""
    G = _clean_graph(graph)
    solution: Set[Any] = set()
    alive = set(G.nodes())
    for v in sorted(G.nodes(), key=G.degree):
        if v in alive:
            solution.add(v)
            alive.discard(v)
            alive.difference_update(G.neighbors(v))
    return solution


def find_independent_dominating_set_brute_force(graph: nx.Graph) -> Optional[Set[Any]]:
    """Exact exponential MIDS solver for tiny tests only."""
    G = _clean_graph(graph)
    if G.number_of_nodes() == 0:
        return set()
    nodes = list(G.nodes())
    for r in range(1, len(nodes) + 1):
        for cand in itertools.combinations(nodes, r):
            S = set(cand)
            if verify_independent_dominating_set(G, S):
                return S
    return None
