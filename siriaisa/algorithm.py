# Created on 08/07/2026
# Author: Frank Vega

"""
Linear-time heuristic pool for Minimum Weighted Independent Dominating Set
(MWIDS).

The public function ``find_weighted_independent_dominating_set`` always returns
an independent dominating set and chooses the best candidate by total vertex
weight. It uses a fixed pool of linear Furones-style signals plus a
Salvador-oriented-incidence auxiliary candidate solved by the weighted
Baker-layer IDS routine with eps=1.0.

The candidate pool and selection rule are unchanged; the implementation runs
on flat index-based adjacency arrays (see ``baker_ids.FlatGraph``) built once
per input. This removes NetworkX per-call overhead, graph/subgraph copies and
redundant O(n+m) re-verifications (duplicate candidates are deduplicated, and
candidates that are feasible by construction are not re-checked). Local-swap
verification is done on the touched neighbourhoods only, which is exact and
O(deg) instead of O(n+m) per probe.

Guarantee: feasibility and O(|V|+|E|) time for a fixed candidate pool, assuming
expected O(1) Python dict/set operations. This is a heuristic, not an exact
solver and not a proved constant-approximation algorithm.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import networkx as nx

try:
    from . import baker_ids
except Exception:
    import baker_ids

_EPS = 1.0e-12

FlatGraph = baker_ids.FlatGraph
_flat_order = baker_ids.flat_order
_flat_bucket_order = baker_ids.flat_bucket_order
_flat_repair_prune = baker_ids.flat_repair_prune
_flat_verify = baker_ids.flat_verify_ids
_flat_weight = baker_ids.flat_solution_weight
_extend_unique = baker_ids.extend_unique


# ---------------------------------------------------------------------------
# Public helpers (NetworkX-facing, backward compatible)
# ---------------------------------------------------------------------------


def _clean_graph(graph: nx.Graph) -> nx.Graph:
    if not isinstance(graph, nx.Graph) or graph.is_directed():
        raise ValueError("Input must be an undirected NetworkX Graph.")
    G = nx.Graph(graph)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def _closed(G: nx.Graph, v: Any) -> Iterable[Any]:
    yield v
    yield from G.neighbors(v)


def _node_weight(G: nx.Graph, v: Any, weight: str = "weight") -> float:
    return float(G.nodes[v].get(weight, 1.0))


def _ratio_denominator(G: nx.Graph, v: Any, weight: str = "weight") -> float:
    return max(_node_weight(G, v, weight), _EPS)


def calculate_solution_weight(G: nx.Graph, solution: Iterable[Any], weight: str = "weight") -> float:
    return sum(_node_weight(G, v, weight) for v in solution)


def verify_independent_dominating_set(G: nx.Graph, S: Optional[Set[Any]]) -> bool:
    return baker_ids.verify_independent_dominating_set(G, S)


def _is_dominating_set(G: nx.Graph, D: Set[Any]) -> bool:
    if not D.issubset(G.nodes):
        return False
    dominated = {v: False for v in G.nodes()}
    for v in D:
        dominated[v] = True
        for u in G.neighbors(v):
            dominated[u] = True
    return all(dominated.values())


def repair_prune_weighted(
    G: nx.Graph,
    candidate: Optional[Set[Any]] = None,
    order: Optional[Sequence[Any]] = None,
    weight: str = "weight",
) -> Set[Any]:
    """
    Repair a seed into a maximal independent set, then prune by weight.

    Runtime: O(n+m), up to fixed bucket constants.
    """
    if not isinstance(G, nx.Graph) or G.is_directed():
        raise ValueError("G must be an undirected NetworkX Graph.")
    return baker_ids.repair_prune_weighted(G, candidate, order, weight)


repair_prune = repair_prune_weighted


# ---------------------------------------------------------------------------
# Flat candidate machinery
# ---------------------------------------------------------------------------


class _Cand:
    __slots__ = ("vertices", "order", "name")

    def __init__(self, vertices: Iterable[int], order: Iterable[int], name: str):
        self.vertices = set(vertices)
        self.order = list(order)
        self.name = name


class _Orders:
    """Per-component precomputed data shared by all candidate generators."""

    __slots__ = ("ratio", "cheap", "degree", "natural", "denom")

    def __init__(self, F: FlatGraph):
        self.natural = list(range(F.n))
        self.ratio = _flat_order(F, self.natural, "coverage_per_weight")
        self.cheap = _flat_order(F, self.natural, "cheap")
        self.degree = _flat_order(F, self.natural, "degree")
        self.denom = baker_ids.flat_denominators(F)


def _prune_ds_flat(F: FlatGraph, D: Set[int]) -> Set[int]:
    """Remove redundant vertices from a dominating set, expensive first."""
    D = set(D)
    n = F.n
    adj = F.adj
    count = [0] * n
    for v in D:
        count[v] += 1
        for u in adj[v]:
            count[u] += 1

    for v in _flat_bucket_order(sorted(D), F.w, 256, True):
        if count[v] < 2:
            continue
        removable = True
        for u in adj[v]:
            if count[u] < 2:
                removable = False
                break
        if removable:
            D.remove(v)
            count[v] -= 1
            for u in adj[v]:
                count[u] -= 1
    return D


def _coverage_sweep_flat(F: FlatGraph, order: Sequence[int], name: str) -> _Cand:
    n = F.n
    adj = F.adj
    dominated = bytearray(n)
    ndom = 0
    chosen: List[int] = []
    for v in order:
        if dominated[v]:
            for u in adj[v]:
                if not dominated[u]:
                    break
            else:
                continue
        chosen.append(v)
        if not dominated[v]:
            dominated[v] = 1
            ndom += 1
        for u in adj[v]:
            if not dominated[u]:
                dominated[u] = 1
                ndom += 1
        if ndom == n:
            break
    D = _prune_ds_flat(F, set(chosen))
    return _Cand(D, [v for v in chosen if v in D], name)


def _dynamic_weighted_coverage_flat(F: FlatGraph, O: _Orders) -> _Cand:
    n = F.n
    if n == 0:
        return _Cand(set(), [], "dynamic_weighted_coverage")
    adj = F.adj
    denom = O.denom

    gain = [len(adj[v]) + 1 for v in range(n)]
    max_score = max(gain[v] / denom[v] for v in range(n))
    bucket_count = 256
    buckets: List[List[int]] = [[] for _ in range(bucket_count)]

    def bucket_index(v: int) -> int:
        if max_score <= 0:
            return 0
        idx = int((gain[v] / denom[v] / max_score) * (bucket_count - 1))
        if idx < 0:
            return 0
        if idx > bucket_count - 1:
            return bucket_count - 1
        return idx

    for v in range(n):
        buckets[bucket_index(v)].append(v)

    undom = bytearray(b"\x01") * n
    nundom = n
    in_sel = bytearray(n)
    order: List[int] = []
    top = bucket_count - 1

    while nundom:
        s = -1
        while top >= 0:
            b = buckets[top]
            while b:
                v = b.pop()
                if in_sel[v] or gain[v] <= 0:
                    continue
                iv = bucket_index(v)
                if iv != top:
                    buckets[iv].append(v)
                    continue
                s = v
                break
            if s >= 0:
                break
            top -= 1
        if s < 0:
            break

        in_sel[s] = 1
        order.append(s)
        newly = [s] if undom[s] else []
        for u in adj[s]:
            if undom[u]:
                newly.append(u)
        for x in newly:
            undom[x] = 0
            nundom -= 1
            if not in_sel[x]:
                gain[x] -= 1
                if gain[x] > 0:
                    buckets[bucket_index(x)].append(x)
            for z in adj[x]:
                if in_sel[z]:
                    continue
                gain[z] -= 1
                if gain[z] > 0:
                    buckets[bucket_index(z)].append(z)

    D = _prune_ds_flat(F, set(order))
    return _Cand(D, [v for v in order if v in D], "dynamic_weighted_coverage")


def _witness_score_flat(F: FlatGraph, threshold: int, name: str, O: _Orders) -> _Cand:
    n = F.n
    adj = F.adj
    denom = O.denom
    score = [0.0] * n
    for v in range(n):
        if len(adj[v]) <= threshold:
            contribution = 1.0 / denom[v]
            score[v] += contribution
            for u in adj[v]:
                score[u] += contribution

    key = [score[v] / denom[v] for v in range(n)]
    order = _flat_bucket_order(O.natural, key, 256, True)
    _extend_unique(order, O.ratio, n)
    return _coverage_sweep_flat(F, order, name)


def _ownership_flat(F: FlatGraph, mode: str, O: _Orders) -> _Cand:
    n = F.n
    if n == 0:
        return _Cand(set(), [], f"ownership_{mode}")
    adj = F.adj
    denom = O.denom

    avg = (2 * F.m) // max(1, n)
    threshold = max(2, avg)
    score = [0.0] * n
    np1 = n + 1
    late = mode == "late"

    for x in range(n):
        dx = len(adj[x])
        if dx > threshold:
            continue
        owner = -1
        owner_key = 0.0
        for v in itertools.chain((x,), adj[x]):
            dv = len(adj[v])
            if dv < dx:
                continue
            base = (dv + 1) / denom[v]
            tie = v if late else -v
            key = base * np1 + tie / np1
            if owner < 0 or key > owner_key:
                owner, owner_key = v, key
        if owner >= 0:
            score[owner] += 1.0 / denom[x]

    key2 = [score[v] / denom[v] for v in range(n)]
    order = _flat_bucket_order(O.natural, key2, 256, True)
    _extend_unique(order, O.ratio, n)
    return _coverage_sweep_flat(F, order, f"ownership_{mode}")


def _reverse_delete_flat(F: FlatGraph, mode: str, O: _Orders) -> _Cand:
    n = F.n
    if n == 0:
        return _Cand(set(), [], f"reverse_delete_{mode}")
    adj = F.adj

    if mode == "input":
        order = O.natural
    elif mode == "reverse_input":
        order = O.natural[::-1]
    elif mode == "expensive":
        order = _flat_order(F, O.natural, "expensive")
    elif mode == "cheap":
        order = O.cheap
    elif mode == "coverage_per_weight":
        order = O.ratio
    else:
        raise ValueError(f"unknown reverse-delete mode: {mode}")

    in_d = bytearray(b"\x01") * n
    count = [len(adj[v]) + 1 for v in range(n)]
    for v in order:
        if not in_d[v] or count[v] < 2:
            continue
        removable = True
        for u in adj[v]:
            if count[u] < 2:
                removable = False
                break
        if removable:
            in_d[v] = 0
            count[v] -= 1
            for u in adj[v]:
                count[u] -= 1

    members = [v for v in range(n) if in_d[v]]
    return _Cand(set(members), members, f"reverse_delete_{mode}")


def _seed_complete_flat(
    F: FlatGraph,
    O: _Orders,
    seed_limit: int = 16,
    residual_passes: int = 3,
) -> _Cand:
    n = F.n
    if n == 0:
        return _Cand(set(), [], "seed_complete")
    adj = F.adj
    denom = O.denom

    seed_order = O.ratio
    fallback_order = O.cheap
    # Exact upper bound on any residual score of v: (deg(v)+1)/w(v).
    bound = [(len(adj[v]) + 1) / denom[v] for v in range(n)]

    best: Optional[Set[int]] = None
    best_w = float("inf")
    best_order: List[int] = []

    for seed in seed_order[:seed_limit]:
        in_d = bytearray(n)
        order: List[int] = [seed]
        in_d[seed] = 1
        dom = bytearray(n)
        ndom = 1
        dom[seed] = 1
        for u in adj[seed]:
            if not dom[u]:
                dom[u] = 1
                ndom += 1

        passes = residual_passes
        while ndom < n and passes > 0:
            passes -= 1
            best_v, best_score, best_gain = -1, -1.0, 0
            for v in range(n):
                if in_d[v] or bound[v] <= best_score:
                    continue
                gain = 0 if dom[v] else 1
                for u in adj[v]:
                    if not dom[u]:
                        gain += 1
                score = gain / denom[v]
                if score > best_score:
                    best_v, best_score, best_gain = v, score, gain
            if best_v < 0 or best_gain <= 0:
                break
            in_d[best_v] = 1
            order.append(best_v)
            if not dom[best_v]:
                dom[best_v] = 1
                ndom += 1
            for u in adj[best_v]:
                if not dom[u]:
                    dom[u] = 1
                    ndom += 1

        if ndom < n:
            for sweep in (seed_order, fallback_order):
                if ndom == n:
                    break
                for v in sweep:
                    if ndom == n:
                        break
                    if in_d[v]:
                        continue
                    if not dom[v]:
                        covers = True
                    else:
                        covers = False
                        for u in adj[v]:
                            if not dom[u]:
                                covers = True
                                break
                    if covers:
                        in_d[v] = 1
                        order.append(v)
                        if not dom[v]:
                            dom[v] = 1
                            ndom += 1
                        for u in adj[v]:
                            if not dom[u]:
                                dom[u] = 1
                                ndom += 1

        D = _prune_ds_flat(F, {v for v in order})
        if ndom == n:
            w = _flat_weight(F, D)
            if w < best_w:
                best, best_w = D, w
                best_order = [v for v in order if v in D]

    return _Cand(best or set(), best_order, "seed_complete")


def _salvador_baker_flat(F: FlatGraph, O: _Orders) -> _Cand:
    name = "salvador_baker_weighted_ids"
    n = F.n
    if n == 0:
        return _Cand(set(), [], name)
    if F.m == 0:
        return _Cand(set(range(n)), O.cheap, name)

    adj = F.adj
    w = F.w

    # Linear-size oriented-incidence auxiliary graph, built directly as flat
    # arrays (no NetworkX construction, no graph copy). Each original
    # incidence (u,v) gets one auxiliary node with weight w(u)/max(1,deg(u)).
    removed = bytearray(n)
    aux_adj: List[List[int]] = []
    aux_w: List[float] = []
    aux_src: List[int] = []

    for u in range(n):
        nbrs = [v for v in adj[u] if not removed[v]]
        removed[u] = 1
        first = -1
        prev = -1
        wu = w[u] / max(1, len(adj[u]))
        for v in nbrs:
            a = len(aux_adj)
            aux_adj.append([])
            aux_w.append(wu)
            aux_src.append(u)
            b = len(aux_adj)
            aux_adj.append([])
            aux_w.append(w[v] / max(1, len(adj[v])))
            aux_src.append(v)
            aux_adj[a].append(b)
            aux_adj[b].append(a)
            if prev < 0:
                first = a
            else:
                aux_adj[a].append(prev)
                aux_adj[prev].append(a)
            prev = b
        if len(nbrs) > 1:
            aux_adj[first].append(prev)
            aux_adj[prev].append(first)

    auxF = FlatGraph(aux_adj, aux_w)
    aux_cand = baker_ids.flat_baker_candidate(auxF, eps=1.0)
    aux_order = _flat_order(auxF, list(range(auxF.n)), "coverage_per_weight")

    order: List[int] = []
    seen = bytearray(n)
    for x in sorted(aux_cand):
        v = aux_src[x]
        if not seen[v]:
            seen[v] = 1
            order.append(v)
    for x in aux_order:
        v = aux_src[x]
        if not seen[v]:
            seen[v] = 1
            order.append(v)
    _extend_unique(order, O.ratio, n)

    D = _prune_ds_flat(F, set(order))
    return _Cand(D, [v for v in order if v in D], name)


def _ds_candidates_flat(F: FlatGraph, O: _Orders) -> Iterator[_Cand]:
    avg = (2 * F.m) // max(1, F.n)

    yield _coverage_sweep_flat(F, O.ratio, "ratio_sweep")
    yield _coverage_sweep_flat(F, O.cheap, "cheap_sweep")
    yield _coverage_sweep_flat(F, O.degree, "degree_sweep")
    yield _dynamic_weighted_coverage_flat(F, O)
    yield _witness_score_flat(F, 2, "low_witness", O)
    yield _witness_score_flat(F, max(2, avg), "medium_witness", O)
    yield _ownership_flat(F, "late", O)
    yield _ownership_flat(F, "early", O)
    yield _seed_complete_flat(F, O)
    yield _salvador_baker_flat(F, O)
    yield _reverse_delete_flat(F, "input", O)
    yield _reverse_delete_flat(F, "reverse_input", O)
    yield _reverse_delete_flat(F, "expensive", O)
    yield _reverse_delete_flat(F, "cheap", O)
    yield _reverse_delete_flat(F, "coverage_per_weight", O)


def _priority_order(primary: Iterable[int], secondary: Iterable[int], n: int) -> List[int]:
    out: List[int] = []
    seen = bytearray(n)
    for src in (primary, secondary, range(n)):
        for v in src:
            if not seen[v]:
                seen[v] = 1
                out.append(v)
    return out


def _ids_from_ds_flat(F: FlatGraph, c: _Cand, O: _Orders) -> Iterator[Tuple[Set[int], bool]]:
    """Yield (candidate, feasible_by_construction) pairs from a DS candidate."""
    D = c.vertices
    if not D:
        return
    n = F.n
    if _flat_verify(F, D):
        yield set(D), True

    cheap = O.cheap
    ratio = O.ratio
    yield _flat_repair_prune(F, D, _priority_order(c.order, cheap, n)), True
    yield _flat_repair_prune(F, D, _priority_order(reversed(c.order), cheap, n)), True
    yield _flat_repair_prune(F, D, _priority_order((v for v in cheap if v in D), cheap, n)), True
    yield _flat_repair_prune(F, D, _priority_order((v for v in ratio if v in D), ratio, n)), True


def _weighted_absorb_once_flat(
    F: FlatGraph,
    S: Set[int],
    probe_order: Sequence[int],
) -> Set[int]:
    """
    One safe weighted IDS swap pass.

    For each constant-many probe vertex u, try adding u and removing selected
    neighbours of u when the total weight decreases. Feasibility of a swap is
    checked exactly on the touched closed neighbourhoods only (O(deg) per
    probe instead of a full O(n+m) re-verification). Because the probe list
    is capped by a fixed constant in _best_component, this remains O(n+m).
    """
    n = F.n
    adj = F.adj
    w = F.w

    best = set(S)
    in_best = bytearray(n)
    dom = [0] * n
    for v in best:
        in_best[v] = 1
        dom[v] += 1
        for u in adj[v]:
            dom[u] += 1

    for u in probe_order:
        if in_best[u]:
            continue
        R = [r for r in adj[u] if in_best[r]]
        if not R:
            continue

        delta = w[u]
        for r in R:
            delta -= w[r]
        if delta >= -_EPS:
            continue

        # Exact local domination check on touched vertices.
        touched: Dict[int, int] = {}
        for r in R:
            touched[r] = touched.get(r, 0) - 1
            for x in adj[r]:
                touched[x] = touched.get(x, 0) - 1
        touched[u] = touched.get(u, 0) + 1
        for x in adj[u]:
            touched[x] = touched.get(x, 0) + 1

        feasible = True
        for x, d in touched.items():
            if dom[x] + d < 1:
                feasible = False
                break
        if not feasible:
            continue

        for r in R:
            in_best[r] = 0
            best.remove(r)
        in_best[u] = 1
        best.add(u)
        for x, d in touched.items():
            dom[x] += d

    return best


def _pool_flat(F: FlatGraph, O: _Orders) -> Iterator[Tuple[Set[int], bool]]:
    n = F.n
    natural = O.natural

    for order in (O.ratio, O.cheap, O.degree, natural, natural[::-1]):
        yield _flat_repair_prune(F, set(order), order), True

    ds_list = list(_ds_candidates_flat(F, O))
    for c in ds_list:
        yield from _ids_from_ds_flat(F, c, O)

    probes: List[int] = []
    _extend_unique(probes, O.cheap[:16], n)
    _extend_unique(probes, O.ratio[:16], n)
    for c in ds_list:
        _extend_unique(probes, c.order[:8], n)
        if len(probes) >= 64:
            break

    for seed in probes[:64]:
        yield _flat_repair_prune(F, {seed}, _priority_order([seed], O.cheap, n)), True
        yield _flat_repair_prune(F, {seed}, _priority_order([seed], O.ratio, n)), True


def _best_component_flat(F: FlatGraph) -> Set[int]:
    n = F.n
    O = _Orders(F)
    w = F.w

    probes: List[int] = []
    _extend_unique(probes, O.cheap[:32], n)
    _extend_unique(probes, O.ratio[:32], n)
    probes = probes[:64]

    best: Optional[Set[int]] = None
    best_w = float("inf")
    seen_cands: Set[frozenset] = set()

    for cand, feasible in _pool_flat(F, O):
        key = frozenset(cand)
        if key in seen_cands:
            continue
        seen_cands.add(key)
        if not feasible and not _flat_verify(F, cand):
            continue

        improved = _weighted_absorb_once_flat(F, cand, probes)
        for opt in (cand, improved):
            opt_w = sum(w[v] for v in opt)
            if opt_w < best_w:
                best, best_w = set(opt), opt_w
            if improved is cand or improved == cand:
                break

    if best is None:
        best = _flat_repair_prune(F, set(range(n)), list(range(n)))
        if not _flat_verify(F, best):
            raise RuntimeError("failed to construct a valid weighted IDS")

    return best


def _flat_components(F: FlatGraph) -> List[List[int]]:
    """Connected components as index lists in global node order; O(n+m)."""
    n = F.n
    adj = F.adj
    comp_id = [-1] * n
    cid = 0
    stack: List[int] = []
    for root in range(n):
        if comp_id[root] >= 0:
            continue
        comp_id[root] = cid
        stack.append(root)
        while stack:
            v = stack.pop()
            for u in adj[v]:
                if comp_id[u] < 0:
                    comp_id[u] = cid
                    stack.append(u)
        cid += 1
    comps: List[List[int]] = [[] for _ in range(cid)]
    for v in range(n):
        comps[comp_id[v]].append(v)
    return comps


# ---------------------------------------------------------------------------
# Public solvers
# ---------------------------------------------------------------------------


def find_weighted_independent_dominating_set(graph: nx.Graph, weight: str = "weight") -> Set[Any]:
    """Return a valid weighted independent dominating set heuristic in O(n+m)."""
    G = _clean_graph(graph)
    if G.number_of_nodes() == 0:
        return set()

    F, labels, _ = baker_ids.flat_from_nx(G, weight)
    comps = _flat_components(F)

    solution_idx: Set[int] = set()
    if len(comps) == 1:
        solution_idx = _best_component_flat(F)
    else:
        for comp in comps:
            local = {v: i for i, v in enumerate(comp)}
            adj_c = [[local[u] for u in F.adj[v]] for v in comp]
            w_c = [F.w[v] for v in comp]
            sol = _best_component_flat(FlatGraph(adj_c, w_c))
            solution_idx.update(comp[i] for i in sol)

    if not _flat_verify(F, solution_idx):
        raise RuntimeError("internal error: invalid independent dominating set")

    return {labels[v] for v in solution_idx}


def find_independent_dominating_set(graph: nx.Graph) -> Set[Any]:
    """Unweighted-compatible wrapper; all missing weights are 1.0."""
    return find_weighted_independent_dominating_set(graph, weight="weight")


def find_weighted_independent_dominating_set_brute_force(
    graph: nx.Graph,
    weight: str = "weight",
) -> Optional[Set[Any]]:
    """Exact exponential MWIDS solver for tiny tests only."""
    G = _clean_graph(graph)
    if G.number_of_nodes() == 0:
        return set()

    nodes = list(G.nodes())
    best: Optional[Set[Any]] = None
    best_w = float("inf")

    for r in range(len(nodes) + 1):
        for cand in itertools.combinations(nodes, r):
            S = set(cand)
            if verify_independent_dominating_set(G, S):
                w = calculate_solution_weight(G, S, weight)
                if w < best_w:
                    best, best_w = S, w

    return best

find_independent_dominating_set_brute_force = find_weighted_independent_dominating_set_brute_force
find_independent_dominating_set_approximation = find_weighted_independent_dominating_set
calculate_solution_weight_unweighted_name = calculate_solution_weight
