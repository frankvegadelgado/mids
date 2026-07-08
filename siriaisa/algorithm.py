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

Guarantee: feasibility and O(|V|+|E|) time for a fixed candidate pool, assuming
expected O(1) Python dict/set operations. This is a heuristic, not an exact
solver and not a proved constant-approximation algorithm.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import networkx as nx

try:
    from . import baker_ids
except Exception:
    import baker_ids


_EPS = 1.0e-12


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


def _append_unique(out: List[Any], values: Iterable[Any], allowed: Set[Any]) -> None:
    seen = set(out)
    for v in values:
        if v in allowed and v not in seen:
            out.append(v)
            seen.add(v)


def _fixed_bucket_order_from_scores(
    nodes: List[Any],
    scores: Dict[Any, float],
    bucket_count: int = 256,
    high_to_low: bool = True,
) -> List[Any]:
    if not nodes:
        return []
    bucket_count = max(2, bucket_count)
    lo = min(scores[v] for v in nodes)
    hi = max(scores[v] for v in nodes)
    if hi <= lo:
        return list(nodes)

    buckets: List[List[Any]] = [[] for _ in range(bucket_count)]
    scale = (bucket_count - 1) / (hi - lo)
    for v in nodes:
        idx = int((scores[v] - lo) * scale)
        idx = max(0, min(idx, bucket_count - 1))
        buckets[idx].append(v)

    out: List[Any] = []
    rng = range(bucket_count - 1, -1, -1) if high_to_low else range(bucket_count)
    for idx in rng:
        out.extend(buckets[idx])
    return out


def _weighted_order(
    G: nx.Graph,
    allowed: Iterable[Any],
    weight: str = "weight",
    mode: str = "coverage_per_weight",
    bucket_count: int = 256,
) -> List[Any]:
    nodes = list(allowed)
    if not nodes:
        return []

    if mode == "coverage_per_weight":
        scores = {v: (G.degree(v) + 1) / _ratio_denominator(G, v, weight) for v in nodes}
        return _fixed_bucket_order_from_scores(nodes, scores, bucket_count, True)
    if mode == "cheap":
        scores = {v: _node_weight(G, v, weight) for v in nodes}
        return _fixed_bucket_order_from_scores(nodes, scores, bucket_count, False)
    if mode == "expensive":
        scores = {v: _node_weight(G, v, weight) for v in nodes}
        return _fixed_bucket_order_from_scores(nodes, scores, bucket_count, True)
    if mode == "degree":
        scores = {v: float(G.degree(v)) for v in nodes}
        return _fixed_bucket_order_from_scores(nodes, scores, bucket_count, True)
    if mode == "low_degree":
        scores = {v: float(G.degree(v)) for v in nodes}
        return _fixed_bucket_order_from_scores(nodes, scores, bucket_count, False)
    raise ValueError(f"unknown order mode: {mode}")


def verify_independent_dominating_set(G: nx.Graph, S: Optional[Set[Any]]) -> bool:
    if S is None:
        return False
    S = set(S)
    if not S.issubset(G.nodes):
        return False

    selected = {v: False for v in G.nodes()}
    dominated = {v: False for v in G.nodes()}
    for v in S:
        selected[v] = True
        dominated[v] = True

    for u, v in G.edges():
        if selected[u] and selected[v]:
            return False
        if selected[u]:
            dominated[v] = True
        if selected[v]:
            dominated[u] = True
    return all(dominated.values())


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
    if G.number_of_nodes() == 0:
        return set()

    allowed = set(G.nodes())
    sweep: List[Any] = []
    _append_unique(sweep, list(G.nodes()) if order is None else order, allowed)
    _append_unique(sweep, G.nodes(), allowed)
    candidate = set() if candidate is None else set(candidate) & allowed

    selected: Set[Any] = set()
    dom_count: Dict[Any, int] = {v: 0 for v in G.nodes()}

    def add(v: Any) -> None:
        selected.add(v)
        dom_count[v] += 1
        for u in G.neighbors(v):
            dom_count[u] += 1

    for v in sweep:
        if v in candidate and dom_count[v] == 0:
            add(v)
    for v in sweep:
        if dom_count[v] == 0:
            add(v)

    for v in _weighted_order(G, selected, weight=weight, mode="expensive"):
        if v not in selected or dom_count[v] <= 1:
            continue
        removable = True
        for u in G.neighbors(v):
            if dom_count[u] <= 1:
                removable = False
                break
        if removable:
            selected.remove(v)
            dom_count[v] -= 1
            for u in G.neighbors(v):
                dom_count[u] -= 1
    return selected


repair_prune = repair_prune_weighted


class _Cand:
    __slots__ = ("vertices", "order", "name")

    def __init__(self, vertices: Iterable[Any], order: Iterable[Any], name: str):
        self.vertices = set(vertices)
        self.order = list(order)
        self.name = name


def _prune_ds_weighted(G: nx.Graph, D: Set[Any], weight: str) -> Set[Any]:
    D = set(D) & set(G.nodes())
    count: Dict[Any, int] = {v: 0 for v in G.nodes()}
    for v in D:
        for u in _closed(G, v):
            count[u] += 1

    for v in _weighted_order(G, D, weight=weight, mode="expensive"):
        if v not in D:
            continue
        if all(count[u] >= 2 for u in _closed(G, v)):
            D.remove(v)
            for u in _closed(G, v):
                count[u] -= 1
    return D


def _coverage_sweep(G: nx.Graph, order: Sequence[Any], name: str, weight: str) -> _Cand:
    dominated: Set[Any] = set()
    chosen: List[Any] = []
    n = G.number_of_nodes()
    for v in order:
        if any(u not in dominated for u in _closed(G, v)):
            chosen.append(v)
            for u in _closed(G, v):
                dominated.add(u)
            if len(dominated) == n:
                break
    D = _prune_ds_weighted(G, set(chosen), weight)
    return _Cand(D, [v for v in chosen if v in D], name)


def _dynamic_weighted_coverage(G: nx.Graph, weight: str) -> _Cand:
    n = G.number_of_nodes()
    if n == 0:
        return _Cand(set(), [], "dynamic_weighted_coverage")

    gain = {v: G.degree(v) + 1 for v in G.nodes()}
    max_score = max(gain[v] / _ratio_denominator(G, v, weight) for v in G.nodes())
    bucket_count = 256
    buckets: List[List[Any]] = [[] for _ in range(bucket_count)]

    def bucket_index(v: Any) -> int:
        if max_score <= 0:
            return 0
        score = gain[v] / _ratio_denominator(G, v, weight)
        idx = int((score / max_score) * (bucket_count - 1))
        return max(0, min(idx, bucket_count - 1))

    for v in G.nodes():
        buckets[bucket_index(v)].append(v)

    undominated = set(G.nodes())
    selected: Set[Any] = set()
    order: List[Any] = []
    top = bucket_count - 1

    while undominated:
        s = None
        while top >= 0:
            while buckets[top]:
                v = buckets[top].pop()
                if v in selected or gain[v] <= 0:
                    continue
                if bucket_index(v) != top:
                    buckets[bucket_index(v)].append(v)
                    continue
                s = v
                break
            if s is not None:
                break
            top -= 1
        if s is None:
            break

        selected.add(s)
        order.append(s)
        newly = [w for w in _closed(G, s) if w in undominated]
        for w in newly:
            undominated.remove(w)
            for z in _closed(G, w):
                if z in selected:
                    continue
                gain[z] -= 1
                if gain[z] > 0:
                    buckets[bucket_index(z)].append(z)

    D = _prune_ds_weighted(G, set(order), weight)
    return _Cand(D, [v for v in order if v in D], "dynamic_weighted_coverage")


def _witness_score(G: nx.Graph, threshold: int, name: str, weight: str) -> _Cand:
    score = {v: 0.0 for v in G.nodes()}
    deg = dict(G.degree())
    for w, d in deg.items():
        if d <= threshold:
            contribution = 1.0 / _ratio_denominator(G, w, weight)
            score[w] += contribution
            for v in G.neighbors(w):
                score[v] += contribution

    order = _fixed_bucket_order_from_scores(
        list(G.nodes()),
        {v: score[v] / _ratio_denominator(G, v, weight) for v in G.nodes()},
        256,
        True,
    )
    _append_unique(order, _weighted_order(G, G.nodes(), weight=weight, mode="coverage_per_weight"), set(G.nodes()))
    return _coverage_sweep(G, order, name, weight)


def _ownership(G: nx.Graph, mode: str, weight: str) -> _Cand:
    nodes = list(G.nodes())
    if not nodes:
        return _Cand(set(), [], f"ownership_{mode}")

    pos = {v: i for i, v in enumerate(nodes)}
    deg = dict(G.degree())
    avg = (2 * G.number_of_edges()) // max(1, len(nodes))
    threshold = max(2, avg)
    score = {v: 0.0 for v in nodes}

    for w in nodes:
        if deg[w] > threshold:
            continue
        owner, owner_key = None, None
        for v in _closed(G, w):
            if deg.get(v, 0) < deg[w]:
                continue
            base = (deg.get(v, 0) + 1) / _ratio_denominator(G, v, weight)
            tie = pos[v] if mode == "late" else -pos[v]
            key = base * (len(nodes) + 1) + tie / max(1, len(nodes) + 1)
            if owner is None or key > owner_key:
                owner, owner_key = v, key
        if owner is not None:
            score[owner] += 1.0 / _ratio_denominator(G, w, weight)

    order = _fixed_bucket_order_from_scores(
        nodes,
        {v: score[v] / _ratio_denominator(G, v, weight) for v in nodes},
        256,
        True,
    )
    _append_unique(order, _weighted_order(G, nodes, weight=weight, mode="coverage_per_weight"), set(G.nodes()))
    return _coverage_sweep(G, order, f"ownership_{mode}", weight)


def _reverse_delete(G: nx.Graph, mode: str, weight: str) -> _Cand:
    nodes = list(G.nodes())
    if not nodes:
        return _Cand(set(), [], f"reverse_delete_{mode}")

    if mode == "input":
        order = nodes
    elif mode == "reverse_input":
        order = list(reversed(nodes))
    elif mode == "expensive":
        order = _weighted_order(G, nodes, weight=weight, mode="expensive")
    elif mode == "cheap":
        order = _weighted_order(G, nodes, weight=weight, mode="cheap")
    elif mode == "coverage_per_weight":
        order = _weighted_order(G, nodes, weight=weight, mode="coverage_per_weight")
    else:
        raise ValueError(f"unknown reverse-delete mode: {mode}")

    D = set(nodes)
    count = {v: G.degree(v) + 1 for v in nodes}
    for v in order:
        if v in D and all(count[u] >= 2 for u in _closed(G, v)):
            D.remove(v)
            for u in _closed(G, v):
                count[u] -= 1
    return _Cand(D, [v for v in nodes if v in D], f"reverse_delete_{mode}")


def _seed_complete(G: nx.Graph, weight: str, seed_limit: int = 16, residual_passes: int = 3) -> _Cand:
    n = G.number_of_nodes()
    if n == 0:
        return _Cand(set(), [], "seed_complete")

    seed_order = _weighted_order(G, G.nodes(), weight=weight, mode="coverage_per_weight")
    fallback_order = _weighted_order(G, G.nodes(), weight=weight, mode="cheap")
    ratio_order = seed_order
    best: Optional[Set[Any]] = None
    best_order: List[Any] = []

    for seed in seed_order[:seed_limit]:
        D: Set[Any] = {seed}
        order = [seed]
        dominated = set(_closed(G, seed))

        passes = residual_passes
        while len(dominated) < n and passes > 0:
            passes -= 1
            best_v, best_score, best_gain = None, -1.0, 0
            for v in G.nodes():
                if v in D:
                    continue
                gain = sum(1 for u in _closed(G, v) if u not in dominated)
                score = gain / _ratio_denominator(G, v, weight)
                if score > best_score:
                    best_v, best_score, best_gain = v, score, gain
            if best_v is None or best_gain <= 0:
                break
            D.add(best_v)
            order.append(best_v)
            for u in _closed(G, best_v):
                dominated.add(u)

        if len(dominated) < n:
            for v in ratio_order:
                if v not in D and any(u not in dominated for u in _closed(G, v)):
                    D.add(v)
                    order.append(v)
                    for u in _closed(G, v):
                        dominated.add(u)
                    if len(dominated) == n:
                        break
            for v in fallback_order:
                if len(dominated) == n:
                    break
                if v not in D and any(u not in dominated for u in _closed(G, v)):
                    D.add(v)
                    order.append(v)
                    for u in _closed(G, v):
                        dominated.add(u)

        D = _prune_ds_weighted(G, D, weight)
        if _is_dominating_set(G, D):
            if best is None or calculate_solution_weight(G, D, weight) < calculate_solution_weight(G, best, weight):
                best, best_order = D, [v for v in order if v in D]

    return _Cand(best or set(), best_order, "seed_complete")


def _build_salvador_aux(G: nx.Graph, weight: str) -> nx.Graph:
    """
    Build a linear-size oriented-incidence auxiliary graph.

    Each original incidence (u,v) gets node ("inc",u,v). Its auxiliary weight
    is w(u)/max(1,deg(u)).
    """
    B = nx.Graph()
    W = G.copy()
    deg = dict(G.degree())

    for u in list(G.nodes()):
        if u not in W:
            continue
        nbrs = list(W.neighbors(u))
        W.remove_node(u)
        first, prev = None, None

        for v in nbrs:
            x_uv = ("inc", u, v)
            x_vu = ("inc", v, u)
            B.add_node(x_uv, weight=_node_weight(G, u, weight) / max(1, deg.get(u, 1)))
            B.add_node(x_vu, weight=_node_weight(G, v, weight) / max(1, deg.get(v, 1)))
            B.add_edge(x_uv, x_vu)
            if prev is None:
                first = x_uv
            else:
                B.add_edge(x_uv, prev)
            prev = x_vu

        if len(nbrs) > 1 and first is not None and prev is not None:
            B.add_edge(first, prev)

    return B


def _salvador_baker_weighted_ids(G: nx.Graph, weight: str) -> _Cand:
    if G.number_of_nodes() == 0:
        return _Cand(set(), [], "salvador_baker_weighted_ids")
    if G.number_of_edges() == 0:
        nodes = set(G.nodes())
        return _Cand(nodes, _weighted_order(G, nodes, weight=weight, mode="cheap"), "salvador_baker_weighted_ids")

    B = _build_salvador_aux(G, weight)
    if B.number_of_nodes() == 0:
        return _Cand(set(), [], "salvador_baker_weighted_ids")

    aux = baker_ids.baker_layer_weighted_ids_candidate(B, eps=1.0, weight="weight")
    aux_order = baker_ids.weighted_bucket_order(B, B.nodes(), weight="weight", mode="coverage_per_weight")

    order: List[Any] = []
    seen: Set[Any] = set()

    def decode(x: Any) -> None:
        if isinstance(x, tuple) and len(x) == 3 and x[0] == "inc":
            v = x[1]
            if v in G and v not in seen:
                order.append(v)
                seen.add(v)

    for x in aux:
        decode(x)
    for x in aux_order:
        decode(x)

    _append_unique(order, _weighted_order(G, G.nodes(), weight=weight, mode="coverage_per_weight"), set(G.nodes()))

    D = set(order)
    if _is_dominating_set(G, D):
        D = _prune_ds_weighted(G, D, weight)
        order = [v for v in order if v in D]

    return _Cand(D, order, "salvador_baker_weighted_ids")


def _ds_candidates(G: nx.Graph, weight: str) -> Iterator[_Cand]:
    nodes = set(G.nodes())
    avg = (2 * G.number_of_edges()) // max(1, G.number_of_nodes())

    yield _coverage_sweep(G, _weighted_order(G, nodes, weight=weight, mode="coverage_per_weight"), "ratio_sweep", weight)
    yield _coverage_sweep(G, _weighted_order(G, nodes, weight=weight, mode="cheap"), "cheap_sweep", weight)
    yield _coverage_sweep(G, _weighted_order(G, nodes, weight=weight, mode="degree"), "degree_sweep", weight)
    yield _dynamic_weighted_coverage(G, weight)
    yield _witness_score(G, 2, "low_witness", weight)
    yield _witness_score(G, max(2, avg), "medium_witness", weight)
    yield _ownership(G, "late", weight)
    yield _ownership(G, "early", weight)
    yield _seed_complete(G, weight)
    yield _salvador_baker_weighted_ids(G, weight)
    yield _reverse_delete(G, "input", weight)
    yield _reverse_delete(G, "reverse_input", weight)
    yield _reverse_delete(G, "expensive", weight)
    yield _reverse_delete(G, "cheap", weight)
    yield _reverse_delete(G, "coverage_per_weight", weight)


def _priority_order(primary: Iterable[Any], secondary: Iterable[Any], G: nx.Graph) -> List[Any]:
    out: List[Any] = []
    allowed = set(G.nodes())
    _append_unique(out, primary, allowed)
    _append_unique(out, secondary, allowed)
    _append_unique(out, G.nodes(), allowed)
    return out


def _ids_from_ds(G: nx.Graph, c: _Cand, ratio: List[Any], cheap: List[Any], weight: str) -> Iterator[Set[Any]]:
    D = set(c.vertices) & set(G.nodes())
    if not D:
        return
    if verify_independent_dominating_set(G, D):
        yield D

    yield repair_prune_weighted(G, D, _priority_order(c.order, cheap, G), weight)
    yield repair_prune_weighted(G, D, _priority_order(reversed(c.order), cheap, G), weight)
    yield repair_prune_weighted(G, D, _priority_order((v for v in cheap if v in D), cheap, G), weight)
    yield repair_prune_weighted(G, D, _priority_order((v for v in ratio if v in D), ratio, G), weight)


def _weighted_absorb_once(G: nx.Graph, S: Set[Any], probe_order: Sequence[Any], weight: str) -> Set[Any]:
    """
    One safe weighted IDS swap pass.

    For each constant-many probe vertex u, try adding u and removing selected
    neighbours of u when the total weight decreases. The candidate is accepted
    only after a full linear verification. Because the probe list is capped by
    a fixed constant in _best_component, this remains O(n+m).
    """
    best = set(S)
    best_w = calculate_solution_weight(G, best, weight)

    for u in probe_order:
        if u in best:
            continue

        R = {r for r in G.neighbors(u) if r in best}
        if not R:
            continue

        cand = set(best)
        cand.difference_update(R)

        if any(v in cand for v in G.neighbors(u)):
            continue

        cand.add(u)
        cand_w = calculate_solution_weight(G, cand, weight)
        if cand_w + _EPS >= best_w:
            continue

        if verify_independent_dominating_set(G, cand):
            best = cand
            best_w = cand_w

    return best


def _pool(G: nx.Graph, weight: str) -> Iterator[Set[Any]]:
    nodes = set(G.nodes())
    ratio = _weighted_order(G, nodes, weight=weight, mode="coverage_per_weight")
    cheap = _weighted_order(G, nodes, weight=weight, mode="cheap")
    degree = _weighted_order(G, nodes, weight=weight, mode="degree")
    natural = list(G.nodes())

    for order in (ratio, cheap, degree, natural, list(reversed(natural))):
        yield repair_prune_weighted(G, set(order), order, weight)

    ds_list = list(_ds_candidates(G, weight))
    for c in ds_list:
        yield from _ids_from_ds(G, c, ratio, cheap, weight)

    probes: List[Any] = []
    _append_unique(probes, cheap[:16], nodes)
    _append_unique(probes, ratio[:16], nodes)
    for c in ds_list:
        _append_unique(probes, c.order[:8], nodes)
        if len(probes) >= 64:
            break

    for seed in probes[:64]:
        yield repair_prune_weighted(G, {seed}, _priority_order([seed], cheap, G), weight)
        yield repair_prune_weighted(G, {seed}, _priority_order([seed], ratio, G), weight)


def _best_component(G: nx.Graph, weight: str) -> Set[Any]:
    ratio = _weighted_order(G, G.nodes(), weight=weight, mode="coverage_per_weight")
    cheap = _weighted_order(G, G.nodes(), weight=weight, mode="cheap")

    probes: List[Any] = []
    _append_unique(probes, cheap[:32], set(G.nodes()))
    _append_unique(probes, ratio[:32], set(G.nodes()))

    best: Optional[Set[Any]] = None
    for cand in _pool(G, weight):
        if not verify_independent_dominating_set(G, cand):
            continue

        improved = _weighted_absorb_once(G, cand, probes[:64], weight)
        for opt in (cand, improved):
            if not verify_independent_dominating_set(G, opt):
                continue
            if best is None or calculate_solution_weight(G, opt, weight) < calculate_solution_weight(G, best, weight):
                best = set(opt)

    if best is None:
        best = repair_prune_weighted(G, set(G.nodes()), list(G.nodes()), weight)
        if not verify_independent_dominating_set(G, best):
            raise RuntimeError("failed to construct a valid weighted IDS")

    return best


def find_weighted_independent_dominating_set(graph: nx.Graph, weight: str = "weight") -> Set[Any]:
    """Return a valid weighted independent dominating set heuristic in O(n+m)."""
    G = _clean_graph(graph)
    if G.number_of_nodes() == 0:
        return set()

    solution: Set[Any] = set()
    for comp in nx.connected_components(G):
        H = G.subgraph(comp).copy()
        solution.update(_best_component(H, weight))

    if not verify_independent_dominating_set(G, solution):
        raise RuntimeError("internal error: invalid independent dominating set")

    return solution


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


find_independent_dominating_set_approximation = find_weighted_independent_dominating_set
calculate_solution_weight_unweighted_name = calculate_solution_weight