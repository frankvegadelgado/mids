# Created on 08/07/2026
# Author: Frank Vega

"""
Weighted Baker-style candidate for the Minimum Weighted Independent
Dominating Set problem (MWIDS).

The public function ``baker_layer_weighted_ids_candidate`` is a fixed-epsilon
linear-time candidate generator.  With eps=1.0 it performs only a constant
number of BFS-layer shifts.  It always returns a valid independent dominating
set, but it is not a true Baker PTAS: exact bounded-treewidth MWIDS dynamic
programming is intentionally not implemented because the requested final
routine must remain linear-time.

Weights are read from node attribute ``weight`` by default.  Missing weights
are treated as 1.0.  Non-positive weights are accepted for summation, but the
ordering ratios use a tiny positive denominator to avoid division by zero.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import networkx as nx


_EPS = 1.0e-12


def _clean_graph(graph: nx.Graph) -> nx.Graph:
    """Return a simple undirected copy with self-loops removed."""
    if not isinstance(graph, nx.Graph) or graph.is_directed():
        raise ValueError("G must be an undirected NetworkX graph.")
    G = nx.Graph(graph)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def _closed(G: nx.Graph, v: Any) -> Iterable[Any]:
    """Yield the closed neighbourhood N[v]."""
    yield v
    yield from G.neighbors(v)


def _node_weight(G: nx.Graph, v: Any, weight: str = "weight") -> float:
    """Return the objective weight of v."""
    return float(G.nodes[v].get(weight, 1.0))


def _ratio_denominator(G: nx.Graph, v: Any, weight: str = "weight") -> float:
    """Positive denominator used only for ordering ratios."""
    return max(_node_weight(G, v, weight), _EPS)


def solution_weight(G: nx.Graph, S: Iterable[Any], weight: str = "weight") -> float:
    """Return total solution weight."""
    return sum(_node_weight(G, v, weight) for v in S)


def verify_independent_dominating_set(G: nx.Graph, S: Optional[Set[Any]]) -> bool:
    """Return True iff S is independent and dominating in O(n+m)."""
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


def _append_unique(out: List[Any], values: Iterable[Any], allowed: Set[Any]) -> None:
    """Append allowed values to out without duplicates."""
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
    """Bucket-order arbitrary numeric scores in linear time for fixed buckets."""
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
        if idx < 0:
            idx = 0
        elif idx >= bucket_count:
            idx = bucket_count - 1
        buckets[idx].append(v)

    order: List[Any] = []
    rng = range(bucket_count - 1, -1, -1) if high_to_low else range(bucket_count)
    for idx in rng:
        order.extend(buckets[idx])
    return order


def weighted_bucket_order(
    G: nx.Graph,
    allowed: Iterable[Any],
    weight: str = "weight",
    mode: str = "coverage_per_weight",
    bucket_count: int = 256,
) -> List[Any]:
    """
    Return a linear-time order for weighted IDS repair.

    Supported modes:
      * coverage_per_weight: large |N[v]| / w(v) first.
      * cheap: small w(v) first.
      * expensive: large w(v) first.
      * degree: large degree first.
    """
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
    raise ValueError(f"unknown weighted order mode: {mode}")


def _bfs_layers_all_components(G: nx.Graph) -> Dict[Any, int]:
    """Return BFS depths for every component in O(n+m)."""
    layers: Dict[Any, int] = {}
    for root in G.nodes():
        if root in layers:
            continue
        layers[root] = 0
        q = deque([root])
        while q:
            v = q.popleft()
            for u in G.neighbors(v):
                if u not in layers:
                    layers[u] = layers[v] + 1
                    q.append(u)
    return layers


def repair_prune_weighted(
    G: nx.Graph,
    candidate: Optional[Set[Any]] = None,
    order: Optional[Sequence[Any]] = None,
    weight: str = "weight",
) -> Set[Any]:
    """
    Repair a seed into a maximal independent set, then prune by weight.

    Greedy maximal independent repair guarantees domination. The prune phase
    removes a selected vertex whenever domination remains valid. Runtime is
    O(n+m).
    """
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

    prune_order = weighted_bucket_order(G, selected, weight=weight, mode="expensive")
    for v in prune_order:
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


def _one_layer_candidate(
    G: nx.Graph,
    kept: Set[Any],
    separator: Set[Any],
    weight: str,
) -> Set[Any]:
    """Build one weighted Baker-layer candidate and repair on G."""
    all_nodes = set(G.nodes())
    kept_order = weighted_bucket_order(G, kept, weight=weight, mode="coverage_per_weight")
    sep_order = weighted_bucket_order(G, separator, weight=weight, mode="coverage_per_weight")
    cheap_order = weighted_bucket_order(G, all_nodes, weight=weight, mode="cheap")
    ratio_order = weighted_bucket_order(G, all_nodes, weight=weight, mode="coverage_per_weight")

    seed: Set[Any] = set()
    if kept:
        H = G.subgraph(kept).copy()
        seed = repair_prune_weighted(H, set(kept_order), kept_order, weight=weight)

    order_a: List[Any] = []
    _append_unique(order_a, seed, all_nodes)
    _append_unique(order_a, kept_order, all_nodes)
    _append_unique(order_a, sep_order, all_nodes)
    _append_unique(order_a, cheap_order, all_nodes)
    _append_unique(order_a, ratio_order, all_nodes)
    cand_a = repair_prune_weighted(G, seed, order_a, weight=weight)

    order_b: List[Any] = []
    _append_unique(order_b, sep_order, all_nodes)
    _append_unique(order_b, kept_order, all_nodes)
    _append_unique(order_b, cheap_order, all_nodes)
    _append_unique(order_b, ratio_order, all_nodes)
    cand_b = repair_prune_weighted(G, set(sep_order), order_b, weight=weight)

    return min((cand_a, cand_b), key=lambda S: (solution_weight(G, S, weight), len(S)))


def baker_layer_weighted_ids_candidate(
    graph: nx.Graph,
    eps: float = 1.0,
    weight: str = "weight",
) -> Set[Any]:
    """
    Return a weighted Baker-style IDS candidate.

    Runtime is O((ceil(1/eps)+1)(n+m)). For eps=1.0 this is linear.
    """
    if eps <= 0:
        raise ValueError("eps must be positive.")
    G = _clean_graph(graph)
    if G.number_of_nodes() == 0:
        return set()

    period = max(2, int(1.0 / eps + 0.999999999) + 1)
    layers = _bfs_layers_all_components(G)
    nodes = set(G.nodes())
    candidates: List[Set[Any]] = []

    for residue in range(period):
        separator = {v for v in nodes if layers[v] % period == residue}
        kept = nodes - separator
        cand = _one_layer_candidate(G, kept, separator, weight)
        if verify_independent_dominating_set(G, cand):
            candidates.append(cand)

    ratio_order = weighted_bucket_order(G, nodes, weight=weight, mode="coverage_per_weight")
    cheap_order = weighted_bucket_order(G, nodes, weight=weight, mode="cheap")
    for order in (ratio_order, cheap_order):
        cand = repair_prune_weighted(G, set(order), order, weight=weight)
        if verify_independent_dominating_set(G, cand):
            candidates.append(cand)

    if not candidates:
        fallback = repair_prune_weighted(G, set(G.nodes()), list(G.nodes()), weight=weight)
        if not verify_independent_dominating_set(G, fallback):
            raise RuntimeError("failed to construct a valid weighted IDS")
        return fallback

    return min(candidates, key=lambda S: (solution_weight(G, S, weight), len(S)))


# Backward-compatible alias.
baker_ptas_ids = baker_layer_weighted_ids_candidate