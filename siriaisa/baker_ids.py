# Created on 08/07/2026
# Author: Frank Vega

"""
Linear Baker-style candidate for Minimum Independent Dominating Set (MIDS).

This module exposes ``baker_ptas_ids(G, eps=1.0, weights=None)``.  It is
intended as a fixed-epsilon candidate generator for the main ``algorithm.py``.
It always returns a valid independent dominating set, and for fixed epsilon it
runs in O(|V| + |E|) time under expected O(1) Python set/dict operations.

Important note
--------------
The classical Baker PTAS solves bounded-layer planar subproblems exactly by
bounded-treewidth dynamic programming.  Exact MIDS dynamic programming is not
implemented here, because the user's target was a practical linear-time final
routine.  Instead, this file uses Baker-style BFS layer deletion plus safe
maximal-independent-set repair on the original graph.  Thus this is a linear
Baker-style IDS candidate, not a universal PTAS proof.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import networkx as nx


def _clean_graph(graph: nx.Graph) -> nx.Graph:
    """Return a simple undirected copy with self-loops removed."""
    if not isinstance(graph, nx.Graph) or graph.is_directed():
        raise ValueError("G must be an undirected NetworkX graph.")
    G = nx.Graph(graph)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def _closed_neighborhood(G: nx.Graph, v: Any) -> Iterable[Any]:
    """Yield v and then all neighbours of v."""
    yield v
    yield from G.neighbors(v)


def verify_independent_dominating_set(G: nx.Graph, S: Optional[Set[Any]]) -> bool:
    """Return True iff S is independent and dominating in O(n + m)."""
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
    """Append allowed values to out without duplicates, preserving order."""
    seen = set(out)
    for v in values:
        if v in allowed and v not in seen:
            out.append(v)
            seen.add(v)


def _weighted_bucket_order(
    G: nx.Graph,
    allowed: Iterable[Any],
    weights: Optional[Dict[Any, float]] = None,
    bucket_count: int = 256,
) -> List[Any]:
    """
    Return a linear-time weighted closed-degree order.

    Vertices are ranked by approximately (degree(v)+1)/weight(v).  A fixed
    number of buckets avoids comparison sorting and keeps the routine linear.
    """
    nodes = list(allowed)
    if not nodes:
        return []
    weights = {} if weights is None else weights

    scores = []
    min_score = float("inf")
    max_score = float("-inf")
    for v in nodes:
        w = max(float(weights.get(v, 1.0)), 1e-12)
        s = (G.degree(v) + 1) / w
        scores.append((v, s))
        if s < min_score:
            min_score = s
        if s > max_score:
            max_score = s

    if max_score <= min_score:
        return nodes

    bucket_count = max(2, bucket_count)
    buckets: List[List[Any]] = [[] for _ in range(bucket_count)]
    scale = (bucket_count - 1) / (max_score - min_score)
    for v, s in scores:
        idx = int((s - min_score) * scale)
        if idx < 0:
            idx = 0
        elif idx >= bucket_count:
            idx = bucket_count - 1
        buckets[idx].append(v)

    order: List[Any] = []
    for idx in range(bucket_count - 1, -1, -1):
        order.extend(buckets[idx])
    return order


def _bfs_layers_all_components(G: nx.Graph) -> Dict[Any, int]:
    """Return BFS depth labels for every component in O(n + m)."""
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


def repair_prune(
    G: nx.Graph,
    candidate: Optional[Set[Any]] = None,
    order: Optional[Sequence[Any]] = None,
) -> Set[Any]:
    """
    Repair candidate into a maximal independent set, hence an IDS.

    A maximal independent set is dominating: every vertex not selected has a
    selected neighbour, otherwise it could be added.  This routine is linear.
    """
    if G.number_of_nodes() == 0:
        return set()

    nodes = set(G.nodes())
    order_list = list(G.nodes()) if order is None else list(order)
    sweep: List[Any] = []
    _append_unique(sweep, order_list, nodes)
    _append_unique(sweep, G.nodes(), nodes)

    candidate = set() if candidate is None else set(candidate) & nodes
    selected: Set[Any] = set()
    dominated_count: Dict[Any, int] = {v: 0 for v in G.nodes()}

    def add_vertex(v: Any) -> None:
        selected.add(v)
        dominated_count[v] += 1
        for u in G.neighbors(v):
            dominated_count[u] += 1

    for v in sweep:
        if v in candidate and dominated_count[v] == 0:
            add_vertex(v)

    for v in sweep:
        if dominated_count[v] == 0:
            add_vertex(v)

    return selected


def _one_layer_candidate(
    G: nx.Graph,
    kept: Set[Any],
    separator: Set[Any],
    weights: Optional[Dict[Any, float]],
) -> Set[Any]:
    """Build one Baker-shift candidate and repair it on the original graph."""
    all_nodes = set(G.nodes())

    kept_order = _weighted_bucket_order(G, kept, weights)
    sep_order = _weighted_bucket_order(G, separator, weights)
    full_order = _weighted_bucket_order(G, all_nodes, weights)

    seed: Set[Any] = set()
    if kept:
        H = G.subgraph(kept).copy()
        seed = repair_prune(H, set(kept_order), kept_order)

    order_a: List[Any] = []
    _append_unique(order_a, seed, all_nodes)
    _append_unique(order_a, kept_order, all_nodes)
    _append_unique(order_a, sep_order, all_nodes)
    _append_unique(order_a, full_order, all_nodes)
    cand_a = repair_prune(G, seed, order_a)

    order_b: List[Any] = []
    _append_unique(order_b, sep_order, all_nodes)
    _append_unique(order_b, kept_order, all_nodes)
    _append_unique(order_b, full_order, all_nodes)
    cand_b = repair_prune(G, set(sep_order), order_b)

    return min((cand_a, cand_b), key=lambda S: (_weight_sum(S, weights), len(S)))


def _weight_sum(S: Set[Any], weights: Optional[Dict[Any, float]]) -> float:
    """Return total weight for tie-breaking."""
    if weights is None:
        return float(len(S))
    return sum(float(weights.get(v, 1.0)) for v in S)


def baker_ptas_ids(
    graph: nx.Graph,
    eps: float = 1.0,
    weights: Optional[Dict[Any, float]] = None,
) -> Set[Any]:
    """
    Return a Baker-style independent dominating set candidate.

    Runtime:
        O((ceil(1/eps)+1) * (|V|+|E|)).
        For eps=1 this is O(|V|+|E|).
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
        cand = _one_layer_candidate(G, kept, separator, weights)
        if verify_independent_dominating_set(G, cand):
            candidates.append(cand)

    full_order = _weighted_bucket_order(G, nodes, weights)
    full_cand = repair_prune(G, set(full_order), full_order)
    if verify_independent_dominating_set(G, full_cand):
        candidates.append(full_cand)

    if not candidates:
        fallback = repair_prune(G, set(G.nodes()), list(G.nodes()))
        if not verify_independent_dominating_set(G, fallback):
            raise RuntimeError("failed to construct a valid independent dominating set")
        return fallback

    return min(candidates, key=lambda S: (_weight_sum(S, weights), len(S)))