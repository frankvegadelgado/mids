# Created on 08/07/2026
# Author: Frank Vega

"""
Weighted Baker-style candidate for the Minimum Weighted Independent
Dominating Set problem (MWIDS) -- flat-array engine.

The public function ``baker_layer_weighted_ids_candidate`` is a fixed-epsilon
linear-time candidate generator.  With eps=1.0 it performs only a constant
number of BFS-layer shifts.  It always returns a valid independent dominating
set, but it is not a true Baker PTAS: exact bounded-treewidth MWIDS dynamic
programming is intentionally not implemented because the requested final
routine must remain linear-time.

This module also exposes the shared flat (index-based) primitives used by
``algorithm.py``.  All heavy lifting happens on ``FlatGraph`` -- plain Python
lists indexed by integer vertex ids -- which removes per-call NetworkX
overhead and graph/subgraph copies.  Every routine remains O(n+m) with small
constants.

Weights are read from node attribute ``weight`` by default.  Missing weights
are treated as 1.0.  Non-positive weights are accepted for summation, but the
ordering ratios use a tiny positive denominator to avoid division by zero.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx


_EPS = 1.0e-12


# ---------------------------------------------------------------------------
# Flat graph representation
# ---------------------------------------------------------------------------


class FlatGraph:
    """Immutable index-based graph: adjacency lists of ints plus weights."""

    __slots__ = ("n", "m", "adj", "w")

    def __init__(self, adj: List[List[int]], w: List[float]):
        self.adj = adj
        self.w = w
        self.n = len(adj)
        self.m = sum(len(a) for a in adj) // 2


def flat_from_nx(G: nx.Graph, weight: str = "weight") -> Tuple[FlatGraph, List[Any], Dict[Any, int]]:
    """Convert a simple undirected nx graph once into flat arrays."""
    labels = list(G.nodes())
    idx = {v: i for i, v in enumerate(labels)}
    adj: List[List[int]] = []
    for v in labels:
        adj.append([idx[u] for u in G.neighbors(v) if u != v])
    w = [float(G.nodes[v].get(weight, 1.0)) for v in labels]
    return FlatGraph(adj, w), labels, idx


def flat_denominators(F: FlatGraph) -> List[float]:
    """Positive per-vertex denominators used only for ordering ratios."""
    return [x if x > _EPS else _EPS for x in F.w]


def flat_solution_weight(F: FlatGraph, S: Iterable[int]) -> float:
    w = F.w
    return sum(w[v] for v in S)


def flat_verify_ids(F: FlatGraph, S: Optional[Iterable[int]]) -> bool:
    """Return True iff S is independent and dominating; O(n+m)."""
    if S is None:
        return False
    n = F.n
    adj = F.adj
    in_s = bytearray(n)
    for v in S:
        if not 0 <= v < n:
            return False
        in_s[v] = 1
    for v in range(n):
        if in_s[v]:
            for u in adj[v]:
                if in_s[u]:
                    return False
        else:
            for u in adj[v]:
                if in_s[u]:
                    break
            else:
                return False
    return True


# ---------------------------------------------------------------------------
# Linear-time bucket orders
# ---------------------------------------------------------------------------


def flat_bucket_order(
    nodes: Sequence[int],
    score: Sequence[float],
    bucket_count: int = 256,
    high_to_low: bool = True,
) -> List[int]:
    """Bucket-order nodes by ``score[v]`` in linear time for fixed buckets."""
    if not nodes:
        return []
    bucket_count = max(2, bucket_count)
    it = iter(nodes)
    first = next(it)
    lo = hi = score[first]
    for v in it:
        s = score[v]
        if s < lo:
            lo = s
        elif s > hi:
            hi = s
    if hi <= lo:
        return list(nodes)

    buckets: List[List[int]] = [[] for _ in range(bucket_count)]
    scale = (bucket_count - 1) / (hi - lo)
    top = bucket_count - 1
    for v in nodes:
        idx = int((score[v] - lo) * scale)
        if idx < 0:
            idx = 0
        elif idx > top:
            idx = top
        buckets[idx].append(v)

    out: List[int] = []
    if high_to_low:
        for b in reversed(buckets):
            out.extend(b)
    else:
        for b in buckets:
            out.extend(b)
    return out


def flat_order(
    F: FlatGraph,
    nodes: Sequence[int],
    mode: str = "coverage_per_weight",
    bucket_count: int = 256,
) -> List[int]:
    """
    Return a linear-time order for weighted IDS repair.

    Supported modes:
      * coverage_per_weight: large |N[v]| / w(v) first.
      * cheap: small w(v) first.
      * expensive: large w(v) first.
      * degree: large degree first.
      * low_degree: small degree first.
    """
    if not nodes:
        return []
    adj = F.adj
    w = F.w
    if mode == "coverage_per_weight":
        score = [(len(adj[v]) + 1) / (w[v] if w[v] > _EPS else _EPS) for v in range(F.n)]
        return flat_bucket_order(nodes, score, bucket_count, True)
    if mode == "cheap":
        return flat_bucket_order(nodes, w, bucket_count, False)
    if mode == "expensive":
        return flat_bucket_order(nodes, w, bucket_count, True)
    if mode == "degree":
        score = [float(len(a)) for a in adj]
        return flat_bucket_order(nodes, score, bucket_count, True)
    if mode == "low_degree":
        score = [float(len(a)) for a in adj]
        return flat_bucket_order(nodes, score, bucket_count, False)
    raise ValueError(f"unknown weighted order mode: {mode}")


def extend_unique(out: List[int], more: Iterable[int], n: int) -> None:
    """Append new ids to ``out`` without duplicates (O(len) with a mask)."""
    seen = bytearray(n)
    for v in out:
        seen[v] = 1
    for v in more:
        if not seen[v]:
            seen[v] = 1
            out.append(v)


# ---------------------------------------------------------------------------
# Repair + prune (the core feasibility routine)
# ---------------------------------------------------------------------------


def flat_repair_prune(
    F: FlatGraph,
    candidate: Optional[Iterable[int]] = None,
    order: Optional[Sequence[int]] = None,
    allowed: Optional[bytearray] = None,
) -> Set[int]:
    """
    Repair a seed into a maximal independent set, then prune by weight.

    ``allowed`` optionally restricts the routine to an induced subgraph
    (mask of length n) without building a copy.  Runtime O(n+m).
    """
    n = F.n
    if n == 0:
        return set()
    adj = F.adj

    seen = bytearray(n)
    sweep: List[int] = []
    if order is not None:
        for v in order:
            if not seen[v] and (allowed is None or allowed[v]):
                seen[v] = 1
                sweep.append(v)
    if allowed is None:
        for v in range(n):
            if not seen[v]:
                seen[v] = 1
                sweep.append(v)
    else:
        for v in range(n):
            if allowed[v] and not seen[v]:
                seen[v] = 1
                sweep.append(v)

    in_cand: Optional[bytearray] = None
    if candidate is not None:
        in_cand = bytearray(n)
        for v in candidate:
            if allowed is None or allowed[v]:
                in_cand[v] = 1

    dom = [0] * n
    in_sel = bytearray(n)
    selected: List[int] = []

    if allowed is None:
        if in_cand is not None:
            for v in sweep:
                if in_cand[v] and dom[v] == 0:
                    in_sel[v] = 1
                    selected.append(v)
                    dom[v] += 1
                    for u in adj[v]:
                        dom[u] += 1
        for v in sweep:
            if dom[v] == 0:
                in_sel[v] = 1
                selected.append(v)
                dom[v] += 1
                for u in adj[v]:
                    dom[u] += 1
    else:
        if in_cand is not None:
            for v in sweep:
                if in_cand[v] and dom[v] == 0:
                    in_sel[v] = 1
                    selected.append(v)
                    dom[v] += 1
                    for u in adj[v]:
                        if allowed[u]:
                            dom[u] += 1
        for v in sweep:
            if dom[v] == 0:
                in_sel[v] = 1
                selected.append(v)
                dom[v] += 1
                for u in adj[v]:
                    if allowed[u]:
                        dom[u] += 1

    # Prune, most expensive selected vertices first.
    prune_order = flat_bucket_order(selected, F.w, 256, True)
    if allowed is None:
        for v in prune_order:
            if dom[v] <= 1:
                continue
            removable = True
            for u in adj[v]:
                if dom[u] <= 1:
                    removable = False
                    break
            if removable:
                in_sel[v] = 0
                dom[v] -= 1
                for u in adj[v]:
                    dom[u] -= 1
    else:
        for v in prune_order:
            if dom[v] <= 1:
                continue
            removable = True
            for u in adj[v]:
                if allowed[u] and dom[u] <= 1:
                    removable = False
                    break
            if removable:
                in_sel[v] = 0
                dom[v] -= 1
                for u in adj[v]:
                    if allowed[u]:
                        dom[u] -= 1

    return {v for v in selected if in_sel[v]}


# ---------------------------------------------------------------------------
# Flat Baker-layer candidate
# ---------------------------------------------------------------------------


def _flat_bfs_layers(F: FlatGraph) -> List[int]:
    """BFS depths for every component in O(n+m), roots in index order."""
    n = F.n
    adj = F.adj
    layers = [-1] * n
    dq: deque = deque()
    for root in range(n):
        if layers[root] >= 0:
            continue
        layers[root] = 0
        dq.append(root)
        while dq:
            v = dq.popleft()
            lv = layers[v] + 1
            for u in adj[v]:
                if layers[u] < 0:
                    layers[u] = lv
                    dq.append(u)
    return layers


def _flat_one_layer(
    F: FlatGraph,
    kept: List[int],
    separator: List[int],
    cheap_order: List[int],
    ratio_order: List[int],
) -> Set[int]:
    """Build one weighted Baker-layer candidate and repair on F."""
    n = F.n
    kept_order = flat_order(F, kept, "coverage_per_weight")
    sep_order = flat_order(F, separator, "coverage_per_weight")

    seed: Set[int] = set()
    if kept:
        mask = bytearray(n)
        for v in kept:
            mask[v] = 1
        seed = flat_repair_prune(F, set(kept_order), kept_order, allowed=mask)

    order_a: List[int] = sorted(seed)
    extend_unique(order_a, kept_order, n)
    extend_unique(order_a, sep_order, n)
    extend_unique(order_a, cheap_order, n)
    extend_unique(order_a, ratio_order, n)
    cand_a = flat_repair_prune(F, seed, order_a)

    order_b: List[int] = list(sep_order)
    extend_unique(order_b, kept_order, n)
    extend_unique(order_b, cheap_order, n)
    extend_unique(order_b, ratio_order, n)
    cand_b = flat_repair_prune(F, set(sep_order), order_b)

    wa = (flat_solution_weight(F, cand_a), len(cand_a))
    wb = (flat_solution_weight(F, cand_b), len(cand_b))
    return cand_a if wa <= wb else cand_b


def flat_baker_candidate(F: FlatGraph, eps: float = 1.0) -> Set[int]:
    """
    Return a weighted Baker-style IDS candidate on a flat graph.

    Runtime is O((ceil(1/eps)+1)(n+m)). For eps=1.0 this is linear.
    """
    if eps <= 0:
        raise ValueError("eps must be positive.")
    n = F.n
    if n == 0:
        return set()

    period = max(2, int(1.0 / eps + 0.999999999) + 1)
    layers = _flat_bfs_layers(F)
    all_nodes = list(range(n))
    ratio_order = flat_order(F, all_nodes, "coverage_per_weight")
    cheap_order = flat_order(F, all_nodes, "cheap")

    candidates: List[Set[int]] = []
    for residue in range(period):
        separator = [v for v in all_nodes if layers[v] % period == residue]
        kept = [v for v in all_nodes if layers[v] % period != residue]
        cand = _flat_one_layer(F, kept, separator, cheap_order, ratio_order)
        if flat_verify_ids(F, cand):
            candidates.append(cand)

    # Repairs are feasible by construction; no re-verification needed.
    for order in (ratio_order, cheap_order):
        candidates.append(flat_repair_prune(F, set(order), order))

    if not candidates:
        fallback = flat_repair_prune(F, set(all_nodes), all_nodes)
        if not flat_verify_ids(F, fallback):
            raise RuntimeError("failed to construct a valid weighted IDS")
        return fallback

    return min(candidates, key=lambda S: (flat_solution_weight(F, S), len(S)))


# ---------------------------------------------------------------------------
# Public NetworkX-facing API (backward compatible)
# ---------------------------------------------------------------------------


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

    adjacency = G.adj
    for v in S:
        for u in adjacency[v]:
            if u in S:
                return False
    for v in G.nodes():
        if v in S:
            continue
        for u in adjacency[v]:
            if u in S:
                break
        else:
            return False
    return True


def weighted_bucket_order(
    G: nx.Graph,
    allowed: Iterable[Any],
    weight: str = "weight",
    mode: str = "coverage_per_weight",
    bucket_count: int = 256,
) -> List[Any]:
    """Return a linear-time order for weighted IDS repair (nx wrapper)."""
    F, labels, idx = flat_from_nx(G, weight)
    nodes = [idx[v] for v in allowed]
    return [labels[v] for v in flat_order(F, nodes, mode, bucket_count)]


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
    F, labels, idx = flat_from_nx(G, weight)
    cand = None if candidate is None else {idx[v] for v in candidate if v in idx}
    ordr = None if order is None else [idx[v] for v in order if v in idx]
    return {labels[v] for v in flat_repair_prune(F, cand, ordr)}


def _bfs_layers_all_components(G: nx.Graph) -> Dict[Any, int]:
    """Return BFS depths for every component in O(n+m)."""
    F, labels, _ = flat_from_nx(G)
    layers = _flat_bfs_layers(F)
    return {labels[v]: layers[v] for v in range(F.n)}


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
    F, labels, _ = flat_from_nx(G, weight)
    return {labels[v] for v in flat_baker_candidate(F, eps)}


# Backward-compatible alias.
baker_ptas_ids = baker_layer_weighted_ids_candidate
