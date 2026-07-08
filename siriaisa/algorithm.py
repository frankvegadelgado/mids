# Created on 08/07/2026
# Author: Frank Vega

"""
Linear-time heuristic pool for Minimum Independent Dominating Set (MIDS).

The public function ``find_independent_dominating_set`` always returns a valid
independent dominating set.  It uses a fixed pool of linear Furones-style
signals, plus a Salvador oriented-incidence auxiliary candidate solved with
``baker_ids.baker_ptas_ids(..., eps=1.0)``.

Guarantee: feasibility and O(|V| + |E|) time for a fixed candidate pool,
assuming expected O(1) Python dict/set operations.  This is not an exact solver
and not a proved universal constant-approximation algorithm.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import networkx as nx

try:
    from . import baker_ids
except Exception:  # allows standalone copy-paste next to baker_ids.py
    import baker_ids


# ---------------------------------------------------------------------------
# Basic helpers
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


def _append_unique(out: List[Any], values: Iterable[Any], allowed: Set[Any]) -> None:
    seen = set(out)
    for v in values:
        if v in allowed and v not in seen:
            out.append(v)
            seen.add(v)


def _degree_orders(G: nx.Graph) -> Tuple[List[Any], List[Any]]:
    nodes = list(G.nodes())
    if not nodes:
        return [], []
    max_d = max((G.degree(v) for v in nodes), default=0)
    buckets: List[List[Any]] = [[] for _ in range(max_d + 1)]
    for v in nodes:
        buckets[G.degree(v)].append(v)
    low = [v for d in range(max_d + 1) for v in buckets[d]]
    high = [v for d in range(max_d, -1, -1) for v in buckets[d]]
    return high, low


def _weighted_bucket_order(
    G: nx.Graph,
    allowed: Iterable[Any],
    weights: Optional[Dict[Any, float]] = None,
    bucket_count: int = 256,
) -> List[Any]:
    nodes = list(allowed)
    if not nodes:
        return []
    weights = {} if weights is None else weights
    scored: List[Tuple[Any, float]] = []
    lo, hi = float("inf"), float("-inf")
    for v in nodes:
        w = max(float(weights.get(v, 1.0)), 1e-12)
        s = (G.degree(v) + 1) / w
        scored.append((v, s))
        lo = min(lo, s)
        hi = max(hi, s)
    if hi <= lo:
        return nodes
    buckets: List[List[Any]] = [[] for _ in range(max(2, bucket_count))]
    scale = (len(buckets) - 1) / (hi - lo)
    for v, s in scored:
        idx = int((s - lo) * scale)
        idx = max(0, min(idx, len(buckets) - 1))
        buckets[idx].append(v)
    return [v for i in range(len(buckets) - 1, -1, -1) for v in buckets[i]]


def _is_dom(G: nx.Graph, D: Set[Any]) -> bool:
    if not D.issubset(G.nodes):
        return False
    dominated = {v: False for v in G.nodes()}
    for v in D:
        dominated[v] = True
        for u in G.neighbors(v):
            dominated[u] = True
    return all(dominated.values())


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


# ---------------------------------------------------------------------------
# Safe IDS repair
# ---------------------------------------------------------------------------

def repair_prune(
    G: nx.Graph,
    candidate: Optional[Set[Any]] = None,
    order: Optional[Sequence[Any]] = None,
) -> Set[Any]:
    """Repair a seed into a maximal independent set, hence an IDS, in O(n+m)."""
    if not isinstance(G, nx.Graph) or G.is_directed():
        raise ValueError("G must be an undirected NetworkX Graph.")
    if G.number_of_nodes() == 0:
        return set()

    nodes = set(G.nodes())
    sweep: List[Any] = []
    _append_unique(sweep, list(G.nodes()) if order is None else order, nodes)
    _append_unique(sweep, G.nodes(), nodes)
    candidate = set() if candidate is None else set(candidate) & nodes

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
    return selected


# ---------------------------------------------------------------------------
# Furones-style dominating-set signals
# ---------------------------------------------------------------------------

class _Cand:
    __slots__ = ("vertices", "order", "name")

    def __init__(self, vertices: Iterable[Any], order: Iterable[Any], name: str):
        self.vertices = set(vertices)
        self.order = list(order)
        self.name = name


def _prune_ds(G: nx.Graph, D: Set[Any]) -> Set[Any]:
    D = set(D) & set(G.nodes())
    count: Dict[Any, int] = {v: 0 for v in G.nodes()}
    for v in D:
        for u in _closed(G, v):
            count[u] += 1
    for v in list(D):
        if all(count[u] >= 2 for u in _closed(G, v)):
            D.remove(v)
            for u in _closed(G, v):
                count[u] -= 1
    return D


def _coverage_sweep(G: nx.Graph, order: Sequence[Any], name: str) -> _Cand:
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
    D = _prune_ds(G, set(chosen))
    return _Cand(D, [v for v in chosen if v in D], name)


def _dynamic_coverage(G: nx.Graph) -> _Cand:
    if G.number_of_nodes() == 0:
        return _Cand(set(), [], "dynamic_coverage")
    gain = {v: G.degree(v) + 1 for v in G.nodes()}
    max_gain = max(gain.values(), default=0)
    buckets: List[List[Any]] = [[] for _ in range(max_gain + 1)]
    for v in G.nodes():
        buckets[gain[v]].append(v)

    undom = set(G.nodes())
    selected: Set[Any] = set()
    order: List[Any] = []
    top = max_gain
    while undom:
        s = None
        while top >= 1:
            while buckets[top]:
                v = buckets[top].pop()
                if v not in selected and gain[v] == top:
                    s = v
                    break
            if s is not None:
                break
            top -= 1
        if s is None:
            break
        selected.add(s)
        order.append(s)
        newly = [w for w in _closed(G, s) if w in undom]
        for w in newly:
            undom.remove(w)
            for z in _closed(G, w):
                if z in selected:
                    continue
                gain[z] -= 1
                if gain[z] >= 1:
                    buckets[gain[z]].append(z)
    D = _prune_ds(G, set(order))
    return _Cand(D, [v for v in order if v in D], "dynamic_coverage")


def _witness_score(G: nx.Graph, threshold: int, name: str) -> _Cand:
    score = {v: 0 for v in G.nodes()}
    deg = dict(G.degree())
    for w, d in deg.items():
        if d <= threshold:
            score[w] += 1
            for v in G.neighbors(w):
                score[v] += 1
    order = _weighted_bucket_order(G, G.nodes(), {v: 1.0 / max(1, score[v]) for v in G.nodes()})
    high, _ = _degree_orders(G)
    _append_unique(order, high, set(G.nodes()))
    return _coverage_sweep(G, order, name)


def _ownership(G: nx.Graph, mode: str) -> _Cand:
    nodes = list(G.nodes())
    if not nodes:
        return _Cand(set(), [], f"ownership_{mode}")
    pos = {v: i for i, v in enumerate(nodes)}
    deg = dict(G.degree())
    avg = (2 * G.number_of_edges()) // max(1, len(nodes))
    threshold = max(2, avg)
    score = {v: 0 for v in nodes}
    for w in nodes:
        if deg[w] > threshold:
            continue
        owner, owner_key = None, None
        for v in _closed(G, w):
            if v == w and G.degree(w) > 0:
                continue
            if deg.get(v, 0) < deg[w]:
                continue
            key = pos[v] if mode == "late" else -pos[v]
            if owner is None or key > owner_key:
                owner, owner_key = v, key
        if owner is not None:
            score[owner] += 1
    order = _weighted_bucket_order(G, nodes, {v: 1.0 / max(1, score[v]) for v in nodes})
    high, _ = _degree_orders(G)
    _append_unique(order, high, set(G.nodes()))
    return _coverage_sweep(G, order, f"ownership_{mode}")


def _reverse_delete(G: nx.Graph, mode: str) -> _Cand:
    nodes = list(G.nodes())
    if not nodes:
        return _Cand(set(), [], f"reverse_delete_{mode}")
    high, low = _degree_orders(G)
    if mode == "input":
        order = nodes
    elif mode == "reverse_input":
        order = list(reversed(nodes))
    elif mode == "high_degree":
        order = high
    elif mode == "low_degree":
        order = low
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


def _seed_complete(G: nx.Graph, seed_limit: int = 16, residual_passes: int = 3) -> _Cand:
    if G.number_of_nodes() == 0:
        return _Cand(set(), [], "seed_complete")
    high, _ = _degree_orders(G)
    best: Optional[Set[Any]] = None
    best_order: List[Any] = []
    n = G.number_of_nodes()

    for seed in high[:seed_limit]:
        D, order = {seed}, [seed]
        dominated = set(_closed(G, seed))
        passes = residual_passes
        while len(dominated) < n and passes > 0:
            passes -= 1
            best_v, best_gain = None, 0
            for v in G.nodes():
                if v in D:
                    continue
                gain = sum(1 for u in _closed(G, v) if u not in dominated)
                if gain > best_gain:
                    best_v, best_gain = v, gain
            if best_v is None or best_gain <= 0:
                break
            D.add(best_v)
            order.append(best_v)
            for u in _closed(G, best_v):
                dominated.add(u)
        if len(dominated) < n:
            for v in high:
                if v not in D and any(u not in dominated for u in _closed(G, v)):
                    D.add(v)
                    order.append(v)
                    for u in _closed(G, v):
                        dominated.add(u)
                    if len(dominated) == n:
                        break
        D = _prune_ds(G, D)
        if _is_dom(G, D) and (best is None or len(D) < len(best)):
            best, best_order = D, [v for v in order if v in D]
            if len(best) <= 2:
                break
    return _Cand(best or set(), best_order, "seed_complete")


# ---------------------------------------------------------------------------
# Salvador reduction + Baker IDS candidate with eps=1
# ---------------------------------------------------------------------------

def _build_salvador_aux(G: nx.Graph) -> Tuple[nx.Graph, Dict[Any, float]]:
    B = nx.Graph()
    weights: Dict[Any, float] = {}
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
            B.add_node(x_uv)
            B.add_node(x_vu)
            weights[x_uv] = 1.0 / max(1, deg.get(u, 1))
            weights[x_vu] = 1.0 / max(1, deg.get(v, 1))
            B.add_edge(x_uv, x_vu)
            if prev is None:
                first = x_uv
            else:
                B.add_edge(x_uv, prev)
            prev = x_vu
        if len(nbrs) > 1 and first is not None and prev is not None:
            B.add_edge(first, prev)
    B.graph["salvador_planar_bipartite_by_construction"] = True
    B.graph["baker_ids_eps"] = 1.0
    return B, weights


def _salvador_baker_ids(G: nx.Graph) -> _Cand:
    if G.number_of_nodes() == 0:
        return _Cand(set(), [], "salvador_baker_ids")
    if G.number_of_edges() == 0:
        return _Cand(set(G.nodes()), list(G.nodes()), "salvador_baker_ids")

    B, weights = _build_salvador_aux(G)
    if B.number_of_nodes() == 0:
        return _Cand(set(), [], "salvador_baker_ids")

    aux_ids = baker_ids.baker_ptas_ids(B, eps=1.0, weights=weights)
    order: List[Any] = []
    seen: Set[Any] = set()

    def decode(node: Any) -> None:
        if isinstance(node, tuple) and len(node) == 3 and node[0] == "inc":
            v = node[1]
            if v in G and v not in seen:
                order.append(v)
                seen.add(v)

    for node in aux_ids:
        decode(node)
    for node in _weighted_bucket_order(B, B.nodes(), weights):
        decode(node)

    D = set(order)
    if _is_dom(G, D):
        D = _prune_ds(G, D)
        order = [v for v in order if v in D]
    return _Cand(D, order, "salvador_baker_ids")


def _ds_candidates(G: nx.Graph) -> Iterator[_Cand]:
    high, low = _degree_orders(G)
    avg = (2 * G.number_of_edges()) // max(1, G.number_of_nodes())
    yield _coverage_sweep(G, high, "high_degree_sweep")
    yield _coverage_sweep(G, low, "low_degree_sweep")
    yield _dynamic_coverage(G)
    yield _witness_score(G, 2, "low_witness")
    yield _witness_score(G, max(2, avg), "medium_witness")
    yield _ownership(G, "late")
    yield _ownership(G, "early")
    yield _seed_complete(G)
    yield _salvador_baker_ids(G)
    yield _reverse_delete(G, "input")
    yield _reverse_delete(G, "reverse_input")
    yield _reverse_delete(G, "high_degree")
    yield _reverse_delete(G, "low_degree")


# ---------------------------------------------------------------------------
# Candidate conversion and selection
# ---------------------------------------------------------------------------

def _priority_order(primary: Iterable[Any], secondary: Iterable[Any], G: nx.Graph) -> List[Any]:
    out: List[Any] = []
    allowed = set(G.nodes())
    _append_unique(out, primary, allowed)
    _append_unique(out, secondary, allowed)
    _append_unique(out, G.nodes(), allowed)
    return out


def _ids_from_ds(G: nx.Graph, c: _Cand, high: List[Any], low: List[Any]) -> Iterator[Set[Any]]:
    D = set(c.vertices) & set(G.nodes())
    if not D:
        return
    if verify_independent_dominating_set(G, D):
        yield D
    for order in (
        _priority_order(c.order, high, G),
        _priority_order(reversed(c.order), high, G),
        _priority_order((v for v in high if v in D), high, G),
        _priority_order((v for v in low if v in D), low, G),
    ):
        yield repair_prune(G, D, order)


def _absorb_once(G: nx.Graph, S: Set[Any], probe_order: Sequence[Any]) -> Set[Any]:
    """One safe IDS swap pass: add u, remove its selected neighbours if valid."""
    S = set(S)
    if not S:
        return S
    count = {v: 0 for v in G.nodes()}
    unique: Dict[Any, Optional[Any]] = {v: None for v in G.nodes()}
    for s in S:
        for x in _closed(G, s):
            count[x] += 1
            unique[x] = s if count[x] == 1 else None

    private_count = {s: 0 for s in S}
    cover: Dict[Tuple[Any, Any], int] = defaultdict(int)
    for x, cnt in count.items():
        if cnt != 1:
            continue
        r = unique[x]
        if r not in S:
            continue
        private_count[r] += 1
        cover[(x, r)] += 1
        for u in G.neighbors(x):
            cover[(u, r)] += 1

    newS = set(S)
    used_removed: Set[Any] = set()
    added: Set[Any] = set()
    for u in probe_order:
        if u in newS or any(a in G[u] for a in added):
            continue
        R = [r for r in G.neighbors(u) if r in newS and r not in used_removed]
        if len(R) <= 1:
            continue
        if all(cover.get((u, r), 0) >= private_count.get(r, 0) for r in R):
            for r in R:
                newS.remove(r)
                used_removed.add(r)
            newS.add(u)
            added.add(u)
    return newS if verify_independent_dominating_set(G, newS) else S


def _pool(G: nx.Graph) -> Iterator[Set[Any]]:
    high, low = _degree_orders(G)
    natural = list(G.nodes())
    for order in (high, low, natural, list(reversed(natural))):
        yield repair_prune(G, set(order), order)

    ds_list = list(_ds_candidates(G))
    for c in ds_list:
        yield from _ids_from_ds(G, c, high, low)

    probes: List[Any] = []
    _append_unique(probes, high[:16], set(G.nodes()))
    for c in ds_list:
        _append_unique(probes, c.order[:16], set(G.nodes()))
        if len(probes) >= 64:
            break
    for seed in probes[:64]:
        yield repair_prune(G, {seed}, _priority_order([seed], high, G))
        yield repair_prune(G, {seed}, _priority_order([seed], low, G))


def _best_component(G: nx.Graph) -> Set[Any]:
    high, _ = _degree_orders(G)
    best: Optional[Set[Any]] = None
    for cand in _pool(G):
        if not verify_independent_dominating_set(G, cand):
            continue
        improved = _absorb_once(G, cand, high[:32])
        for opt in (cand, improved):
            if verify_independent_dominating_set(G, opt):
                if best is None or len(opt) < len(best):
                    best = set(opt)
                    if len(best) == 1:
                        return best
    if best is None:
        best = repair_prune(G, set(G.nodes()), list(G.nodes()))
        if not verify_independent_dominating_set(G, best):
            raise RuntimeError("failed to construct a valid IDS")
    return best


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_independent_dominating_set(graph: nx.Graph) -> Set[Any]:
    """Return a valid independent dominating set in O(|V|+|E|)."""
    G = _clean_graph(graph)
    if G.number_of_nodes() == 0:
        return set()
    solution: Set[Any] = set()
    for comp in nx.connected_components(G):
        H = G.subgraph(comp).copy()
        solution.update(_best_component(H))
    if not verify_independent_dominating_set(G, solution):
        raise RuntimeError("internal error: invalid independent dominating set")
    return solution


def find_independent_dominating_set_brute_force(graph: nx.Graph) -> Optional[Set[Any]]:
    """Exact exponential solver for small tests."""
    G = _clean_graph(graph)
    if G.number_of_nodes() == 0:
        return set()
    nodes = list(G.nodes())
    for size in range(1, len(nodes) + 1):
        for cand in itertools.combinations(nodes, size):
            S = set(cand)
            if verify_independent_dominating_set(G, S):
                return S
    return None


def find_independent_dominating_set_approximation(G: nx.Graph) -> Set[Any]:
    return find_independent_dominating_set(G)


def calculate_solution_weight(G: nx.Graph, solution: Set[Any]) -> float:
    return sum(G.nodes[v].get("weight", 1.0) for v in solution)