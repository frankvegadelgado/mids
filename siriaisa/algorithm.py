# Created on 01/06/2025
# Author: Frank Vega

"""
Linear-time heuristic for the Minimum Independent Dominating Set problem.

The public routine `find_independent_dominating_set` always returns a valid
independent dominating set for a finite undirected NetworkX graph.  It is not
an exact solver for MIDS, which is NP-hard; the guarantee here is feasibility
and O(|V| + |E|) running time.

The implementation follows the Siriaisa/MIDS idea of repairing deterministic
seeds into maximal independent sets, but removes all non-linear pieces: no LP,
no sorting, no quadratic verification, no repeated reverse-delete, and no local
exchange search.  Degree priority is implemented with buckets, so the high- and
low-degree sweeps are linear for simple graphs.
"""

import itertools
from typing import Iterable, List, Sequence, Set, Tuple

import networkx as nx


# ---------------------------------------------------------------------------
# Linear verification and repair
# ---------------------------------------------------------------------------

def _clean_graph(graph: nx.Graph) -> nx.Graph:
    """Return a simple undirected copy with self-loops ignored."""
    if graph.is_directed():
        raise ValueError("Input must be an undirected NetworkX Graph.")

    # nx.Graph(graph) is an O(n + m) copy and collapses possible parallel edges
    # if a graph-like object is passed.
    G = nx.Graph(graph)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def _degree_bucket_orders(G: nx.Graph) -> Tuple[List, List]:
    """
    Return (high_degree_order, low_degree_order) in O(|V| + |E|).

    For a simple graph, Delta <= |V|-1, so bucket allocation is O(|V|).  Ties
    preserve NetworkX node iteration order instead of sorting by labels.
    """
    nodes = list(G.nodes())
    if not nodes:
        return [], []

    degrees = {}
    max_degree = 0
    for v in nodes:
        degree = G.degree(v)
        degrees[v] = degree
        if degree > max_degree:
            max_degree = degree

    buckets = [[] for _ in range(max_degree + 1)]
    for v in nodes:
        buckets[degrees[v]].append(v)

    low_order = []
    for degree in range(max_degree + 1):
        low_order.extend(buckets[degree])

    high_order = []
    for degree in range(max_degree, -1, -1):
        high_order.extend(buckets[degree])

    return high_order, low_order


def repair_prune(G: nx.Graph, candidate: Set = None, order: Sequence = None) -> Set:
    """
    Repair a seed into an independent dominating set, then do one linear prune.

    The repair phase is a greedy maximal-independent-set sweep.  A maximal
    independent set is automatically dominating: if an unselected vertex had no
    selected neighbor, it could be added, contradicting maximality.

    Parameters
    ----------
    G:
        Undirected simple NetworkX graph.  Self-loops are ignored by the public
        entry point before this helper is called.
    candidate:
        Vertices that should be tried first.  Invalid vertices are ignored.
    order:
        Linear order used both to process the candidate seed and to complete the
        maximal independent set.  If omitted, G.nodes() order is used.

    Runtime
    -------
    O(|V| + |E|), expected under Python set/dict operations.
    """
    if not isinstance(G, nx.Graph) or G.is_directed():
        raise ValueError("G must be an undirected NetworkX Graph.")

    if G.number_of_nodes() == 0:
        return set()

    nodes = list(G.nodes()) if order is None else list(order)

    # Ensure all graph nodes appear once in the sweep, even if a custom order is
    # incomplete.  This append is linear because every membership test is O(1).
    seen = set()
    sweep = []
    for v in nodes:
        if v in G and v not in seen:
            sweep.append(v)
            seen.add(v)
    for v in G.nodes():
        if v not in seen:
            sweep.append(v)
            seen.add(v)

    candidate = set() if candidate is None else set(candidate)
    candidate.intersection_update(G.nodes)

    selected = set()
    dominated_count = {v: 0 for v in G.nodes()}

    def add_vertex(v):
        selected.add(v)
        dominated_count[v] += 1
        for u in G.neighbors(v):
            dominated_count[u] += 1

    # First keep as much of the supplied seed as possible, in the requested
    # order, while maintaining independence.
    for v in sweep:
        if v in candidate and dominated_count[v] == 0:
            add_vertex(v)

    # Then complete to a maximal independent set, hence domination is repaired.
    for v in sweep:
        if dominated_count[v] == 0:
            add_vertex(v)

    # One-pass reverse-delete pruning.  Each selected vertex is tested once and,
    # if removed, its closed-neighborhood counters are updated once.
    for v in list(selected):
        if dominated_count[v] <= 1:
            continue

        removable = True
        for u in G.neighbors(v):
            if dominated_count[u] <= 1:
                removable = False
                break

        if removable:
            selected.remove(v)
            dominated_count[v] -= 1
            for u in G.neighbors(v):
                dominated_count[u] -= 1

    return selected


def _linear_candidates(G: nx.Graph) -> Iterable[Set]:
    """Generate a constant number of linear-time candidates."""
    natural = list(G.nodes())
    high_degree, low_degree = _degree_bucket_orders(G)

    # Natural and reverse-natural keep previous deterministic behavior robust to
    # insertion order.  High-degree is the main repair for universal/hub traps.
    orders = (
        high_degree,
        low_degree,
        natural,
        list(reversed(natural)),
    )

    for order in orders:
        yield repair_prune(G, set(order), order)


def _best_linear_component_solution(component_graph: nx.Graph) -> Set:
    """Return the smallest verified candidate for one component in linear time."""
    best = None
    for candidate in _linear_candidates(component_graph):
        if verify_independent_dominating_set(component_graph, candidate):
            if best is None or len(candidate) < len(best):
                best = candidate

    if best is None:
        # This should not happen: every maximal independent set is independent
        # and dominating.  Keep an explicit failure mode for corrupted inputs.
        raise RuntimeError("No verified independent dominating set candidate found")

    return best


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_independent_dominating_set(graph):
    """
    Return a valid independent dominating set in O(|V| + |E|).

    This is a linear-time heuristic, not an exact MIDS solver.  It uses a
    constant number of bucket-ordered maximal-independent-set repairs and keeps
    the smallest verified candidate per connected component.
    """
    if not isinstance(graph, nx.Graph):
        raise ValueError("Input must be an undirected NetworkX Graph.")

    if graph.number_of_nodes() == 0:
        return set()

    working_graph = _clean_graph(graph)
    solution = set()

    # Building each component subgraph copies each edge/vertex exactly once in
    # total over all components, so the total remains O(n + m).
    for component in nx.connected_components(working_graph):
        component_graph = working_graph.subgraph(component).copy()
        solution.update(_best_linear_component_solution(component_graph))

    return solution


def find_independent_dominating_set_brute_force(graph):
    """
    Compute an exact minimum independent dominating set in exponential time.

    This helper is intentionally not linear; it is kept only for testing small
    graphs against the heuristic.
    """
    if graph.number_of_nodes() == 0:
        return set()

    G = _clean_graph(graph)
    nodes = list(G.nodes())
    for size in range(1, len(nodes) + 1):
        for candidate in itertools.combinations(nodes, size):
            candidate_set = set(candidate)
            if verify_independent_dominating_set(G, candidate_set):
                return candidate_set

    return None


def find_independent_dominating_set_approximation(G):
    """
    Backward-compatible alias for the linear public routine.

    The previous repository variant used an LP-guided approximation here; that
    is not linear because LP solving is not O(|V| + |E|).  This alias preserves
    the API while keeping the promised linear-time end-to-end behavior.
    """
    return find_independent_dominating_set(G)


def calculate_solution_weight(G, solution):
    """Calculate the total weight of the nodes in the solution."""
    return sum(G.nodes[v].get("weight", 1.0) for v in solution)


def verify_independent_dominating_set(G, solution):
    """Return True iff solution is independent and dominating, in O(n + m)."""
    if solution is None or not isinstance(G, nx.Graph) or G.is_directed():
        return False

    solution = set(solution)
    if not solution.issubset(G.nodes):
        return False

    selected = {v: False for v in G.nodes()}
    dominated = {v: False for v in G.nodes()}

    for v in solution:
        selected[v] = True
        dominated[v] = True

    for u, v in G.edges():
        if u == v:
            # The algorithm treats self-loops as irrelevant, preserving the
            # previous public behavior where pairwise combinations ignored them.
            continue

        if selected[u] and selected[v]:
            return False

        if selected[u]:
            dominated[v] = True
        if selected[v]:
            dominated[u] = True

    return all(dominated.values())
