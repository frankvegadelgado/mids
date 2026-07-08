# Created on 01/06/2025
# Author: Frank Vega

import itertools

import networkx as nx

from . import maxcut

def repair_prune(G: nx.Graph, candidate: set):
    """
    Repair a proposed candidate into an independent dominating set, then prune
    redundant vertices.

    Runtime: O(|V| + |E|).
    """
    if not isinstance(G, nx.Graph):
        raise ValueError("G must be an undirected NetworkX Graph.")

    if G.number_of_nodes() == 0:
        return set()

    candidate = set() if candidate is None else set(candidate)
    candidate.intersection_update(G.nodes)

    selected = set()
    dominated_count = {v: 0 for v in G.nodes}

    def add_vertex(v):
        selected.add(v)
        dominated_count[v] += 1
        for u in G.neighbors(v):
            dominated_count[u] += 1

    # Keep as much of candidate as possible while enforcing independence.
    for v in G.nodes:
        if v in candidate and dominated_count[v] == 0:
            add_vertex(v)

    # Repair domination by extending to a maximal independent set.
    for v in G.nodes:
        if dominated_count[v] == 0:
            add_vertex(v)

    # Prune redundant selected vertices.
    # Removing vertices cannot break independence, only domination.
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


def max_cut_bipartite(G: nx.Graph):
    B = nx.Graph()
    for u, v in G.edges():
        B.add_edges_from([((u, 0), (v, 1)), ((u, 1), (v, 0))])
    result = maxcut.maxcut_bipartite_min_side_linear(B, minimize_side=1)
    D = {u for u, _ in result["side_1"]}
    return D


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_independent_dominating_set(graph):
    """
    Approximate a minimum independent dominating set for an undirected graph.
    """
    if not isinstance(graph, nx.Graph):
        raise ValueError("Input must be an undirected NetworkX Graph.")

    if graph.number_of_nodes() == 0:
        return set()

    working_graph = graph.copy()
    working_graph.remove_edges_from(list(nx.selfloop_edges(working_graph)))

    # Isolated vertices must be selected to dominate themselves.
    solution = set(nx.isolates(working_graph))
    working_graph.remove_nodes_from(solution)

    if working_graph.number_of_nodes() == 0:
        return solution

    for component in nx.connected_components(working_graph):
        component_graph = working_graph.subgraph(component).copy()
        solution.update(repair_prune(component_graph, max_cut_bipartite(component_graph)))

    return solution


def find_independent_dominating_set_brute_force(graph):
    """
    Compute an exact minimum independent dominating set in exponential time.

    Args:
        graph: A NetworkX graph.

    Returns:
        A minimum independent dominating set.
    """
    if graph.number_of_nodes() == 0:
        return set()

    nodes = list(graph.nodes())
    for size in range(1, len(nodes) + 1):
        for candidate in itertools.combinations(nodes, size):
            candidate_set = set(candidate)
            if verify_independent_dominating_set(graph, candidate_set):
                return candidate_set

    return None


def find_independent_dominating_set_approximation(G):
    """
    Run the LP-guided MIDS approximation directly on the input graph.
    """
    if len(G) == 0:
        return set()

    solution = approx.mids_lp(G).independent_dominating_set
    if not verify_independent_dominating_set(G, solution):
        raise RuntimeError("LP-guided MIDS routine returned an invalid independent dominating set")

    return solution


def calculate_solution_weight(G, solution):
    """Calculate the total weight of the nodes in the solution."""
    return sum(G.nodes[v].get("weight", 1.0) for v in solution)


def verify_independent_dominating_set(G, solution):
    """Return True when solution is both independent and dominating."""
    if solution is None:
        return False

    solution = set(solution)
    if not solution.issubset(G.nodes):
        return False

    is_independent = all(not G.has_edge(u, v) for u, v in itertools.combinations(solution, 2))
    return is_independent and nx.dominating.is_dominating_set(G, solution)
