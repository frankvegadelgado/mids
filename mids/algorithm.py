# Created on 01/06/2025
# Author: Frank Vega

import itertools

import networkx as nx

from . import approx


def _degree_four_auxiliary_graph(component_graph):
    """Build the sequential degree-four auxiliary graph used by Iris."""
    auxiliary = component_graph.copy()

    for u in component_graph.nodes():
        neighbors = list(auxiliary.neighbors(u))
        auxiliary.remove_node(u)

        first_auxiliary = None
        previous_neighbor = None
        for i, v in enumerate(neighbors):
            aux_vertex = (u, i)
            auxiliary.add_edge(aux_vertex, v)
            if previous_neighbor is None:
                first_auxiliary = aux_vertex
            else:
                auxiliary.add_edge(aux_vertex, previous_neighbor)
            previous_neighbor = v

        if len(neighbors) > 1:
            auxiliary.add_edge(first_auxiliary, previous_neighbor)

    max_degree = max(dict(auxiliary.degree()).values()) if auxiliary.number_of_nodes() else 0
    if max_degree > 4:
        raise RuntimeError(f"Degree-four reduction failed: max degree is {max_degree}, expected <= 4")

    return auxiliary


def _find_component_solution(component_graph):
    """Return the smallest verified Iris candidate for one connected component."""
    auxiliary = _degree_four_auxiliary_graph(component_graph)
    auxiliary_solution = approx.mids_lp(auxiliary).independent_dominating_set

    projected_solution = {
        vertex[0]
        for vertex in auxiliary_solution
        if isinstance(vertex, tuple) and len(vertex) == 2
    }
    complement_solution = set(component_graph) - projected_solution
    direct_solution = approx.mids_lp(component_graph).independent_dominating_set

    candidates = sorted(
        (projected_solution, complement_solution, direct_solution),
        key=len,
    )
    for candidate in candidates:
        if verify_independent_dominating_set(component_graph, candidate):
            return candidate

    raise RuntimeError("Degree-four reduction failed: no verified candidate found")


def find_independent_dominating_set(graph):
    """
    Approximate a minimum independent dominating set for an undirected graph.

    Args:
        graph (nx.Graph): Input graph.

    Returns:
        set: A verified independent dominating set.
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
        solution.update(_find_component_solution(component_graph))

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

    Parameters:
        G: NetworkX graph.

    Returns:
        A set of nodes representing an independent dominating set.
    """
    if len(G) == 0:
        return set()

    return approx.mids_lp(G).independent_dominating_set


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
