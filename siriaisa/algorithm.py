# Created on 01/06/2025
# Author: Frank Vega

import itertools

import networkx as nx

from . import approx


# ---------------------------------------------------------------------------
# Basic verification / repair
# ---------------------------------------------------------------------------

def _node_key(vertex):
    """Stable tie-breaker that also works when vertex types differ."""
    return (type(vertex).__name__, repr(vertex))


def _maximal_independent_set_from_seed(graph, seed_vertices):
    """Repair a seed order into a maximal independent set of graph."""
    ordered_vertices = []
    seen = set()

    for vertex in seed_vertices:
        if vertex in graph and vertex not in seen:
            ordered_vertices.append(vertex)
            seen.add(vertex)

    # Deterministic tail: avoid depending on NetworkX insertion order.
    for vertex in sorted(graph.nodes(), key=_node_key):
        if vertex not in seen:
            ordered_vertices.append(vertex)
            seen.add(vertex)

    independent_set = set()
    excluded = set()
    for vertex in ordered_vertices:
        if vertex not in excluded:
            independent_set.add(vertex)
            excluded.add(vertex)
            excluded.update(graph.neighbors(vertex))

    return independent_set


def _reverse_delete_redundant(graph, solution):
    """Remove selected vertices that are not needed for domination."""
    solution = set(solution)
    changed = True
    while changed:
        changed = False
        for vertex in sorted(solution, key=lambda v: (graph.degree(v), _node_key(v))):
            trial = solution - {vertex}
            if verify_independent_dominating_set(graph, trial):
                solution = trial
                changed = True
                break
    return solution


def _local_exchange_compress(graph, solution):
    """
    Improve a verified independent dominating set by bounded local exchanges.

    The key repair is a 2-add exchange: if two independent outside vertices can
    replace three or more selected vertices while preserving domination, accept
    the compression. This defeats node-order failures where greedy MIS chooses
    many low-degree witnesses before two high-coverage dominators.
    """
    solution = set(solution)
    if not verify_independent_dominating_set(graph, solution):
        return solution

    solution = _reverse_delete_redundant(graph, solution)

    changed = True
    while changed:
        changed = False
        outside = [v for v in graph.nodes() if v not in solution]
        outside.sort(key=lambda v: (-graph.degree(v), _node_key(v)))

        # 1-add compression: one outside vertex replaces at least two selected
        # conflicting vertices.
        for x in outside:
            conflicts = {s for s in solution if graph.has_edge(x, s)}
            if len(conflicts) < 2:
                continue
            trial = (solution - conflicts) | {x}
            if len(trial) < len(solution) and verify_independent_dominating_set(graph, trial):
                solution = _reverse_delete_redundant(graph, trial)
                changed = True
                break
        if changed:
            continue

        # 2-add compression: two independent outside vertices replace at least
        # three selected conflicting vertices.
        outside = [v for v in graph.nodes() if v not in solution]
        outside.sort(key=lambda v: (-graph.degree(v), _node_key(v)))
        outside_set = set(outside)
        for i, x in enumerate(outside):
            forbidden = set(graph.neighbors(x)) | {x}
            for y in outside[i + 1:]:
                if y in forbidden:
                    continue
                conflicts = {
                    s for s in solution
                    if graph.has_edge(x, s) or graph.has_edge(y, s)
                }
                if len(conflicts) < 3:
                    continue
                trial = (solution - conflicts) | {x, y}
                if len(trial) < len(solution) and verify_independent_dominating_set(graph, trial):
                    solution = _reverse_delete_redundant(graph, trial)
                    changed = True
                    break
            if changed:
                break

    return solution


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def _degree_four_auxiliary_graph(component_graph):
    """Build the sequential degree-four auxiliary graph used by Siriaisa."""
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


def _lp_values(graph):
    """Return an LP-value dictionary when approx.solve_mids_lp is available."""
    nodes = list(graph.nodes())
    if not hasattr(approx, "solve_mids_lp"):
        return {v: 0.0 for v in nodes}
    try:
        x, _, _ = approx.solve_mids_lp(graph)
        return {v: float(x[i]) for i, v in enumerate(nodes)}
    except Exception:
        return {v: 0.0 for v in nodes}


def _ordered_candidates_from_scores(component_graph, lp):
    """Generate deterministic MIS candidates from several useful orderings."""
    nodes = list(component_graph.nodes())

    orders = []
    orders.append(("HIGH_DEGREE", sorted(nodes, key=lambda v: (-component_graph.degree(v), _node_key(v)))))
    orders.append(("LOW_DEGREE", sorted(nodes, key=lambda v: (component_graph.degree(v), _node_key(v)))))
    orders.append(("LP_HIGH_DEGREE", sorted(nodes, key=lambda v: (-lp.get(v, 0.0), -component_graph.degree(v), _node_key(v)))))
    orders.append(("LP_LOW_DEGREE", sorted(nodes, key=lambda v: (-lp.get(v, 0.0), component_graph.degree(v), _node_key(v)))))
    orders.append(("LP_REVERSE_HIGH_DEGREE", sorted(nodes, key=lambda v: (lp.get(v, 0.0), -component_graph.degree(v), _node_key(v)))))

    # A domination-power order. Since closed-neighborhood size is degree+1 in a
    # simple graph, this is effectively a deterministic high-coverage order, but
    # kept separate to make the intent explicit and easy to extend.
    orders.append(("DOMINATION_POWER", sorted(nodes, key=lambda v: (-(component_graph.degree(v) + 1), _node_key(v)))))

    for label, order in orders:
        yield label, set(order), _maximal_independent_set_from_seed(component_graph, order)


def _component_candidate_details(component_graph):
    """Return raw seeds and repaired candidates for one connected component."""
    candidates = []
    lp = _lp_values(component_graph)

    # Original degree-four auxiliary route.
    auxiliary = _degree_four_auxiliary_graph(component_graph)
    auxiliary_solution = approx.mids_lp(auxiliary).independent_dominating_set

    projected_seed = [
        vertex[0]
        for vertex in auxiliary_solution
        if isinstance(vertex, tuple) and len(vertex) == 2
    ]
    projected_seed_set = set(projected_seed)
    complement_seed = [
        vertex
        for vertex in component_graph.nodes()
        if vertex not in projected_seed_set
    ]

    candidates.append(("S1_AUX_PROJECTED", projected_seed_set,
                       _maximal_independent_set_from_seed(component_graph, projected_seed)))
    candidates.append(("S2_AUX_COMPLEMENT", set(complement_seed),
                       _maximal_independent_set_from_seed(component_graph, complement_seed)))

    # Direct LP-guided candidate from approx.mids_lp.
    direct_solution = approx.mids_lp(component_graph).independent_dominating_set
    candidates.append(("S3_DIRECT_LP", set(direct_solution), set(direct_solution)))

    # New deterministic order candidates. The important one for the found
    # counterexample is LP_HIGH_DEGREE: it breaks fractional LP ties toward the
    # vertex with larger domination coverage instead of preserving insertion order.
    candidates.extend(_ordered_candidates_from_scores(component_graph, lp))

    # Seed candidates around the top LP/degree vertices. This helps when a good
    # solution requires committing to one or two high-coverage vertices before the
    # greedy sweep starts.
    top = sorted(
        component_graph.nodes(),
        key=lambda v: (-lp.get(v, 0.0), -component_graph.degree(v), _node_key(v)),
    )[: min(12, component_graph.number_of_nodes())]
    for v in top:
        candidates.append((f"SEED_ONE_{repr(v)}", {v},
                           _maximal_independent_set_from_seed(component_graph, [v])))
    for i, u in enumerate(top):
        for v in top[i + 1:]:
            if component_graph.has_edge(u, v):
                continue
            candidates.append((f"SEED_PAIR_{repr(u)}_{repr(v)}", {u, v},
                               _maximal_independent_set_from_seed(component_graph, [u, v])))

    return candidates


def _find_component_solution(component_graph):
    """Return the smallest repaired, locally compressed, verified candidate."""
    best = None
    best_label = None

    for label, _, candidate in _component_candidate_details(component_graph):
        if not verify_independent_dominating_set(component_graph, candidate):
            continue
        candidate = _local_exchange_compress(component_graph, candidate)
        if verify_independent_dominating_set(component_graph, candidate):
            if best is None or len(candidate) < len(best):
                best = candidate
                best_label = label

    if best is not None:
        return best

    raise RuntimeError("No verified independent dominating set candidate found")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_independent_dominating_set(graph):
    """
    Approximate a minimum independent dominating set for an undirected graph.

    This strengthened version keeps the previous auxiliary and direct LP
    candidates, but adds deterministic tie-breaking, high-coverage candidates,
    seed-one/seed-pair candidates, and bounded local exchange compression.
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
