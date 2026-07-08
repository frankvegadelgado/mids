# Created on 01/06/2025
# Author: Frank Vega
# Linear-time guarded implementation for independent dominating set.

from collections import deque
import itertools

import networkx as nx

from . import maxcut

# ---------------------------------------------------------------------------
# Linear validation and pruning helpers
# ---------------------------------------------------------------------------

def _verify_independent_dominating_set_linear(G: nx.Graph, solution: set) -> bool:
    """Return True iff solution is independent and dominating in O(|V| + |E|)."""
    if solution is None:
        return False

    solution = set(solution)
    if not solution.issubset(G.nodes):
        return False

    dominated = {v: False for v in G.nodes}

    for v in solution:
        dominated[v] = True

    for u, v in G.edges():
        if u in solution and v in solution:
            return False
        if u in solution:
            dominated[v] = True
        if v in solution:
            dominated[u] = True

    return all(dominated.values())


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


# ---------------------------------------------------------------------------
# Linear candidate generators
# ---------------------------------------------------------------------------


def _exact_small_component_mids(G: nx.Graph, limit: int = 16):
    """
    Exact MIDS for components of size at most `limit`, using bit masks.

    Because `limit` is a fixed constant, this is O(1) per small component in
    asymptotic terms and the whole algorithm remains O(|V| + |E|). It is an
    adversarial guard: every small component is solved exactly instead of being
    left to a heuristic order.
    """
    n = G.number_of_nodes()
    if n > limit:
        return None
    if n == 0:
        return set()

    nodes = list(G.nodes)
    index = {v: i for i, v in enumerate(nodes)}
    closed = [0] * n

    for i, v in enumerate(nodes):
        mask = 1 << i
        for u in G.neighbors(v):
            mask |= 1 << index[u]
        closed[i] = mask

    all_vertices = (1 << n) - 1

    for size in range(1, n + 1):
        for combo in itertools.combinations(range(n), size):
            chosen = 0
            dominated = 0
            feasible = True

            for i in combo:
                # closed[i] without bit i is the open-neighborhood bit mask.
                if chosen & (closed[i] ^ (1 << i)):
                    feasible = False
                    break
                chosen |= 1 << i
                dominated |= closed[i]

            if feasible and dominated == all_vertices:
                return {nodes[i] for i in combo}

    return None


def _dominating_singleton(G: nx.Graph):
    """
    Return {v} if v alone dominates this connected component; otherwise None.

    For a component with no isolates, this is equivalent to deg(v) = n - 1.
    This exactly catches the universal-vertex adversary in O(|V|).
    """
    n = G.number_of_nodes()
    if n == 0:
        return set()

    target_degree = n - 1
    for v in G.nodes:
        if G.degree(v) == target_degree:
            return {v}

    return None


def _degree_bucket_order(G: nx.Graph, descending: bool):
    """
    Yield vertices in nondecreasing or nonincreasing degree order using buckets.

    Runtime: O(|V| + |E|), since degrees are integers in [0, |V|-1].
    This avoids Python comparison sorting.
    """
    n = G.number_of_nodes()
    buckets = [[] for _ in range(n)]

    for v in G.nodes:
        buckets[G.degree(v)].append(v)

    if descending:
        degree_range = range(n - 1, -1, -1)
    else:
        degree_range = range(n)

    for degree in degree_range:
        for v in buckets[degree]:
            yield v


def _maximal_independent_set_from_order(G: nx.Graph, order):
    """
    Build a maximal independent set from the supplied order.

    A maximal independent set is always an independent dominating set.
    Runtime: O(|V| + |E|).
    """
    selected = set()
    dominated = {v: False for v in G.nodes}

    def add_vertex(v):
        selected.add(v)
        dominated[v] = True
        for u in G.neighbors(v):
            dominated[u] = True

    for v in order:
        if not dominated[v]:
            add_vertex(v)

    return selected


def _original_order_candidate(G: nx.Graph):
    return _maximal_independent_set_from_order(G, G.nodes)


def _reverse_order_candidate(G: nx.Graph):
    return _maximal_independent_set_from_order(G, reversed(list(G.nodes)))


def _high_degree_candidate(G: nx.Graph):
    return _maximal_independent_set_from_order(
        G, _degree_bucket_order(G, descending=True)
    )


def _low_degree_candidate(G: nx.Graph):
    return _maximal_independent_set_from_order(
        G, _degree_bucket_order(G, descending=False)
    )



def _seeded_candidate(G: nx.Graph, seed, order):
    """
    Build a maximal independent set after forcing one seed vertex first.

    Runtime: O(|V| + |E|). This is a constant-factor safeguard for cases where
    the right high-coverage vertex is skipped by a pure order scan.
    """
    if seed not in G:
        return set()

    def seeded_order():
        yield seed
        for v in order:
            if v != seed:
                yield v

    return _maximal_independent_set_from_order(G, seeded_order())


def _top_degree_seeds(G: nx.Graph, limit: int = 8):
    """
    Return up to `limit` highest-degree vertices using buckets, no sorting.

    Runtime: O(|V| + |E|) for fixed limit.
    """
    seeds = []
    for v in _degree_bucket_order(G, descending=True):
        seeds.append(v)
        if len(seeds) >= limit:
            break
    return seeds



def _seed_then_best_residual_candidate(G: nx.Graph, seed, completion_order):
    """
    Force `seed`, choose one nonadjacent vertex that covers the largest number
    of still-undominated vertices, then complete to a maximal independent set.

    For a fixed seed this is O(|V| + |E|). Since only constantly many seeds are
    tried, the total routine remains linear.
    """
    if seed not in G:
        return set()

    selected = {seed}
    dominated = {v: False for v in G.nodes}
    dominated[seed] = True
    for u in G.neighbors(seed):
        dominated[u] = True

    forbidden = {seed}
    forbidden.update(G.neighbors(seed))

    score = {v: 0 for v in G.nodes}
    for x in G.nodes:
        if dominated[x]:
            continue

        if x not in forbidden:
            score[x] += 1

        for y in G.neighbors(x):
            if y not in forbidden:
                score[y] += 1

    best = None
    best_score = 0
    for v in G.nodes:
        if score[v] > best_score:
            best = v
            best_score = score[v]

    def order():
        yield seed
        if best is not None and best_score > 0:
            yield best
        for v in completion_order:
            if v != seed and v != best:
                yield v

    return _maximal_independent_set_from_order(G, order())


def _max_cut_bipartite_candidate_pair(G: nx.Graph):
    """
    Return two projected max-cut candidates from the lifted bipartite graph.

    Both projections are repaired/pruned later. Returning both sides is a
    constant-factor linear-time safeguard against an unlucky orientation.
    """
    B = nx.Graph()
    for u, v in G.edges():
        B.add_edges_from([((u, 0), (v, 1)), ((u, 1), (v, 0))])

    result = maxcut.maxcut_bipartite_min_side_linear(B, minimize_side=1)
    if not result.get("feasible", False):
        return []

    return [
        {u for u, _ in result["side_1"]},
        {u for u, _ in result["side_0"]},
    ]


def max_cut_bipartite(G: nx.Graph):
    """
    Backward-compatible public helper: return the side_1 projection.
    """
    pair = _max_cut_bipartite_candidate_pair(G)
    return pair[0] if pair else set()


def _best_linear_component_solution(G: nx.Graph):
    """
    Compute a guarded linear-time independent dominating set for one component.

    Strategy
    --------
    1. Exact constant-size guard: components with at most 16 vertices are solved
       exactly. The cutoff is fixed, so the global asymptotic time is linear.
    2. Exact singleton guard: if a universal vertex exists, return it.
       This fixes the universal-vertex/triangle adversary exactly.
    3. Build several constant-many linear candidates:
       - max-cut side_1 projection;
       - max-cut side_0 projection;
       - original-order maximal independent set;
       - reverse-order maximal independent set;
       - high-degree-first maximal independent set;
       - low-degree-first maximal independent set.
    3. Repair/prune every candidate and return the smallest valid result.

    Runtime: O(|V| + |E|), with a constant number of linear passes.
    """
    exact_small = _exact_small_component_mids(G, limit=16)
    if exact_small is not None:
        return exact_small

    singleton = _dominating_singleton(G)
    if singleton is not None:
        return singleton

    raw_candidates = []
    raw_candidates.extend(_max_cut_bipartite_candidate_pair(G))
    raw_candidates.append(_original_order_candidate(G))
    raw_candidates.append(_reverse_order_candidate(G))
    raw_candidates.append(_high_degree_candidate(G))
    raw_candidates.append(_low_degree_candidate(G))

    # Seed the highest-degree vertices, then complete by different linear orders.
    # The seed limit is a fixed constant, so this preserves O(|V| + |E|).
    original_nodes = list(G.nodes)
    reverse_nodes = list(reversed(original_nodes))
    for seed in _top_degree_seeds(G, limit=8):
        raw_candidates.append(_seeded_candidate(G, seed, original_nodes))
        raw_candidates.append(_seeded_candidate(G, seed, reverse_nodes))
        raw_candidates.append(_seeded_candidate(G, seed, _degree_bucket_order(G, descending=True)))
        raw_candidates.append(_seeded_candidate(G, seed, _degree_bucket_order(G, descending=False)))

        # After forcing a promising seed, also try the best residual-covering
        # second vertex. This detects many two-center adversaries in linear time.
        raw_candidates.append(_seed_then_best_residual_candidate(G, seed, original_nodes))
        raw_candidates.append(_seed_then_best_residual_candidate(G, seed, reverse_nodes))
        raw_candidates.append(_seed_then_best_residual_candidate(G, seed, _degree_bucket_order(G, descending=True)))
        raw_candidates.append(_seed_then_best_residual_candidate(G, seed, _degree_bucket_order(G, descending=False)))

    best = None

    for candidate in raw_candidates:
        repaired = repair_prune(G, candidate)
        if not _verify_independent_dominating_set_linear(G, repaired):
            continue
        if best is None or len(repaired) < len(best):
            best = repaired

    if best is None:
        # This should not happen because every maximal independent set is an IDS,
        # but keep a safe linear fallback.
        best = repair_prune(G, set())

    return best


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_independent_dominating_set(graph):
    """
    Approximate a minimum independent dominating set for an undirected graph.

    The implementation is linear-time up to a constant number of passes over
    each connected component. It contains an exact singleton-dominator guard
    and chooses the smallest valid result among several linear candidates.

    This overcomes the previous adversarial family where a universal vertex is
    hidden after many leaves and a small odd cycle makes the old order-driven
    repair return almost all leaves.
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
        component_graph = working_graph.subgraph(component)
        solution.update(_best_linear_component_solution(component_graph))

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

    This function is preserved for API compatibility with the previous file.
    It requires an external `approx` module in the package namespace.
    """
    if len(G) == 0:
        return set()

    if approx is None:
        raise RuntimeError("LP-guided approximation requires the package's approx module")

    solution = approx.mids_lp(G).independent_dominating_set
    if not verify_independent_dominating_set(G, solution):
        raise RuntimeError("LP-guided MIDS routine returned an invalid independent dominating set")

    return solution


def calculate_solution_weight(G, solution):
    """Calculate the total weight of the nodes in the solution."""
    return sum(G.nodes[v].get("weight", 1.0) for v in solution)


def verify_independent_dominating_set(G, solution):
    """Return True when solution is both independent and dominating."""
    return _verify_independent_dominating_set_linear(G, solution)
