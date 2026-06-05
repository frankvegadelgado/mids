# Created on 01/06/2025
# Author: Frank Vega

import itertools

import networkx as nx
from . import approx
def find_independent_dominating_set(graph):
    """
    Approximate minimum independent dominating set for an undirected graph.

    Args:
        graph (nx.Graph): A NetworkX Graph object representing the input graph.

    Returns:
        set: A set of vertex indices representing the approximate minimum independent dominating set.
             Returns an empty set if the graph is empty or has no edges.
    """
    
    def dominating_via_reduction_max_degree_4(graph):
        """
        Args:
            graph (nx.Graph): Connected component subgraph to process
            
        Returns:
            set: Vertices in the approximate independent dominating set for this component
            
        Raises:
            RuntimeError: If reduction fails (resulting graph has max degree > 4)
        """
        # Create a working copy to avoid modifying the original graph
        G = graph.copy()
        
        # Reduction step: Replace each vertex with auxiliary vertices
        # This transforms the problem into a maximum degree 4 case
        for u in graph.nodes():
            neighbors = list(G.neighbors(u))  # Get neighbors before removing node
            G.remove_node(u)  # Remove original vertex
            k = len(neighbors)  # Degree of original vertex
            
            # Create auxiliary vertices and connect each to one neighbor
            initial, previous = None, None
            for i, v in enumerate(neighbors):
                aux_vertex = (u, i)  # Auxiliary vertex naming: (original_vertex, index)
                G.add_edge(aux_vertex, v)
                if previous is not None:
                    G.add_edge(aux_vertex, previous)
                else:
                    initial = aux_vertex
                previous = v
            if len(neighbors) > 1:
                G.add_edge(initial, previous)              
        # Verify the reduction was successful (max degree should be 4)
        max_degree = max(dict(G.degree()).values()) if G.number_of_nodes() > 0 else 0
        if max_degree > 4:
            raise RuntimeError(f"Polynomial-time reduction failed: max degree is {max_degree}, expected ≤ 4")
        
        
        # LP-based Unweighted Minimum Independent Dominating Set (MIDS)
        # with (Delta+1)/2 approximation guarantee (approximate for Δ=4)
        dominating_set = approx.mids_lp(G).independent_dominating_set
        # Extract original vertices from auxiliary vertex pairs
        solution1 = {u for u, _ in dominating_set}
        solution2 = set(graph) - solution1
        solution3 = approx.mids_lp(graph).independent_dominating_set
        min_solution = sorted([solution1, solution2, solution3], key=len)
        for solution in min_solution:
            if verify_independent_dominating_set(graph, solution):
                return solution
            
        raise RuntimeError(f"Polynomial-time reduction failed: No solution found")
    # Input validation: Ensure we have a proper NetworkX Graph
    if not isinstance(graph, nx.Graph):
        raise ValueError("Input must be an undirected NetworkX Graph.")
   
    # Handle trivial cases where no dominating set is needed
    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        return set()  # Empty graph or no edges means empty dominating set
   
    # Create a working copy to avoid modifying the input graph
    working_graph = graph.copy()
   
    # Preprocessing: Clean the graph by removing unnecessary elements
    # Remove self-loops since they don't affect dominating set (dominating sets itself)
    working_graph.remove_edges_from(list(nx.selfloop_edges(working_graph)))
   
    # Initialize the dominating set with all isolated nodes, as they must be included to dominate themselves
    approximate_dominating_set = set(nx.isolates(working_graph))
    # Remove isolated nodes (degree 0) as they don't contribute to any edge coverage
    working_graph.remove_nodes_from(approximate_dominating_set)
   
    # Check if preprocessing left us with an empty graph
    if working_graph.number_of_nodes() == 0:
        return approximate_dominating_set
   
    # Process each connected component independently for efficiency
    # This is optimal since components don't share edges, so their dominating sets are independent
    for component in nx.connected_components(working_graph):
        # Extract the induced subgraph for this connected component
        component_subgraph = working_graph.subgraph(component)
        
        # Apply the reduction-based algorithm to find independent dominating set for this component
        solution = dominating_via_reduction_max_degree_4(component_subgraph)
        
        # Add the component's independent dominating set to the overall solution
        approximate_dominating_set.update(solution)                  
    
    return approximate_dominating_set

def find_independent_dominating_set_brute_force(graph):
    """
    Computes an exact minimum independent dominating set in exponential time.

    Args:
        graph: A NetworkX Graph.

    Returns:
        A set of vertex indices representing the exact dominating set, or None if the graph is empty.
    """

    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        return None

    n_vertices = len(graph.nodes())

    for k in range(1, n_vertices + 1): # Iterate through all possible sizes of the cover
        for candidate in itertools.combinations(graph.nodes(), k):
            cover_candidate = set(candidate)
            if verify_independent_dominating_set(graph, cover_candidate):
                return cover_candidate
                
    return None

def find_independent_dominating_set_approximation(G):
    """
    LP-based Unweighted Minimum Independent Dominating Set (MIDS)
    with O(Delta) approximation guarantee.
    
    Parameters:
    - G: NetworkX graph with 'weight' attributes on nodes
    
    Returns:
    - A set of nodes representing an approximate independent weight independent dominating set
    """
    # Check if graph is empty
    if len(G) == 0:
        return set()
    
    solution = approx.mids_lp(G)
    return solution.independent_dominating_set

def calculate_solution_weight(G, solution):
    """Calculate the total weight of the nodes in the solution"""
    return sum(G.nodes[v].get('weight', 1.0) for v in solution)

def verify_independent_dominating_set(G, solution):
    """Verify that the solution is both independent and dominating"""
    # Check independence: no two nodes in solution should be adjacent
    for u in solution:
      for v in solution:
        if u != v and G.has_edge(u, v):
          return False
    
    # Check domination
    if not nx.dominating.is_dominating_set(G, solution):
      return False
    
    return True