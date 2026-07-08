from collections import deque
import networkx as nx


def maxcut_bipartite_min_side_linear(G, minimize_side=1):
    """
    Exact ordinary Max-Cut for bipartite graphs in O(n + m),
    while minimizing one side of the assignment as much as possible.

    Problem
    -------
    For a bipartite graph G, ordinary Max-Cut is trivial:

        MaxCut(G) = |E(G)|.

    But there may be many maximum-cut assignments, especially when G has
    several connected components. This function returns a maximum-cut
    assignment that minimizes the number of vertices assigned to one chosen
    side.

    If minimize_side=1:
        minimize the number of vertices assigned 1.

    If minimize_side=0:
        minimize the number of vertices assigned 0.

    Isolated vertices do not affect the cut, so they are placed on the
    opposite side of minimize_side.

    Why this is optimal
    -------------------
    In every non-isolated connected bipartite component, the two bipartition
    classes must be placed on opposite sides to cut every edge. The only
    freedom is flipping the whole component. Therefore, to minimize one side,
    put the smaller bipartition class of each component on that side.

    Runtime
    -------
    O(n + m).

    Returns
    -------
    dict:
        {
            "feasible": True/False,
            "is_bipartite": True/False,
            "maxcut_value": int,
            "assignment": dict node -> 0/1,
            "side_0": set,
            "side_1": set,
            "minimized_side": 0 or 1,
            "minimized_side_size": int,
            "bad_edge": edge if non-bipartite else None
        }
    """

    if G.is_directed():
        raise ValueError("This implementation expects an undirected graph.")

    if minimize_side not in (0, 1):
        raise ValueError("minimize_side must be 0 or 1.")

    color = {}
    assignment = {}

    for start in G.nodes():
        if start in color:
            continue

        # Isolated vertices do not affect Max-Cut.
        # Put them outside the minimized side.
        if G.degree(start) == 0:
            color[start] = 0
            assignment[start] = minimize_side ^ 1
            continue

        # BFS 2-color this connected component.
        color[start] = 0
        q = deque([start])
        part_0 = {start}
        part_1 = set()

        while q:
            u = q.popleft()

            for v in G.neighbors(u):
                expected = color[u] ^ 1

                if v not in color:
                    color[v] = expected

                    if expected == 0:
                        part_0.add(v)
                    else:
                        part_1.add(v)

                    q.append(v)

                elif color[v] != expected:
                    return {
                        "feasible": False,
                        "is_bipartite": False,
                        "maxcut_value": None,
                        "assignment": None,
                        "side_0": None,
                        "side_1": None,
                        "minimized_side": minimize_side,
                        "minimized_side_size": None,
                        "bad_edge": (u, v),
                    }

        # Put the smaller bipartition class into minimize_side.
        if len(part_0) <= len(part_1):
            small = part_0
            large = part_1
        else:
            small = part_1
            large = part_0

        for v in small:
            assignment[v] = minimize_side

        for v in large:
            assignment[v] = minimize_side ^ 1

    side_0 = {v for v, bit in assignment.items() if bit == 0}
    side_1 = {v for v, bit in assignment.items() if bit == 1}

    maxcut_value = G.number_of_edges()

    return {
        "feasible": True,
        "is_bipartite": True,
        "maxcut_value": maxcut_value,
        "assignment": assignment,
        "side_0": side_0,
        "side_1": side_1,
        "minimized_side": minimize_side,
        "minimized_side_size": len(side_1) if minimize_side == 1 else len(side_0),
        "bad_edge": None,
    }


def cut_value(G, assignment):
    """
    Count edges crossing the cut.
    """
    return sum(
        1
        for u, v in G.edges()
        if assignment[u] != assignment[v]
    )


def verify_maxcut_min_side_result(G, result):
    """
    Verify that all edges are cut.
    """
    if not result["feasible"]:
        return False

    return cut_value(G, result["assignment"]) == G.number_of_edges()

if __name__ == "__main__":
    tests = {
        "path P5": nx.path_graph(5),
        "path P6": nx.path_graph(6),
        "cycle C4": nx.cycle_graph(4),
        "cycle C6": nx.cycle_graph(6),
        "star K1,5": nx.star_graph(5),
        "K3,4": nx.complete_bipartite_graph(3, 4),
        "disconnected": nx.disjoint_union(nx.path_graph(5), nx.star_graph(4)),
        "triangle non-bipartite": nx.cycle_graph(3),
    }

    for name, G in tests.items():
        result = maxcut_bipartite_min_side_linear(G, minimize_side=1)

        print()
        print(name)
        print("bipartite:", result["is_bipartite"])

        if result["feasible"]:
            print("Max-Cut value:", result["maxcut_value"])
            print("cut check:", cut_value(G, result["assignment"]))
            print("side 0 size:", len(result["side_0"]))
            print("side 1 size:", len(result["side_1"]))
            print("minimized side:", result["minimized_side"])
            print("minimized side size:", result["minimized_side_size"])
            print("assignment:", result["assignment"])

            assert verify_maxcut_min_side_result(G, result)
        else:
            print("not bipartite; bad edge:", result["bad_edge"])