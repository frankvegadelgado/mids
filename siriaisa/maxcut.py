from collections import deque
import networkx as nx


def maxcut_bipartite_min_side_linear(G, minimize_side=1):
    """
    Exact ordinary Max-Cut for bipartite graphs in O(n + m), with a
    lexicographic tie-break:

        1. maximize the cut value;
        2. among maximum cuts, minimize the requested side size;
        3. among those, minimize the number of non-cut edges of the form
           (minimize_side, minimize_side).

    If minimize_side=1, the third objective minimizes the number of edges
    whose two endpoints are both assigned 1.

    If minimize_side=0, the third objective minimizes the number of edges
    whose two endpoints are both assigned 0.

    For a genuinely bipartite graph, every returned maximum cut cuts every
    edge, so the final number of (1, 1) and (0, 0) non-cut edges is always 0.
    The secondary tie-break is still implemented explicitly and costs only
    linear time. It is relevant only when there are equal-size bipartition
    classes and a future variant supplies additional same-side/penalty edges.

    Runtime
    -------
    O(|V| + |E|).

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
            "minimized_same_side_edges": int,
            "same_side_0_edges": int,
            "same_side_1_edges": int,
            "bad_edge": edge if non-bipartite else None
        }
    """

    if G.is_directed():
        raise ValueError("This implementation expects an undirected graph.")

    if minimize_side not in (0, 1):
        raise ValueError("minimize_side must be 0 or 1.")

    color = {}
    assignment = {}

    def internal_edge_count(part):
        """Count edges with both endpoints in part in O(sum degrees of part)."""
        part = set(part)
        count_twice = 0
        for u in part:
            for v in G.neighbors(u):
                if v in part:
                    count_twice += 1
        return count_twice // 2

    for start in G.nodes():
        if start in color:
            continue

        # Isolated vertices do not affect Max-Cut or same-side edge counts.
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
                        "minimized_same_side_edges": None,
                        "same_side_0_edges": None,
                        "same_side_1_edges": None,
                        "bad_edge": (u, v),
                    }

        # Two maximum-cut orientations are possible for this component:
        #   A: part_0 goes to minimize_side
        #   B: part_1 goes to minimize_side
        # Choose lexicographically by:
        #   (size of minimize_side, number of same-side minimized edges).
        # For simple bipartite components, both internal counts are 0, but
        # computing them keeps the requested secondary objective explicit.
        cost_0 = (len(part_0), internal_edge_count(part_0))
        cost_1 = (len(part_1), internal_edge_count(part_1))

        if cost_0 <= cost_1:
            minimized_part = part_0
            other_part = part_1
        else:
            minimized_part = part_1
            other_part = part_0

        for v in minimized_part:
            assignment[v] = minimize_side

        other_side = minimize_side ^ 1
        for v in other_part:
            assignment[v] = other_side

    side_0 = {v for v, bit in assignment.items() if bit == 0}
    side_1 = {v for v, bit in assignment.items() if bit == 1}

    same_side_0_edges = 0
    same_side_1_edges = 0
    cut_edges = 0

    for u, v in G.edges():
        au = assignment[u]
        av = assignment[v]
        if au != av:
            cut_edges += 1
        elif au == 0:
            same_side_0_edges += 1
        else:
            same_side_1_edges += 1

    minimized_same_side_edges = (
        same_side_1_edges if minimize_side == 1 else same_side_0_edges
    )

    return {
        "feasible": True,
        "is_bipartite": True,
        "maxcut_value": G.number_of_edges(),
        "assignment": assignment,
        "side_0": side_0,
        "side_1": side_1,
        "minimized_side": minimize_side,
        "minimized_side_size": len(side_1) if minimize_side == 1 else len(side_0),
        "minimized_same_side_edges": minimized_same_side_edges,
        "same_side_0_edges": same_side_0_edges,
        "same_side_1_edges": same_side_1_edges,
        "bad_edge": None,
    }


def cut_value(G, assignment):
    """Count edges crossing the cut."""
    return sum(1 for u, v in G.edges() if assignment[u] != assignment[v])


def same_side_edge_count(G, assignment, side):
    """Count edges whose two endpoints are both assigned to side."""
    if side not in (0, 1):
        raise ValueError("side must be 0 or 1.")
    return sum(1 for u, v in G.edges() if assignment[u] == side and assignment[v] == side)


def verify_maxcut_min_side_result(G, result):
    """Verify that all edges are cut."""
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
            print("same-side 0 edges:", result["same_side_0_edges"])
            print("same-side 1 edges:", result["same_side_1_edges"])
            print("assignment:", result["assignment"])

            assert verify_maxcut_min_side_result(G, result)
            assert same_side_edge_count(G, result["assignment"], 1) == result["same_side_1_edges"]
        else:
            print("not bipartite; bad edge:", result["bad_edge"])
