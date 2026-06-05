"""
LP-based Unweighted Minimum Independent Dominating Set (MIDS)
with O(Delta) approximation guarantee.

Algorithm:
  1. Solve the LP relaxation of MIDS to obtain fractional values x_v in [0,1].
  2. Use LP solution to define a priority score for each node.
  3. Run a greedy Maximal Independent Set (MIS) sweep in priority order.

Key insight (Boppana/Halldorsson):
  Any maximal independent set IS a valid MIDS (independent + dominating).
  Greedy MIS gives |MIDS| <= (Delta+1)/2 * |OPT| for unweighted graphs,
  i.e., an O(Delta) approximation ratio.

LP relaxation (MIDS-LP):
  min   sum x_v
  s.t.  sum_{u in N[v]} x_u >= 1    for all v  (domination)
        x_u + x_v          <= 1    for all (u,v) in E  (independence)
        0 <= x_v <= 1
"""

import time
from dataclasses import dataclass
from typing import Set

import networkx as nx
import numpy as np
from scipy.optimize import linprog


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

@dataclass
class MIDSResult:
    independent_dominating_set: Set
    size: int
    lp_lower_bound: float
    approx_ratio: float          # size / lp_lower_bound
    delta: int                   # max degree
    lp_solve_time: float
    greedy_time: float
    lp_success: bool
    verified: bool


def solve_mids_lp(G: nx.Graph) -> tuple[np.ndarray, float, bool]:
    """
    Solve the LP relaxation of MIDS.
    Returns (x, objective, success).
    """
    nodes = list(G.nodes())
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}

    c = np.ones(n)   # minimize sum x_v (unweighted)

    A_ub, b_ub = [], []

    # Domination: for each v, -sum_{u in N[v]} x_u <= -1
    for v in nodes:
        row = np.zeros(n)
        for u in list(G.neighbors(v)) + [v]:
            row[idx[u]] = -1.0
        A_ub.append(row)
        b_ub.append(-1.0)

    # Independence: x_u + x_v <= 1 for each edge
    for u, v in G.edges():
        row = np.zeros(n)
        row[idx[u]] = 1.0
        row[idx[v]] = 1.0
        A_ub.append(row)
        b_ub.append(1.0)

    bounds = [(0.0, 1.0)] * n

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                     options={'presolve': True})

    if result.success:
        return result.x, result.fun, True
    else:
        # Fallback: uniform 1/(deg+1) fractional solution
        x = np.array([1.0 / (G.degree(v) + 1) for v in nodes])
        return x, float(np.sum(x)), False


def greedy_mis_from_priority(G: nx.Graph, priority_order: list) -> Set:
    """
    Build a Maximal Independent Set by processing nodes in priority_order.
    First node not yet excluded is added to the set and its neighbors excluded.
    The resulting set is always both independent and dominating.
    """
    independent_set = set()
    excluded = set()

    for v in priority_order:
        if v not in excluded:
            independent_set.add(v)
            excluded.add(v)
            excluded.update(G.neighbors(v))

    return independent_set


def mids_lp(G: nx.Graph) -> MIDSResult:
    """
    Compute a Minimum Independent Dominating Set approximation.

    Steps:
      1. Solve LP relaxation -> fractional x_v values.
      2. Sort nodes by x_v descending (higher LP value = prefer first).
      3. Run greedy MIS in that order -> valid MIDS.

    Approximation guarantee: O(Delta) for unweighted graphs.
    """
    if len(G) == 0:
        return MIDSResult(set(), 0, 0.0, 1.0, 0, 0.0, 0.0, True, True)

    nodes = list(G.nodes())
    delta = max(dict(G.degree()).values()) if G.number_of_nodes() > 0 else 0

    # Step 1: Solve LP
    t0 = time.perf_counter()
    x, lp_obj, lp_ok = solve_mids_lp(G)
    lp_time = time.perf_counter() - t0

    # Step 2: Priority order; nodes with higher fractional x_v go first
    # (LP "votes" them into the solution; greedy respects that preference)
    idx = {v: i for i, v in enumerate(nodes)}
    priority_order = sorted(nodes, key=lambda v: -x[idx[v]])

    # Step 3: Greedy MIS
    t0 = time.perf_counter()
    ids = greedy_mis_from_priority(G, priority_order)
    greedy_time = time.perf_counter() - t0

    # Verify
    is_independent = all(
        not G.has_edge(u, v) for u in ids for v in ids if u != v
    )
    is_dominating = nx.is_dominating_set(G, ids)
    verified = is_independent and is_dominating

    size = len(ids)
    ratio = size / lp_obj if lp_obj > 1e-9 else float('inf')

    return MIDSResult(
        independent_dominating_set=ids,
        size=size,
        lp_lower_bound=lp_obj,
        approx_ratio=ratio,
        delta=delta,
        lp_solve_time=lp_time,
        greedy_time=greedy_time,
        lp_success=lp_ok,
        verified=verified,
    )


# ---------------------------------------------------------------------------
# Baseline: pure greedy MIS (no LP guidance) for comparison
# ---------------------------------------------------------------------------

def mids_greedy_baseline(G: nx.Graph) -> MIDSResult:
    """
    Pure greedy MIDS: pick minimum-degree node first (no LP).
    Used as a baseline to compare against LP-guided version.
    """
    if len(G) == 0:
        return MIDSResult(set(), 0, 0.0, 1.0, 0, 0.0, 0.0, True, True)

    nodes = list(G.nodes())
    delta = max(dict(G.degree()).values()) if nodes else 0

    t0 = time.perf_counter()
    # Heuristic: process low-degree nodes first (they're cheaper to include)
    priority_order = sorted(nodes, key=lambda v: G.degree(v))
    ids = greedy_mis_from_priority(G, priority_order)
    greedy_time = time.perf_counter() - t0

    verified = (
        all(not G.has_edge(u, v) for u in ids for v in ids if u != v)
        and nx.is_dominating_set(G, ids)
    )

    # Compute LP bound for fair ratio comparison
    _, lp_obj, lp_ok = solve_mids_lp(G)
    ratio = len(ids) / lp_obj if lp_obj > 1e-9 else float('inf')

    return MIDSResult(
        independent_dominating_set=ids,
        size=len(ids),
        lp_lower_bound=lp_obj,
        approx_ratio=ratio,
        delta=delta,
        lp_solve_time=0.0,
        greedy_time=greedy_time,
        lp_success=lp_ok,
        verified=verified,
    )


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

def run_tests():
    import math

    test_cases = []

    # 1. Small hand-verifiable graphs
    G1 = nx.path_graph(6)
    test_cases.append(("Path P6",         G1))

    G2 = nx.cycle_graph(8)
    test_cases.append(("Cycle C8",        G2))

    G3 = nx.complete_graph(6)
    test_cases.append(("Complete K6",     G3))

    G4 = nx.complete_bipartite_graph(4, 4)
    test_cases.append(("Complete Bip K4,4", G4))

    G5 = nx.petersen_graph()
    test_cases.append(("Petersen",        G5))

    G6 = nx.star_graph(8)
    test_cases.append(("Star S8",         G6))

    # 2. Random graphs (various densities)
    rng = np.random.default_rng(42)
    G7 = nx.erdos_renyi_graph(30, 0.15, seed=42)
    test_cases.append(("ER n=30 p=0.15",  G7))

    G8 = nx.erdos_renyi_graph(30, 0.40, seed=7)
    test_cases.append(("ER n=30 p=0.40",  G8))

    G9 = nx.barabasi_albert_graph(40, 2, seed=0)
    test_cases.append(("BA n=40 m=2",     G9))

    G10 = nx.barabasi_albert_graph(50, 4, seed=1)
    test_cases.append(("BA n=50 m=4",     G10))

    # 3. Grid graphs
    G11 = nx.grid_2d_graph(5, 5)
    test_cases.append(("Grid 5x5",        G11))

    G12 = nx.grid_2d_graph(6, 6)
    test_cases.append(("Grid 6x6",        G12))

    # 4. Trees
    G13 = nx.random_labeled_tree(20, seed=3)
    test_cases.append(("Random Tree n=20",G13))

    G14 = nx.balanced_tree(3, 3)
    test_cases.append(("Balanced Tree r=3 h=3", G14))

    # 5. Regular graphs
    G15 = nx.random_regular_graph(3, 20, seed=5)
    test_cases.append(("3-Regular n=20",  G15))

    G16 = nx.random_regular_graph(4, 20, seed=6)
    test_cases.append(("4-Regular n=20",  G16))

    # 6. Larger instance
    G17 = nx.erdos_renyi_graph(80, 0.10, seed=99)
    # ensure connected
    if not nx.is_connected(G17):
        G17 = G17.subgraph(max(nx.connected_components(G17), key=len)).copy()
    test_cases.append(("ER n=80 p=0.10",  G17))

    # Header
    col = {
        'name':    26,
        'n':        5,
        'm':        6,
        'delta':    5,
        'lp_lb':    7,
        'lp_sz':    6,
        'gr_sz':    6,
        'lp_rat':   7,
        'gr_rat':   7,
        'ok':       5,
        'lp_t':     8,
    }

    hdr = (
        f"{'Graph':<{col['name']}} "
        f"{'n':>{col['n']}} "
        f"{'m':>{col['m']}} "
        f"{'Delta':>{col['delta']}} "
        f"{'LP lb':>{col['lp_lb']}} "
        f"{'LP sz':>{col['lp_sz']}} "
        f"{'GR sz':>{col['gr_sz']}} "
        f"{'LP rat':>{col['lp_rat']}} "
        f"{'GR rat':>{col['gr_rat']}} "
        f"{'OK?':>{col['ok']}} "
        f"{'LP time':>{col['lp_t']}}"
    )
    sep = "-" * len(hdr)
    print(sep)
    print("  LP-Guided vs Greedy Baseline - Unweighted MIDS Approximation")
    print(sep)
    print(hdr)
    print(sep)

    all_passed = True
    lp_wins = 0
    ties = 0
    gr_wins = 0

    for name, G in test_cases:
        # Ensure simple undirected
        G = nx.Graph(G)
        if len(G) == 0:
            continue

        r_lp = mids_lp(G)
        r_gr  = mids_greedy_baseline(G)

        ok = "yes" if r_lp.verified else "no"
        if not r_lp.verified:
            all_passed = False

        if r_lp.size < r_gr.size:
            lp_wins += 1
        elif r_lp.size == r_gr.size:
            ties += 1
        else:
            gr_wins += 1

        print(
            f"{name:<{col['name']}} "
            f"{G.number_of_nodes():>{col['n']}} "
            f"{G.number_of_edges():>{col['m']}} "
            f"{r_lp.delta:>{col['delta']}} "
            f"{r_lp.lp_lower_bound:>{col['lp_lb']}.2f} "
            f"{r_lp.size:>{col['lp_sz']}} "
            f"{r_gr.size:>{col['gr_sz']}} "
            f"{r_lp.approx_ratio:>{col['lp_rat']}.3f} "
            f"{r_gr.approx_ratio:>{col['gr_rat']}.3f} "
            f"{ok:>{col['ok']}} "
            f"{r_lp.lp_solve_time*1000:>{col['lp_t']}.1f}ms"
        )

    print(sep)
    print(f"\nAll verified: {'YES' if all_passed else 'NO'}")
    print(f"LP-guided wins: {lp_wins}  |  Ties: {ties}  |  Greedy wins: {gr_wins}")
    print()
    print("Column legend:")
    print("  LP lb  = LP relaxation lower bound on |OPT|")
    print("  LP sz  = MIDS size from LP-guided greedy")
    print("  GR sz  = MIDS size from degree-based greedy (baseline)")
    print("  LP rat = LP sz / LP lb  (approximation ratio, LP-guided)")
    print("  GR rat = GR sz / LP lb  (approximation ratio, baseline)")
    print("  OK?    = verified independent + dominating")
    print(sep)

    # --- Approximation ratio vs Delta analysis ---
    print("\n  Approximation Ratio vs O(Delta) Bound")
    print(sep)
    print(f"  {'Graph':<{col['name']}}  {'Delta':>5}  {'(Delta+1)/2':>12}  {'LP ratio':>9}  {'within bound?':>14}")
    print(sep)
    for name, G in test_cases:
        G = nx.Graph(G)
        if len(G) == 0:
            continue
        r = mids_lp(G)
        bound = (r.delta + 1) / 2
        within = "yes" if r.approx_ratio <= bound + 1e-6 else "no"
        print(f"  {name:<{col['name']}}  {r.delta:>5}  {bound:>12.1f}  {r.approx_ratio:>9.3f}  {within:>14}")
    print(sep)


if __name__ == "__main__":
    run_tests()
