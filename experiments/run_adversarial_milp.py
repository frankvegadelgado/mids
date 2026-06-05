"""Run Iris against exact SciPy MILP on the adversarial DIMACS suite."""

from __future__ import annotations

import time
from pathlib import Path
import sys

import networkx as nx
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mids import algorithm, parser


def exact_mids_milp(graph: nx.Graph) -> tuple[set[int], float]:
    nodes = sorted(graph.nodes())
    index = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    for v in nodes:
        row = np.zeros(n)
        row[index[v]] = 1.0
        for u in graph.neighbors(v):
            row[index[u]] = 1.0
        rows.append(row)
        lower.append(1.0)
        upper.append(np.inf)

    for u, v in graph.edges():
        row = np.zeros(n)
        row[index[u]] = 1.0
        row[index[v]] = 1.0
        rows.append(row)
        lower.append(-np.inf)
        upper.append(1.0)

    start = time.perf_counter()
    result = milp(
        c=np.ones(n),
        integrality=np.ones(n),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(np.vstack(rows), np.array(lower), np.array(upper)),
        options={"time_limit": 60},
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if not result.success:
        raise RuntimeError(f"MILP failed on {graph}: {result.message}")

    x = np.rint(result.x).astype(int)
    solution = {nodes[i] for i, bit in enumerate(x) if bit == 1}
    if not algorithm.verify_independent_dominating_set(graph, solution):
        raise RuntimeError("MILP solution failed independent-domination verification")
    return solution, elapsed_ms


def main() -> None:
    base = Path(__file__).resolve().parent
    for path in sorted(base.glob("adv_*.dimacs")):
        graph = parser.read(path)
        delta = max(dict(graph.degree()).values()) if graph.number_of_nodes() else 0

        start = time.perf_counter()
        iris = algorithm.find_independent_dominating_set(graph)
        iris_ms = (time.perf_counter() - start) * 1000.0
        if not algorithm.verify_independent_dominating_set(graph, iris):
            raise RuntimeError(f"Iris solution failed verification on {path.name}")

        optimum, milp_ms = exact_mids_milp(graph)
        ratio = len(iris) / len(optimum)
        print(
            f"{path.name}: n={graph.number_of_nodes()} m={graph.number_of_edges()} "
            f"Delta={delta} Iris={len(iris)} Opt={len(optimum)} "
            f"ratio={ratio:.3f} Iris_ms={iris_ms:.3f} MILP_ms={milp_ms:.3f}"
        )


if __name__ == "__main__":
    main()
