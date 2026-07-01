"""
car -- Constant Approximation Ratio suite.

Runs Siriaisa (``siriaisa.algorithm.find_independent_dominating_set``) against an
exact SciPy MILP optimum on ten thousand instances drawn from the structured
graph families and random-graph models defined in ``generators.py``, then writes
per-instance rows and a per-family summary to CSV.

The suite is the empirical counterpart of the manuscript's family theorems:

  * "bounded" families must stay within their degree constant (2, 3, 4, r, ...),
  * "rigid" families must report ratio exactly 1,
  * "random" instances must obey the maximal-independent-set bound (ratio <= Delta).

Any row whose exact ratio exceeds the family's expected constant is flagged as a
violation, which would indicate either a bug or a broken theorem.

Usage
-----
    python run_car.py                     # 10000 instances (default)
    python run_car.py --count 200         # quick smoke run
    python run_car.py --count 10000 --seed 7 --outdir results
    python run_car.py --dump-dimacs       # also save each instance as DIMACS

Outputs (under --outdir, default ``car/results``):
    car_results.csv   one row per instance
    car_summary.csv   one row per family
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
import sys

import networkx as nx
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

# Make the parent repository importable so ``siriaisa`` is found when this file
# is run directly from the ``car`` directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from siriaisa import algorithm  # noqa: E402

import generators  # noqa: E402  (local module)

EPS = 1e-6


# ---------------------------------------------------------------------------
# Exact optimum via SciPy MILP (same model as experiments/run_adversarial_milp)
# ---------------------------------------------------------------------------

def exact_mids_milp(graph: nx.Graph, time_limit: float = 30.0) -> tuple[set, float]:
    nodes = sorted(graph.nodes())
    index = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    if n == 0:
        return set(), 0.0

    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    # Domination: sum_{u in N[v]} x_u >= 1
    for v in nodes:
        row = np.zeros(n)
        row[index[v]] = 1.0
        for u in graph.neighbors(v):
            row[index[u]] = 1.0
        rows.append(row)
        lower.append(1.0)
        upper.append(np.inf)

    # Independence: x_u + x_v <= 1
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
        options={"time_limit": time_limit},
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if not result.success:
        raise RuntimeError(f"MILP failed: {result.message}")

    x = np.rint(result.x).astype(int)
    solution = {nodes[i] for i, bit in enumerate(x) if bit == 1}
    if not algorithm.verify_independent_dominating_set(graph, solution):
        raise RuntimeError("MILP solution failed independent-domination verification")
    return solution, elapsed_ms


# ---------------------------------------------------------------------------
# DIMACS dump (matches siriaisa.parser: 1-based, ``p edge n m`` + ``e u v``)
# ---------------------------------------------------------------------------

def write_dimacs(graph: nx.Graph, path: Path) -> None:
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    with path.open("w") as handle:
        handle.write("c car suite instance\n")
        handle.write(f"p edge {n} {m}\n")
        for u, v in graph.edges():
            handle.write(f"e {u + 1} {v + 1}\n")


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def run(count: int, seed: int, outdir: Path, time_limit: float, dump_dimacs: bool) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    results_path = outdir / "car_results.csv"
    summary_path = outdir / "car_summary.csv"
    dimacs_dir = outdir / "instances"
    if dump_dimacs:
        dimacs_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "index", "family", "kind", "n", "m", "delta",
        "siriaisa_size", "optimum", "ratio", "expected_constant",
        "within_bound", "siriaisa_ms", "milp_ms",
    ]

    # Per-family accumulators.
    summary: dict[str, dict] = {}
    total_violations = 0
    milp_failures = 0

    with results_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for index in range(count):
            family, graph = generators.generate_instance(index, seed=seed)
            n = graph.number_of_nodes()
            m = graph.number_of_edges()
            delta = generators.max_degree(graph)

            if n == 0:
                continue

            t0 = time.perf_counter()
            approx_set = algorithm.find_independent_dominating_set(graph)
            siriaisa_ms = (time.perf_counter() - t0) * 1000.0

            if not algorithm.verify_independent_dominating_set(graph, approx_set):
                raise RuntimeError(f"Siriaisa produced an invalid set on instance {index} ({family.name})")

            try:
                optimum_set, milp_ms = exact_mids_milp(graph, time_limit=time_limit)
            except RuntimeError:
                milp_failures += 1
                continue

            optimum = len(optimum_set)
            size = len(approx_set)
            ratio = size / optimum if optimum > 0 else float("inf")
            expected = family.expected_constant(graph)
            within = ratio <= expected + EPS
            if not within:
                total_violations += 1

            if dump_dimacs:
                write_dimacs(graph, dimacs_dir / f"{index:05d}_{family.name}.dimacs")

            writer.writerow({
                "index": index,
                "family": family.name,
                "kind": family.kind,
                "n": n,
                "m": m,
                "delta": delta,
                "siriaisa_size": size,
                "optimum": optimum,
                "ratio": f"{ratio:.4f}",
                "expected_constant": f"{expected:.4f}",
                "within_bound": int(within),
                "siriaisa_ms": f"{siriaisa_ms:.3f}",
                "milp_ms": f"{milp_ms:.3f}",
            })

            acc = summary.setdefault(family.name, {
                "kind": family.kind, "count": 0, "sum_ratio": 0.0,
                "max_ratio": 0.0, "violations": 0, "sum_delta": 0,
                "worst_expected": 0.0,
            })
            acc["count"] += 1
            acc["sum_ratio"] += ratio
            acc["max_ratio"] = max(acc["max_ratio"], ratio)
            acc["sum_delta"] += delta
            acc["worst_expected"] = max(acc["worst_expected"], expected)
            if not within:
                acc["violations"] += 1

            if (index + 1) % 500 == 0:
                print(f"  ...{index + 1}/{count} instances processed")

    # Per-family summary.
    with summary_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "family", "kind", "count", "mean_ratio", "max_ratio",
            "expected_constant", "mean_delta", "violations",
        ])
        for name in sorted(summary):
            acc = summary[name]
            c = acc["count"]
            writer.writerow([
                name, acc["kind"], c,
                f"{acc['sum_ratio'] / c:.4f}",
                f"{acc['max_ratio']:.4f}",
                f"{acc['worst_expected']:.4f}",
                f"{acc['sum_delta'] / c:.2f}",
                acc["violations"],
            ])

    # Console report.
    print("\n" + "=" * 78)
    print("  car -- Constant Approximation Ratio suite: per-family summary")
    print("=" * 78)
    header = f"{'family':<20}{'kind':<9}{'count':>6}{'mean':>8}{'max':>7}{'const':>7}{'viol':>6}"
    print(header)
    print("-" * len(header))
    for name in sorted(summary):
        acc = summary[name]
        c = acc["count"]
        print(f"{name:<20}{acc['kind']:<9}{c:>6}"
              f"{acc['sum_ratio'] / c:>8.3f}{acc['max_ratio']:>7.3f}"
              f"{acc['worst_expected']:>7.2f}{acc['violations']:>6}")
    print("-" * len(header))
    print(f"MILP failures (skipped): {milp_failures}")
    print(f"TOTAL family-constant violations: {total_violations}")
    print(f"\nPer-instance rows : {results_path}")
    print(f"Per-family summary: {summary_path}")
    if total_violations == 0:
        print("\nAll instances satisfied their family constant "
              "(bounded <= Delta bound, rigid == 1).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the car (Constant Approximation Ratio) suite.")
    parser.add_argument("--count", type=int, default=10000, help="number of instances (default 10000)")
    parser.add_argument("--seed", type=int, default=12345, help="master RNG seed")
    parser.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent / "results",
                        help="output directory for CSV files")
    parser.add_argument("--time-limit", type=float, default=30.0, help="MILP time limit (s) per instance")
    parser.add_argument("--dump-dimacs", action="store_true", help="also save each instance as a DIMACS file")
    args = parser.parse_args()

    print(f"Running car suite: count={args.count} seed={args.seed} outdir={args.outdir}")
    start = time.perf_counter()
    run(args.count, args.seed, args.outdir, args.time_limit, args.dump_dimacs)
    print(f"\nTotal wall time: {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
