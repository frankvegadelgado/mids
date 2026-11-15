"""
car_baselines -- Siriaisa against baseline heuristics on the car instances.

The instances are the ones of ``run_car.py`` (``generators.generate_instance``
with the same seed).  On every instance the script computes an independent
dominating set with

  * ``siriaisa``      Siriaisa (``siriaisa.algorithm.find_independent_dominating_set``),
  * ``random_mis``    a random maximal independent set of NetworkX
                      (``networkx.maximal_independent_set``),
  * ``greedy_min``    the minimum-degree greedy: select a vertex of smallest
                      degree among the free vertices and delete its closed
                      neighbourhood,
  * ``greedy_max``    the maximum-degree greedy: select a free vertex with the
                      largest number of free neighbours and delete its closed
                      neighbourhood,
  * ``greedy_swap``   ``greedy_max`` followed by a local search with the 2-swap
                      move (two vertices of the set are dropped and one vertex
                      is added) until no such move exists,

and compares the sizes with the exact optimum.  The optimum is read from
``results/car_results.csv`` when the row of the instance is there with the same
family, number of vertices and number of edges; otherwise it is computed with
the integer program of ``run_car.py``.

Every set is verified to be an independent dominating set before its size is
used.  The script uses the repository's own ``siriaisa`` folder, so it runs
whether or not the package is installed.

Usage
-----
    python car_baselines.py                 # 10000 instances (default)
    python car_baselines.py --count 200     # quick run
    python car_baselines.py --seed 12345 --outdir results

Outputs (under --outdir, default ``car/results``):
    car_baselines.csv           one row per instance
    car_baselines_summary.csv   one row per family and one row ALL
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import networkx as nx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from siriaisa import algorithm  # noqa: E402

import generators  # noqa: E402  (local module)

METHODS = ["siriaisa", "random_mis", "greedy_min", "greedy_max", "greedy_swap"]


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def random_mis(graph: nx.Graph, seed: int) -> set:
    """A random maximal independent set (NetworkX)."""
    if graph.number_of_nodes() == 0:
        return set()
    return set(nx.maximal_independent_set(graph, seed=seed))


def _greedy(graph: nx.Graph, largest: bool) -> set:
    """Select a free vertex with the smallest (or largest) number of free
    neighbours, delete its closed neighbourhood, and repeat.  Ties are broken
    by the smallest label."""
    free = set(graph.nodes())
    free_degree = {v: graph.degree(v) for v in free}
    chosen = set()
    while free:
        if largest:
            v = min(free, key=lambda x: (-free_degree[x], x))
        else:
            v = min(free, key=lambda x: (free_degree[x], x))
        chosen.add(v)
        removed = [v] + [u for u in graph.neighbors(v) if u in free]
        for u in removed:
            free.discard(u)
        for u in removed:
            for w in graph.neighbors(u):
                if w in free:
                    free_degree[w] -= 1
    return chosen


def greedy_min(graph: nx.Graph) -> set:
    return _greedy(graph, largest=False)


def greedy_max(graph: nx.Graph) -> set:
    return _greedy(graph, largest=True)


def two_swap(graph: nx.Graph, start: set) -> set:
    """Local search with the 2-swap move on a maximal independent set.

    A move drops two vertices u, w of the set and adds one vertex v.  The new
    set is an independent dominating set exactly when u and w are the only
    neighbours of v in the set and every vertex whose neighbours in the set
    are among u and w is v or a neighbour of v.  Moves are applied until none
    exists; every move makes the set one vertex smaller.
    """
    current = set(start)
    count = {v: 0 for v in graph.nodes()}
    for s in current:
        for x in graph.neighbors(s):
            count[x] += 1
    improved = True
    while improved:
        improved = False
        for v in sorted(graph.nodes()):
            if v in current or count[v] != 2:
                continue
            u, w = [x for x in graph.neighbors(v) if x in current]
            closed_v = set(graph.neighbors(v))
            closed_v.add(v)
            ok = True
            for x in set(graph.neighbors(u)) | set(graph.neighbors(w)):
                if x in closed_v:
                    continue
                others = count[x] - (1 if graph.has_edge(x, u) else 0) \
                    - (1 if graph.has_edge(x, w) else 0)
                if others == 0:
                    ok = False
                    break
            if not ok:
                continue
            for s in (u, w):
                current.discard(s)
                for x in graph.neighbors(s):
                    count[x] -= 1
            current.add(v)
            for x in graph.neighbors(v):
                count[x] += 1
            improved = True
            break
    return current


# ---------------------------------------------------------------------------
# Reference optimum
# ---------------------------------------------------------------------------

def load_reference(path: Path) -> dict:
    """index -> (family, n, m, optimum) from car_results.csv, if present."""
    reference = {}
    if not path.exists():
        return reference
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                reference[int(row["index"])] = (
                    row["family"], int(row["n"]), int(row["m"]),
                    int(row["optimum"]))
            except (KeyError, ValueError):
                continue
    return reference


def optimum_of(index, family, graph, reference, time_limit):
    """The exact optimum and its origin ("csv" or "milp")."""
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    row = reference.get(index)
    if row is not None and row[0] == family.name and row[1] == n and row[2] == m:
        return row[3], "csv"
    import run_car  # the integer program of run_car.py
    solution, _ = run_car.exact_mids_milp(graph, time_limit=time_limit)
    return len(solution), "milp"


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def timed(function, *arguments):
    start = time.perf_counter()
    result = function(*arguments)
    return result, (time.perf_counter() - start) * 1000.0


def run(count: int, seed: int, outdir: Path, time_limit: float) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    results_path = outdir / "car_baselines.csv"
    summary_path = outdir / "car_baselines_summary.csv"
    reference = load_reference(outdir / "car_results.csv")
    print(f"Reference optima read from car_results.csv: {len(reference)}",
          flush=True)

    fieldnames = ["index", "family", "kind", "n", "m", "delta", "optimum",
                  "optimum_from"]
    for name in METHODS:
        fieldnames += [f"{name}_size", f"{name}_ratio", f"{name}_ms"]

    summary: dict[str, dict] = {}
    skipped = 0

    def account(key, kind, sizes, times, optimum):
        acc = summary.setdefault(key, {"kind": kind, "count": 0})
        acc["count"] += 1
        for name in METHODS:
            ratio = sizes[name] / optimum
            acc[f"{name}_optimal"] = acc.get(f"{name}_optimal", 0) + (
                1 if sizes[name] == optimum else 0)
            acc[f"{name}_sum"] = acc.get(f"{name}_sum", 0.0) + ratio
            acc[f"{name}_max"] = max(acc.get(f"{name}_max", 0.0), ratio)
            acc[f"{name}_ms"] = acc.get(f"{name}_ms", 0.0) + times[name]

    with results_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(count):
            family, graph = generators.generate_instance(index, seed=seed)
            n = graph.number_of_nodes()
            if n == 0:
                continue
            try:
                optimum, origin = optimum_of(index, family, graph, reference,
                                             time_limit)
            except RuntimeError:
                skipped += 1
                continue

            sets, times = {}, {}
            sets["siriaisa"], times["siriaisa"] = timed(
                algorithm.find_independent_dominating_set, graph)
            sets["random_mis"], times["random_mis"] = timed(
                random_mis, graph, seed + index)
            sets["greedy_min"], times["greedy_min"] = timed(greedy_min, graph)
            sets["greedy_max"], times["greedy_max"] = timed(greedy_max, graph)
            swapped, extra = timed(two_swap, graph, sets["greedy_max"])
            sets["greedy_swap"] = swapped
            times["greedy_swap"] = times["greedy_max"] + extra

            sizes = {}
            for name in METHODS:
                if not algorithm.verify_independent_dominating_set(
                        graph, sets[name]):
                    raise RuntimeError(
                        f"{name} returned an invalid set on instance {index} "
                        f"({family.name})")
                sizes[name] = len(sets[name])
                if sizes[name] < optimum:
                    raise RuntimeError(
                        f"{name} is below the reference optimum on instance "
                        f"{index} ({family.name})")

            row = {
                "index": index, "family": family.name, "kind": family.kind,
                "n": n, "m": graph.number_of_edges(),
                "delta": generators.max_degree(graph),
                "optimum": optimum, "optimum_from": origin,
            }
            for name in METHODS:
                row[f"{name}_size"] = sizes[name]
                row[f"{name}_ratio"] = f"{sizes[name] / optimum:.4f}"
                row[f"{name}_ms"] = f"{times[name]:.3f}"
            writer.writerow(row)

            account(family.name, family.kind, sizes, times, optimum)
            account("ALL", "all", sizes, times, optimum)

            if (index + 1) % 500 == 0:
                print(f"  ...{index + 1}/{count} instances processed",
                      flush=True)

    names = sorted(k for k in summary if k != "ALL") + (
        ["ALL"] if "ALL" in summary else [])
    with summary_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["family", "kind", "count"]
        for name in METHODS:
            header += [f"{name}_optimal", f"{name}_mean_ratio",
                       f"{name}_max_ratio", f"{name}_total_ms"]
        writer.writerow(header)
        for key in names:
            acc = summary[key]
            c = acc["count"]
            line = [key, acc["kind"], c]
            for name in METHODS:
                line += [acc[f"{name}_optimal"], f"{acc[f'{name}_sum'] / c:.4f}",
                         f"{acc[f'{name}_max']:.4f}", f"{acc[f'{name}_ms']:.1f}"]
            writer.writerow(line)

    print("\n" + "=" * 100)
    print("  car_baselines: optimal instances / mean ratio / max ratio")
    print("=" * 100)
    head = f"{'family':<20}{'count':>6}"
    for name in METHODS:
        head += f"{name:>24}"
    print(head)
    print("-" * len(head))
    for key in names:
        acc = summary[key]
        c = acc["count"]
        line = f"{key:<20}{c:>6}"
        for name in METHODS:
            cell = (f"{acc[f'{name}_optimal']}/{acc[f'{name}_sum'] / c:.3f}"
                    f"/{acc[f'{name}_max']:.3f}")
            line += f"{cell:>24}"
        print(line)
    print("-" * len(head))
    print(f"Instances skipped (no optimum): {skipped}")
    print(f"\nPer-instance rows : {results_path}")
    print(f"Per-family summary: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Siriaisa against baseline heuristics on the car instances.")
    parser.add_argument("--count", type=int, default=10000,
                        help="number of instances (default 10000)")
    parser.add_argument("--seed", type=int, default=12345,
                        help="master seed, the one of run_car.py")
    parser.add_argument("--outdir", type=Path, default=HERE / "results",
                        help="output directory; car_results.csv is read from it")
    parser.add_argument("--time-limit", type=float, default=30.0,
                        help="integer-program time limit (s) when an optimum "
                             "is not in car_results.csv")
    args = parser.parse_args()

    print(f"Running car_baselines: count={args.count} seed={args.seed} "
          f"outdir={args.outdir}", flush=True)
    start = time.perf_counter()
    run(args.count, args.seed, args.outdir, args.time_limit)
    print(f"\nTotal wall time: {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
