"""car/ circle-graph study: the splitting excess s(G) and Siriaisa 0.0.8 on circle graphs.

MIDS is NP-complete on circle graphs and, unless P = NP, not approximable within
n^eps for any eps < 1 (Damian-Iordache and Pemmaraju, ISAAC 1999). The paper proves
(Proposition "Splitting excess on circle graphs"):

* (i) for a chord model with endpoints numbered 1..2n around the circle and the chord
  of v written (a_v, b_v), a_v < b_v,
      deg(v) = b_v - a_v - 1 - 2 |{u : a_v < a_u < b_u < b_v}|,
  so the degrees, B and s(G) are computed in O(n log n) time from the model, without
  listing the edges (Fenwick tree);
* (ii) K_n and K_{a,b} are circle graphs, so circle graphs attain the largest excess
  s(K_n) of all graphs, and the output is optimal on them.

This script builds every instance as a chord model, computes the degrees and s(G) from
the model by (i) and checks them against the graph built from the crossings (an
AssertionError stops the script), and then runs the measurements of car_split.py
(car_split.run_instance): copy graph, the decoded sets S_0 of the three inner solvers
and their independence, c13, the output c18, the certified lower bound and, for
n <= --ip-max-n, the MIDS integer program (HiGHS); the inequality
|S_0| <= k (U + s(G)) of Theorem "Degree splitting" is asserted.

Families (chord models; endpoints are real numbers in [0, 1), ranked into 1..2n):

* ``random_uniform``     both endpoints uniform (dense: two random chords cross with
                         probability 1/3);
* ``random_local_d``     left endpoint uniform, length uniform in [0, 2d/n]
                         (sparse, expected degree of order d), d = 3, 10;
* ``random_local_sqrt``  length uniform in [0, 2/sqrt(n)] (degrees of order sqrt(n),
                         the splitting threshold);
* ``permutation``        left endpoints in [0, 1/2), right endpoints in [1/2, 1) in a
                         random order (permutation graphs, a subclass of circle graphs);
* ``complete``           the n diameters of a regular 2n-gon (K_n, the maximum s);
* ``complete_bipartite_bal`` / ``_1to3``  nested horizontal and vertical chord families
                         (K_{n/2,n/2}, K_{n/4,3n/4});
* ``star``               a long chord crossed by n - 1 nested short chords around one of
                         its endpoints (K_{1,n-1});
* ``hubs_h``             h = 2, 4 long pairwise disjoint chords, each crossed by nested
                         short chords, plus random local chords;
* ``path`` / ``cycle``   consecutive overlapping chords (Delta = 2).

Run from the repository root (the repository's own siriaisa/ is used):

    python car/car_circle.py              # n = 100, 400, 1600
    python car/car_circle.py --quick      # n = 100, 400
    python car/car_circle.py --ip-max-n 400 --ip-time 30

Every instance runs in a worker process with a time limit (--timeout, default 900 s).
Outputs (in car/results/): car_circle.csv, car_circle_summary.csv and car_circle.json.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import sys
import time
from pathlib import Path

_CAR = Path(__file__).resolve().parent
_REPO_ROOT = _CAR.parent
for p in (str(_REPO_ROOT), str(_CAR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import scipy  # noqa: E402

import siriaisa  # noqa: E402

import car_split as cs  # noqa: E402  (per-instance measurements, splitting excess, worker pool)

SEED = 20261103
OUT_DIR = _CAR / "results"
DEFAULT_TIMEOUT = 900.0


# ----------------------------------------------------------------------------
# Chord models
# ----------------------------------------------------------------------------
def rank_model(chords: list[tuple[float, float]]) -> list[tuple[int, int]]:
    """Rank the 2n real endpoints into 1..2n; every chord becomes (a, b) with a < b.
    Raises if two endpoints coincide (probability 0 for the random families)."""
    ends = sorted((x, v) for v, c in enumerate(chords) for x in c)
    if any(ends[i][0] == ends[i + 1][0] for i in range(len(ends) - 1)):
        raise ValueError("chord model with a repeated endpoint")
    pos: list[list[int]] = [[] for _ in chords]
    for r, (_, v) in enumerate(ends, start=1):
        pos[v].append(r)
    return [(p[0], p[1]) for p in pos]


def model_graph(model: list[tuple[int, int]]) -> nx.Graph:
    """The circle graph of the model: u, v adjacent iff exactly one endpoint of u lies
    strictly between the endpoints of v. Built by a sweep over the endpoints that keeps
    the open chords: when chord v closes, it crosses every chord opened after a_v that is
    still open."""
    n = len(model)
    G = nx.empty_graph(n)
    at = [None] * (2 * n + 1)
    for v, (a, b) in enumerate(model):
        at[a] = (v, True)
        at[b] = (v, False)
    opened: list[int] = []          # chords in order of their left endpoint
    index = [-1] * n
    alive = [False] * n
    for p in range(1, 2 * n + 1):
        v, is_open = at[p]
        if is_open:
            index[v] = len(opened)
            opened.append(v)
            alive[v] = True
        else:
            alive[v] = False
            for u in opened[index[v] + 1:]:
                if alive[u]:
                    G.add_edge(u, v)
    return G


def model_degrees(model: list[tuple[int, int]]) -> list[int]:
    """Degrees by Proposition 'Splitting excess on circle graphs'(i), in O(n log n):
    deg(v) = b_v - a_v - 1 - 2 |{u : a_v < a_u < b_u < b_v}|, the count by a Fenwick tree
    over the right endpoints, scanning the chords by decreasing left endpoint."""
    n = len(model)
    size = 2 * n
    tree = [0] * (size + 1)

    def add(i: int) -> None:
        while i <= size:
            tree[i] += 1
            i += i & -i

    def prefix(i: int) -> int:
        t = 0
        while i > 0:
            t += tree[i]
            i -= i & -i
        return t

    deg = [0] * n
    for v in sorted(range(n), key=lambda v: -model[v][0]):
        a, b = model[v]
        inside = prefix(b - 1)      # chords with a_u > a_v (already added) and b_u < b_v
        deg[v] = b - a - 1 - 2 * inside
        add(b)
    return deg


# ----------------------------------------------------------------------------
# Families
# ----------------------------------------------------------------------------
def random_uniform(n: int, rng: random.Random):
    return [(rng.random(), rng.random()) for _ in range(n)]


def random_local(n: int, max_len: float, rng: random.Random):
    out = []
    for _ in range(n):
        x = rng.random()
        y = (x + rng.uniform(1e-9, max_len)) % 1.0
        out.append((x, y))
    return out


def permutation(n: int, rng: random.Random):
    perm = list(range(n))
    rng.shuffle(perm)
    return [((i + 0.5) / (2 * n), 0.5 + (perm[i] + 0.5) / (2 * n)) for i in range(n)]


def complete(n: int):
    return [(i / (2 * n), (i + n) / (2 * n)) for i in range(n)]


def complete_bipartite(a: int, b: int):
    """Positions of the proof of Proposition 'Splitting excess on circle graphs'(ii):
    A_i = (i, 2a + b + 1 - i), B_j = (a + b + 1 - j, 2a + b + j) on 1..2(a + b)."""
    A = [(i, 2 * a + b + 1 - i) for i in range(1, a + 1)]
    B = [(a + b + 1 - j, 2 * a + b + j) for j in range(1, b + 1)]
    return [(float(x), float(y)) for x, y in A + B]


def star(n: int):
    eps = 0.2 / n
    return [(0.25, 0.75)] + [(0.25 - i * eps, 0.25 + i * eps) for i in range(1, n)]


def hubs(n: int, h: int, rng: random.Random):
    """h pairwise disjoint long chords (consecutive arcs of the circle), each crossed by
    about n / (2h) nested short chords around its left endpoint; the rest are local chords."""
    chords = []
    per = n // (2 * h)
    width = 1.0 / h
    for j in range(h):
        x, y = j * width + 0.1 * width, j * width + 0.9 * width
        chords.append((x, y))
        eps = 0.05 * width / (per + 1)
        chords += [(x - i * eps, x + i * eps) for i in range(1, per + 1)]
    rest = n - len(chords)
    chords += random_local(rest, 3.0 / n, rng)
    return chords


def path(n: int, cyclic: bool):
    step = 1.0 / (n + 2)
    chords = [(i * step, (i + 1.5) * step) for i in range(n)]
    if cyclic:  # the last chord also crosses the first one: wrap it around the origin
        chords[-1] = ((n - 0.75) * step, 0.75 * step)
    return chords


def instances(quick: bool):
    """(name, family, source, build(rng) -> real chord model)."""
    sizes = (100, 400) if quick else (100, 400, 1600)
    for n in sizes:
        for r in range(2):
            t = f"_n{n}_r{r}"
            yield ("random_uniform" + t, "random_uniform", "circle (dense random)",
                   lambda rng, n=n: random_uniform(n, rng))
            for d in (3, 10):
                yield (f"random_local_d{d}" + t, f"random_local_d{d}", "circle (sparse random)",
                       lambda rng, n=n, d=d: random_local(n, 2.0 * d / n, rng))
            yield ("random_local_sqrt" + t, "random_local_sqrt", "circle (degrees ~ sqrt n)",
                   lambda rng, n=n: random_local(n, 2.0 / math.sqrt(n), rng))
            yield ("permutation" + t, "permutation", "permutation graph",
                   lambda rng, n=n: permutation(n, rng))
            for h in (2, 4):
                yield (f"hubs_{h}" + t, f"hubs_{h}", "circle (few hubs)", lambda rng, n=n, h=h: hubs(n, h, rng))
        yield (f"complete_n{n}", "complete", "circle (K_n)", lambda rng, n=n: complete(n))
        yield (f"complete_bipartite_bal_n{n}", "complete_bipartite_bal", "circle (K_{n/2,n/2})",
               lambda rng, n=n: complete_bipartite(n // 2, n - n // 2))
        yield (f"complete_bipartite_1to3_n{n}", "complete_bipartite_1to3", "circle (K_{n/4,3n/4})",
               lambda rng, n=n: complete_bipartite(n // 4, n - n // 4))
        yield (f"star_n{n}", "star", "circle (K_{1,n-1})", lambda rng, n=n: star(n))
        yield (f"path_n{n}", "path", "circle (path)", lambda rng, n=n: path(n, False))
        yield (f"cycle_n{n}", "cycle", "circle (cycle)", lambda rng, n=n: path(n, True))


# ----------------------------------------------------------------------------
# Measurements
# ----------------------------------------------------------------------------
def _job(name: str, args) -> dict:
    for nm, family, source, build in instances(args.quick):
        if nm != name:
            continue
        rng = random.Random(f"{SEED}:{name}")
        model = rank_model(build(rng))
        n = len(model)
        t0 = time.perf_counter()
        deg = model_degrees(model)
        k = max(1, math.isqrt(n))
        nB, s = cs.excess(deg, k)
        model_seconds = time.perf_counter() - t0
        t1 = time.perf_counter()
        G = model_graph(model)
        graph_seconds = time.perf_counter() - t1
        if deg != [G.degree(v) for v in range(n)]:
            raise AssertionError(f"{name}: degrees from the chord model differ from the circle graph "
                                 "-- contradicts Proposition 'Splitting excess on circle graphs'(i)")
        if family == "complete" and G.number_of_edges() != n * (n - 1) // 2:
            raise AssertionError(f"{name}: the diameters do not form K_n")
        if family.startswith("complete_bipartite"):
            a = n // 2 if family.endswith("bal") else n // 4
            if not nx.is_isomorphic(G, nx.complete_bipartite_graph(a, n - a)):
                raise AssertionError(f"{name}: the chord families do not form K_(a,b)")
        if family == "star" and not nx.is_isomorphic(G, nx.star_graph(n - 1)):
            raise AssertionError(f"{name}: the model does not form K_(1,n-1)")
        if family in ("path", "cycle") and not nx.is_isomorphic(G, nx.path_graph(n) if family == "path" else nx.cycle_graph(n)):
            raise AssertionError(f"{name}: the model does not form the {family}")
        row = cs.run_instance(name, family, source, G, args)
        if row["s"] != s or row["B"] != nB:
            raise AssertionError(f"{name}: s(G) from the model ({s}) differs from s(G) of the graph ({row['s']})")
        row.update({"s_model": s, "model_seconds": model_seconds, "graph_seconds": graph_seconds})
        return row
    raise KeyError(name)


def summarise(rows: list[dict]) -> dict:
    out = {}
    for f in sorted({r["family"] for r in rows}) + ["ALL"]:
        g = rows if f == "ALL" else [r for r in rows if r["family"] == f]
        out[f] = {
            "source": g[0]["source"] if f != "ALL" else "",
            "instances": len(g),
            "n_range": f"{min(r['n'] for r in g)}-{max(r['n'] for r in g)}",
            "max_m": max(r["m"] for r in g),
            "max_Delta": max(r["Delta"] for r in g),
            "s_zero": sum(r["s"] == 0 for r in g),
            "max_s": max(r["s"] for r in g),
            "max_s_over_sKn": max(r["s"] / r["s_Kn"] if r["s_Kn"] else 0.0 for r in g),
            "max_s_over_n32": max(r["s_over_n32"] for r in g),
            "s_le_i": sum(r["s_le_i"] for r in g),
            "S0_independent": sum(r["any_S0_independent"] for r in g),
            "exact": sum(r["exact"] for r in g),
            "c13_optimal": sum(r["c13_optimal"] for r in g),
            "c18_optimal": sum(r["c18_optimal"] for r in g),
            "max_c18_ratio": max(r["c18"] / r["lower"] for r in g if r["lower"]),
            "model_seconds": sum(r["model_seconds"] for r in g),
            "graph_seconds": sum(r["graph_seconds"] for r in g),
            "siriaisa_seconds": sum(r["siriaisa_seconds"] for r in g),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="n = 100 and 400 only")
    ap.add_argument("--ip-max-n", type=int, default=400, help="largest n for the MIDS integer program (default 400)")
    ap.add_argument("--ip-time", type=float, default=30.0, help="integer-program time limit in seconds (default 30)")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                    help=f"per-instance time limit in seconds (default {DEFAULT_TIMEOUT:g}; 0 = none)")
    args = ap.parse_args()

    print(f"siriaisa {siriaisa.__version__} from {Path(siriaisa.__file__).resolve().parent}", flush=True)
    started = time.time()
    rows, skipped = [], []
    with cs.InstanceRunner(args.timeout) as runner:
        for name, family, _, _ in instances(args.quick):
            row = runner.run(_job, name, args)
            if row is None:
                print(f"  [circle] {family:24} {name:34} skipped (more than {args.timeout:g} s)", flush=True)
                skipped.append(name)
            else:
                rows.append(row)
    summary = summarise(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["family", "source", "name", "n", "m", "Delta", "k", "B", "s", "s_model", "bound_ii", "s_Kn", "s_over_n",
            "s_over_n32", "H_nodes", "H_max_degree", "best_S0", "any_S0_independent", "c13", "c18", "lower", "upper",
            "exact", "ip_status", "c13_optimal", "c18_optimal", "s_le_i", "theorem_bound", "model_seconds",
            "graph_seconds", "siriaisa_seconds"]
    with (OUT_DIR / "car_circle.csv").open("w", encoding="utf-8") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(f"\"{r[c]}\"" if c == "source" else
                              (f"{r[c]:.6f}" if isinstance(r[c], float) else str(r[c])) for c in cols) + "\n")
    scols = list(next(iter(summary.values())).keys())
    with (OUT_DIR / "car_circle_summary.csv").open("w", encoding="utf-8") as fh:
        fh.write("family," + ",".join(scols) + "\n")
        for f, s in summary.items():
            fh.write(f + "," + ",".join(f"\"{s[c]}\"" if c == "source" else
                                        (f"{s[c]:.6f}" if isinstance(s[c], float) else str(s[c])) for c in scols) + "\n")
    (OUT_DIR / "car_circle.json").write_text(json.dumps(
        {"siriaisa_version": siriaisa.__version__,
         "environment": {"python": platform.python_version(), "platform": platform.platform(),
                         "networkx": nx.__version__, "scipy": scipy.__version__, "numpy": np.__version__},
         "seed": SEED, "quick": args.quick, "ip_max_n": args.ip_max_n, "ip_time": args.ip_time,
         "timeout": args.timeout, "skipped_instances": skipped, "summary": summary, "rows": rows,
         "elapsed_seconds": time.time() - started}, indent=2, default=str), encoding="utf-8")

    print("\nCircle graphs: splitting excess s(G) (k = floor(sqrt(n)))")
    print(f"{'family':24} {'inst':>4} {'n':>9} {'Delta':>5} {'s=0':>4} {'max s':>8} {'s/s(Kn)':>7} {'s<=i':>5}"
          f" {'S0 ind':>6} {'exact':>5} {'c13 opt':>7} {'c18 opt':>7} {'max ratio':>9} {'model s':>8} {'graph s':>8}")
    for f, s in summary.items():
        print(f"{f:24} {s['instances']:>4} {s['n_range']:>9} {s['max_Delta']:>5} {s['s_zero']:>4} {s['max_s']:>8}"
              f" {s['max_s_over_sKn']:7.3f} {s['s_le_i']:>5} {s['S0_independent']:>6} {s['exact']:>5}"
              f" {s['c13_optimal']:>7} {s['c18_optimal']:>7} {s['max_c18_ratio']:9.4f}"
              f" {s['model_seconds']:8.3f} {s['graph_seconds']:8.3f}")
    print("s/s(Kn): s(G) relative to the maximum s(K_n); model s / graph s: seconds to get s(G) from the chord "
          "model (O(n log n)) and to build the circle graph; max ratio: c18 / lower bound")
    print(f"elapsed {time.time() - started:.1f} s; skipped {len(skipped)}")


if __name__ == "__main__":
    main()
