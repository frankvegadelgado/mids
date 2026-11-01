"""car/ bounded-degree study: graphs on which s(G) = 0 and MIDS is hard to approximate.

Chlebik and Chlebikova (Information and Computation 2008) proved that there are
absolute constants delta > 0 and B_0 such that, for every B >= B_0, MIDS is NP-hard to
approximate within delta * B on graphs of maximum degree at most B, bipartite or not.
The paper proves (Proposition "Splitting excess on graph families"(a)) that every
graph with Delta <= B and n >= B^2 has s(G) = 0: strategy c13 splits nothing and runs
its inner solvers on G itself. These graphs are the family on which the splitting
excess is feasible (zero) while the hardness factor grows linearly with the degree
bound. This script runs Siriaisa 0.0.8 on them for B = 4, 8, 16, 32, always with
n >= B^2, and asserts on every instance (an AssertionError stops the script):

* s(G) = 0 (Proposition "Splitting excess on graph families"(a));
* |S| <= n/2 on regular graphs (Theorem "Regular graphs");
* the certified lower bound never exceeds the output.

Families (labels shuffled; one generator per instance, derived from SEED and the name):

* ``regular_B``          random B-regular graphs;
* ``regular_bipartite_B`` double covers H x K2 of random B-regular graphs H
                         (B-regular bipartite);
* ``planted_code_B``     B-regular graphs with a planted perfect code D: |D| = n/(B+1)
                         vertices, each with B private neighbours, the other vertices
                         joined by a random (B-1)-regular graph. D is an independent
                         dominating set of size n/(B+1) = ceil(n/(Delta+1)), so
                         i(G) = n/(B+1) is known exactly (Lemma "Certified lower bound");
* ``scids_B``            SC-IDS graphs of random set systems with sets of size B-1 and
                         element frequency at most B (degree B, bipartite);
                         i = sc + |S| (Chlebik--Chlebikova), sc bounded by an integer program;
* ``vcids_B``            VC-IDS graphs h(H) of random B-regular graphs H (degree B);
                         i = tau(H) + 2|E(H)| (Proposition "VC-IDS"), tau(H) bounded by
                         its LP and an integer program;
* ``kr_bounded_s``       Khot--Regev YES graphs on 3-regular constraint graphs with
                         s = 4, 8 samples (degree at most 3s; N >= 9s blocks, so n >= Delta^2).

Measurements (car_split.run_instance): n, m, Delta, k, s(G), the copy graph, the
decoded sets S_0 of the three inner solvers, c13, the output c18, the certified lower
bound LB and, for n <= --ip-max-n, the MIDS integer program (HiGHS, --ip-time);
then the reference of the family (planted value, SC-IDS or VC-IDS identity) narrows
the interval [L, U] for i(G), and c18 / L is an upper bound on the ratio.

Run from the repository root (the repository's own siriaisa/ is used):

    python car/car_bounded.py             # n = max(B^2, 256) and 4 times that
    python car/car_bounded.py --quick     # n = max(B^2, 256)
    python car/car_bounded.py --ip-max-n 1100 --ip-time 60

Every instance runs in a worker process with a time limit (--timeout, default 900 s).
Outputs (in car/results/): car_bounded.csv, car_bounded_summary.csv and car_bounded.json.
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

import car_bipartite as cb  # noqa: E402  (double covers, label shuffling)
import car_split as cs  # noqa: E402  (per-instance measurements, worker pool)
import car_ugc as cu  # noqa: E402  (SC-IDS and VC-IDS builders and bounds)

SEED = 20261104
OUT_DIR = _CAR / "results"
DEFAULT_TIMEOUT = 900.0
DEGREES = (4, 8, 16, 32)


# ----------------------------------------------------------------------------
# Generators: each returns (graph, reference) with reference a dict of the family's
# identity for i(G): {"kind": ..., "lower": int, "upper": int | None}.
# ----------------------------------------------------------------------------
def regular(n: int, B: int, rng: random.Random):
    n += (n * B) % 2
    return nx.random_regular_graph(B, n, seed=rng.randrange(2**31)), None


def regular_bipartite(n: int, B: int, rng: random.Random):
    return cb.double_cover(n // 2, B, rng), None


def planted_code(n: int, B: int, rng: random.Random):
    t = max(2, -(-n // (B + 1)))
    R = nx.random_regular_graph(B - 1, t * B, seed=rng.randrange(2**31))
    G = nx.relabel_nodes(R, {v: ("r", v) for v in R})
    others = [("r", v) for v in R]
    rng.shuffle(others)
    for j in range(t):
        for x in others[j * B:(j + 1) * B]:
            G.add_edge(("d", j), x)
    return G, {"kind": "planted perfect code", "lower": t, "upper": t, "D": t}


def scids(n: int, B: int, rng: random.Random, time_limit: float):
    sets = cu.random_set_system(max(4, -(-n // 3)), B, rng)
    G = cu.sc_ids(sets)
    lo, primal, _ = cu.sc_bounds(sets, time_limit)
    return G, {"kind": "SC-IDS i = sc + |S|", "lower": lo + len(sets),
               "upper": primal + len(sets) if primal is not None else None}


def vcids(n: int, B: int, rng: random.Random, time_limit: float):
    nh = max(B + 2, round(n / (1 + 3 * B)))
    nh += (nh * B) % 2
    H = nx.random_regular_graph(B, nh, seed=rng.randrange(2**31))
    G, _ = cu.vc_ids(H)
    lo, primal, _ = cu.vc_bounds(H, time_limit)
    mh = H.number_of_edges()
    return G, {"kind": "VC-IDS i = tau + 2 m_H", "lower": lo + 2 * mh,
               "upper": primal + 2 * mh if primal is not None else None}


def kr_bounded(N: int, s: int, rng: random.Random):
    return cs.kr_graph(N, 3, s, rng), None


def instances(quick: bool):
    """(name, family, source, B, build(rng, args) -> (graph, reference))."""
    src = "NP-hard within delta*B (Chlebik--Chlebikova), s(G) = 0"
    for B in DEGREES:
        base = max(B * B, 256)
        for n in ((base,) if quick else (base, 4 * base)):
            for r in range(2):
                t = f"_B{B}_n{n}_r{r}"
                yield (f"regular{t}", f"regular_B{B}", src, B, lambda rng, a, n=n, B=B: regular(n, B, rng))
                yield (f"regular_bipartite{t}", f"regular_bipartite_B{B}", src + ", bipartite", B,
                       lambda rng, a, n=n, B=B: regular_bipartite(n, B, rng))
                yield (f"planted_code{t}", f"planted_code_B{B}", src + ", planted optimum", B,
                       lambda rng, a, n=n, B=B: planted_code(n, B, rng))
                yield (f"scids{t}", f"scids_B{B}", "SC-IDS reduction, degree B, s(G) = 0", B,
                       lambda rng, a, n=n, B=B: scids(n, B, rng, a.ip_time))
                yield (f"vcids{t}", f"vcids_B{B}", "VC-IDS reduction, degree B, s(G) = 0", B,
                       lambda rng, a, n=n, B=B: vcids(n, B, rng, a.ip_time))
    for s in (4, 8):
        for N in ((9 * s + 4,) if quick else (9 * s + 4, 18 * s)):
            for r in range(2):
                yield (f"kr_bounded_s{s}_N{N}_r{r}", f"kr_bounded_s{s}",
                       "UG-hard vertex cover (Khot--Regev), degree <= 3s, s(G) = 0", 3 * s,
                       lambda rng, a, N=N, s=s: kr_bounded(N, s, rng))


# ----------------------------------------------------------------------------
# Measurements
# ----------------------------------------------------------------------------
def _job(name: str, args) -> dict:
    for nm, family, source, B, build in instances(args.quick):
        if nm != name:
            continue
        rng = random.Random(f"{SEED}:{name}")
        t0 = time.perf_counter()
        G, ref = build(rng, args)
        G = cb.shuffle_labels(G, rng)
        reference_seconds = time.perf_counter() - t0
        n = G.number_of_nodes()
        Delta = max((d for _, d in G.degree()), default=0)
        row = cs.run_instance(nm, family, source, G, args)
        if n >= Delta * Delta and row["s"] != 0:
            raise AssertionError(f"{nm}: n = {n} >= Delta^2 = {Delta * Delta} but s(G) = {row['s']} "
                                 "-- contradicts Proposition 'Splitting excess on graph families'(a)")
        degs = {d for _, d in G.degree()}
        regular_graph = len(degs) == 1 and Delta >= 1
        if regular_graph and 2 * row["c18"] > n:
            raise AssertionError(f"{nm}: regular, |S| = {row['c18']} > n/2 -- contradicts Theorem 'Regular graphs'")
        lower, upper = row["lower"], row["upper"]
        if ref is not None:
            lower = max(lower, ref["lower"])
            if ref["upper"] is not None:
                upper = min(upper, ref["upper"])
        if ref is not None and ref["kind"] == "planted perfect code":
            if cs.lower_bound(G) != ref["D"]:
                raise AssertionError(f"{nm}: planted code size {ref['D']} differs from ceil(n/(Delta+1))")
        if lower > upper:
            raise AssertionError(f"{nm}: lower bound {lower} > upper bound {upper}")
        exact = lower == upper
        row.update({
            "degree_bound": B, "regular": regular_graph, "reference": ref["kind"] if ref else "LB / MILP",
            "lower": lower, "upper": upper, "exact": exact,
            "c13_optimal": exact and row["c13"] == lower, "c18_optimal": exact and row["c18"] == lower,
            "ratio_upper": row["c18"] / lower if lower else 1.0, "reference_seconds": reference_seconds,
        })
        print(f"  [bounded] {family:22} -> i in [{lower},{upper}] c18={row['c18']} ratio<={row['ratio_upper']:.4f}"
              f"{' opt' if row['c18_optimal'] else ''}", flush=True)
        return row
    raise KeyError(name)


def summarise(rows: list[dict]) -> dict:
    out = {}
    for f in list(dict.fromkeys(r["family"] for r in rows)) + ["ALL"]:
        g = rows if f == "ALL" else [r for r in rows if r["family"] == f]
        out[f] = {
            "source": g[0]["source"] if f != "ALL" else "",
            "reference": g[0]["reference"] if f != "ALL" else "",
            "instances": len(g),
            "n_range": f"{min(r['n'] for r in g)}-{max(r['n'] for r in g)}",
            "max_Delta": max(r["Delta"] for r in g),
            "s_zero": sum(r["s"] == 0 for r in g),
            "exact": sum(r["exact"] for r in g),
            "c13_optimal": sum(r["c13_optimal"] for r in g),
            "c18_optimal": sum(r["c18_optimal"] for r in g),
            "max_ratio_upper": max(r["ratio_upper"] for r in g),
            "mean_ratio_upper": sum(r["ratio_upper"] for r in g) / len(g),
            "siriaisa_seconds": sum(r["siriaisa_seconds"] for r in g),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="n = max(B^2, 256) only")
    ap.add_argument("--ip-max-n", type=int, default=1100, help="largest n for the MIDS integer program (default 1100)")
    ap.add_argument("--ip-time", type=float, default=60.0, help="integer-program time limit in seconds (default 60)")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                    help=f"per-instance time limit in seconds (default {DEFAULT_TIMEOUT:g}; 0 = none)")
    args = ap.parse_args()

    print(f"siriaisa {siriaisa.__version__} from {Path(siriaisa.__file__).resolve().parent}", flush=True)
    started = time.time()
    rows, skipped = [], []
    with cs.InstanceRunner(args.timeout) as runner:
        for name, family, _, _, _ in instances(args.quick):
            row = runner.run(_job, name, args)
            if row is None:
                print(f"  [bounded] {family:22} {name:34} skipped (more than {args.timeout:g} s)", flush=True)
                skipped.append(name)
            else:
                rows.append(row)
    summary = summarise(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["family", "source", "reference", "name", "degree_bound", "n", "m", "Delta", "k", "B", "s", "regular", "H_nodes",
            "best_S0", "any_S0_independent", "c13", "c18", "lower", "upper", "exact", "ip_status", "c13_optimal",
            "c18_optimal", "ratio_upper", "siriaisa_seconds", "reference_seconds"]
    with (OUT_DIR / "car_bounded.csv").open("w", encoding="utf-8") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(f"\"{r[c]}\"" if c in ("source", "reference") else
                              (f"{r[c]:.6f}" if isinstance(r[c], float) else str(r[c])) for c in cols) + "\n")
    scols = list(next(iter(summary.values())).keys())
    with (OUT_DIR / "car_bounded_summary.csv").open("w", encoding="utf-8") as fh:
        fh.write("family," + ",".join(scols) + "\n")
        for f, s in summary.items():
            fh.write(f + "," + ",".join(f"\"{s[c]}\"" if c in ("source", "reference") else
                                        (f"{s[c]:.6f}" if isinstance(s[c], float) else str(s[c])) for c in scols) + "\n")
    (OUT_DIR / "car_bounded.json").write_text(json.dumps(
        {"siriaisa_version": siriaisa.__version__,
         "environment": {"python": platform.python_version(), "platform": platform.platform(),
                         "networkx": nx.__version__, "scipy": scipy.__version__, "numpy": np.__version__},
         "seed": SEED, "quick": args.quick, "ip_max_n": args.ip_max_n, "ip_time": args.ip_time,
         "timeout": args.timeout, "skipped_instances": skipped, "summary": summary, "rows": rows,
         "elapsed_seconds": time.time() - started}, indent=2, default=str), encoding="utf-8")

    print("\nBounded degree B, n >= B^2 (s(G) = 0; NP-hard within delta*B)")
    print(f"{'family':24} {'inst':>4} {'n':>11} {'Delta':>5} {'s=0':>4} {'exact':>5} {'c13 opt':>7} {'c18 opt':>7}"
          f" {'max ratio':>9} {'mean ratio':>10}")
    for f, s in summary.items():
        print(f"{f:24} {s['instances']:>4} {s['n_range']:>11} {s['max_Delta']:>5} {s['s_zero']:>4} {s['exact']:>5}"
              f" {s['c13_optimal']:>7} {s['c18_optimal']:>7} {s['max_ratio_upper']:9.4f} {s['mean_ratio_upper']:10.4f}")
    print("ratio: c18 / certified lower bound on i(G) (an upper bound on the true ratio; exact for planted_code)")
    print(f"elapsed {time.time() - started:.1f} s; skipped {len(skipped)}")


if __name__ == "__main__":
    main()
