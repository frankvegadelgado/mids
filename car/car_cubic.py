"""car/ cubic study: Siriaisa 0.0.8 on cubic and subcubic graphs.

MIDS is APX-hard on cubic graphs (Bazgan, Brankovic, Casel and Fernau, DAM 2020,
via Manlove's reduction), NP-hard on cubic planar graphs (Manlove, DAM 1999) and
NP-complete on cubic planar bipartite graphs (Loverov and Orlovich, 2020); for
maximum degree at most 3 it is NP-hard to approximate within 1 + 1/680
(Chlebik and Chlebikova, 2008). The paper proves, for k = max(1, floor(sqrt(n))):

* Proposition "Splitting excess on cubic and bounded-degree graphs":
  s(G) = n for cubic graphs with n in {4, 6, 8}, s(G) = 0 for cubic graphs with
  n >= 10; s(G) <= 8 when Delta <= 3, and s(G) = 0 when Delta <= 3 and n >= 9;
* Theorem "Regular graphs": on a d-regular graph every independent dominating
  set, in particular the output, has at most n/2 vertices, so the ratio is at most
  (d + 1)/2, that is 2 on cubic graphs (i(G) >= n/4).

This script checks both statements on every instance (an AssertionError stops it)
and measures:

* n, m, Delta, delta, bipartite, planar, k, |B|, s(G), whether the copy graph of
  strategy c13 equals G (no vertex split);
* the sizes of all candidates c1..c18 of final_candidates, the strategies that
  attain the output c18, c18 / n (at most 1/2 on regular graphs);
* the reference: the certified lower bound LB = sum over components of
  ceil(n_C / (Delta_C + 1)) (= ceil(n/4) on connected cubic graphs) and, for
  n <= --ip-max-n, the MIDS integer program (scipy.optimize.milp, HiGHS, time limit
  --ip-time) giving i(G) or an interval [L, U].

Families (two instances per size unless the family is a single graph):

* ``small_cubic_n4/6/8``  every cubic graph on 4, 6 and 8 vertices (isomorphism
                          classes collected from random 3-regular graphs);
* ``named_cubic``         Petersen, Heawood, Moebius--Kantor, Pappus, Desargues,
                          dodecahedron, truncated tetrahedron, truncated cube,
                          Frucht, Tutte, McGee, Foster graphs;
* ``random_cubic``        random 3-regular graphs (APX-hard class);
* ``cubic_bipartite``     random 3-regular bipartite graphs;
* ``cubic_double_cover``  G x K2 of a random 3-regular graph (bipartite, cubic);
* ``cubic_planar``        duals of random Delaunay triangulations with a triangular
                          outer face (3-connected cubic planar graphs);
* ``cubic_planar_bipartite`` random cubic planar bipartite graphs, grown from the
                          cube by inserting a pair of rungs across an even face
                          (the class of Loverov--Orlovich);
* ``prism``, ``moebius_ladder``  C_t x K2 (bipartite for even t) and C_{2t} plus
                          its long diagonals;
* ``gen_petersen_j``      generalized Petersen graphs GP(t, j), j = 2, 3;
* ``subcubic``            random cubic graphs with 20% of the edges deleted;
* ``small_subcubic``      random graphs of maximum degree at most 3 on 3..8 vertices.

Run from the repository root (the repository's own siriaisa/ is used):

    python car/car_cubic.py               # n = 100, 400, 1600
    python car/car_cubic.py --quick       # n = 100, 400
    python car/car_cubic.py --ip-max-n 400 --ip-time 30

Every instance runs in a worker process with a time limit (--timeout, default
900 s). Outputs (in car/results/): car_cubic.csv, car_cubic_summary.csv and
car_cubic.json.
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
from scipy.spatial import Delaunay  # noqa: E402

import siriaisa  # noqa: E402
from siriaisa.algorithm import final_candidates, verify_independent_dominating_set  # noqa: E402
from siriaisa.degree_split import copy_graph  # noqa: E402

import car_bipartite as cb  # noqa: E402  (double covers, MIDS integer program)
import car_split as cs  # noqa: E402  (splitting excess, lower bound, per-instance worker)

SEED = 20261102
OUT_DIR = _CAR / "results"
DEFAULT_TIMEOUT = 900.0
STRATEGIES = [f"c{i}" for i in range(1, 19)]


# ----------------------------------------------------------------------------
# Generators
# ----------------------------------------------------------------------------
def small_cubic(n: int, rng: random.Random, tries: int = 400) -> list[nx.Graph]:
    """Representatives of the isomorphism classes of cubic graphs on n vertices met in
    ``tries`` random 3-regular graphs (n = 4: 1 class; 6: 2; 8: 6, five connected and 2K4)."""
    reps: list[nx.Graph] = []
    for _ in range(tries):
        G = nx.random_regular_graph(3, n, seed=rng.randrange(2**31))
        if not any(nx.is_isomorphic(G, H) for H in reps):
            reps.append(G)
    reps.sort(key=lambda H: (nx.number_connected_components(H), sorted(d for _, d in nx.triangles(H).items())))
    return reps


def small_subcubic(n: int, rng: random.Random) -> nx.Graph:
    """Random graph on n vertices with maximum degree at most 3 (random edges, degree-capped)."""
    G = nx.empty_graph(n)
    pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    rng.shuffle(pairs)
    for u, v in pairs:
        if G.degree(u) < 3 and G.degree(v) < 3 and rng.random() < 0.7:
            G.add_edge(u, v)
    return G


def named_cubic() -> list[tuple[str, nx.Graph]]:
    return [
        ("petersen", nx.petersen_graph()),
        ("heawood", nx.heawood_graph()),
        ("moebius_kantor", nx.moebius_kantor_graph()),
        ("pappus", nx.pappus_graph()),
        ("desargues", nx.desargues_graph()),
        ("dodecahedral", nx.dodecahedral_graph()),
        ("truncated_tetrahedron", nx.truncated_tetrahedron_graph()),
        ("truncated_cube", nx.truncated_cube_graph()),
        ("frucht", nx.frucht_graph()),
        ("tutte", nx.tutte_graph()),
        ("mcgee", nx.LCF_graph(24, [12, 7, -7], 8)),
        ("foster", nx.LCF_graph(90, [17, -9, 37, -37, 9, -17], 15)),
    ]


def cubic_planar(n: int, rng: random.Random) -> nx.Graph:
    """Dual of a Delaunay triangulation of n/2 - 1 random points inside a large triangle:
    the triangulation is maximal planar with n/2 + 2 vertices, so its dual is a 3-connected
    cubic planar graph with n faces-turned-vertices (n even)."""
    n -= n % 2
    pts = [(rng.random(), rng.random()) for _ in range(n // 2 - 1)]
    pts += [(-10.0, -10.0), (11.0, -10.0), (0.5, 12.0)]
    tri = Delaunay(np.array(pts))
    T = len(tri.simplices)
    G = nx.empty_graph(T + 1)  # node T is the outer face
    for t in range(T):
        for u in tri.neighbors[t]:
            G.add_edge(t, int(u) if u >= 0 else T)
    return G


def _faces(emb: nx.PlanarEmbedding) -> list[list[int]]:
    seen, faces = set(), []
    for u, v in emb.edges():
        if (u, v) not in seen:
            faces.append(emb.traverse_face(u, v, mark_half_edges=seen))
    return faces


def cubic_planar_bipartite(n: int, rng: random.Random) -> nx.Graph:
    """Random cubic planar bipartite graph on n vertices (n divisible by 4, n >= 8), grown
    from the cube Q3. A step picks a face a..b..c..d.. and two of its edges ab, cd at an even
    distance along the face, subdivides them as a-x1-x2-b and c-y1-y2-d and adds the chords
    x2-y1 and x1-y2 inside the face: planarity, bipartiteness and degree 3 are preserved, and
    the graph grows by 4 vertices."""
    n = max(8, n - n % 4)
    G = nx.convert_node_labels_to_integers(nx.hypercube_graph(3))
    nxt = G.number_of_nodes()
    while G.number_of_nodes() < n:
        ok, emb = nx.check_planarity(G)
        faces = _faces(emb)
        F = faces[rng.randrange(len(faces))]
        L = len(F)
        i = rng.randrange(L)
        j = (i + 2 * rng.randrange(1, L // 2)) % L if L >= 4 else i
        a, b, c, d = F[i], F[(i + 1) % L], F[j], F[(j + 1) % L]
        x1, x2, y1, y2 = nxt, nxt + 1, nxt + 2, nxt + 3
        nxt += 4
        G.remove_edge(a, b)
        G.remove_edge(c, d)
        G.add_edges_from([(a, x1), (x1, x2), (x2, b), (c, y1), (y1, y2), (y2, d), (x2, y1), (x1, y2)])
    return G


def cubic_bipartite(n: int, rng: random.Random) -> nx.Graph:
    """Random 3-regular bipartite graph on n vertices (n even, n >= 6): the union of three
    edge-disjoint random perfect matchings between the two sides (a matching that repeats
    an edge is redrawn)."""
    half = max(3, n // 2)
    G = nx.empty_graph(2 * half)
    for _ in range(3):
        while True:
            perm = list(range(half))
            rng.shuffle(perm)
            if not any(G.has_edge(i, half + perm[i]) for i in range(half)):
                G.add_edges_from((i, half + perm[i]) for i in range(half))
                break
    return G


def prism(t: int) -> nx.Graph:
    return nx.circular_ladder_graph(t)


def moebius_ladder(t: int) -> nx.Graph:
    return nx.circulant_graph(2 * t, [1, t])


def generalized_petersen(t: int, j: int) -> nx.Graph:
    G = nx.cycle_graph(t)
    for v in range(t):
        G.add_edge(v, t + v)
        G.add_edge(t + v, t + (v + j) % t)
    return G


def subcubic(n: int, rng: random.Random) -> nx.Graph:
    G = nx.random_regular_graph(3, n - n % 2, seed=rng.randrange(2**31))
    G.remove_edges_from([e for e in list(G.edges()) if rng.random() < 0.2])
    return G


def instances(quick: bool):
    """(name, family, source, build(rng))."""
    for n in (4, 6, 8):
        count = {4: 1, 6: 2, 8: 6}[n]
        for idx in range(count):
            yield (f"small_cubic_n{n}_{idx}", f"small_cubic_n{n}", "cubic, n <= 8",
                   lambda rng, n=n, idx=idx: small_cubic(n, random.Random(f"{SEED}:small{n}"))[idx])
    for n in range(3, 9):
        yield (f"small_subcubic_n{n}", "small_subcubic", "Delta <= 3, n <= 8", lambda rng, n=n: small_subcubic(n, rng))
    for name, G in named_cubic():
        yield (f"named_{name}", "named_cubic", "cubic", lambda rng, G=G: G.copy())
    sizes = (100, 400) if quick else (100, 400, 1600)
    for n in sizes:
        for r in range(2):
            t = f"_n{n}_r{r}"
            yield ("random_cubic" + t, "random_cubic", "APX-hard (cubic)",
                   lambda rng, n=n: nx.random_regular_graph(3, n, seed=rng.randrange(2**31)))
            yield ("cubic_bipartite" + t, "cubic_bipartite", "cubic bipartite",
                   lambda rng, n=n: cubic_bipartite(n, rng))
            yield ("cubic_double_cover" + t, "cubic_double_cover", "cubic bipartite",
                   lambda rng, n=n: cb.double_cover(n // 2, 3, rng))
            yield ("cubic_planar" + t, "cubic_planar", "NP-hard (cubic planar)", lambda rng, n=n: cubic_planar(n, rng))
            yield ("cubic_planar_bipartite" + t, "cubic_planar_bipartite", "NP-complete (cubic planar bipartite)",
                   lambda rng, n=n: cubic_planar_bipartite(n, rng))
            yield ("subcubic" + t, "subcubic", "NP-hard within 1+1/680 (Delta <= 3)", lambda rng, n=n: subcubic(n, rng))
        yield (f"prism_n{n}", "prism", "cubic planar (bipartite for even t)", lambda rng, n=n: prism(n // 2))
        yield (f"prism_odd_n{n + 2}", "prism", "cubic planar", lambda rng, n=n: prism(n // 2 + 1 - (n // 2) % 2))
        yield (f"moebius_ladder_n{n}", "moebius_ladder", "cubic", lambda rng, n=n: moebius_ladder(n // 2))
        for j in (2, 3):
            yield (f"gen_petersen_j{j}_n{n}", f"gen_petersen_j{j}", "cubic",
                   lambda rng, n=n, j=j: generalized_petersen(n // 2, j))


# ----------------------------------------------------------------------------
# Measurements
# ----------------------------------------------------------------------------
def run_instance(name: str, family: str, source: str, G: nx.Graph, args) -> dict:
    G = nx.convert_node_labels_to_integers(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    n, m = G.number_of_nodes(), G.number_of_edges()
    adj = [list(G.adj[v]) for v in range(n)]
    deg = [len(a) for a in adj]
    Delta, delta = max(deg, default=0), min(deg, default=0)
    regular = n > 0 and Delta == delta and Delta >= 1
    cubic = regular and Delta == 3
    k = max(1, math.isqrt(n))
    nB, s = cs.excess(deg, k)

    # Proposition "Splitting excess on cubic and bounded-degree graphs".
    if cubic and s != (n if n <= 8 else 0):
        raise AssertionError(f"{name}: cubic, n = {n}, s(G) = {s} -- contradicts the cubic proposition (i)")
    if Delta <= 3 and (s > 8 or (n >= 9 and s != 0)):
        raise AssertionError(f"{name}: Delta <= 3, n = {n}, s(G) = {s} -- contradicts the cubic proposition (ii)")
    hadj, owner, split = copy_graph(adj, k)
    H_equals_G = split == 0 and len(hadj) == n
    if s == 0 and not H_equals_G:
        raise AssertionError(f"{name}: s(G) = 0 but the copy graph differs from G")

    t0 = time.perf_counter()
    cands = final_candidates(G)
    seconds = time.perf_counter() - t0
    for key in ("c13", "c18"):
        if not verify_independent_dominating_set(G, cands[key]):
            raise AssertionError(f"{name}: {key} is not an independent dominating set")
    sizes = {key: len(cands[key]) for key in STRATEGIES if key in cands}
    c18 = sizes["c18"]
    # Theorem "Regular graphs".
    if regular and 2 * c18 > n:
        raise AssertionError(f"{name}: {Delta}-regular, |S| = {c18} > n/2 -- contradicts Theorem 'Regular graphs'")

    LB = cs.lower_bound(G)
    ref = {"status": "not run", "value": None, "lower": LB, "seconds": 0.0}
    if n <= args.ip_max_n:
        ref = cb.mids_milp(G, args.ip_time)
        ref["lower"] = max(ref["lower"], LB)
    lower = ref["lower"]
    upper = min(c18, ref["value"]) if ref["value"] is not None else c18
    if lower > upper:
        raise AssertionError(f"{name}: lower bound {lower} > upper bound {upper}")
    if cubic and c18 > 2 * lower:
        raise AssertionError(f"{name}: cubic, |S| = {c18} > 2 L = {2 * lower} -- contradicts Theorem 'Regular graphs'")
    exact = lower == upper
    base = [key for key in STRATEGIES[:13] if key in sizes]
    best_base = min(sizes[key] for key in base)
    row = {
        "name": name, "family": family, "source": source, "n": n, "m": m, "Delta": Delta, "delta": delta,
        "regular": regular, "cubic": cubic, "bipartite": nx.is_bipartite(G), "planar": nx.check_planarity(G)[0],
        "connected": n > 0 and nx.is_connected(G), "k": k, "B": nB, "s": s, "H_equals_G": H_equals_G,
        "sizes": sizes, "c13": sizes["c13"], "c18": c18, "c18_over_n": c18 / n if n else 0.0,
        "best_base": best_base, "best_base_strategies": [key for key in base if sizes[key] == best_base],
        "c18_strategies": [key for key in STRATEGIES[:17] if sizes.get(key) == c18],
        "LB": LB, "lower": lower, "upper": upper, "exact": exact, "ip_status": ref["status"],
        "ip_seconds": ref["seconds"], "c13_optimal": exact and sizes["c13"] == lower,
        "c18_optimal": exact and c18 == lower, "ratio_upper": c18 / lower if lower else 1.0,
        "siriaisa_seconds": seconds,
    }
    print(f"  [cubic] {family:24} {name:34} n={n:>5} m={m:>5} Delta={Delta} k={k:>2} s={s:>2}"
          f"{' bip' if row['bipartite'] else '    '}{' pl' if row['planar'] else '   '}"
          f" c13={sizes['c13']:>4} c18={c18:>4} c18/n={row['c18_over_n']:.3f} i in [{lower},{upper}]"
          f"{' opt' if row['c18_optimal'] else ''} {seconds:6.1f}s", flush=True)
    return row


def _job(name: str, args) -> dict:
    for nm, family, source, build in instances(args.quick):
        if nm == name:
            rng = random.Random(f"{SEED}:{name}")
            return run_instance(nm, family, source, build(rng), args)
    raise KeyError(name)


def summarise(rows: list[dict]) -> dict:
    out = {}
    for f in sorted({r["family"] for r in rows}) + ["ALL"]:
        g = rows if f == "ALL" else [r for r in rows if r["family"] == f]
        wins = {key: sum(key in r["c18_strategies"] for r in g) for key in STRATEGIES[:17]}
        out[f] = {
            "source": g[0]["source"] if f != "ALL" else "",
            "instances": len(g),
            "n_range": f"{min(r['n'] for r in g)}-{max(r['n'] for r in g)}",
            "cubic": sum(r["cubic"] for r in g),
            "s_zero": sum(r["s"] == 0 for r in g),
            "max_s": max(r["s"] for r in g),
            "exact": sum(r["exact"] for r in g),
            "c13_optimal": sum(r["c13_optimal"] for r in g),
            "c18_optimal": sum(r["c18_optimal"] for r in g),
            "max_ratio_upper": max(r["ratio_upper"] for r in g),
            "max_c18_over_n": max(r["c18_over_n"] for r in g),
            "mean_c18_over_n": sum(r["c18_over_n"] for r in g) / len(g),
            "wins": " ".join(f"{key}:{v}" for key, v in wins.items() if v),
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
                print(f"  [cubic] {family:24} {name:34} skipped (more than {args.timeout:g} s)", flush=True)
                skipped.append(name)
            else:
                rows.append(row)
    summary = summarise(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["family", "source", "name", "n", "m", "Delta", "delta", "regular", "cubic", "bipartite", "planar",
            "connected", "k", "B", "s", "H_equals_G", "c13", "c18", "c18_over_n", "best_base", "LB", "lower",
            "upper", "exact", "ip_status", "c13_optimal", "c18_optimal", "ratio_upper", "siriaisa_seconds"]
    with (OUT_DIR / "car_cubic.csv").open("w", encoding="utf-8") as fh:
        fh.write(",".join(cols + STRATEGIES + ["c18_strategies"]) + "\n")
        for r in rows:
            fh.write(",".join(f"\"{r[c]}\"" if c == "source" else
                              (f"{r[c]:.6f}" if isinstance(r[c], float) else str(r[c])) for c in cols)
                     + "," + ",".join(str(r["sizes"].get(key, "")) for key in STRATEGIES)
                     + ",\"" + " ".join(r["c18_strategies"]) + "\"\n")
    scols = ["source", "instances", "n_range", "cubic", "s_zero", "max_s", "exact", "c13_optimal", "c18_optimal",
             "max_ratio_upper", "max_c18_over_n", "mean_c18_over_n", "wins", "siriaisa_seconds"]
    with (OUT_DIR / "car_cubic_summary.csv").open("w", encoding="utf-8") as fh:
        fh.write("family," + ",".join(scols) + "\n")
        for f, s in summary.items():
            fh.write(f + "," + ",".join(f"\"{s[c]}\"" if c in ("source", "wins") else
                                        (f"{s[c]:.6f}" if isinstance(s[c], float) else str(s[c])) for c in scols) + "\n")
    (OUT_DIR / "car_cubic.json").write_text(json.dumps(
        {"siriaisa_version": siriaisa.__version__,
         "environment": {"python": platform.python_version(), "platform": platform.platform(),
                         "networkx": nx.__version__, "scipy": scipy.__version__, "numpy": np.__version__},
         "seed": SEED, "quick": args.quick, "ip_max_n": args.ip_max_n, "ip_time": args.ip_time,
         "timeout": args.timeout, "skipped_instances": skipped, "summary": summary, "rows": rows,
         "elapsed_seconds": time.time() - started}, indent=2, default=str), encoding="utf-8")

    print("\nCubic and subcubic graphs (k = floor(sqrt(n)))")
    print(f"{'family':24} {'inst':>4} {'n':>9} {'cubic':>5} {'s=0':>4} {'max s':>5} {'exact':>5} {'c13 opt':>7}"
          f" {'c18 opt':>7} {'max ratio':>9} {'max |S|/n':>9}")
    for f, s in summary.items():
        print(f"{f:24} {s['instances']:>4} {s['n_range']:>9} {s['cubic']:>5} {s['s_zero']:>4} {s['max_s']:>5}"
              f" {s['exact']:>5} {s['c13_optimal']:>7} {s['c18_optimal']:>7} {s['max_ratio_upper']:9.4f}"
              f" {s['max_c18_over_n']:9.4f}")
    print("max ratio: c18 / certified lower bound (an upper bound on the ratio; at most 2 on cubic graphs);"
          " max |S|/n: at most 1/2 on regular graphs")
    print(f"elapsed {time.time() - started:.1f} s; skipped {len(skipped)}")


if __name__ == "__main__":
    main()
