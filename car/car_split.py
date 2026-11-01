"""car/ splitting study: the splitting excess s(G) of strategy c13 on hard and structured families.

Strategy c13 of Siriaisa 0.0.8 (siriaisa/degree_split.py) replaces every vertex of
degree larger than k = floor(sqrt(n)) by unconnected copies, each joined to at
most k of its neighbours, solves the copy graph H with three inner solvers, maps
the solution back (owner set S_0) and repairs it. The paper proves

    |S_0| <= k (i(G) + s(G)),    s(G) = sum over split v of (ceil(deg v / k) - 1),

and the same bound for the output when S_0 is independent. This script measures,
on every instance,

* n, m, Delta, k, |B| (vertices of degree > k), s(G), the bound
  min((2m - |B|)/k, |B|(n - 2)/k) and s(K_n) of Proposition "Splitting excess";
* the copy graph: |V(H)| (checked against n - |B| + sum ceil(deg/k)), Delta(H) <= k;
* for each inner solver (greedy completion, Min-to-Min, induced-forest layers):
  |S_H|, |S_0|, whether S_0 is independent in G, and the repaired size;
* c13 and the output c18 of final_candidates;
* the reference: the certified lower bound LB = sum ceil(n_C / (Delta_C + 1)) and,
  for n <= --ip-max-n, the MIDS integer program (HiGHS, --ip-time) giving i(G) or an
  interval [L, U]; the inequality |S_0| <= k (U + s(G)) is checked (it holds for
  every upper bound U >= i(G)).

Families. NP-hard or UGC-hard families (the source of the hardness is the label):

* ``cubic_bipartite``     random 3-regular bipartite (MIDS NP-complete on cubic
                          planar bipartite graphs, Loverov--Orlovich);
* ``planar_delaunay``     Delaunay triangulations of random points (planar, MIDS
                          NP-hard on planar graphs), unbounded degree;
* ``regular_d``           random d-regular, d = 4, 5, 6 (bounded degree,
                          Chlebik--Chlebikova);
* ``regular_bipartite_d`` random d-regular bipartite, d = 4, 5 (bounded-degree
                          bipartite, Chlebik--Chlebikova);
* ``scids_B``             SC-IDS of random set systems, B = 4, 6 (bipartite, degree B);
* ``vcids_cubic``         VC-IDS of random cubic graphs (degree 3);
* ``kr``                  Khot--Regev YES graphs (UG-hard vertex cover instances),
                          base parameters, N = 40 and 80 blocks of 16 samples;
* ``kr_bounded``          Khot--Regev YES graphs with constraint degree 3 and s = 4, 8
                          samples (degree at most 12, 24: the bounded-degree setting
                          of Austrin--Khot--Safra);
* ``vcids_kr``            VC-IDS of Khot--Regev YES graphs (N = 40);
* ``gnp_dense``           G(n, 1/2) (general graphs, Halldorsson's n^(1-eps) setting).

Structured and random families: complete K_n; complete bipartite (balanced and
1:3); complete multipartite (4 equal parts, and random part sizes); crown; star;
wheel; double star; split graph (clique of sqrt(n) vertices joined at random to
an independent set); G(n, 3/n); Barabasi--Albert (m = 3); bipartite power law;
hypercube; 2D grid.

Run from the repository root (the repository's own siriaisa/ is used):

    python car/car_split.py               # n = 100, 400, 1600
    python car/car_split.py --quick       # n = 100, 400
    python car/car_split.py --ip-max-n 400 --ip-time 30

Every instance runs in a worker process with a time limit (--timeout, default
900 s). Outputs (in car/results/): car_split.csv, car_split_summary.csv and
car_split.json.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
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
from siriaisa import forest_layers, heuristics, local_search  # noqa: E402
from siriaisa.algorithm import final_candidates, verify_independent_dominating_set  # noqa: E402
from siriaisa.bipartite import two_colouring  # noqa: E402
from siriaisa.degree_split import copy_graph  # noqa: E402

import car_bipartite as cb  # noqa: E402  (generators and the sparse MIDS integer program)
import car_ugc as cu  # noqa: E402  (Khot--Regev, SC-IDS and VC-IDS builders)

SEED = 20261101
OUT_DIR = _CAR / "results"
DEFAULT_TIMEOUT = 900.0


# ----------------------------------------------------------------------------
# Generators
# ----------------------------------------------------------------------------
def complete_multipartite_random(n: int, parts: int, rng: random.Random) -> nx.Graph:
    cuts = sorted(rng.sample(range(1, n), parts - 1))
    sizes = [b - a for a, b in zip([0, *cuts], [*cuts, n])]
    return nx.complete_multipartite_graph(*sizes)


def double_star(n: int) -> nx.Graph:
    a = (n - 2) // 2
    G = nx.Graph()
    G.add_edge("x", "y")
    G.add_edges_from(("x", ("l", i)) for i in range(a))
    G.add_edges_from(("y", ("r", i)) for i in range(n - 2 - a))
    return G


def split_graph(n: int, rng: random.Random) -> nx.Graph:
    c = max(2, math.isqrt(n))
    G = nx.complete_graph(c)
    for v in range(c, n):
        G.add_node(v)
        for u in rng.sample(range(c), rng.randint(1, c)):
            G.add_edge(u, v)
    return G


def delaunay_planar(n: int, rng: random.Random) -> nx.Graph:
    pts = np.array([[rng.random(), rng.random()] for _ in range(n)])
    tri = Delaunay(pts)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for a, b, c in tri.simplices:
        G.add_edges_from([(int(a), int(b)), (int(b), int(c)), (int(a), int(c))])
    return G


def kr_graph(N: int, d: int, s: int, rng: random.Random) -> nx.Graph:
    H, _ = cu.khot_regev_yes(N=N, d=d, k=4, p=0.45, eta=0.0, s=s, linear=False, rng=rng)
    return H


def instances(quick: bool):
    """(name, family, source, build(rng))."""
    sizes = (100, 400) if quick else (100, 400, 1600)
    for n in sizes:
        for r in range(2):
            t = f"_n{n}_r{r}"
            yield ("cubic_bipartite" + t, "cubic_bipartite", "NP-complete (cubic planar bipartite)",
                   lambda rng, n=n: cb.regular_bipartite(n // 2, 3, rng))
            yield ("planar_delaunay" + t, "planar_delaunay", "NP-hard (planar)",
                   lambda rng, n=n: delaunay_planar(n, rng))
            for d in (4, 5, 6):
                yield (f"regular_d{d}" + t, f"regular_d{d}", "NP-hard to approximate (bounded degree)",
                       lambda rng, n=n, d=d: nx.random_regular_graph(d, n - (n * d) % 2, seed=rng.randrange(2**31)))
            for d in (4, 5):
                yield (f"regular_bipartite_d{d}" + t, f"regular_bipartite_d{d}",
                       "NP-hard to approximate (bounded-degree bipartite)",
                       lambda rng, n=n, d=d: cb.regular_bipartite(n // 2, d, rng))
            for B in (4, 6):
                yield (f"scids_B{B}" + t, f"scids_B{B}", "SC-IDS reduction (bipartite, degree B)",
                       lambda rng, n=n, B=B: cu.sc_ids(cu.random_set_system(max(4, n // 4), B, rng)))
            yield ("vcids_cubic" + t, "vcids_cubic", "VC-IDS reduction (degree 3)",
                   lambda rng, n=n: cu.vc_ids(nx.random_regular_graph(3, max(4, n // 8) * 2, seed=rng.randrange(2**31)))[0])
            yield ("gnp_dense" + t, "gnp_dense", "general graphs (n^(1-eps) hardness)",
                   lambda rng, n=n: nx.gnp_random_graph(n, 0.5, seed=rng.randrange(2**31)))
            # Structured and random families.
            yield ("complete" + t, "complete", "structured", lambda rng, n=n: nx.complete_graph(n))
            yield ("complete_bipartite_bal" + t, "complete_bipartite_bal", "structured",
                   lambda rng, n=n: nx.complete_bipartite_graph(n // 2, n - n // 2))
            yield ("complete_bipartite_1to3" + t, "complete_bipartite_1to3", "structured",
                   lambda rng, n=n: nx.complete_bipartite_graph(n // 4, n - n // 4))
            yield ("complete_multipartite_4" + t, "complete_multipartite_4", "structured",
                   lambda rng, n=n: nx.complete_multipartite_graph(*([n // 4] * 4)))
            yield ("complete_multipartite_rand" + t, "complete_multipartite_rand", "structured",
                   lambda rng, n=n: complete_multipartite_random(n, 5, rng))
            yield ("crown" + t, "crown", "structured", lambda rng, n=n: cb.crown(n // 2))
            yield ("star" + t, "star", "structured", lambda rng, n=n: nx.star_graph(n - 1))
            yield ("wheel" + t, "wheel", "structured", lambda rng, n=n: nx.wheel_graph(n))
            yield ("double_star" + t, "double_star", "structured", lambda rng, n=n: double_star(n))
            yield ("split" + t, "split", "structured", lambda rng, n=n: split_graph(n, rng))
            yield ("gnp_sparse" + t, "gnp_sparse", "random", lambda rng, n=n: nx.gnp_random_graph(n, 3.0 / n, seed=rng.randrange(2**31)))
            yield ("barabasi_albert" + t, "barabasi_albert", "random",
                   lambda rng, n=n: nx.barabasi_albert_graph(n, 3, seed=rng.randrange(2**31)))
            yield ("powerlaw_bipartite" + t, "powerlaw_bipartite", "random", lambda rng, n=n: cb.power_law_bipartite(n, 2.5, rng))
            side = math.isqrt(n)
            yield ("grid" + t, "grid", "structured", lambda rng, side=side: nx.grid_2d_graph(side, side))
        q = max(2, round(math.log2(n)))
        yield (f"hypercube_Q{q}", "hypercube", "structured", lambda rng, q=q: nx.hypercube_graph(q))
    for N in ((40,) if quick else (40, 80)):
        for r in range(2):
            yield (f"kr_N{N}_r{r}", "kr", "UG-hard vertex cover (Khot--Regev)", lambda rng, N=N: kr_graph(N, 4, 16, rng))
            for s in (4, 8):
                yield (f"kr_bounded_s{s}_N{N}_r{r}", "kr_bounded", "UG-hard bounded degree (Austrin--Khot--Safra setting)",
                       lambda rng, N=N, s=s: kr_graph(N, 3, s, rng))
    for r in range(2):
        yield (f"vcids_kr_N40_r{r}", "vcids_kr", "VC-IDS of Khot--Regev (UGC)",
               lambda rng: cu.vc_ids(kr_graph(40, 4, 16, rng))[0])


# ----------------------------------------------------------------------------
# Measurements
# ----------------------------------------------------------------------------
def excess(degrees: list[int], k: int) -> tuple[int, int]:
    """(|B|, s) for the degree sequence and threshold k."""
    B = [d for d in degrees if d > k]
    return len(B), sum(-(-d // k) - 1 for d in B)


def lower_bound(G: nx.Graph) -> int:
    total = 0
    for comp in nx.connected_components(G):
        nc = len(comp)
        dc = max(G.degree(v) for v in comp)
        total += -(-nc // (dc + 1))
    return total


def run_instance(name: str, family: str, source: str, G: nx.Graph, args) -> dict:
    G = nx.convert_node_labels_to_integers(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    n, m = G.number_of_nodes(), G.number_of_edges()
    adj = [list(G.adj[v]) for v in range(n)]
    deg = [len(a) for a in adj]
    Delta = max(deg, default=0)
    k = max(1, math.isqrt(n))
    nB, s = excess(deg, k)
    bound_ii = min((2 * m - nB) / k, nB * (n - 2) / k) if nB else 0.0
    s_Kn = n * (-(-(n - 1) // k) - 1) if n >= 3 else 0
    if s > bound_ii + 1e-9 or s > s_Kn:
        raise AssertionError(f"{name}: s(G) = {s} exceeds the bounds of Proposition 'Splitting excess'")

    t0 = time.perf_counter()
    hadj, owner, split = copy_graph(adj, k)
    split_vertex = [False] * n
    for x in range(n, len(hadj)):
        split_vertex[owner[x]] = True
    keep = [x for x in range(len(hadj)) if not (x < n and split_vertex[x])]
    index = {x: i for i, x in enumerate(keep)}
    sub = [[index[y] for y in hadj[x]] for x in keep]
    nH = len(keep)
    if nH != n - nB + sum(-(-d // k) for d in deg if d > k) or max((len(a) for a in sub), default=0) > k:
        raise AssertionError(f"{name}: copy graph does not match Lemma 'Copy graph'")
    comp = two_colouring(sub)[1]
    inner = {"greedy": local_search.complete_ids(sub, ()), "min_to_min": heuristics.min_to_min_ids(sub),
             "forest_layers": forest_layers.forest_layers_ids(sub, comp)}
    decoded = {}
    for key, sol in inner.items():
        count = [0] * n
        for i, x in enumerate(keep):
            if sol[i]:
                count[owner[x]] += 1
        S0 = {v for v in range(n) if count[v]}
        independent = all(not (u in S0 and v in S0) for u, v in G.edges())
        dominating = all(v in S0 or any(u in S0 for u in adj[v]) for v in range(n))
        if not dominating:
            raise AssertionError(f"{name}: decoded set S_0 ({key}) does not dominate G (Lemma 'Decoding')")
        repaired = local_search.complete_ids(adj, list(S0), priority=count)
        decoded[key] = {"S_H": sum(sol), "S0": len(S0), "S0_independent": independent, "repaired": sum(repaired)}
    split_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    cands = final_candidates(G)
    siriaisa_seconds = time.perf_counter() - t1
    for key in ("c13", "c18"):
        if not verify_independent_dominating_set(G, cands[key]):
            raise AssertionError(f"{name}: {key} is not an independent dominating set")
    c13, c18 = len(cands["c13"]), len(cands["c18"])

    LB = lower_bound(G)
    ref = {"status": "not run", "value": None, "lower": LB, "seconds": 0.0}
    if n <= args.ip_max_n:
        ref = cb.mids_milp(G, args.ip_time)
        ref["lower"] = max(ref["lower"], LB)
    lower = ref["lower"]
    upper = min(c18, ref["value"]) if ref["value"] is not None else c18
    exact = lower == upper
    best_S0 = min(v["S0"] for v in decoded.values())
    if best_S0 > k * (upper + s):
        raise AssertionError(f"{name}: |S_0| = {best_S0} > k (U + s) = {k * (upper + s)} -- contradicts Theorem 'Degree splitting'")
    row = {
        "name": name, "family": family, "source": source, "n": n, "m": m, "Delta": Delta, "k": k,
        "B": nB, "s": s, "bound_ii": bound_ii, "s_Kn": s_Kn,
        "s_over_n": s / n if n else 0.0, "s_over_n32": s / n ** 1.5 if n else 0.0,
        "H_nodes": nH, "H_max_degree": max((len(a) for a in sub), default=0), "split": split,
        "decoded": decoded, "any_S0_independent": any(v["S0_independent"] for v in decoded.values()),
        "best_S0": best_S0, "c13": c13, "c18": c18,
        "lower": lower, "upper": upper, "exact": exact, "ip_status": ref["status"], "ip_seconds": ref["seconds"],
        "c13_optimal": exact and c13 == lower, "c18_optimal": exact and c18 == lower,
        "s_le_i": s <= lower,  # sufficient: s <= L <= i
        "theorem_bound": k * (upper + s),
        "split_seconds": split_seconds, "siriaisa_seconds": siriaisa_seconds,
    }
    print(f"  [split] {family:27} {name:34} n={n:>6,} Delta={Delta:>5} k={k:>3} |B|={nB:>6,} s={s:>8,}"
          f" s/n^1.5={row['s_over_n32']:.3f} S0={best_S0:>6,}{' ind' if row['any_S0_independent'] else ''}"
          f" c13={c13:>6,} c18={c18:>6,} i in [{lower},{upper}] {siriaisa_seconds:6.1f}s", flush=True)
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
        k = len(g)
        out[f] = {
            "source": g[0]["source"] if f != "ALL" else "",
            "instances": k,
            "n_range": f"{min(r['n'] for r in g)}-{max(r['n'] for r in g)}",
            "max_Delta": max(r["Delta"] for r in g),
            "s_zero": sum(r["s"] == 0 for r in g),
            "max_s": max(r["s"] for r in g),
            "max_s_over_n": max(r["s_over_n"] for r in g),
            "max_s_over_n32": max(r["s_over_n32"] for r in g),
            "s_le_i": sum(r["s_le_i"] for r in g),
            "S0_independent": sum(r["any_S0_independent"] for r in g),
            "exact": sum(r["exact"] for r in g),
            "c13_optimal": sum(r["c13_optimal"] for r in g),
            "c18_optimal": sum(r["c18_optimal"] for r in g),
            "max_c18_ratio": max(r["c18"] / r["lower"] for r in g if r["lower"]),
            "siriaisa_seconds": sum(r["siriaisa_seconds"] for r in g),
        }
    return out


class InstanceRunner:
    """One instance per spawned worker process, with a time limit (as Salvador's car/)."""

    def __init__(self, timeout: float):
        self.timeout = timeout or 0.0
        self._pool = None

    def run(self, func, *args):
        if self.timeout <= 0:
            return func(*args)
        if self._pool is None:
            self._pool = multiprocessing.get_context("spawn").Pool(processes=1)
        result = self._pool.apply_async(func, args)
        try:
            return result.get(self.timeout)
        except multiprocessing.TimeoutError:
            self._pool.terminate()
            self._pool.join()
            self._pool = None
            return None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None


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
    with InstanceRunner(args.timeout) as runner:
        for name, family, _, _ in instances(args.quick):
            row = runner.run(_job, name, args)
            if row is None:
                print(f"  [split] {family:27} {name:34} skipped (more than {args.timeout:g} s)", flush=True)
                skipped.append(name)
            else:
                rows.append(row)
    summary = summarise(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["family", "source", "name", "n", "m", "Delta", "k", "B", "s", "bound_ii", "s_Kn", "s_over_n", "s_over_n32",
            "H_nodes", "H_max_degree", "best_S0", "any_S0_independent", "c13", "c18", "lower", "upper", "exact",
            "ip_status", "c13_optimal", "c18_optimal", "s_le_i", "theorem_bound", "siriaisa_seconds"]
    with (OUT_DIR / "car_split.csv").open("w", encoding="utf-8") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(f"\"{r[c]}\"" if c == "source" else
                              (f"{r[c]:.6f}" if isinstance(r[c], float) else str(r[c])) for c in cols) + "\n")
    scols = ["source", "instances", "n_range", "max_Delta", "s_zero", "max_s", "max_s_over_n", "max_s_over_n32",
             "s_le_i", "S0_independent", "exact", "c13_optimal", "c18_optimal", "max_c18_ratio", "siriaisa_seconds"]
    with (OUT_DIR / "car_split_summary.csv").open("w", encoding="utf-8") as fh:
        fh.write("family," + ",".join(scols) + "\n")
        for f, s in summary.items():
            fh.write(f + "," + ",".join(f"\"{s[c]}\"" if c == "source" else
                                        (f"{s[c]:.6f}" if isinstance(s[c], float) else str(s[c])) for c in scols) + "\n")
    (OUT_DIR / "car_split.json").write_text(json.dumps(
        {"siriaisa_version": siriaisa.__version__,
         "environment": {"python": platform.python_version(), "platform": platform.platform(),
                         "networkx": nx.__version__, "scipy": scipy.__version__, "numpy": np.__version__},
         "seed": SEED, "quick": args.quick, "ip_max_n": args.ip_max_n, "ip_time": args.ip_time,
         "timeout": args.timeout, "skipped_instances": skipped, "summary": summary, "rows": rows,
         "elapsed_seconds": time.time() - started}, indent=2, default=str), encoding="utf-8")

    print("\nSplitting excess s(G) per family (k = floor(sqrt(n)))")
    print(f"{'family':27} {'inst':>4} {'n':>10} {'Delta':>6} {'s=0':>4} {'max s':>8} {'s/n^1.5':>8} {'s<=i':>5}"
          f" {'S0 ind':>6} {'exact':>5} {'c13 opt':>7} {'c18 opt':>7} {'max ratio':>9}")
    for f, s in summary.items():
        print(f"{f:27} {s['instances']:>4} {s['n_range']:>10} {s['max_Delta']:>6} {s['s_zero']:>4} {s['max_s']:>8}"
              f" {s['max_s_over_n32']:8.3f} {s['s_le_i']:>5} {s['S0_independent']:>6} {s['exact']:>5}"
              f" {s['c13_optimal']:>7} {s['c18_optimal']:>7} {s['max_c18_ratio']:9.4f}")
    print("s<=i: s(G) <= certified lower bound on i(G); S0 ind: some inner solver's decoded set is independent "
          "(then Theorem 'Degree splitting' bounds the output); max ratio: c18 / lower bound (upper bound on the ratio)")


if __name__ == "__main__":
    main()
