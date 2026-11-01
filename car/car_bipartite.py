"""car/ bipartite study: how far is each Siriaisa strategy from the optimum on bipartite graphs?

Counterpart, for the Minimum Independent Dominating Set (MIDS), of Salvador's
car/car_bipartite.py. Siriaisa 0.0.8 (siriaisa/algorithm.py) runs the ensemble
c1..c18 (``final_candidates``): Salvador 0.1.4's seventeen strategies and
Esperanza 0.1.6's phases adapted to independent domination, with forests (solved
exactly by the linear-time tree DP) in place of Salvador's bipartite structures,
and c13, the degree splitting at floor(sqrt(n)) (see siriaisa/algorithm.py for
the list); c18, the returned set, is the smallest refinement of c1..c17. The
report gives every strategy's size, the size of each candidate after the c18
refinement, the number of vertices c13 split and copies it created, and, per
instance, the number of c9
induced-forest layers, whether c9 was exact (every component a tree, one layer)
and whether c9 is optimal, and per family how often (``c9 exact``, ``c9 opt``).

Unlike vertex cover, MIDS stays NP-hard on bipartite graphs, so there is no
Koenig-type formula for the optimum. For every instance the minimum size
i(G) is computed *for reference only* (it is never given to Siriaisa) by an
exact SciPy/HiGHS MILP

    minimize  sum_v x_v
    s.t.      x_v + sum_{u in N(v)} x_u >= 1   for every vertex v   (domination)
              x_u + x_v <= 1                   for every edge uv    (independence)
              x_v in {0, 1}

with a time limit (``--milp-time-limit``). When HiGHS proves optimality the
reported ratio |c18| / i(G) is exact (status ``optimal`` or ``suboptimal``).
When the time limit is reached, HiGHS's dual bound L gives the certified
range ceil(L) <= i(G) <= min(|c18|, MILP incumbent): c18 is ``certified`` when
|c18| = ceil(L), and ``open`` otherwise. For even cycles and paths
(i = ceil(n/3)), complete bipartite graphs K_{a,b} (i = min(a, b)) and crown
graphs (i = 2) the closed form is used as well and checked against the MILP.

Families (all bipartite, labels shuffled), as in Salvador's study:

* random bipartite G(n1, n2, p), balanced and unbalanced, average degree 3, 6, 20;
* random d-regular bipartite graphs (union of d random perfect matchings);
* bipartite graphs with power-law degrees (bipartite configuration model);
* bipartite double covers G x K2 of random 3- and 5-regular graphs;
* "hidden matching" graphs: a perfect matching plus random edges from one side
  to the other;
* random trees, paths, even cycles, 2D/3D grids, grids with 30% of the edges
  deleted, complete bipartite K_{a,b}, crown graphs, hypercubes Q_k and random
  halves of hypercubes.

Run from the repository root (the repository's own siriaisa/ is used, so the
package does not need to be installed; an installed siriaisa is ignored):

    python car/car_bipartite.py                      # sizes 100 .. 2000
    python car/car_bipartite.py --quick              # sizes 100 and 200 only
    python car/car_bipartite.py --max-n 10000        # also sizes 5000 and 10000
    python car/car_bipartite.py --milp-time-limit 120 --timeout 600

Every instance is built and evaluated in a worker process with a time limit
(``--timeout``, default 600 s, 0 = no limit): an instance whose construction,
Siriaisa run or MILP takes longer is skipped and listed in
car/results/car_bipartite.json (``skipped_instances``). Each instance has its
own random generator, derived from SEED and its name, so the worker rebuilds
it independently.

Outputs (in car/results/): car_bipartite.json, car_bipartite.csv (one row per
instance) and car_bipartite_summary.csv (one row per family).
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

# Use the repository's siriaisa/ (works whether or not the package is installed).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import scipy  # noqa: E402
from scipy.optimize import Bounds, LinearConstraint, milp  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402

import siriaisa  # noqa: E402
from siriaisa.algorithm import (CANDIDATE_NAMES, FINAL_NAME, final_candidates,  # noqa: E402
                                verify_independent_dominating_set)

SEED = 20261031
NAMES = list(CANDIDATE_NAMES)            # c1..c18
BASE = [k for k in NAMES if k != FINAL_NAME]  # c1..c17
OUT_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_TIMEOUT = 600.0
DEFAULT_MILP_TIME_LIMIT = 60.0


# ----------------------------------------------------------------------------
# Generators (every graph returned is bipartite)
# ----------------------------------------------------------------------------
def shuffle_labels(G: nx.Graph, rng: random.Random) -> nx.Graph:
    nodes = list(G.nodes())
    perm = list(range(len(nodes)))
    rng.shuffle(perm)
    return nx.relabel_nodes(G, {v: perm[i] for i, v in enumerate(nodes)})


def random_bipartite(n1: int, n2: int, avg_deg: float, rng: random.Random) -> nx.Graph:
    p = min(1.0, avg_deg * (n1 + n2) / (2.0 * n1 * n2))
    return nx.bipartite.random_graph(n1, n2, p, seed=rng.randrange(2**31))


def regular_bipartite(half: int, d: int, rng: random.Random) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(2 * half))
    for _ in range(d):
        perm = list(range(half))
        rng.shuffle(perm)
        G.add_edges_from((i, half + perm[i]) for i in range(half))
    return G


def power_law_bipartite(n: int, exponent: float, rng: random.Random) -> nx.Graph:
    half = n // 2

    def seq() -> list[int]:
        return [max(1, min(half - 1, int(rng.paretovariate(exponent - 1)))) for _ in range(half)]

    a, b = seq(), seq()
    diff = sum(a) - sum(b)
    i = 0
    while diff != 0:  # balance the two degree sums
        if diff > 0:
            b[i % half] += 1
            diff -= 1
        else:
            a[i % half] += 1
            diff += 1
        i += 1
    M = nx.bipartite.configuration_model(a, b, seed=rng.randrange(2**31))
    return nx.Graph(M)


def double_cover(n: int, d: int, rng: random.Random) -> nx.Graph:
    H = nx.random_regular_graph(d, n - (n * d) % 2, seed=rng.randrange(2**31))
    return nx.tensor_product(H, nx.complete_graph(2))


def thinned_grid(rows: int, cols: int, keep: float, rng: random.Random) -> nx.Graph:
    G = nx.grid_2d_graph(rows, cols)
    G.remove_edges_from([e for e in list(G.edges()) if rng.random() > keep])
    return G


def hypercube_subgraph(k: int, keep: float, rng: random.Random) -> nx.Graph:
    G = nx.hypercube_graph(k)
    G.remove_edges_from([e for e in list(G.edges()) if rng.random() > keep])
    return G


def crown(k: int) -> nx.Graph:
    G = nx.Graph()
    G.add_edges_from((("a", i), ("b", j)) for i in range(k) for j in range(k) if i != j)
    return G


def hidden_matching(half: int, extra_deg: float, rng: random.Random) -> nx.Graph:
    """Perfect matching a_i - b_i plus random a -> b edges."""
    G = nx.Graph()
    G.add_edges_from((i, half + i) for i in range(half))
    for i in range(half):
        k = int(extra_deg) + (1 if rng.random() < extra_deg - int(extra_deg) else 0)
        for j in rng.sample(range(half), min(k, half)):
            G.add_edge(i, half + j)
    return G


def instances(max_n: int, quick: bool):
    """Yield ``(name, group, build, closed_form)``: ``build(rng)`` returns the graph and
    ``closed_form`` is the known value of i(G) or None."""
    sizes = [n for n in ((100, 200) if quick else (100, 200, 500, 1000, 2000, 5000, 10_000)) if n <= max_n]
    for n in sizes:
        reps = 3 if n <= 1000 else 1
        for r in range(reps):
            for deg in (3, 6, 20):
                yield (f"gnp_bip_bal_n{n}_d{deg}_r{r}", "random bipartite (balanced)",
                       lambda rng, n=n, deg=deg: random_bipartite(n // 2, n - n // 2, deg, rng), None)
                yield (f"gnp_bip_unbal_n{n}_d{deg}_r{r}", "random bipartite (1:2)",
                       lambda rng, n=n, deg=deg: random_bipartite(n // 3, n - n // 3, deg, rng), None)
            for d in (3, 4, 8):
                yield (f"reg_bip_n{n}_d{d}_r{r}", "random d-regular bipartite",
                       lambda rng, n=n, d=d: regular_bipartite(n // 2, d, rng), None)
            yield (f"powerlaw_bip_n{n}_r{r}", "power-law bipartite",
                   lambda rng, n=n: power_law_bipartite(n, 2.5, rng), None)
            for d in (3, 5):
                yield (f"double_cover_n{n}_d{d}_r{r}", "double cover G x K2 of random regular",
                       lambda rng, n=n, d=d: double_cover(n // 2, d, rng), None)
            for ed in (1.5, 4.0):
                yield (f"hidden_matching_n{n}_e{ed}_r{r}", "hidden perfect matching + random edges",
                       lambda rng, n=n, ed=ed: hidden_matching(n // 2, ed, rng), None)
            yield (f"tree_n{n}_r{r}", "random tree",
                   lambda rng, n=n: nx.random_labeled_tree(n, seed=rng.randrange(2**31)), None)
            side = int(n ** 0.5)
            yield (f"thinned_grid_{side}x{side}_r{r}", "2D grid, 70% edges kept",
                   lambda rng, side=side: thinned_grid(side, side, 0.7, rng), None)
        side = int(n ** 0.5)
        yield f"grid_{side}x{side}", "2D grid", lambda rng, side=side: nx.grid_2d_graph(side, side), None
        c = max(2, round(n ** (1 / 3)))
        yield f"grid3d_{c}^3", "3D grid", lambda rng, c=c: nx.grid_graph(dim=[c, c, c]), None
        en = n - n % 2
        yield f"even_cycle_{en}", "even cycle", lambda rng, en=en: nx.cycle_graph(en), math.ceil(en / 3)
        yield f"path_{n}", "path", lambda rng, n=n: nx.path_graph(n), math.ceil(n / 3)
        if n <= 1000:
            a, b = n // 4, n - n // 4
            yield f"K_{a}_{b}", "complete bipartite", lambda rng, a=a, b=b: nx.complete_bipartite_graph(a, b), min(a, b)
            yield f"crown_{n // 2}", "crown graph", lambda rng, n=n: crown(n // 2), 2
    for k in (6, 8, 10, 12, 13):
        if 2 ** k <= max_n and not (quick and k > 8):
            yield f"hypercube_Q{k}", "hypercube", lambda rng, k=k: nx.hypercube_graph(k), None
            yield f"hypercube_Q{k}_half", "hypercube, 50% edges kept", lambda rng, k=k: hypercube_subgraph(k, 0.5, rng), None


# ----------------------------------------------------------------------------
# Exact reference: sparse MILP (HiGHS)
# ----------------------------------------------------------------------------
def mids_milp(G: nx.Graph, time_limit: float) -> dict:
    """Exact MIDS by MILP with a time limit; reference only. Returns ``status`` ("optimal" or
    "time_limit"), ``value`` (best integer solution size found, or None), ``lower`` (certified
    lower bound, an integer), ``seconds``."""
    nodes = list(G.nodes())
    index = {v: i for i, v in enumerate(nodes)}
    n, m = len(nodes), G.number_of_edges()
    rows, cols = [], []
    for v in nodes:  # domination rows 0..n-1
        i = index[v]
        rows.append(i)
        cols.append(i)
        for u in G.adj[v]:
            rows.append(i)
            cols.append(index[u])
    for k, (u, v) in enumerate(G.edges()):  # independence rows n..n+m-1
        rows += [n + k, n + k]
        cols += [index[u], index[v]]
    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n + m, n)).tocsr()
    lb = np.concatenate([np.ones(n), np.full(m, -np.inf)])
    ub = np.concatenate([np.full(n, np.inf), np.ones(m)])
    t0 = time.perf_counter()
    res = milp(c=np.ones(n), integrality=np.ones(n), bounds=Bounds(0, 1),
               constraints=LinearConstraint(A, lb, ub),
               options={"time_limit": time_limit, "disp": False})
    seconds = time.perf_counter() - t0
    value = None
    if res.x is not None:
        x = np.rint(res.x).astype(int)
        S = {nodes[i] for i in range(n) if x[i] == 1}
        if not verify_independent_dominating_set(G, S):
            raise RuntimeError("MILP solution is not an independent dominating set")
        value = len(S)
    if res.status == 0 and value is not None:
        return {"status": "optimal", "value": value, "lower": value, "seconds": seconds}
    dual = getattr(res, "mip_dual_bound", None)
    lower = 1 if n else 0
    if dual is not None and np.isfinite(dual):
        lower = max(lower, math.ceil(dual - 1e-6))
    return {"status": "time_limit", "value": value, "lower": lower, "seconds": seconds,
            "message": str(res.message)}


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def run_instance(name: str, group: str, G: nx.Graph, closed_form, rng: random.Random, args) -> dict:
    G = shuffle_labels(nx.convert_node_labels_to_integers(G), rng)
    G.remove_edges_from(nx.selfloop_edges(G))
    assert nx.is_bipartite(G), name
    info: dict = {}
    t0 = time.perf_counter()
    cands, refined = final_candidates(G, return_refined=True, info=info)
    elapsed = time.perf_counter() - t0
    for k in NAMES:
        if not verify_independent_dominating_set(G, cands[k]):
            raise AssertionError(f"{name}: strategy {k} is not an independent dominating set")
        if not verify_independent_dominating_set(G, refined[k]):
            raise AssertionError(f"{name}: refined strategy {k} is not an independent dominating set")
    sizes = {k: len(cands[k]) for k in NAMES}
    rsizes = {k: len(refined[k]) for k in BASE}
    final = sizes[FINAL_NAME]

    ref = mids_milp(G, args.milp_time_limit)
    if ref["status"] == "optimal":
        opt = ref["value"]
        if closed_form is not None and opt != closed_form:
            raise AssertionError(f"{name}: MILP optimum {opt} differs from the closed form {closed_form}")
        lower = upper = opt
        source = "milp"
    elif closed_form is not None:
        opt = lower = upper = closed_form
        source = "closed form"
    else:
        opt = None
        lower = ref["lower"]
        upper = min(final, ref["value"]) if ref["value"] is not None else final
        source = "milp bound"
    if final < lower:
        raise AssertionError(f"{name}: c18 = {final} is below the certified lower bound {lower} -- impossible")
    if opt is not None:
        status = "optimal" if final == opt else "suboptimal"
    else:
        status = "certified" if final == lower else "open"
    proven = status in ("optimal", "certified")
    row = {
        "group": group, "name": name, "n": G.number_of_nodes(), "m": G.number_of_edges(),
        "opt": opt, "lower": lower, "upper": upper, "reference": source,
        "milp_status": ref["status"], "milp_value": ref["value"], "milp_seconds": ref["seconds"],
        "sizes": sizes, "refined": rsizes, "final": final,
        "status": status, "optimal": proven,
        "ratio": final / lower if lower else 1.0,
        "excess": final - lower,
        "optimal_strategies": [k for k in NAMES if sizes[k] == lower] if proven or opt is not None else [],
        "optimal_refined": [k for k in BASE if rsizes[k] == lower] if proven or opt is not None else [],
        "c11_exact": bool(info.get("c11_exact", False)),
        "c11_core_size": info.get("c11_core_size", 0),
        "c1_exhausted": bool(info.get("c1_exhausted", False)),
        "c1_iterations": info.get("c1_iterations", 0),
        "c9_layers": info.get("c9_max_layers", 0),
        "c13_split": info.get("c13_split", 0),
        "c13_copies": info.get("c13_copies", 0),
        "c9_exact": bool(info.get("c9_exact", False)),
        "c9_optimal": (sizes["c9"] == opt) if opt is not None else (sizes["c9"] == lower),
        "components": info.get("components", 0),
        "elapsed_seconds": elapsed,
    }
    ref_txt = f"i={opt}" if opt is not None else f"i in [{lower},{upper}]"
    print(f"  [bipartite] {group[:34]:34} {name:30} n={row['n']:>6,} m={row['m']:>7,} {ref_txt:>16}"
          f" c18={final:>6,} ratio<={row['ratio']:.6f} {status:10}"
          f" c9={sizes['c9']:>6,} L={row['c9_layers']}{' exact' if row['c9_exact'] else ''}{'' if row['c9_optimal'] else ' (not optimal)'}"
          f" base={min(sizes[k] for k in BASE):>6,} siriaisa={elapsed:6.2f}s milp={ref['seconds']:6.2f}s",
          flush=True)
    return row


def _instance_job(name: str, args) -> dict:
    """Build instance ``name`` with its own generator and evaluate it; runs in the worker process."""
    for nm, group, build, closed_form in instances(args.max_n, args.quick):
        if nm == name:
            rng = random.Random(f"{SEED}:{name}")
            return run_instance(name, group, build(rng), closed_form, rng, args)
    raise KeyError(name)


class InstanceRunner:
    """Runs one instance job in a spawned worker process with a time limit (as Salvador's car/):
    ``run(func, *args)`` returns ``func(*args)``, or None after ``timeout`` seconds (the worker is
    then terminated and a fresh one started). With ``timeout <= 0`` the job runs in this process."""

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


def summarise(rows: list[dict]) -> dict:
    out = {}
    for g in sorted({r["group"] for r in rows}) + ["ALL"]:
        grows = rows if g == "ALL" else [r for r in rows if r["group"] == g]
        k = len(grows)
        known = [r for r in grows if r["opt"] is not None or r["optimal"]]
        out[g] = {
            "instances": k,
            "reference_exact": sum(r["opt"] is not None for r in grows),
            "optimal": sum(r["optimal"] for r in grows),
            "optimal_rate": sum(r["optimal"] for r in grows) / k if k else 0.0,
            "suboptimal": sum(r["status"] == "suboptimal" for r in grows),
            "open": sum(r["status"] == "open" for r in grows),
            "max_ratio": max((r["ratio"] for r in grows), default=1.0),
            "mean_ratio": sum(r["ratio"] for r in grows) / k if k else 1.0,
            "total_excess": sum(r["excess"] for r in grows),
            "total_lower": sum(r["lower"] for r in grows),
            "c11_exact": sum(r["c11_exact"] for r in grows),
            "c11_mean_core_fraction": sum(r["c11_core_size"] / r["n"] for r in grows) / k if k else 0.0,
            "c1_exhausted": sum(r["c1_exhausted"] for r in grows),
            "c9_exact": sum(r["c9_exact"] for r in grows),
            "c9_optimal": sum(r["c9_optimal"] for r in grows),
            "c9_max_layers": max((r["c9_layers"] for r in grows), default=0),
            "optimal_by_strategy": {s: sum(s in r["optimal_strategies"] for r in known) for s in NAMES},
            "optimal_by_refined": {s: sum(s in r["optimal_refined"] for r in known) for s in BASE},
            "siriaisa_seconds": sum(r["elapsed_seconds"] for r in grows),
            "milp_seconds": sum(r["milp_seconds"] for r in grows),
        }
    return out


def print_table(summary: dict) -> None:
    header = (f"{'family':42} {'inst':>5} {'opt':>5} {'opt%':>6} {'subopt':>6} {'open':>5} {'max ratio':>10}"
              f" {'excess':>7} {'c11ex':>5} {'core%':>6} {'c9ex':>5} {'c9opt':>5} {'c9L':>4}  optimal by strategy (" + " ".join(NAMES) + ")"
              "  | after refinement (" + " ".join(BASE) + ")")
    print("\nBipartite graphs: Siriaisa c18 against the exact MIDS optimum i(G) (MILP, closed forms)")
    print(header)
    print("-" * len(header))
    for g, s in summary.items():
        by = " ".join(f"{s['optimal_by_strategy'][k]:>3}" for k in NAMES)
        byr = " ".join(f"{s['optimal_by_refined'][k]:>3}" for k in BASE)
        print(f"{g[:42]:42} {s['instances']:>5} {s['optimal']:>5} {100 * s['optimal_rate']:5.1f}%"
              f" {s['suboptimal']:>6} {s['open']:>5} {s['max_ratio']:10.6f} {s['total_excess']:>7} {s['c11_exact']:>5} {100 * s['c11_mean_core_fraction']:5.1f}% {s['c9_exact']:>5} {s['c9_optimal']:>5} {s['c9_max_layers']:>4}  {by}  | {byr}")
    print("c9ex = every component is a tree solved in one layer (then c9 is optimal); "
          "c9opt = c9 optimal; c9L = largest number of c9 layers; "
          "c11ex = c11 exact (empty feedback core: a forest); core% = mean c11 feedback core size / n")
    print("opt = c18 equals i(G) (proven by MILP / closed form) or the certified lower bound; "
          "open = MILP time limit reached and c18 above the lower bound (ratio is then an upper bound)")
    print("\nLegend: " + "; ".join(f"{k} = {v}" for k, v in CANDIDATE_NAMES.items()))


def environment() -> dict:
    return {"python": platform.python_version(), "platform": platform.platform(),
            "networkx": nx.__version__, "scipy": scipy.__version__, "numpy": np.__version__,
            "siriaisa": siriaisa.__version__, "siriaisa_path": str(Path(siriaisa.__file__).resolve().parent)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="sizes 100 and 200 only")
    ap.add_argument("--max-n", type=int, default=2000, help="largest instance size (default 2000; up to 10000)")
    ap.add_argument("--milp-time-limit", type=float, default=DEFAULT_MILP_TIME_LIMIT,
                    help=f"time limit of the reference MILP per instance (default {DEFAULT_MILP_TIME_LIMIT:g} s)")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                    help=f"skip an instance whose construction and evaluation take longer than this many seconds "
                         f"(default {DEFAULT_TIMEOUT:g}; 0 = no limit, run in this process)")
    args = ap.parse_args()

    print(f"siriaisa {siriaisa.__version__} from {Path(siriaisa.__file__).resolve().parent}", flush=True)
    started = time.time()
    rows, skipped = [], []
    names = [(name, group) for name, group, _, _ in instances(args.max_n, args.quick)]
    with InstanceRunner(args.timeout) as runner:
        for name, group in names:
            row = runner.run(_instance_job, name, args)
            if row is None:
                print(f"  [bipartite] {group[:34]:34} {name:30} skipped (more than {args.timeout:g} s)", flush=True)
                skipped.append(name)
            else:
                rows.append(row)
    summary = summarise(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "car_bipartite.csv").open("w", encoding="utf-8") as fh:
        fh.write("group,name,n,m,opt,lower,upper,reference,milp_status,milp_value,milp_seconds,"
                 + ",".join(NAMES) + "," + ",".join(f"refined_{k}" for k in BASE)
                 + ",final,status,ratio,excess,optimal_strategies,optimal_refined,c11_exact,c11_core_size,c1_exhausted,"
                 "c1_iterations,c9_layers,c9_exact,c9_optimal,c13_split,c13_copies,components,elapsed_seconds\n")
        for r in rows:
            fh.write(f"\"{r['group']}\",{r['name']},{r['n']},{r['m']},{'' if r['opt'] is None else r['opt']},"
                     f"{r['lower']},{r['upper']},{r['reference']},{r['milp_status']},"
                     f"{'' if r['milp_value'] is None else r['milp_value']},{r['milp_seconds']:.3f},"
                     + ",".join(str(r["sizes"][k]) for k in NAMES) + ","
                     + ",".join(str(r["refined"][k]) for k in BASE)
                     + f",{r['final']},{r['status']},{r['ratio']:.6f},{r['excess']},"
                     f"{'+'.join(r['optimal_strategies'])},{'+'.join(r['optimal_refined'])},"
                     f"{r['c11_exact']},{r['c11_core_size']},{r['c1_exhausted']},{r['c1_iterations']},{r['c9_layers']},{r['c9_exact']},{r['c9_optimal']},{r['c13_split']},{r['c13_copies']},{r['components']},"
                     f"{r['elapsed_seconds']:.3f}\n")
    with (OUT_DIR / "car_bipartite_summary.csv").open("w", encoding="utf-8") as fh:
        fh.write("group,instances,reference_exact,optimal,optimal_rate,suboptimal,open,max_ratio,mean_ratio,"
                 "total_excess,total_lower,c11_exact,c11_mean_core_fraction,c1_exhausted,c9_exact,c9_optimal,c9_max_layers,"
                 + ",".join(f"optimal_{k}" for k in NAMES) + ","
                 + ",".join(f"optimal_refined_{k}" for k in BASE) + ",siriaisa_seconds,milp_seconds\n")
        for g, s in summary.items():
            fh.write(f"\"{g}\",{s['instances']},{s['reference_exact']},{s['optimal']},{s['optimal_rate']:.4f},"
                     f"{s['suboptimal']},{s['open']},{s['max_ratio']:.6f},{s['mean_ratio']:.6f},"
                     f"{s['total_excess']},{s['total_lower']},{s['c11_exact']},{s['c11_mean_core_fraction']:.4f},{s['c1_exhausted']},{s['c9_exact']},{s['c9_optimal']},{s['c9_max_layers']},"
                     + ",".join(str(s["optimal_by_strategy"][k]) for k in NAMES) + ","
                     + ",".join(str(s["optimal_by_refined"][k]) for k in BASE)
                     + f",{s['siriaisa_seconds']:.2f},{s['milp_seconds']:.2f}\n")
    (OUT_DIR / "car_bipartite.json").write_text(json.dumps(
        {"siriaisa_version": siriaisa.__version__, "environment": environment(), "seed": SEED,
         "candidates": CANDIDATE_NAMES, "milp_time_limit": args.milp_time_limit, "timeout": args.timeout,
         "max_n": args.max_n, "quick": args.quick, "skipped_instances": skipped,
         "summary": summary, "rows": rows, "elapsed_seconds": time.time() - started},
        indent=2, default=str), encoding="utf-8")
    print_table(summary)


if __name__ == "__main__":
    main()
