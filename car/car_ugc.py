"""car/ hardness study: Siriaisa on the graph families on which MIDS is known to be hard.

What is known (the instances below follow these reductions):

* Minimum Independent Dominating Set (MIDS) cannot be approximated within
  n^(1-eps) unless P = NP (Halldorsson 1993, doi:10.1016/0020-0190(93)90022-2).
* In graphs of maximum degree B it is NP-hard to approximate within delta * B
  for an absolute delta > 0 and every B >= B_0, also on *bipartite* graphs, and
  within 1 + 1/680 (B = 3), 1 + 1/293 (B = 4, bipartite), 1 + 1/151 (B = 5,
  bipartite) (Chlebik and Chlebikova, Information and Computation 206 (2008),
  doi:10.1016/j.ic.2008.07.003). These are NP-hardness results (no UGC needed).
  Their proofs use two reductions, both built here:

  - SC-IDS (Claim 5, bipartite, degree <= B): the incidence graph of a set
    system (sets S, elements U), with a path S - S' - S'' attached to every set;
    ids(G) = sc + |S|.
  - VC-IDS (Claim 6, degree <= max(B, 3)): every edge uv of a graph H is
    replaced by a gadget with 6 vertices and 8 edges; ids(h(H)) = vc(H) + 2m.
    The paper gives the gadget in a figure; the gadget used here is
    u - a - b - c - d - v plus the square b - x - y - c, for which the identity
    holds exactly: an independent dominating set meets a gadget in at least 2
    vertices if it contains u or v and in at least 3 otherwise (x and y are
    dominated only from {b, x, y, c}; a and d need u, v or a gadget vertex), so
    |D| >= |D & V(H)| + 2m + #(edges of H with no end in D) >= vc(H) + 2m; and a
    vertex cover C plus {a_u, y} (v in C only), {x, c} (u, v in C) per gadget is
    an independent dominating set of size |C| + 2m. Hence every independent
    dominating set D of h(H) also *decodes* to a vertex cover of H of size at
    most |D| - 2m: D & V(H) plus one end of every edge with no end in D.

* UGC: no UGC-hardness result specific to MIDS is used or claimed here (the
  NP-hardness results above are unconditional). Vertex cover is UG-hard to
  approximate within 2 - eps (Khot and Regev, doi:10.1016/j.jcss.2007.06.019).
  Through VC-IDS, the Khot-Regev YES-type instances H (generated exactly as in
  Esperanza's car/ugc_candidate.py and Salvador's Part D, same seed and names)
  give MIDS instances h(H), and the vertex cover decoded from Siriaisa's answer
  on h(H) is a vertex-cover algorithm on the UG-hard instances; its ratio
  against certified bounds on tau(H) is reported.

Families

* ``vcids_kr``    VC-IDS of Khot-Regev YES instances (planted UG labelling);
* ``vcids_reg3``  VC-IDS of random 3-regular graphs (the degree-3 family of
                  Theorem 9: h(H) has maximum degree 3);
* ``vcids_reg4``, ``vcids_reg5`` VC-IDS of random 4- and 5-regular graphs
                  (maximum degree 4, 5);
* ``scids_B``     SC-IDS of random set systems with sets of size B - 1 and
                  every element in at most B sets (bipartite, maximum degree B),
                  B = 4, 5, 6.

Certified bounds (the optimum is never given to Siriaisa)

* VC-IDS: L_tau <= tau(H) <= U_tau with L_tau = max(ceil(LP), MILP dual
  bound) and U_tau = min(MILP primal, planted cover, decoded cover); then
  L = L_tau + 2m <= ids(h(H)) <= U_tau + 2m.
* SC-IDS: the set-cover MILP gives L_sc <= sc <= U_sc; L = L_sc + |S|.

Status of Siriaisa's |D|: ``optimal`` when |D| = L (then ids = L), ``suboptimal``
when the optimum is proved (L = U) and |D| > L, ``open`` otherwise (|D| / L is
then an upper bound on the ratio). For VC-IDS the decoded
cover C_D (|C_D| <= |D| - 2m) is reported with |C_D| / L_tau, and ``vc_optimal``
when |C_D| = L_tau. ``--self-check`` also solves the MIDS MILP directly on the
small instances and checks ids(h(H)) = vc(H) + 2m and ids(G) = sc + |S|.

Run from the repository root (the repository's own siriaisa/ is used, so the
package does not need to be installed):

    python car/car_ugc.py                  # default sizes
    python car/car_ugc.py --quick          # small sizes only
    python car/car_ugc.py --self-check     # also verify the two identities by MILP
    python car/car_ugc.py --ip-time 120 --timeout 1800

Every instance is built and evaluated in a worker process with a time limit
(``--timeout``, default 1200 s); an instance that exceeds it is skipped and
listed in the JSON file.

Outputs (in car/results/): car_ugc.csv (one row per instance),
car_ugc_summary.csv (one row per family) and car_ugc.json.
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

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import scipy  # noqa: E402
from scipy.optimize import Bounds, LinearConstraint, milp  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402

import siriaisa  # noqa: E402
from siriaisa.algorithm import find_independent_dominating_set, verify_independent_dominating_set  # noqa: E402

SEED = 20261009  # the seed of Salvador's Part D and Esperanza's ugc_candidate.py
OUT_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_TIMEOUT = 1200.0
DEFAULT_IP_TIME = 60.0


# ----------------------------------------------------------------------------
# Khot-Regev YES-type instances (as in Esperanza's car/ugc_candidate.py)
# ----------------------------------------------------------------------------
def unique_games(N: int, d: int, k: int, eta: float, linear: bool, rng: random.Random):
    H = nx.random_regular_graph(d, N - (N * d) % 2, seed=rng.randrange(2**31))
    L = {v: rng.randrange(k) for v in H}
    cons = []
    for v, w in H.edges():
        if rng.random() < eta:
            if linear:
                c = rng.randrange(k)
                pi = tuple((i + c) % k for i in range(k))
            else:
                pi = list(range(k))
                rng.shuffle(pi)
                pi = tuple(pi)
        elif linear:
            c = (L[v] - L[w]) % k
            pi = tuple((i + c) % k for i in range(k))
        else:
            pi = list(range(k))
            rng.shuffle(pi)
            j = pi.index(L[v])
            pi[j], pi[L[w]] = pi[L[w]], pi[j]
            pi = tuple(pi)
        cons.append((v, w, pi))
    return H, cons, L


def sample_subset(k: int, p: float, rng: random.Random) -> int:
    x = 0
    for i in range(k):
        if rng.random() < p:
            x |= 1 << i
    return x


def apply_perm(y: int, pi: tuple[int, ...]) -> int:
    out = 0
    i = 0
    while y:
        if y & 1:
            out |= 1 << pi[i]
        y >>= 1
        i += 1
    return out


def khot_regev_yes(N: int, d: int, k: int, p: float, eta: float, s: int, linear: bool, rng: random.Random):
    """Sampled Khot-Regev graph of a planted UG instance: (H, planted vertex cover)."""
    Hc, cons, L = unique_games(N, d, k, eta, linear, rng)
    blocks = {v: [sample_subset(k, p, rng) for _ in range(s)] for v in Hc}
    vid = {}
    for v in Hc:
        for t in range(s):
            vid[(v, t)] = len(vid)
    H = nx.Graph()
    H.add_nodes_from(range(len(vid)))
    for v, w, pi in cons:
        bx = blocks[v]
        by = [apply_perm(y, pi) for y in blocks[w]]
        for a, x in enumerate(bx):
            for b, y in enumerate(by):
                if x & y == 0:
                    H.add_edge(vid[(v, a)], vid[(w, b)])
    I = {vid[(v, t)] for v in Hc for t in range(s) if blocks[v][t] >> L[v] & 1}
    for u in sorted(I):
        if u in I and any(w in I for w in H[u]):
            I.discard(u)
    return H, set(H.nodes()) - I


# ----------------------------------------------------------------------------
# Reductions
# ----------------------------------------------------------------------------
def vc_ids(H: nx.Graph) -> tuple[nx.Graph, list]:
    """VC-IDS: every edge uv of H replaced by u - a - b - c - d - v and b - x - y - c.
    Returns (h(H), gadgets) with gadgets[e] = (u, v, a, b, c, d, x, y) (node ids of h(H))."""
    G = nx.Graph()
    nodes = list(H.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    G.add_nodes_from(range(len(nodes)))
    nxt = len(nodes)
    gadgets = []
    for u, v in H.edges():
        a, b, c, d, x, y = range(nxt, nxt + 6)
        nxt += 6
        iu, iv = idx[u], idx[v]
        G.add_edges_from([(iu, a), (a, b), (b, c), (c, d), (d, iv), (b, x), (x, y), (y, c)])
        gadgets.append((iu, iv, a, b, c, d, x, y))
    return G, gadgets


def cover_to_ids(gadgets: list, C: set) -> set:
    """Independent dominating set of h(H) of size |C| + 2m from a vertex cover C (indices of H)."""
    D = set(C)
    for iu, iv, a, b, c, d, x, y in gadgets:
        if iu in C and iv in C:
            D |= {x, c}
        elif iv in C:  # u not in C: a dominates u
            D |= {a, y}
        else:  # only u in C: d dominates v
            D |= {d, x}
    return D


def ids_to_cover(gadgets: list, n_h: int, D: set) -> set:
    """Vertex cover of H of size at most |D| - 2m: D & V(H) plus one end of every edge with no end in D."""
    C = {v for v in D if v < n_h}
    for iu, iv, *_ in gadgets:
        if iu not in C and iv not in C:
            C.add(iu)
    return C


def random_set_system(n_sets: int, B: int, rng: random.Random) -> list[list[int]]:
    """Sets of size B - 1 over a universe in which every element lies in at most B sets
    (configuration model: each element offers B slots)."""
    k = B - 1
    n_elem = max(k, math.ceil(n_sets * k / B))
    slots = [e for e in range(n_elem) for _ in range(B)]
    rng.shuffle(slots)
    sets: list[list[int]] = []
    pos = 0
    for _ in range(n_sets):
        S: list[int] = []
        while len(S) < k and pos < len(slots):
            e = slots[pos]
            pos += 1
            if e not in S:
                S.append(e)
        if S:
            sets.append(S)
    covered = {e for S in sets for e in S}
    remap = {e: i for i, e in enumerate(sorted(covered))}
    return [[remap[e] for e in S] for S in sets]


def sc_ids(sets: list[list[int]]) -> nx.Graph:
    """SC-IDS: element nodes ('e', i), set nodes ('s', j) with the path ('s', j) - ('t', j) - ('u', j)."""
    G = nx.Graph()
    for j, S in enumerate(sets):
        G.add_edge(("s", j), ("t", j))
        G.add_edge(("t", j), ("u", j))
        for e in S:
            G.add_edge(("s", j), ("e", e))
    return G


# ----------------------------------------------------------------------------
# Integer programs (reference only)
# ----------------------------------------------------------------------------
def _milp(c, A, lb, ub, time_limit):
    res = milp(c=c, integrality=np.ones(len(c)), bounds=Bounds(0, 1),
               constraints=LinearConstraint(A, lb, ub), options={"time_limit": time_limit, "disp": False})
    value = int(round(res.fun)) if res.x is not None else None
    dual = getattr(res, "mip_dual_bound", None)
    lower = value if res.status == 0 and value is not None else (
        math.ceil(dual - 1e-6) if dual is not None and np.isfinite(dual) else 0)
    return value, lower, res.status == 0


def vc_bounds(H: nx.Graph, time_limit: float) -> tuple[int, int | None, bool]:
    """(certified lower bound on tau, MILP primal, proved optimal); LP bound included."""
    nodes = list(H.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    E = list(H.edges())
    if not E:
        return 0, 0, True
    rows = [r for r in range(len(E)) for _ in (0, 1)]
    cols = [idx[x] for e in E for x in e]
    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(E), len(nodes))).tocsr()
    value, lower, opt = _milp(np.ones(len(nodes)), A, np.ones(len(E)), np.full(len(E), np.inf), time_limit)
    # LP relaxation = half a maximum matching of the bipartite double cover.
    D = nx.Graph()
    D.add_edges_from(((u, 0), (v, 1)) for u, v in E)
    D.add_edges_from(((v, 0), (u, 1)) for u, v in E)
    M = nx.bipartite.hopcroft_karp_matching(D, top_nodes=[(v, 0) for v in nodes if (v, 0) in D])
    lp = math.ceil((len(M) // 2) / 2 - 1e-9)
    return max(lower, lp), value, opt


def sc_bounds(sets: list[list[int]], time_limit: float) -> tuple[int, int | None, bool]:
    """(certified lower bound on sc, MILP primal, proved optimal)."""
    n_elem = 1 + max((e for S in sets for e in S), default=-1)
    rows = [e for S in sets for e in S]
    cols = [j for j, S in enumerate(sets) for _ in S]
    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_elem, len(sets))).tocsr()
    value, lower, opt = _milp(np.ones(len(sets)), A, np.ones(n_elem), np.full(n_elem, np.inf), time_limit)
    return lower, value, opt


def ids_milp(G: nx.Graph, time_limit: float) -> tuple[int, int | None, bool]:
    nodes = list(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    E = list(G.edges())
    rows, cols = [], []
    for v in nodes:
        for u in (v, *G.adj[v]):
            rows.append(idx[v])
            cols.append(idx[u])
    for k, (u, v) in enumerate(E):
        rows += [n + k, n + k]
        cols += [idx[u], idx[v]]
    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n + len(E), n)).tocsr()
    lb = np.concatenate([np.ones(n), np.full(len(E), -np.inf)])
    ub = np.concatenate([np.full(n, np.inf), np.ones(len(E))])
    value, lower, opt = _milp(np.ones(n), A, lb, ub, time_limit)
    return lower, value, opt


# ----------------------------------------------------------------------------
# Instances
# ----------------------------------------------------------------------------
def instances(quick: bool):
    """(name, family, params)."""
    blocks = (40,) if quick else (40, 80)
    for N in blocks:
        for r in range(2 if quick else 3):
            base = dict(N=N, d=4, k=4, p=0.45, eta=0.0, s=16, linear=False)
            yield f"kr_base_N{N}_r{r}", "vcids_kr", base
            yield f"kr_k3_N{N}_r{r}", "vcids_kr", {**base, "k": 3}
            yield f"kr_p0.35_N{N}_r{r}", "vcids_kr", {**base, "p": 0.35}
            yield f"kr_eta0.1_N{N}_r{r}", "vcids_kr", {**base, "eta": 0.1}
            yield f"kr_linear_N{N}_r{r}", "vcids_kr", {**base, "linear": True}
    sizes = (50, 100, 200) if quick else (50, 100, 200, 500, 1000)
    for n in sizes:
        for r in range(3):
            for d in (3, 4, 5):
                yield f"reg{d}_n{n}_r{r}", f"vcids_reg{d}", {"n": n, "d": d}
    set_counts = (30, 60) if quick else (30, 60, 120, 250, 500)
    for ns in set_counts:
        for r in range(3):
            for B in (4, 5, 6):
                yield f"sc_B{B}_s{ns}_r{r}", f"scids_B{B}", {"sets": ns, "B": B}


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def run_instance(name: str, family: str, params: dict, args) -> dict:
    rng = random.Random(f"{SEED}:{name}")
    row = {"name": name, "family": family, "params": params}
    if family.startswith("vcids"):
        planted = None
        if family == "vcids_kr":
            H, planted = khot_regev_yes(rng=rng, **params)
        else:
            H = nx.random_regular_graph(params["d"], params["n"], seed=rng.randrange(2**31))
        H = nx.convert_node_labels_to_integers(H)
        n_h, m_h = H.number_of_nodes(), H.number_of_edges()
        G, gadgets = vc_ids(H)
        t0 = time.perf_counter()
        D = find_independent_dominating_set(G)
        elapsed = time.perf_counter() - t0
        if not verify_independent_dominating_set(G, D):
            raise AssertionError(f"{name}: Siriaisa's set is not an independent dominating set")
        C_D = ids_to_cover(gadgets, n_h, D)
        if not all(u in C_D or v in C_D for u, v in H.edges()) or len(C_D) > len(D) - 2 * m_h:
            raise AssertionError(f"{name}: decoded cover invalid")
        t1 = time.perf_counter()
        L_tau, ip_value, ip_opt = vc_bounds(H, args.ip_time)
        ip_seconds = time.perf_counter() - t1
        uppers = [len(C_D)] + ([ip_value] if ip_value is not None else []) + ([len(planted)] if planted else [])
        U_tau = min(uppers)
        if planted is not None:
            Dp = cover_to_ids(gadgets, planted)
            if not verify_independent_dominating_set(G, Dp) or len(Dp) != len(planted) + 2 * m_h:
                raise AssertionError(f"{name}: cover-to-IDS construction failed")
        lower = L_tau + 2 * m_h
        upper = min(len(D), U_tau + 2 * m_h)
        if len(D) < lower or len(C_D) < L_tau:
            raise AssertionError(f"{name}: below the certified lower bound -- impossible")
        row.update({
            "H_n": n_h, "H_m": m_h, "H_max_degree": max((d for _, d in H.degree()), default=0),
            "n": G.number_of_nodes(), "m": G.number_of_edges(),
            "max_degree": max((d for _, d in G.degree()), default=0),
            "bipartite": nx.is_bipartite(G),
            "siriaisa": len(D), "lower": lower, "upper": upper,
            "tau_lower": L_tau, "tau_upper": U_tau, "tau_ip_optimal": ip_opt,
            "planted_cover": len(planted) if planted else None,
            "decoded_cover": len(C_D), "vc_ratio": len(C_D) / L_tau if L_tau else 1.0,
            "vc_optimal": len(C_D) == L_tau,
        })
        if args.self_check and G.number_of_nodes() <= args.self_check_max_n:
            lo, val, opt = ids_milp(G, args.ip_time)
            row["self_check"] = {"ids_lower": lo, "ids_value": val, "ids_optimal": opt}
            if opt and ip_opt and val != ip_value + 2 * m_h:
                raise AssertionError(f"{name}: ids(h(H)) = {val} differs from vc(H) + 2m = {ip_value + 2 * m_h}")
    else:
        sets = random_set_system(params["sets"], params["B"], rng)
        G = nx.convert_node_labels_to_integers(sc_ids(sets))
        t0 = time.perf_counter()
        D = find_independent_dominating_set(G)
        elapsed = time.perf_counter() - t0
        if not verify_independent_dominating_set(G, D):
            raise AssertionError(f"{name}: Siriaisa's set is not an independent dominating set")
        t1 = time.perf_counter()
        L_sc, sc_value, sc_opt = sc_bounds(sets, args.ip_time)
        ip_seconds = time.perf_counter() - t1
        lower = L_sc + len(sets)
        upper = min(len(D), sc_value + len(sets)) if sc_value is not None else len(D)
        if len(D) < lower:
            raise AssertionError(f"{name}: below the certified lower bound -- impossible")
        row.update({
            "sets": len(sets), "elements": 1 + max(e for S in sets for e in S),
            "n": G.number_of_nodes(), "m": G.number_of_edges(),
            "max_degree": max((d for _, d in G.degree()), default=0),
            "bipartite": nx.is_bipartite(G),
            "siriaisa": len(D), "lower": lower, "upper": upper,
            "sc_lower": L_sc, "sc_upper": sc_value, "sc_ip_optimal": sc_opt,
        })
        if args.self_check and G.number_of_nodes() <= args.self_check_max_n:
            lo, val, opt = ids_milp(G, args.ip_time)
            row["self_check"] = {"ids_lower": lo, "ids_value": val, "ids_optimal": opt}
            if opt and sc_opt and val != sc_value + len(sets):
                raise AssertionError(f"{name}: ids(G) = {val} differs from sc + |S| = {sc_value + len(sets)}")
    row["ratio"] = row["siriaisa"] / row["lower"] if row["lower"] else 1.0
    if row["siriaisa"] == row["lower"]:
        row["status"] = "optimal"
    elif row["lower"] == row["upper"]:
        row["status"] = "suboptimal"  # the optimum is proved (L = U) and Siriaisa exceeds it
    else:
        row["status"] = "open"
    row["elapsed_seconds"] = elapsed
    row["ip_seconds"] = ip_seconds
    extra = (f" tau in [{row['tau_lower']},{row['tau_upper']}] C_D={row['decoded_cover']} "
             f"vc_ratio<={row['vc_ratio']:.4f}") if family.startswith("vcids") else \
        f" sc in [{row['sc_lower']},{row['sc_upper']}]"
    print(f"  [hard] {family:12} {name:24} n={row['n']:>7,} Delta={row['max_degree']:>3}"
          f" ids in [{row['lower']},{row['upper']}] siriaisa={row['siriaisa']:>7,}"
          f" ratio<={row['ratio']:.6f} {row['status']:8}{extra} {elapsed:6.1f}s", flush=True)
    return row


def _job(name: str, args) -> dict:
    for nm, family, params in instances(args.quick):
        if nm == name:
            return run_instance(nm, family, params, args)
    raise KeyError(name)


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


def summarise(rows: list[dict]) -> dict:
    out = {}
    for f in sorted({r["family"] for r in rows}) + ["ALL"]:
        g = rows if f == "ALL" else [r for r in rows if r["family"] == f]
        k = len(g)
        vc = [r for r in g if "vc_ratio" in r]
        out[f] = {
            "instances": k,
            "optimal": sum(r["status"] == "optimal" for r in g),
            "suboptimal": sum(r["status"] == "suboptimal" for r in g),
            "open": sum(r["status"] == "open" for r in g),
            "max_ratio": max((r["ratio"] for r in g), default=1.0),
            "mean_ratio": sum(r["ratio"] for r in g) / k if k else 1.0,
            "max_degree": max((r["max_degree"] for r in g), default=0),
            "bipartite": all(r["bipartite"] for r in g),
            "vc_instances": len(vc),
            "vc_optimal": sum(r["vc_optimal"] for r in vc),
            "max_vc_ratio": max((r["vc_ratio"] for r in vc), default=None),
            "siriaisa_seconds": sum(r["elapsed_seconds"] for r in g),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="small sizes only")
    ap.add_argument("--ip-time", type=float, default=DEFAULT_IP_TIME,
                    help=f"time limit of each reference integer program (default {DEFAULT_IP_TIME:g} s)")
    ap.add_argument("--self-check", action="store_true",
                    help="also solve the MIDS integer program on small instances and check the identities")
    ap.add_argument("--self-check-max-n", type=int, default=3000,
                    help="largest instance (vertices) for --self-check (default 3000)")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                    help=f"per-instance time limit in seconds (default {DEFAULT_TIMEOUT:g}; 0 = none)")
    args = ap.parse_args()

    print(f"siriaisa {siriaisa.__version__} from {Path(siriaisa.__file__).resolve().parent}", flush=True)
    started = time.time()
    rows, skipped = [], []
    with InstanceRunner(args.timeout) as runner:
        for name, family, _ in instances(args.quick):
            row = runner.run(_job, name, args)
            if row is None:
                print(f"  [hard] {family:12} {name:24} skipped (more than {args.timeout:g} s)", flush=True)
                skipped.append(name)
            else:
                rows.append(row)
    summary = summarise(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["family", "name", "n", "m", "max_degree", "bipartite", "siriaisa", "lower", "upper", "ratio", "status",
            "H_n", "H_m", "H_max_degree", "tau_lower", "tau_upper", "tau_ip_optimal", "planted_cover",
            "decoded_cover", "vc_ratio", "vc_optimal", "sets", "elements", "sc_lower", "sc_upper", "sc_ip_optimal",
            "elapsed_seconds", "ip_seconds"]
    with (OUT_DIR / "car_ugc.csv").open("w", encoding="utf-8") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join("" if r.get(c) is None else
                              (f"{r[c]:.6f}" if isinstance(r.get(c), float) else str(r[c])) for c in cols) + "\n")
    scols = ["instances", "optimal", "suboptimal", "open", "max_ratio", "mean_ratio", "max_degree", "bipartite", "vc_instances",
             "vc_optimal", "max_vc_ratio", "siriaisa_seconds"]
    with (OUT_DIR / "car_ugc_summary.csv").open("w", encoding="utf-8") as fh:
        fh.write("family," + ",".join(scols) + "\n")
        for f, s in summary.items():
            fh.write(f + "," + ",".join("" if s[c] is None else
                                        (f"{s[c]:.6f}" if isinstance(s[c], float) else str(s[c])) for c in scols) + "\n")
    (OUT_DIR / "car_ugc.json").write_text(json.dumps(
        {"siriaisa_version": siriaisa.__version__,
         "environment": {"python": platform.python_version(), "platform": platform.platform(),
                         "networkx": nx.__version__, "scipy": scipy.__version__, "numpy": np.__version__},
         "seed": SEED, "ip_time": args.ip_time, "timeout": args.timeout, "quick": args.quick,
         "self_check": args.self_check, "skipped_instances": skipped, "summary": summary, "rows": rows,
         "elapsed_seconds": time.time() - started}, indent=2, default=str), encoding="utf-8")

    print("\nMIDS hardness families: Siriaisa against certified lower bounds")
    print(f"{'family':14} {'inst':>5} {'opt':>5} {'sub':>5} {'open':>5} {'max ratio':>10} {'Delta':>6} {'bip':>4}"
          f" {'vc opt':>7} {'max vc ratio':>13}")
    for f, s in summary.items():
        mvr = "" if s["max_vc_ratio"] is None else f"{s['max_vc_ratio']:.6f}"
        print(f"{f:14} {s['instances']:>5} {s['optimal']:>5} {s['suboptimal']:>5} {s['open']:>5} {s['max_ratio']:10.6f}"
              f" {s['max_degree']:>6} {str(s['bipartite'])[0]:>4} {s['vc_optimal'] if s['vc_instances'] else '':>7} {mvr:>13}")
    print("opt = Siriaisa's size equals the certified lower bound (then it is optimal); sub = the optimum is proved "
          "and Siriaisa exceeds it; open = otherwise (ratio is an upper bound). vc = the vertex cover of H decoded from Siriaisa's set on h(H).")


if __name__ == "__main__":
    main()
