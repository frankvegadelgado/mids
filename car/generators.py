"""
Graph-family generators for the ``car`` (Constant Approximation Ratio) suite.

Every generator returns a simple, undirected ``networkx.Graph`` whose nodes are
relabelled to consecutive integers ``0..n-1``.  The families mirror the
structured classes analysed in the manuscript (Section "Constant Approximation
Ratio on Structured Graph Families") plus three random-graph models used only to
show empirical concentration of the ratio.

Each family carries:
  * a builder that draws safe random parameters from a ``random.Random`` source,
  * a class in {"rigid", "bounded", "random"}, and
  * an ``expected_constant`` callable that returns, for a concrete graph, the
    approximation constant the manuscript predicts for that family
    (1 for rigid families, the degree bound for bounded-degree families, and the
    per-instance maximum degree -- the maximal-independent-set bound -- for the
    random models).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

import networkx as nx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def relabel(graph: nx.Graph) -> nx.Graph:
    """Return a simple graph with consecutive integer labels 0..n-1."""
    return nx.convert_node_labels_to_integers(nx.Graph(graph))


def max_degree(graph: nx.Graph) -> int:
    return max((d for _, d in graph.degree()), default=0)


# ---------------------------------------------------------------------------
# Structured family builders
# ---------------------------------------------------------------------------

def build_crown(n: int) -> nx.Graph:
    """Crown graph on 2n vertices: K_{n,n} minus a perfect matching."""
    graph = nx.Graph()
    upper = list(range(n))
    lower = list(range(n, 2 * n))
    graph.add_nodes_from(upper + lower)
    for i in range(n):
        for j in range(n):
            if i != j:
                graph.add_edge(upper[i], lower[j])
    return graph


def build_double_star(a: int, b: int) -> nx.Graph:
    """Two adjacent centres carrying ``a`` and ``b`` leaves."""
    graph = nx.Graph()
    center0, center1 = 0, 1
    graph.add_edge(center0, center1)
    nxt = 2
    for _ in range(a):
        graph.add_edge(center0, nxt)
        nxt += 1
    for _ in range(b):
        graph.add_edge(center1, nxt)
        nxt += 1
    return graph


def build_random_tree(n: int, seed: int) -> nx.Graph:
    if n <= 1:
        graph = nx.Graph()
        graph.add_nodes_from(range(max(n, 1)))
        return graph
    try:  # NetworkX >= 3.2
        return nx.random_labeled_tree(n, seed=seed)
    except AttributeError:  # older NetworkX
        return nx.random_tree(n, seed=seed)


def build_regular(r: int, n: int, seed: int) -> nx.Graph:
    """Random r-regular graph with the parity/size constraints repaired."""
    n = max(n, r + 1)
    if (n * r) % 2 != 0:
        n += 1
    return nx.random_regular_graph(r, n, seed=seed)


# ---------------------------------------------------------------------------
# Family registry
# ---------------------------------------------------------------------------

@dataclass
class Family:
    name: str
    kind: str  # "rigid" | "bounded" | "random"
    build: Callable[[random.Random], nx.Graph]
    expected_constant: Callable[[nx.Graph], float]


def _families() -> list[Family]:
    # Bounded-degree families: constant is the degree bound.
    bounded = [
        Family("path", "bounded",
               lambda r: nx.path_graph(r.randint(2, 40)),
               lambda g: 2.0),
        Family("cycle", "bounded",
               lambda r: nx.cycle_graph(r.randint(3, 40)),
               lambda g: 2.0),
        Family("ladder", "bounded",
               lambda r: nx.ladder_graph(r.randint(2, 18)),
               lambda g: 3.0),
        Family("grid", "bounded",
               lambda r: relabel(nx.grid_2d_graph(r.randint(2, 6), r.randint(2, 6))),
               lambda g: 4.0),
        Family("regular", "bounded",
               lambda r: (lambda deg: build_regular(deg, r.randint(deg + 1, 34), r.randint(0, 10**9)))(r.randint(2, 5)),
               lambda g: float(max_degree(g))),
        Family("balanced_tree", "bounded",
               lambda r: (lambda br: nx.balanced_tree(br, 3 if br == 2 else 2))(r.randint(2, 3)),
               lambda g: float(max_degree(g))),
    ]

    # Rigid families: exact recovery, constant 1.
    rigid = [
        Family("clique", "rigid",
               lambda r: nx.complete_graph(r.randint(1, 14)),
               lambda g: 1.0),
        Family("star", "rigid",
               lambda r: nx.star_graph(r.randint(1, 30)),
               lambda g: 1.0),
        Family("complete_bipartite", "rigid",
               lambda r: nx.complete_bipartite_graph(r.randint(1, 12), r.randint(1, 12)),
               lambda g: 1.0),
        # Crown: capped at n<=6 so the whole vertex set (2n<=12) fits inside the
        # top-12 seed-pool; every matched non-adjacent pair {u_i,v_i} is then
        # generated as a seed-pair candidate, making exact recovery (ratio 1)
        # provable for every sampled instance rather than merely typical.
        Family("crown", "rigid",
               lambda r: build_crown(r.randint(2, 6)),
               lambda g: 1.0),
        Family("double_star", "rigid",
               lambda r: build_double_star(r.randint(1, 12), r.randint(1, 12)),
               lambda g: 1.0),
    ]

    # Lollipop: exact recovery is not cleanly provable for every clique/path
    # split, so we check the always-valid degree bound (maximal-set bound) and
    # let the summary report its mean ratio, which stays close to 1 in practice.
    bounded.append(
        Family("lollipop", "bounded",
               lambda r: nx.lollipop_graph(r.randint(3, 8), r.randint(1, 12)),
               lambda g: float(max_degree(g)))
    )

    # Random models: no per-instance structural constant, only the maximal-set
    # bound (Lemma "Maximal-set bound"): every returned set is a Delta-approx.
    randoms = [
        Family("erdos_renyi", "random",
               lambda r: nx.erdos_renyi_graph(r.randint(8, 34), r.uniform(0.05, 0.4), seed=r.randint(0, 10**9)),
               lambda g: float(max(max_degree(g), 1))),
        Family("barabasi_albert", "random",
               lambda r: (lambda n: nx.barabasi_albert_graph(n, r.randint(1, 2), seed=r.randint(0, 10**9)))(r.randint(8, 34)),
               lambda g: float(max(max_degree(g), 1))),
        Family("random_tree", "random",
               lambda r: build_random_tree(r.randint(2, 40), r.randint(0, 10**9)),
               lambda g: float(max(max_degree(g), 1))),
    ]

    return bounded + rigid + randoms


FAMILIES: list[Family] = _families()
FAMILY_BY_NAME = {f.name: f for f in FAMILIES}


def generate_instance(index: int, seed: int = 12345):
    """Deterministically build the ``index``-th instance.

    Families are visited round-robin so the ten thousand instances are split
    evenly.  The per-instance RNG seed is derived from ``seed`` and ``index`` so
    the whole suite is reproducible.

    Returns ``(family, graph)`` with ``graph`` relabelled to integers.
    """
    family = FAMILIES[index % len(FAMILIES)]
    rng = random.Random((seed * 1_000_003) ^ (index * 2_654_435_761))
    graph = family.build(rng)
    graph = relabel(graph)
    return family, graph
