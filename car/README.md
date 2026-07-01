# car — Constant Approximation Ratio suite

Large-scale empirical counterpart to the paper's section *Constant Approximation
Ratio on Structured Graph Families*. It draws **10,000 instances** from the
structured families and three random-graph models, runs **Siriaisa**
(`siriaisa.algorithm.find_independent_dominating_set`), and compares each result
against the **exact SciPy MILP optimum**.

## Run

```bash
cd car
python run_car.py                 # 10,000 instances (default)
python run_car.py --count 200     # quick smoke run
python run_car.py --dump-dimacs   # also write each instance under results/instances/
```

Requires the repository's dependencies (Python ≥ 3.12, NetworkX, NumPy, SciPy).
The runner adds the parent directory to `sys.path`, so it works from inside `car/`.

## What it checks

Each instance is tagged with the family constant the paper predicts:

| kind | families | expected ratio |
|------|----------|----------------|
| `rigid` | clique, star, complete_bipartite, crown, double_star, lollipop | **1** (exact recovery) |
| `bounded` | path (2), cycle (2), ladder (3), grid (4), regular (r), balanced_tree (Δ) | **≤ Δ** (maximal-set bound) |
| `random` | erdos_renyi, barabasi_albert, random_tree | **≤ Δ** (maximal-set bound only) |

A row is flagged `within_bound = 0` if the exact ratio exceeds the family
constant. With a correct implementation the count of violations is **0**:
rigid families report ratio 1, bounded/random families stay within their degree
bound.

## Outputs (`results/`)

- `car_results.csv` — one row per instance: `index, family, kind, n, m, delta,
  siriaisa_size, optimum, ratio, expected_constant, within_bound, siriaisa_ms, milp_ms`
- `car_summary.csv` — one row per family: `count, mean_ratio, max_ratio,
  expected_constant, mean_delta, violations`
- `instances/` — DIMACS files (only with `--dump-dimacs`)

Instances are kept small enough (n ≤ 40) that the exact MILP terminates, so every
reported ratio is exact rather than an LP-bound estimate. The whole run is
reproducible from `--seed`.

The full 10,000-instance run takes on the order of several minutes to tens of
minutes depending on the machine (each instance solves one LP-guided
approximation plus one exact MILP). Use `--count` for a fast smoke test first.
