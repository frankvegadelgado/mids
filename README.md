# Siriaisa: Approximate Independent Dominating Set Solver

![To my mother who I love.](docs/siriaisa.jpg)

This work builds upon [Constant-Factor Approximation of Independent Dominating Sets on Structured Graph Families: The Siriaisa Algorithm](https://github.com/frankvegadelgado/mids).

---

# Overview of the Minimum Independent Dominating Set (MIDS)

## Definition:

An **independent dominating set** in a graph $G = (V, E)$ is a subset $D \subseteq V$ such that no two vertices in $D$ are adjacent and every vertex not in $D$ is adjacent to at least one vertex in $D$. The **minimum independent dominating set (MIDS)** is the smallest possible independent dominating set in terms of the number of vertices.

## Key Concepts:

1. **Graph Representation**:

   - $V$: Set of vertices.
   - $E$: Set of edges connecting the vertices.

2. **Independent Dominating Set**:

   - A set $D$ where no two vertices in $D$ are adjacent, and for every vertex $v \in V$, either $v \in D$ or $v$ is adjacent to some vertex in $D$.

3. **Minimum Independent Dominating Set**:
   - The independent dominating set with the smallest cardinality (i.e., the fewest number of vertices).

## Applications:

- **Network Design**: Ensuring coverage in wireless sensor networks.
- **Social Networks**: Identifying influential nodes.
- **Game Theory**: Strategies in certain types of games.
- **Biology**: Modeling protein-protein interaction networks.

## Computational Complexity:

- **NP-Hard**: Finding the minimum independent dominating set is computationally intensive for large graphs.
- **Approximation Algorithms**: Used to find near-optimal solutions in polynomial time.

## Algorithms:

1. **Greedy Algorithm**:

   - Builds a degree-four auxiliary graph, repairs the two lifted vertex seeds into maximal independent sets, and selects the smallest verified candidate.
   - Provides the implementation studied in the constant-approximation certificate theorem.

2. **Integer Linear Programming (ILP)**:

   - Formulates the problem as an optimization problem.
   - Solvable using ILP solvers for exact solutions, though computationally expensive.

3. **Heuristics and Metaheuristics**:
   - Genetic algorithms, simulated annealing, etc., for large-scale problems.

## Challenges:

- **Scalability**: Exact algorithms are infeasible for very large graphs.
- **Dynamic Graphs**: Maintaining a minimum independent dominating set in graphs that change over time.

## Research Directions:

- **Parallel Algorithms**: Leveraging multi-core processors and distributed computing.
- **Machine Learning**: Using learning-based approaches to predict dominating sets.
- **Hybrid Methods**: Combining exact and heuristic methods for better performance.

## Conclusion:

The minimum independent dominating set problem is a fundamental issue in graph theory with wide-ranging applications. While it is computationally challenging, various algorithms and heuristics provide practical solutions for different scenarios. Ongoing research continues to improve the efficiency and applicability of these methods.

---

## Problem Statement

Input: A Boolean Adjacency Matrix $M$.

Answer: Find a Minimum Independent Dominating Set.

### Example Instance: 5 x 5 matrix

|        | c1  | c2  | c3  | c4  | c5  |
| ------ | --- | --- | --- | --- | --- |
| **r1** | 0   | 0   | 1   | 0   | 1   |
| **r2** | 0   | 0   | 0   | 1   | 0   |
| **r3** | 1   | 0   | 0   | 0   | 1   |
| **r4** | 0   | 1   | 0   | 0   | 0   |
| **r5** | 1   | 0   | 1   | 0   | 0   |

The input for undirected graph is typically provided in [DIMACS](http://dimacs.rutgers.edu/Challenges) format. In this way, the previous adjacency matrix is represented in a text file using the following string representation:

```
p edge 5 4
e 1 3
e 1 5
e 2 4
e 3 5
```

This represents a 5x5 matrix in DIMACS format such that each edge $(v,w)$ appears exactly once in the input file and is not repeated as $(w,v)$. In this format, every edge appears in the form of

```
e W V
```

where the fields W and V specify the endpoints of the edge while the lower-case character `e` signifies that this is an edge descriptor line.

_Example Solution:_

Independent Dominating Set Found `1, 2`: Nodes `1` and `2` constitute an optimal solution.

---

# Compile and Environment

## Prerequisites

- Python >= 3.12

## Installation

```bash
pip install siriaisa
```

## Execution

1. Clone the repository:

   ```bash
   git clone https://github.com/frankvegadelgado/mids.git
   cd mids
   ```

2. Run the script:

   ```bash
   iris -i ./benchmarks/testMatrix1
   ```

   utilizing the `iris` command provided by Siriaisa's library to execute the Boolean adjacency matrix `mids\benchmarks\testMatrix1`. The file `testMatrix1` represents the example described herein. We also support `.xz`, `.lzma`, `.bz2`, and `.bzip2` compressed text files.

   **Example Output:**

   ```
   testMatrix1: Independent Dominating Set Found 1, 2
   ```

   This indicates nodes `1, 2` form an Independent Dominating Set.

---

## Independent Dominating Set Size

Use the `-c` flag to count the nodes in the Independent Dominating Set:

```bash
iris -i ./benchmarks/testMatrix2 -c
```

**Output:**

```
testMatrix2: Independent Dominating Set Size 2
```

---

# Command Options

Display help and options:

```bash
iris -h
```

**Output:**

```bash
usage: iris [-h] -i INPUTFILE [-a] [-b] [-c] [-v] [-l] [--version]

Solve the Approximate Independent Dominating Set for undirected graph encoded in DIMACS format.

options:
  -h, --help            show this help message and exit
  -i INPUTFILE, --inputFile INPUTFILE
                        input file path
  -a, --approximation   enable comparison with a polynomial-time approximation approach within a maximum degree factor
  -b, --bruteForce      enable comparison with the exponential-time brute-force approach
  -c, --count           calculate the size of the Independent Dominating Set
  -v, --verbose         enable verbose output
  -l, --log             enable file logging
  --version             show program's version number and exit
```

---

# Batch Execution

Batch execution allows you to solve multiple graphs within a directory consecutively.

To view available command-line options for the `batch_iris` command, use the following in your terminal or command prompt:

```bash
batch_iris -h
```

This will display the following help information:

```bash
usage: batch_iris [-h] -i INPUTDIRECTORY [-a] [-b] [-c] [-v] [-l] [--version]

Solve the Approximate Independent Dominating Set for all undirected graphs encoded in DIMACS format and stored in a directory.

options:
  -h, --help            show this help message and exit
  -i INPUTDIRECTORY, --inputDirectory INPUTDIRECTORY
                        Input directory path
  -a, --approximation   enable comparison with a polynomial-time approximation approach within a maximum degree factor
  -b, --bruteForce      enable comparison with the exponential-time brute-force approach
  -c, --count           calculate the size of the Independent Dominating Set
  -v, --verbose         enable verbose output
  -l, --log             enable file logging
  --version             show program's version number and exit
```

---

# Testing Application

A command-line utility named `test_iris` is provided for evaluating the Algorithm using randomly generated, large sparse matrices. It supports the following options:

```bash
usage: test_iris [-h] -d DIMENSION [-n NUM_TESTS] [-s SPARSITY] [-a] [-b] [-c] [-w] [-v] [-l] [--version]

The Siriaisa Testing Application using randomly generated, large sparse matrices.

options:
  -h, --help            show this help message and exit
  -d DIMENSION, --dimension DIMENSION
                        an integer specifying the dimensions of the square matrices
  -n NUM_TESTS, --num_tests NUM_TESTS
                        an integer specifying the number of tests to run
  -s SPARSITY, --sparsity SPARSITY
                        sparsity of the matrices (0.0 for dense, close to 1.0 for very sparse)
  -a, --approximation   enable comparison with a polynomial-time approximation approach within a maximum degree factor
  -b, --bruteForce      enable comparison with the exponential-time brute-force approach
  -c, --count           calculate the size of the Independent Dominating Set
  -w, --write           write the generated random matrix to a file in the current directory
  -v, --verbose         enable verbose output
  -l, --log             enable file logging
  --version             show program's version number and exit
```

---

# Reproducible Experiments

All experiments compare **Siriaisa** against an **exact SciPy MILP** optimum (`scipy.optimize.milp`, HiGHS). Siriaisa solves its own LP relaxation with `scipy.optimize.linprog` (HiGHS backend). Every instance is independently verified to be independent and dominating.

## Adversarial DIMACS suite (`experiments/`)

A small hand-built suite of structural traps (paths, cycles, stars, cliques, complete bipartite, crown, double star, grid, ladder, lollipop, and a ratio-1.5 trap):

```bash
cd experiments
python run_adversarial_milp.py
```

## Large-scale study: the `car` suite (`car/`)

The `car/` directory (**C**onstant **A**pproximation **R**atio) generates **10,000 instances** drawn from the structured graph families plus three random-graph models (Erdős–Rényi, Barabási–Albert, random regular), solves each with Siriaisa, and compares against the exact MILP optimum. Each instance is tagged with the approximation constant proved for its family, and any instance whose exact ratio exceeds that constant is flagged.

### Reproduce

```bash
cd car
python run_car.py                 # full 10,000 instances (default)
python run_car.py --count 200     # quick smoke run
python run_car.py --dump-dimacs   # also save each instance as DIMACS under car/results/instances/
```

Requirements: Python >= 3.12, NumPy >= 2.2.1, SciPy >= 1.15.0, NetworkX >= 3.4.2. Results are written to `car/results/` as per-instance `car_results.csv` and per-family `car_summary.csv`. The whole run is reproducible from `--seed` (default `12345`). Instances are kept to `n <= 40` so the exact MILP always terminates and every reported ratio is exact.

### Results

Run on a modern x86-64 laptop (Windows 11, single-threaded). Exact ratios versus MILP over all 10,000 instances:

| Family | Class | Instances | Mean ratio | Max ratio | Guarantee | Violations |
| --- | --- | ---: | ---: | ---: | :---: | ---: |
| Path `P_n` | bounded | 667 | 1.0000 | 1.0000 | ≤ 2 | 0 |
| Cycle `C_n` | bounded | 667 | 1.0000 | 1.0000 | ≤ 2 | 0 |
| Ladder `P2×Pn` | bounded | 667 | 1.0000 | 1.0000 | ≤ 3 | 0 |
| Grid `Pa×Pb` | bounded | 667 | 1.0000 | 1.0000 | ≤ 4 | 0 |
| `r`-regular | bounded | 667 | 1.0029 | 1.2000 | ≤ r | 0 |
| Balanced tree | bounded | 667 | 1.0000 | 1.0000 | ≤ Δ | 0 |
| Lollipop `Kc–Pp` | bounded | 667 | 1.0000 | 1.0000 | ≤ c | 0 |
| Clique `K_n` | rigid | 667 | 1.0000 | 1.0000 | = 1 | 0 |
| Star `K_{1,n}` | rigid | 667 | 1.0000 | 1.0000 | = 1 | 0 |
| Complete bipartite `K_{a,b}` | rigid | 667 | 1.0000 | 1.0000 | = 1 | 0 |
| Crown | rigid | 666 | 1.0000 | 1.0000 | = 1 | 0 |
| Double star `DS(a,b)` | rigid | 666 | 1.0000 | 1.0000 | = 1 | 0 |
| Erdős–Rényi | random | 666 | 1.0002 | 1.1429 | ≤ Δ | 0 |
| Barabási–Albert | random | 666 | 1.0000 | 1.0000 | ≤ Δ | 0 |
| Random tree | random | 666 | 1.0000 | 1.0000 | ≤ Δ | 0 |
| **All** | — | **10000** | **1.0002** | **1.2000** | — | **0** |

Key findings:

- **Zero** family-constant violations across all 10,000 instances.
- **9,985 / 10,000 (99.85%)** solved to exact optimality.
- Overall **mean ratio 1.0002**, and the largest ratio anywhere is **1.20**.
- The only 15 non-optimal instances are sparse `r`-regular graphs (14) and one Erdős–Rényi graph, with ratios between 1.111 and 1.200 — all comfortably within their maximal-independent-set degree bound. The worst case was a 26-vertex 5-regular graph (Siriaisa returned 6 vertices against the optimum 5).

This is the empirical counterpart of the paper's theory: the bounded-degree and random families never leave their degree constant, and the rigid families (cliques, stars, complete bipartite, crowns, double stars) are solved exactly.

---

# Code

- Python implementation by **Frank Vega**.

---

# Complexity

```diff
+ Siriaisa separates feasibility from the approximation certificate: every returned set is verified as independent and dominating, while a universal proof of the constant-approximation certificate would imply P = NP by known MIDS inapproximability results.
```

---

# License

- MIT License.
