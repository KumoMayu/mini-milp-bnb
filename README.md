# Mini MILP Branch-and-Bound Solver

A small Python prototype for mixed-integer linear programming (MILP). The LP relaxation backend and Branch-and-Bound loop are implemented in this repository, so the solver mechanics are inspectable. It is intended for learning and small bounded prototypes, not as a replacement for Gurobi, CPLEX, SCIP, or other production solvers.

The public modeling path is intentionally narrow: build a block MILP with `MILPProblem.from_blocks(...)`, solve it with `solve_milp(...)`, and inspect the returned `MILPResult`.

## Problem Form

The public modeling interface is the block MILP form:

```text
min / max c_x^T x + c_y^T y
s.t.      A x + B y <= b
          lb_x <= x <= ub_x
          lb_y <= y <= ub_y
```

`x` is the continuous-variable block. `y` is the integer or binary-variable block. Internally the problem is concatenated as:

```text
z = [x; y]
G = [A B]
c = [c_x; c_y]
```

and passed to the solver as:

```text
min / max c^T z
s.t.      G z <= b
          lb <= z <= ub
```

## Usage

Use `MILPProblem.from_blocks(...)` as the main interface:

```python
from solver import MILPProblem, solve_milp

problem = MILPProblem.from_blocks(
    c_x=c_x,
    c_y=c_y,
    A=A,
    B=B,
    b=b,
    x_lb=x_lb,
    x_ub=x_ub,
    y_lb=y_lb,
    y_ub=y_ub,
    y_types=["I", "B"],
    sense="min",
)

result = solve_milp(problem)
print(result.simple_summary())
```

The internal `MILPProblem.from_standard(...)` representation is available for tests and solver internals, but it is not the recommended public modeling entry point.

## Repository Contents

Public source files:

- `solver/`: MILP problem definition, active-set LP relaxation, B&B solver, and result structures.
- `examples/`: small block-interface MILP examples.
- `benchmarks/`: repeatable benchmark runner for the core examples.
- `verification/`: optional same-sample comparison with Gurobi.
- `tests/`: pytest suite for problem construction, LP relaxation, B&B behavior, and examples.

Local/private notes, old experiments, virtual environments, and reports are intentionally excluded from version control.

## Algorithm

- Original `min` models are converted internally to canonical maximization.
- Each B&B node solves an LP relaxation using active-set enumeration.
- The LP relaxation value is the node upper bound.
- The incumbent is the current best integer feasible solution, i.e. the primal lower bound for maximization.
- Nodes are pruned by infeasibility when the LP relaxation is infeasible.
- Nodes are pruned by bound when the node upper bound cannot improve the incumbent.
- Nodes are pruned by optimality when the LP solution already satisfies integer constraints.
- If no pruning is possible, the solver branches with floor/ceil bounds on a fractional integer variable.

The LP backend builds all node constraints as `M z <= q`, including original constraints and current node lower/upper bounds. It is suitable only for small prototypes; a simplex backend is a natural next step.

## Installation

Dependencies are intentionally limited to:

```text
numpy
pytest
```

Mac / zsh:

```zsh
cd /path/to/optimization-learning
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
cd C:\Users\DonghaoWu\Projects\optimization-learning
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Examples

```zsh
.venv/bin/python examples/fixed_charge_block.py
.venv/bin/python examples/general_integer_block.py
.venv/bin/python examples/unit_commitment_tiny.py
```

Core examples:

- `fixed_charge_block.py`: continuous production plus binary activation and linking constraints.
- `general_integer_block.py`: continuous resource plus general integer decisions.
- `unit_commitment_tiny.py`: tiny one-period unit commitment with dispatch and binary commitment.

## Benchmark

```zsh
.venv/bin/python benchmarks/run_core_cases.py
```

Output columns:

```text
case | status | objective | nodes | lp_solved | prune_inf | prune_bound | prune_opt | time_sec
```

Windows:

```powershell
.\.venv\Scripts\python.exe examples\fixed_charge_block.py
.\.venv\Scripts\python.exe benchmarks\run_core_cases.py
.\.venv\Scripts\python.exe -m pytest
```

## Optional Gurobi Comparison

```zsh
.venv/bin/python verification/compare_with_gurobi_optional.py
```

`gurobipy` is optional and is not listed in `requirements.txt`. If you want to run the comparison, install it separately in the environment you are using:

```zsh
python -m pip install gurobipy
```

If `gurobipy` is not installed, the script exits with a short message. The comparison only checks same-sample objective values and rough timing.

Recent same-sample results on Python 3.12:

| case | mini_obj | gurobi_obj | match | mini_nodes | mini_lp_solved |
|---|---:|---:|---|---:|---:|
| `fixed_charge_block` | 28.5 | 28.5 | True | 7 | 7 |
| `general_integer_block` | 19 | 19 | True | 3 | 3 |
| `unit_commitment_tiny` | 26 | 26 | True | 5 | 5 |

## Tests

```zsh
.venv/bin/python -m pytest
```

## Limitations

- Active-set LP backend is only suitable for small problems.
- No simplex backend yet.
- No presolve, cutting planes, or primal heuristics.
- No warm start or basis reuse between B&B nodes.
- No large-scale sparse matrix support.
