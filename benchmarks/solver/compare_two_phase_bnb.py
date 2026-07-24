from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path
from statistics import median
from time import perf_counter


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.solver.cases import BATCH_CASES, CORE_CASES, SCALING_CASES
from solver import solve_milp


BACKENDS = ("active_set", "two_phase_simplex", "scipy_highs", "gurobi")
FIELDNAMES = [
    "suite",
    "case",
    "seed",
    "units",
    "backend",
    "status",
    "objective",
    "match_gurobi",
    "nodes",
    "lp_solved",
    "candidates_checked",
    "simplex_iterations",
    "lp_total_runtime_sec",
    "total_runtime_median_sec",
    "time_run_1_sec",
    "time_run_2_sec",
    "time_run_3_sec",
    "iteration_limit",
    "numerical_error",
    "note",
]

MAX_NODES = 200
MAX_LP_CANDIDATES = 250_000
MAX_LP_ITERATIONS = 10_000
TOL = 1e-7


def _fmt(value) -> str:
    if value is None:
        return ""
    return f"{float(value):.10g}"


def scipy_highs_available() -> bool:
    if importlib.util.find_spec("scipy.optimize") is None:
        return False
    from scipy.optimize import linprog

    return bool(linprog(c=[1.0], bounds=[(0.0, 1.0)], method="highs").success)


def gurobi_available() -> tuple[bool, str]:
    if importlib.util.find_spec("gurobipy") is None:
        return False, "gurobipy not installed"
    try:
        import gurobipy as gp

        model = gp.Model("license_check")
        model.Params.OutputFlag = 0
        variable = model.addVar(lb=0.0, ub=1.0)
        model.setObjective(variable, gp.GRB.MINIMIZE)
        model.optimize()
    except Exception as exc:
        return False, f"Gurobi unavailable: {exc}"
    return True, ""


def _mini_once(builder, backend: str) -> dict:
    problem = builder()
    start = perf_counter()
    result = solve_milp(
        problem,
        lp_backend=backend,
        use_matrix_presolve=True,
        max_nodes=MAX_NODES,
        max_lp_candidates=MAX_LP_CANDIDATES,
        max_lp_iterations=MAX_LP_ITERATIONS,
    )
    elapsed = perf_counter() - start
    return {
        "status": result.status,
        "objective": result.objective_value,
        "nodes": result.num_nodes,
        "lp_solved": result.num_lp_solved,
        "candidates_checked": result.num_lp_candidates_checked,
        "simplex_iterations": result.num_simplex_iterations,
        "lp_runtime_sec": result.lp_runtime_sec,
        "elapsed": elapsed,
    }


def _gurobi_once(builder) -> dict:
    import gurobipy as gp

    start = perf_counter()
    problem = builder()
    model = gp.Model(problem.name)
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    variables = []
    for index, variable_type in enumerate(problem.var_types):
        if variable_type == "B":
            vtype = gp.GRB.BINARY
        elif variable_type == "I":
            vtype = gp.GRB.INTEGER
        else:
            vtype = gp.GRB.CONTINUOUS
        variables.append(
            model.addVar(
                lb=float(problem.lb[index]),
                ub=float(problem.ub[index]),
                vtype=vtype,
                name=f"z{index}",
            )
        )
    model.setObjective(
        gp.quicksum(
            float(problem.c[index]) * variables[index]
            for index in range(problem.num_vars)
        ),
        gp.GRB.MAXIMIZE if problem.sense == "max" else gp.GRB.MINIMIZE,
    )
    for row in range(problem.num_constraints):
        model.addConstr(
            gp.quicksum(
                float(problem.G[row, column]) * variables[column]
                for column in range(problem.num_vars)
            )
            <= float(problem.h[row])
        )
    model.optimize()
    elapsed = perf_counter() - start
    if model.Status == gp.GRB.OPTIMAL:
        status = "optimal"
        objective = float(model.ObjVal)
    elif model.Status == gp.GRB.INFEASIBLE:
        status = "infeasible"
        objective = None
    elif model.Status == gp.GRB.TIME_LIMIT:
        status = "time_limit"
        objective = float(model.ObjVal) if model.SolCount else None
    else:
        status = f"gurobi_status_{model.Status}"
        objective = None
    return {
        "status": status,
        "objective": objective,
        "nodes": int(model.NodeCount),
        "lp_solved": None,
        "candidates_checked": None,
        "simplex_iterations": None,
        "lp_runtime_sec": None,
        "elapsed": elapsed,
    }


def _same_run_signature(first: dict, other: dict) -> bool:
    if first["status"] != other["status"]:
        return False
    if first["nodes"] != other["nodes"] or first["lp_solved"] != other["lp_solved"]:
        return False
    if first["objective"] is None or other["objective"] is None:
        return first["objective"] is other["objective"]
    return abs(float(first["objective"]) - float(other["objective"])) <= TOL


def benchmark_backend(
    suite: str,
    case_name: str,
    builder,
    backend: str,
    *,
    seed: int | None,
    units: int | None,
    warmups: int,
    repeats: int,
    available: bool,
    unavailable_note: str = "",
) -> dict[str, str]:
    if not available:
        row = {field: "" for field in FIELDNAMES}
        row.update(
            {
                "suite": suite,
                "case": case_name,
                "seed": "" if seed is None else str(seed),
                "units": "" if units is None else str(units),
                "backend": backend,
                "status": "SKIPPED",
                "iteration_limit": "False",
                "numerical_error": "False",
                "note": unavailable_note,
            }
        )
        return row

    solve_once = _gurobi_once if backend == "gurobi" else lambda case: _mini_once(case, backend)
    for _ in range(warmups):
        solve_once(builder)
    runs = [solve_once(builder) for _ in range(repeats)]
    first = runs[0]
    stable = all(_same_run_signature(first, run) for run in runs[1:])
    times = [float(run["elapsed"]) for run in runs]
    lp_times = [
        float(run["lp_runtime_sec"])
        for run in runs
        if run["lp_runtime_sec"] is not None
    ]
    row = {field: "" for field in FIELDNAMES}
    row.update(
        {
            "suite": suite,
            "case": case_name,
            "seed": "" if seed is None else str(seed),
            "units": "" if units is None else str(units),
            "backend": backend,
            "status": str(first["status"]),
            "objective": _fmt(first["objective"]),
            "nodes": "" if first["nodes"] is None else str(first["nodes"]),
            "lp_solved": "" if first["lp_solved"] is None else str(first["lp_solved"]),
            "candidates_checked": (
                "-" if first["candidates_checked"] is None or backend != "active_set"
                else str(first["candidates_checked"])
            ),
            "simplex_iterations": (
                "-" if first["simplex_iterations"] is None or backend != "two_phase_simplex"
                else str(first["simplex_iterations"])
            ),
            "lp_total_runtime_sec": "-" if not lp_times else f"{median(lp_times):.9f}",
            "total_runtime_median_sec": f"{median(times):.9f}",
            "iteration_limit": str(first["status"] == "iteration_limit"),
            "numerical_error": str(first["status"] == "numerical_error"),
            "note": "" if stable else "status/objective/node metrics varied across timed runs",
        }
    )
    for index in range(3):
        row[f"time_run_{index + 1}_sec"] = (
            f"{times[index]:.9f}" if index < len(times) else ""
        )
    return row


def all_cases(selected_suites: set[str]):
    if "core" in selected_suites:
        for name, builder in CORE_CASES:
            yield "core", name, builder, None, None
    if "scaling" in selected_suites:
        for name, builder in SCALING_CASES:
            units = int(name.rsplit("_", 1)[1])
            yield "scaling", name, builder, None, units
    if "batch" in selected_suites:
        for name, builder, units, seed in BATCH_CASES:
            yield "batch", name, builder, seed, units


def add_gurobi_matches(rows: list[dict[str, str]]) -> None:
    references = {
        row["case"]: float(row["objective"])
        for row in rows
        if row["backend"] == "gurobi"
        and row["status"] == "optimal"
        and row["objective"]
    }
    for row in rows:
        if row["status"] != "optimal" or not row["objective"] or row["case"] not in references:
            row["match_gurobi"] = ""
            continue
        row["match_gurobi"] = str(
            abs(float(row["objective"]) - references[row["case"]]) <= TOL
        )


def print_summary(rows: list[dict[str, str]]) -> None:
    print("suite | case | backend | status | objective | nodes | LP | iterations | candidates | median_sec | match")
    print("-" * 132)
    for row in rows:
        print(
            " | ".join(
                [
                    row["suite"],
                    row["case"],
                    row["backend"],
                    row["status"],
                    row["objective"] or "-",
                    row["nodes"] or "-",
                    row["lp_solved"] or "-",
                    row["simplex_iterations"] or "-",
                    row["candidates_checked"] or "-",
                    row["total_runtime_median_sec"] or "-",
                    row["match_gurobi"] or "-",
                ]
            )
        )


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Compare B&B LP backends with repeated timings.")
    parser.add_argument(
        "--suites",
        nargs="+",
        choices=["core", "scaling", "batch"],
        default=["core", "scaling", "batch"],
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "two_phase_simplex_bnb_results.csv",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> list[dict[str, str]]:
    args = parse_args(argv)
    if args.warmups < 0 or args.repeats <= 0:
        raise ValueError("warmups must be nonnegative and repeats must be positive")
    if args.repeats > 3:
        raise ValueError("the fixed CSV schema stores at most three timed runs")

    scipy_ok = scipy_highs_available()
    gurobi_ok, gurobi_note = gurobi_available()
    availability = {
        "active_set": (True, ""),
        "two_phase_simplex": (True, ""),
        "scipy_highs": (scipy_ok, "SciPy-HiGHS not available"),
        "gurobi": (gurobi_ok, gurobi_note),
    }
    rows = []
    for suite, case_name, builder, seed, units in all_cases(set(args.suites)):
        for backend in BACKENDS:
            available, note = availability[backend]
            row = benchmark_backend(
                suite,
                case_name,
                builder,
                backend,
                seed=seed,
                units=units,
                warmups=args.warmups,
                repeats=args.repeats,
                available=available,
                unavailable_note=note,
            )
            rows.append(row)
            print(
                f"completed {suite}/{case_name}/{backend}: "
                f"{row['status']} {row['total_runtime_median_sec']}"
            )

    add_gurobi_matches(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print_summary(rows)
    print(f"wrote {args.output}")
    return rows


if __name__ == "__main__":
    main()
