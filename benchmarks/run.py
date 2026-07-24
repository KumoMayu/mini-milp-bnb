from __future__ import annotations

import argparse
import csv
import importlib.util
import platform
import signal
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from benchmarks.cases import (
    GeneralLPProblem,
    available_families,
    build_case,
)
from benchmarks.config import LIMITS, seeds_for_scale
from solver import TwoPhaseTableauSimplexSolver, solve_milp


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIR = ROOT / "reports"
TOL = 1e-6
CORE_BACKENDS = ("custom", "scipy_highs", "gurobi")
LIMIT_STATUSES = {
    "candidate_limit",
    "iteration_limit",
    "node_limit",
    "time_limit",
}
CSV_FIELDS = [
    "case",
    "family",
    "scale",
    "seed",
    "category",
    "backend",
    "status",
    "termination",
    "expected_status",
    "objective",
    "best_bound",
    "gap",
    "match_gurobi",
    "build_time",
    "solve_time",
    "total_time",
    "nodes",
    "lp_calls",
    "lp_total_time",
    "simplex_iterations",
    "pivot_count",
    "phase_one_iterations",
    "phase_two_iterations",
    "infeasible_prunes",
    "bound_prunes",
    "residual",
    "integrality_violation",
    "num_variables",
    "num_integer_variables",
    "num_constraints",
    "density",
    "note",
]


class BenchmarkTimeout(TimeoutError):
    pass


@contextmanager
def wall_time_limit(seconds: float):
    alarm = getattr(signal, "SIGALRM", None)
    setitimer = getattr(signal, "setitimer", None)
    if alarm is None or setitimer is None:
        yield
        return

    def _raise_timeout(signum, frame):
        del signum, frame
        raise BenchmarkTimeout(f"wall time limit {seconds:g}s reached")

    previous = signal.getsignal(alarm)
    signal.signal(alarm, _raise_timeout)
    setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(alarm, previous)


def _empty_row(case, backend: str, build_time: float) -> dict[str, str]:
    row = {field: "" for field in CSV_FIELDS}
    row.update(
        {
            "case": case.case_id,
            "family": case.family,
            "scale": case.scale,
            "seed": str(case.seed),
            "category": case.category,
            "backend": backend,
            "expected_status": case.expected_status or "",
            "build_time": f"{build_time:.9f}",
            "num_variables": str(case.metadata["num_variables"]),
            "num_integer_variables": str(case.metadata["num_integer_variables"]),
            "num_constraints": str(case.metadata["num_constraints"]),
            "density": f"{float(case.metadata['density']):.6f}",
        }
    )
    return row


def _format(value) -> str:
    return "" if value is None else f"{float(value):.10g}"


def _display_status(raw_status: str) -> str:
    if raw_status in LIMIT_STATUSES:
        return "LIMIT"
    return raw_status


def _tableau_bytes(problem: GeneralLPProblem | object) -> int:
    if isinstance(problem, GeneralLPProblem):
        n = int(problem.c.size)
        m = int(problem.b.size)
        finite_uppers = int(np.count_nonzero(np.isfinite(problem.ub)))
        senses = problem.constraint_senses
    else:
        n = int(problem.num_vars)
        m = int(problem.num_constraints)
        finite_uppers = int(np.count_nonzero(np.isfinite(problem.ub)))
        senses = ("<=",) * m
    rows = m + finite_uppers
    artificial_or_slack = sum(2 if sense == ">=" else 1 for sense in senses)
    columns = n + artificial_or_slack + finite_uppers
    return int((rows + 1) * (columns + 1) * 8)


def _lp_residual(problem: GeneralLPProblem, x: np.ndarray | None) -> float | None:
    if x is None:
        return None
    violations = [float(np.max(problem.lb - x, initial=0.0))]
    finite_upper = np.isfinite(problem.ub)
    if finite_upper.any():
        violations.append(float(np.max(x[finite_upper] - problem.ub[finite_upper], initial=0.0)))
    activity = problem.A @ x
    for value, rhs, sense in zip(activity, problem.b, problem.constraint_senses):
        if sense == "<=":
            violations.append(float(value - rhs))
        elif sense == ">=":
            violations.append(float(rhs - value))
        else:
            violations.append(float(abs(value - rhs)))
    return max(0.0, max(violations, default=0.0))


def _milp_residual(problem, x: np.ndarray | None) -> tuple[float | None, float | None]:
    if x is None:
        return None, None
    residual = max(
        0.0,
        float(np.max(problem.G @ x - problem.h, initial=0.0)),
        float(np.max(problem.lb - x, initial=0.0)),
        float(np.max(x - problem.ub, initial=0.0)),
    )
    integrality = max(
        (
            abs(float(x[index]) - round(float(x[index])))
            for index in problem.integer_indices
        ),
        default=0.0,
    )
    return residual, integrality


def _solve_custom_lp(case, build_time: float) -> dict[str, str]:
    problem = case.problem
    limits = LIMITS[case.scale]
    row = _empty_row(case, "two_phase_simplex", build_time)
    estimated_bytes = _tableau_bytes(problem)
    if estimated_bytes > limits.tableau_memory_limit:
        row.update(
            {
                "status": "SKIPPED_RESOURCE",
                "termination": "tableau_memory_estimate",
                "note": f"estimated tableau bytes={estimated_bytes}",
            }
        )
        return row

    start = perf_counter()
    try:
        with wall_time_limit(limits.wall_time_sec):
            result = TwoPhaseTableauSimplexSolver(
                tolerance=1e-8,
                max_iterations=limits.simplex_iteration_limit,
            ).solve(
                c=problem.c,
                A=problem.A,
                b=problem.b,
                constraint_senses=problem.constraint_senses,
                lb=problem.lb,
                ub=problem.ub,
                sense=problem.sense,
            )
        raw_status = result.status
        objective = result.objective_value
        x = result.x
        solve_time = perf_counter() - start
        row.update(
            {
                "status": _display_status(raw_status),
                "termination": raw_status,
                "objective": _format(objective),
                "solve_time": f"{solve_time:.9f}",
                "total_time": f"{build_time + solve_time:.9f}",
                "simplex_iterations": str(result.num_iterations),
                "pivot_count": str(result.num_iterations),
                "phase_one_iterations": str(result.phase_one_iterations),
                "phase_two_iterations": str(result.phase_two_iterations),
                "residual": _format(_lp_residual(problem, x)),
                "note": result.message,
            }
        )
    except BenchmarkTimeout as exc:
        solve_time = perf_counter() - start
        row.update(
            {
                "status": "LIMIT",
                "termination": "time_limit",
                "solve_time": f"{solve_time:.9f}",
                "total_time": f"{build_time + solve_time:.9f}",
                "note": str(exc),
            }
        )
    return row


def _scipy_matrices(problem: GeneralLPProblem):
    A_ub = []
    b_ub = []
    A_eq = []
    b_eq = []
    for coefficients, rhs, sense in zip(
        problem.A,
        problem.b,
        problem.constraint_senses,
    ):
        if sense == "<=":
            A_ub.append(coefficients)
            b_ub.append(rhs)
        elif sense == ">=":
            A_ub.append(-coefficients)
            b_ub.append(-rhs)
        else:
            A_eq.append(coefficients)
            b_eq.append(rhs)
    return (
        None if not A_ub else np.asarray(A_ub),
        None if not b_ub else np.asarray(b_ub),
        None if not A_eq else np.asarray(A_eq),
        None if not b_eq else np.asarray(b_eq),
    )


def _solve_scipy_lp(case, build_time: float) -> dict[str, str]:
    row = _empty_row(case, "scipy_highs", build_time)
    if importlib.util.find_spec("scipy") is None:
        row.update(
            {
                "status": "SKIPPED",
                "termination": "missing_dependency",
                "note": "SciPy is not installed",
            }
        )
        return row

    problem = case.problem
    A_ub, b_ub, A_eq, b_eq = _scipy_matrices(problem)
    c = problem.c if problem.sense == "min" else -problem.c
    bounds = [
        (
            None if not np.isfinite(lower) else float(lower),
            None if not np.isfinite(upper) else float(upper),
        )
        for lower, upper in zip(problem.lb, problem.ub)
    ]
    start = perf_counter()
    try:
        with wall_time_limit(LIMITS[case.scale].wall_time_sec):
            from scipy.optimize import linprog

            result = linprog(
                c=c,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method="highs",
                options={"time_limit": LIMITS[case.scale].wall_time_sec},
            )
    except BenchmarkTimeout as exc:
        solve_time = perf_counter() - start
        row.update(
            {
                "status": "LIMIT",
                "termination": "time_limit",
                "solve_time": f"{solve_time:.9f}",
                "total_time": f"{build_time + solve_time:.9f}",
                "note": str(exc),
            }
        )
        return row
    solve_time = perf_counter() - start
    status_map = {
        0: "optimal",
        1: "time_limit",
        2: "infeasible",
        3: "unbounded",
        4: "numerical_error",
    }
    raw_status = status_map.get(int(result.status), "lp_error")
    x = None if result.x is None else np.asarray(result.x, dtype=float)
    objective = None
    if result.fun is not None and raw_status == "optimal":
        objective = float(result.fun if problem.sense == "min" else -result.fun)
    row.update(
        {
            "status": _display_status(raw_status),
            "termination": raw_status,
            "objective": _format(objective),
            "solve_time": f"{solve_time:.9f}",
            "total_time": f"{build_time + solve_time:.9f}",
            "simplex_iterations": str(int(result.nit)),
            "residual": _format(_lp_residual(problem, x)),
            "note": str(result.message),
        }
    )
    return row


def _solve_mini_milp(case, backend: str, build_time: float) -> dict[str, str]:
    problem = case.problem
    limits = LIMITS[case.scale]
    label = "bnb_two_phase_simplex" if backend == "two_phase_simplex" else "bnb_scipy_highs"
    row = _empty_row(case, label, build_time)
    if backend == "scipy_highs" and importlib.util.find_spec("scipy.optimize") is None:
        row.update(
            {
                "status": "SKIPPED",
                "termination": "missing_dependency",
                "note": "SciPy is not installed",
            }
        )
        return row
    if backend == "two_phase_simplex":
        estimated_bytes = _tableau_bytes(problem)
        if estimated_bytes > limits.tableau_memory_limit:
            row.update(
                {
                    "status": "SKIPPED_RESOURCE",
                    "termination": "tableau_memory_estimate",
                    "note": f"estimated root tableau bytes={estimated_bytes}",
                }
            )
            return row

    start = perf_counter()
    try:
        with wall_time_limit(limits.wall_time_sec):
            result = solve_milp(
                problem,
                lp_backend=backend,
                max_nodes=limits.node_limit,
                time_limit_sec=limits.wall_time_sec,
                max_lp_iterations=limits.simplex_iteration_limit,
                use_matrix_presolve=True,
            )
        solve_time = perf_counter() - start
        residual, integrality = _milp_residual(problem, result.x)
        row.update(
            {
                "status": _display_status(result.status),
                "termination": result.status,
                "objective": _format(result.objective_value),
                "best_bound": _format(result.global_bound),
                "gap": _format(result.relative_gap),
                "solve_time": f"{solve_time:.9f}",
                "total_time": f"{build_time + solve_time:.9f}",
                "nodes": str(result.num_nodes),
                "lp_calls": str(result.num_lp_solved),
                "lp_total_time": f"{result.lp_runtime_sec:.9f}",
                "simplex_iterations": (
                    str(result.num_simplex_iterations)
                    if backend == "two_phase_simplex"
                    else ""
                ),
                "pivot_count": (
                    str(result.num_simplex_iterations)
                    if backend == "two_phase_simplex"
                    else ""
                ),
                "infeasible_prunes": str(result.num_pruned_infeasible),
                "bound_prunes": str(result.num_pruned_bound),
                "residual": _format(residual),
                "integrality_violation": _format(integrality),
            }
        )
    except BenchmarkTimeout as exc:
        solve_time = perf_counter() - start
        row.update(
            {
                "status": "LIMIT",
                "termination": "time_limit",
                "solve_time": f"{solve_time:.9f}",
                "total_time": f"{build_time + solve_time:.9f}",
                "note": str(exc),
            }
        )
    return row


def _new_gurobi_model(case):
    import gurobipy as gp

    problem = case.problem
    model = gp.Model(case.case_id)
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = LIMITS[case.scale].wall_time_sec
    model.Params.DualReductions = 0
    if case.category == "milp":
        model.Params.NodeLimit = LIMITS[case.scale].node_limit
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
    else:
        variables = [
            model.addVar(
                lb=float(problem.lb[index]),
                ub=float(problem.ub[index]),
                vtype=gp.GRB.CONTINUOUS,
                name=f"x{index}",
            )
            for index in range(problem.c.size)
        ]
        model.setObjective(
            gp.quicksum(
                float(problem.c[index]) * variables[index]
                for index in range(problem.c.size)
            ),
            gp.GRB.MAXIMIZE if problem.sense == "max" else gp.GRB.MINIMIZE,
        )
        for row, sense in enumerate(problem.constraint_senses):
            expression = gp.quicksum(
                float(problem.A[row, column]) * variables[column]
                for column in range(problem.c.size)
            )
            if sense == "<=":
                model.addConstr(expression <= float(problem.b[row]))
            elif sense == ">=":
                model.addConstr(expression >= float(problem.b[row]))
            else:
                model.addConstr(expression == float(problem.b[row]))
    return model, variables


def _solve_gurobi(case, build_time: float) -> dict[str, str]:
    row = _empty_row(case, "gurobi", build_time)
    if importlib.util.find_spec("gurobipy") is None:
        row.update(
            {
                "status": "SKIPPED",
                "termination": "missing_dependency",
                "note": "gurobipy is not installed",
            }
        )
        return row
    try:
        with wall_time_limit(LIMITS[case.scale].wall_time_sec):
            import gurobipy as gp

            model, variables = _new_gurobi_model(case)
            start = perf_counter()
            model.optimize()
        solve_time = perf_counter() - start
    except BenchmarkTimeout as exc:
        row.update(
            {
                "status": "LIMIT",
                "termination": "time_limit",
                "note": str(exc),
            }
        )
        return row
    except Exception as exc:
        row.update(
            {
                "status": "SKIPPED",
                "termination": "gurobi_unavailable",
                "note": str(exc),
            }
        )
        return row

    status_map = {
        gp.GRB.OPTIMAL: "optimal",
        gp.GRB.INFEASIBLE: "infeasible",
        gp.GRB.UNBOUNDED: "unbounded",
        gp.GRB.INF_OR_UNBD: "infeasible_or_unbounded",
        gp.GRB.TIME_LIMIT: "time_limit",
        gp.GRB.NODE_LIMIT: "node_limit",
        gp.GRB.NUMERIC: "numerical_error",
    }
    raw_status = status_map.get(model.Status, f"gurobi_status_{model.Status}")
    x = None
    objective = None
    if model.SolCount:
        x = np.asarray([variable.X for variable in variables], dtype=float)
        objective = float(model.ObjVal)
    if case.category == "lp":
        residual = _lp_residual(case.problem, x)
        integrality = None
        nodes = None
    else:
        residual, integrality = _milp_residual(case.problem, x)
        nodes = int(model.NodeCount)
    row.update(
        {
            "status": _display_status(raw_status),
            "termination": raw_status,
            "objective": _format(objective),
            "best_bound": _format(model.ObjBound if case.category == "milp" else objective),
            "gap": _format(model.MIPGap if case.category == "milp" and model.SolCount else 0.0),
            "solve_time": f"{solve_time:.9f}",
            "total_time": f"{build_time + solve_time:.9f}",
            "nodes": "" if nodes is None else str(nodes),
            "simplex_iterations": str(int(model.IterCount)),
            "residual": _format(residual),
            "integrality_violation": _format(integrality),
        }
    )
    return row


def run_backend(case, backend: str, build_time: float) -> dict[str, str]:
    if backend == "gurobi":
        return _solve_gurobi(case, build_time)
    if case.category == "lp":
        if backend == "custom":
            return _solve_custom_lp(case, build_time)
        return _solve_scipy_lp(case, build_time)
    if backend == "custom":
        return _solve_mini_milp(case, "two_phase_simplex", build_time)
    return _solve_mini_milp(case, "scipy_highs", build_time)


def add_reference_matches(rows: list[dict[str, str]]) -> None:
    references = {
        row["case"]: row
        for row in rows
        if row["backend"] == "gurobi"
        and row["status"] not in {"SKIPPED", "SKIPPED_RESOURCE", "LIMIT"}
    }
    for row in rows:
        reference = references.get(row["case"])
        if reference is None or row["status"] in {"SKIPPED", "SKIPPED_RESOURCE", "LIMIT"}:
            row["match_gurobi"] = ""
            continue
        if row["status"] != reference["status"]:
            row["match_gurobi"] = "False"
            continue
        if row["status"] == "optimal":
            if not row["objective"] or not reference["objective"]:
                row["match_gurobi"] = "False"
            else:
                difference = abs(float(row["objective"]) - float(reference["objective"]))
                scale = max(1.0, abs(float(reference["objective"])))
                row["match_gurobi"] = str(difference <= TOL * scale)
        else:
            row["match_gurobi"] = "True"


def _console_table(rows: list[dict[str, str]]) -> str:
    headers = [
        "case",
        "scale",
        "backend",
        "status",
        "objective",
        "solve_time",
        "nodes",
        "LP iter",
        "gap",
        "match",
    ]
    values = []
    for row in rows:
        values.append(
            [
                row["case"],
                row["scale"],
                row["backend"],
                row["status"],
                row["objective"] or "-",
                row["solve_time"] or "-",
                row["nodes"] or "-",
                row["simplex_iterations"] or "-",
                row["gap"] or ("LIMIT" if row["status"] == "LIMIT" else "-"),
                row["match_gurobi"] or "-",
            ]
        )
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in values))
        for index in range(len(headers))
    ]
    lines = [
        " | ".join(headers[index].ljust(widths[index]) for index in range(len(headers))),
        "-+-".join("-" * width for width in widths),
    ]
    lines.extend(
        " | ".join(
            row[index].ljust(widths[index]) for index in range(len(headers))
        ).rstrip()
        for row in values
    )
    return "\n".join(lines)


def _bottleneck_text(rows: list[dict[str, str]]) -> str:
    custom_lp = [
        row
        for row in rows
        if row["backend"] == "two_phase_simplex"
    ]
    custom_milp = [
        row
        for row in rows
        if row["backend"] == "bnb_two_phase_simplex"
    ]
    if any(row["termination"] == "numerical_error" for row in custom_lp + custom_milp):
        return "当前首先暴露的是数值稳定性：自写 tableau 出现 numerical_error。"
    if any(row["status"] in {"LIMIT", "SKIPPED_RESOURCE"} for row in custom_lp):
        return "当前首先暴露的是 LP tableau：纯 LP 已触发时间、迭代或内存限制。"
    limited_milp = [row for row in custom_milp if row["status"] == "LIMIT"]
    if limited_milp:
        lp_heavy = []
        for row in limited_milp:
            if row["lp_total_time"] and row["solve_time"]:
                lp_heavy.append(float(row["lp_total_time"]) / max(float(row["solve_time"]), 1e-12))
        if lp_heavy and max(lp_heavy) >= 0.7:
            return "当前 MILP 限制主要耗在节点 LP tableau，而不是节点队列本身。"
        return "当前 MILP 限制主要表现为 B&B 节点树增长。"
    return "当前规模尚未触发硬限制；下一步应优先继续观察 large 的 LP 时间与节点数分解。"


def _family_summary_table(rows: list[dict[str, str]]) -> str:
    headers = ["family", "backend", "expected_ok", "LIMIT", "median_sec"]
    values: list[list[str]] = []
    keys = sorted({(row["family"], row["backend"]) for row in rows})
    for family, backend in keys:
        group = [
            row
            for row in rows
            if row["family"] == family and row["backend"] == backend
        ]
        successful = [
            row
            for row in group
            if row["expected_status"] and row["status"] == row["expected_status"]
        ]
        times = [float(row["solve_time"]) for row in successful if row["solve_time"]]
        values.append(
            [
                family,
                backend,
                f"{len(successful)}/{len(group)}",
                str(sum(row["status"] == "LIMIT" for row in group)),
                "-" if not times else f"{median(times):.6f}",
            ]
        )
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in values))
        for index in range(len(headers))
    ]
    lines = [
        " | ".join(headers[index].ljust(widths[index]) for index in range(len(headers))),
        "-+-".join("-" * width for width in widths),
    ]
    lines.extend(
        " | ".join(
            row[index].ljust(widths[index]) for index in range(len(headers))
        ).rstrip()
        for row in values
    )
    return "\n".join(lines)


def write_reports(rows: list[dict[str, str]], report_dir: Path, mode: str) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "benchmark_latest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    completed = [
        row
        for row in rows
        if row["status"] not in {"SKIPPED", "SKIPPED_RESOURCE", "LIMIT"}
    ]
    expected_matches = sum(
        row["status"] == row["expected_status"] for row in completed if row["expected_status"]
    )
    reference_rows = [row for row in rows if row["match_gurobi"]]
    reference_matches = sum(row["match_gurobi"] == "True" for row in reference_rows)
    issues = [
        (
            f"`{row['case']}` / `{row['backend']}`: "
            f"`{row['termination']}`，预期 `{row['expected_status']}`"
        )
        for row in rows
        if row["status"] in {"LIMIT", "SKIPPED_RESOURCE"}
        or (
            row["expected_status"]
            and row["status"] != row["expected_status"]
        )
    ]
    first_issue = issues[0] if issues else "本次没有 LIMIT、资源跳过或预期状态失败。"
    markdown = [
        "# Benchmark Latest",
        "",
        f"- 测试时间：`{datetime.now().astimezone().isoformat(timespec='seconds')}`",
        f"- 模式：`{mode}`",
        f"- 环境：Python `{platform.python_version()}`，`{platform.platform()}`",
        f"- 完成状态与预期一致：`{expected_matches}/{len(completed)}`",
        f"- 与 Gurobi 状态/目标一致：`{reference_matches}/{len(reference_rows)}`",
        f"- 首个限制/失败：{first_issue}",
        "",
        "```text",
        _console_table(rows),
        "```",
        "",
        "## 按模型族汇总",
        "",
        "```text",
        _family_summary_table(rows),
        "```",
        "",
        "## LIMIT / Resource / Failure",
        "",
    ]
    markdown.extend(f"- {item}" for item in issues)
    if not issues:
        markdown.append("- 本次没有 LIMIT、资源跳过或预期状态失败。")
    markdown.extend(
        [
            "",
            "## 当前判断",
            "",
            _bottleneck_text(rows),
            "",
            "详细 build、LP、剪枝、残差和迭代字段见 `reports/benchmark_latest.csv`。",
            "",
        ]
    )
    (report_dir / "benchmark_latest.md").write_text(
        "\n".join(markdown),
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Run unified LP/MILP benchmarks.")
    parser.add_argument("mode", choices=["small", "large", "all"])
    parser.add_argument("--family", choices=available_families())
    parser.add_argument("--backend", choices=CORE_BACKENDS, action="append")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> list[dict[str, str]]:
    args = parse_args(argv)
    scales = ("small", "large") if args.mode == "all" else (args.mode,)
    families = available_families() if args.family is None else (args.family,)
    backends = tuple(args.backend) if args.backend else CORE_BACKENDS
    rows: list[dict[str, str]] = []

    for scale in scales:
        for family in families:
            for seed in seeds_for_scale(scale):
                build_start = perf_counter()
                case = build_case(family, scale, seed)
                build_time = perf_counter() - build_start
                for backend in backends:
                    row = run_backend(case, backend, build_time)
                    rows.append(row)
                    print(
                        f"{case.case_id} {row['backend']}: "
                        f"{row['status']} {row['solve_time'] or '-'}"
                    )

    add_reference_matches(rows)
    print()
    print(_console_table(rows))
    write_reports(rows, args.report_dir, args.mode)
    print(f"\nwrote {args.report_dir / 'benchmark_latest.csv'}")
    print(f"wrote {args.report_dir / 'benchmark_latest.md'}")
    return rows


if __name__ == "__main__":
    main()
