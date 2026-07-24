from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.ml_branching.training.oracle.strong_branching import StrongBranchingPolicy
from research.ml_branching.unit_commitment import UnitCommitmentGenerator
from solver import MILPProblem, solve_milp
from solver.branch_and_bound import fractional_binary_candidates
from solver.branching import MostFractionalPolicy
from solver.lp_active_set import solve_lp_relaxation
from solver.matrix_presolve import MatrixPresolveOptions


CONFIG_PATH = Path("research/ml_branching/configs/unified_solver_comparison.json")
REPORT_DIR = Path("research/ml_branching/reports")
DATA_DIR = REPORT_DIR / "data"
REPORT_PATH = REPORT_DIR / "GNN稳定性与通用性验证.md"
PRESOLVE_QUEUE_CSV = DATA_DIR / "presolve_and_queue_comparison.csv"
BRANCHING_CSV = DATA_DIR / "branching_comparison.csv"
STABILITY_CSV = DATA_DIR / "stability_results.csv"
UNIFIED_CSV = DATA_DIR / "unified_comparison.csv"


STATUS_ZH = {
    "optimal": "最优",
    "infeasible": "不可行",
    "candidate_limit": "候选组合上限",
    "node_limit": "达到节点上限",
    "time_limit": "达到时间上限",
    "lp_error": "LP错误",
}

STRATEGY_ZH = {
    "most_fractional": "最接近0.5分支（原规则）",
    "gnn_seed_1": "GNN图学习分支（种子1）",
    "gnn_seed_2": "GNN图学习分支（种子2）",
    "gnn_seed_3": "GNN图学习分支（种子3）",
    "gnn_main": "GNN图学习分支",
    "strong_branching": "强分支（专家参考）",
}

VERSION_ZH = {
    "active_original": "自制求解器基础版",
    "active_gnn": "自制求解器GNN增强版",
    "highs_original": "开源LP增强版",
    "highs_gnn": "开源LP＋GNN完整版",
}


def load_config(path: str | Path = CONFIG_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _multi_cover_problem(spec: dict) -> MILPProblem:
    seed = int(spec["seed"])
    n_y = int(spec["binary_count"])
    n_x = int(spec["continuous_count"])
    rng = np.random.default_rng(seed)
    capacity = rng.uniform(2.0, 8.0, (n_x, n_y)).round(2)
    fixed_cost = rng.uniform(1.0, 6.0, n_y).round(2)
    variable_cost = rng.uniform(6.0, 10.0, n_x).round(2)
    demand = (float(spec.get("demand_fraction", 0.45)) * capacity.sum(axis=1)).round(2)
    backup_fraction = float(spec.get("backup_fraction", 0.25))

    A_rows = []
    B_rows = []
    b_values = []
    for row in range(n_x):
        a = np.zeros(n_x)
        a[row] = -1.0
        A_rows.append(a)
        B_rows.append(-capacity[row])
        b_values.append(-float(demand[row]))

        a = np.zeros(n_x)
        a[row] = 1.0
        A_rows.append(a)
        B_rows.append(np.zeros(n_y))
        b_values.append(float(backup_fraction * demand[row]))

    resource = rng.uniform(0.2, 1.2, n_y).round(2)
    A_rows.append(np.zeros(n_x))
    B_rows.append(resource)
    b_values.append(float(spec.get("resource_fraction", 0.75)) * float(resource.sum()))

    return MILPProblem.from_blocks(
        c_x=variable_cost,
        c_y=fixed_cost,
        A=np.vstack(A_rows),
        B=np.vstack(B_rows),
        b=np.asarray(b_values, dtype=float),
        x_lb=np.zeros(n_x),
        x_ub=backup_fraction * demand,
        sense="min",
        name=str(spec["id"]),
    )


def _sparse_block_problem(spec: dict) -> MILPProblem:
    seed = int(spec["seed"])
    n = int(spec["size"])
    rng = np.random.default_rng(seed)
    x_ub = rng.uniform(2.0, 6.0, n).round(2)
    x_ref = 0.45 * x_ub
    rows = int(spec.get("rows", n + 2))
    A = np.zeros((rows, n))
    B = np.zeros((rows, n))
    b = np.zeros(rows)
    A[0] = -rng.uniform(0.2, 1.0, n)
    b[0] = float(A[0] @ x_ref) * 0.8
    A[1] = rng.uniform(0.1, 1.0, n)
    b[1] = float(0.75 * (A[1] @ x_ub))
    for index in range(n):
        row = 2 + index
        A[row, index] = 1.0
        B[row, index] = -x_ub[index]
    return MILPProblem.from_blocks(
        c_x=rng.uniform(1.0, 3.0, n),
        c_y=rng.uniform(2.0, 6.0, n),
        A=A,
        B=B,
        b=b,
        x_lb=np.zeros(n),
        x_ub=x_ub,
        sense="min",
        name=str(spec["id"]),
    )


def _unit_commitment_one_period_problem(spec: dict) -> MILPProblem:
    seed = int(spec["seed"])
    units = int(spec["units"])
    rng = np.random.default_rng(seed)
    p_min = rng.uniform(0.5, 1.5, units).round(2)
    p_max = rng.uniform(3.5, 7.0, units).round(2)
    variable_cost = rng.uniform(1.0, 4.0, units).round(2)
    fixed_cost = rng.uniform(1.0, 6.0, units).round(2)
    demand = round(float(spec.get("demand_fraction", 0.58)) * float(p_max.sum()), 2)
    A_rows = [-np.ones(units)]
    B_rows = [np.zeros(units)]
    b_values = [-demand]
    for index in range(units):
        a = np.zeros(units)
        y = np.zeros(units)
        a[index] = 1.0
        y[index] = -p_max[index]
        A_rows.append(a)
        B_rows.append(y)
        b_values.append(0.0)

        a = np.zeros(units)
        y = np.zeros(units)
        a[index] = -1.0
        y[index] = p_min[index]
        A_rows.append(a)
        B_rows.append(y)
        b_values.append(0.0)
    return MILPProblem.from_blocks(
        c_x=variable_cost,
        c_y=fixed_cost,
        A=np.vstack(A_rows),
        B=np.vstack(B_rows),
        b=np.asarray(b_values, dtype=float),
        x_lb=np.zeros(units),
        x_ub=p_max,
        sense="min",
        name=str(spec["id"]),
    )


def _branch_sensitive_cover_problem(spec: dict) -> MILPProblem:
    seed = int(spec["seed"])
    n_x = int(spec["continuous_count"])
    n_y = int(spec["binary_count"])
    kind = str(spec["kind"])
    rng = np.random.default_rng(seed)
    capacity = rng.uniform(2.0, 8.0, (n_x, n_y)).round(2)
    fixed_cost = rng.uniform(1.0, 8.0, n_y).round(2)
    variable_cost = rng.uniform(5.0, 12.0, n_x).round(2)
    demand = (rng.uniform(0.38, 0.62, n_x) * capacity.sum(axis=1)).round(2)
    backup_fraction = rng.uniform(0.10, 0.35, n_x).round(2)

    A_rows = []
    B_rows = []
    b_values = []
    for row in range(n_x):
        a = np.zeros(n_x)
        a[row] = -1.0
        A_rows.append(a)
        B_rows.append(-capacity[row])
        b_values.append(-float(demand[row]))

        a = np.zeros(n_x)
        a[row] = 1.0
        A_rows.append(a)
        B_rows.append(np.zeros(n_y))
        b_values.append(float(backup_fraction[row] * demand[row]))

    if kind in {"budget", "multi"}:
        for _ in range(2):
            resource = rng.uniform(0.1, 1.4, n_y).round(2)
            A_rows.append(np.zeros(n_x))
            B_rows.append(resource)
            b_values.append(float(rng.uniform(0.45, 0.90) * resource.sum()))

    if kind in {"cardinality", "multi"}:
        A_rows.append(np.zeros(n_x))
        B_rows.append(np.ones(n_y))
        b_values.append(float(rng.integers(max(2, n_y // 2), n_y)))

    if kind == "mutual":
        pairs = rng.choice(n_y, size=(max(1, n_y // 2), 2), replace=True)
        for i, j in pairs:
            if int(i) == int(j):
                continue
            row = np.zeros(n_y)
            row[int(i)] = 1.0
            row[int(j)] = 1.0
            A_rows.append(np.zeros(n_x))
            B_rows.append(row)
            b_values.append(1.0)

    return MILPProblem.from_blocks(
        c_x=variable_cost,
        c_y=fixed_cost,
        A=np.vstack(A_rows),
        B=np.vstack(B_rows),
        b=np.asarray(b_values, dtype=float),
        x_lb=np.zeros(n_x),
        x_ub=backup_fraction * demand,
        sense="min",
        name=str(spec["id"]),
    )


def build_problem(spec: dict) -> MILPProblem:
    family = str(spec["family"])
    if family in {"fixed_charge_multi_resource", "capacity_expansion"}:
        return _multi_cover_problem(spec)
    if family == "random_sparse_block":
        return _sparse_block_problem(spec)
    if family == "unit_commitment":
        return _unit_commitment_one_period_problem(spec)
    if family == "branch_sensitive_cover":
        return _branch_sensitive_cover_problem(spec)
    raise ValueError(f"unsupported unified family: {family}")


def full_presolve_options() -> MatrixPresolveOptions:
    return MatrixPresolveOptions(
        eliminate_fixed_variables=True,
        remove_redundant_rows=True,
        tighten_bounds=True,
        max_rounds=3,
    )


def fixed_only_options() -> MatrixPresolveOptions:
    return MatrixPresolveOptions(
        eliminate_fixed_variables=True,
        remove_redundant_rows=False,
        tighten_bounds=False,
        max_rounds=1,
    )


def fixed_rows_options() -> MatrixPresolveOptions:
    return MatrixPresolveOptions(
        eliminate_fixed_variables=True,
        remove_redundant_rows=True,
        tighten_bounds=False,
        max_rounds=1,
    )


def _policy(strategy_id: str, config: dict):
    if strategy_id == "most_fractional":
        return MostFractionalPolicy()
    if strategy_id == "strong_branching":
        settings = config["run_settings"]
        return StrongBranchingPolicy(
            lp_backend=str(settings.get("strong_branching_lp_backend", "active_set")),
            max_lp_candidates=None if settings.get("strong_branching_lp_backend") == "scipy_highs" else int(settings["max_lp_candidates"]),
            use_matrix_presolve=True,
        )
    if strategy_id.startswith("gnn_seed_"):
        from research.ml_branching.runtime.inference import LearnedGNNBranchingPolicy

        checkpoint = config["gnn_checkpoints"][strategy_id]
        return LearnedGNNBranchingPolicy.from_checkpoint(checkpoint, device=config.get("device", "cpu"))
    raise ValueError(f"unknown strategy_id={strategy_id!r}")


def _result_row_base(spec: dict, problem: MILPProblem) -> dict:
    return {
        "case": str(spec["id"]),
        "family": str(spec["family"]),
        "seed": str(spec["seed"]),
        "num_variables": str(problem.num_vars),
        "num_constraints": str(problem.num_constraints),
        "num_binary": str(len(problem.binary_indices)),
    }


def _status_zh(status: str) -> str:
    return STATUS_ZH.get(str(status), str(status))


def _format_float(value, digits: int = 10) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.{digits}g}"


def _solve_once(spec: dict, solver_spec: dict, config: dict) -> tuple[dict, object]:
    problem = build_problem(spec)
    settings = config["run_settings"]
    policy = _policy(solver_spec["strategy_id"], config)
    start = perf_counter()
    result = solve_milp(
        problem,
        tol=float(settings["tolerance"]),
        max_nodes=int(settings["max_nodes"]),
        branching_policy=policy,
        lp_backend=str(solver_spec.get("lp_backend", "active_set")),
        max_lp_candidates=int(settings["max_lp_candidates"]),
        use_matrix_presolve=bool(solver_spec["use_matrix_presolve"]),
        matrix_presolve_options=full_presolve_options() if solver_spec["use_matrix_presolve"] else None,
        node_selection=str(solver_spec["node_selection"]),
        time_limit_sec=float(settings["time_limit_sec"]),
    )
    total_elapsed = perf_counter() - start
    row = _result_row_base(spec, problem)
    row.update(
        {
            "status": _status_zh(result.status),
            "objective": _format_float(result.objective_value),
            "incumbent": _format_float(result.objective_value),
            "global_bound": _format_float(result.global_bound),
            "relative_gap": _format_float(result.relative_gap),
            "bb_nodes": str(result.num_nodes),
            "formal_lp_solved": str(result.num_lp_solved),
            "active_set_candidates": str(result.num_lp_candidates_checked),
            "fixed_variables": str(result.num_fixed_vars_eliminated),
            "removed_rows": str(result.num_removed_rows),
            "tightened_bounds": str(result.num_tightened_bounds),
            "probe_lp_solved": str(int(getattr(policy, "probe_lp_solved", 0))),
            "probe_time_sec": _format_float(getattr(policy, "probe_time_sec", 0.0), 8),
            "inference_time_sec": _format_float(getattr(policy, "inference_time_sec", 0.0), 8),
            "solver_runtime_sec": _format_float(result.runtime_sec, 8),
            "total_runtime_sec": _format_float(total_elapsed, 8),
            "limit_type": "" if result.status == "optimal" else _status_zh(result.status),
            "lp_backend": "SciPy-HiGHS" if str(solver_spec.get("lp_backend", "active_set")) == "scipy_highs" else "自制active-set",
        }
    )
    return row, result


def _median_run(spec: dict, solver_spec: dict, config: dict) -> dict:
    if config["run_settings"].get("warmup", True):
        _solve_once(spec, solver_spec, config)
    rows = []
    for _ in range(int(config["run_settings"]["repeats"])):
        row, _ = _solve_once(spec, solver_spec, config)
        rows.append(row)
    times = [float(row["total_runtime_sec"]) for row in rows]
    median_time = median(times)
    chosen_index = min(range(len(rows)), key=lambda i: (abs(times[i] - median_time), i))
    row = dict(rows[chosen_index])
    row["time_repeats_sec"] = ";".join(_format_float(t, 8) for t in times)
    row["total_runtime_sec"] = _format_float(median_time, 8)
    return row


def _write_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fieldnames} for row in rows)


def _representative_bounds(problem: MILPProblem, settings: dict) -> tuple[np.ndarray, np.ndarray]:
    root = solve_lp_relaxation(
        problem,
        problem.lb,
        problem.ub,
        tol=float(settings["tolerance"]),
        max_candidates=int(settings["max_lp_candidates"]),
        use_matrix_presolve=True,
        matrix_presolve_options=full_presolve_options(),
    )
    lb = problem.lb.copy()
    ub = problem.ub.copy()
    if root.status != "optimal" or root.x is None:
        return lb, ub
    candidates = fractional_binary_candidates(root.x, problem.binary_indices, float(settings["tolerance"]))
    if not candidates:
        return lb, ub
    branch_var = MostFractionalPolicy().select_variable(
        type(
            "_Context",
            (),
            {
                "candidate_indices": candidates,
                "lp_result": root,
            },
        )()
    )
    ub[int(branch_var)] = math.floor(float(root.x[int(branch_var)]))
    return lb, ub


def run_presolve_table(config: dict) -> list[dict]:
    settings = config["run_settings"]
    variants = [
        ("未开启筛选", False, None),
        ("仅固定变量消元", True, fixed_only_options()),
        ("固定变量消元+冗余删除/不可行判断", True, fixed_rows_options()),
        ("三项筛选", True, full_presolve_options()),
    ]
    rows: list[dict] = []
    for spec in config["instances"]:
        problem = build_problem(spec)
        node_lb, node_ub = _representative_bounds(problem, settings)
        for label, use_presolve, options in variants:
            start = perf_counter()
            lp = solve_lp_relaxation(
                problem,
                node_lb,
                node_ub,
                tol=float(settings["tolerance"]),
                max_candidates=int(settings["max_lp_candidates"]),
                use_matrix_presolve=use_presolve,
                matrix_presolve_options=options,
            )
            elapsed = perf_counter() - start
            row = _result_row_base(spec, problem)
            row.update(
                {
                    "section": "三项筛选",
                    "presolve_config": label,
                    "fixed_variables": str(lp.num_fixed_vars),
                    "removed_rows": str(lp.num_removed_rows),
                    "tightened_bounds": str(lp.num_tightened_bounds),
                    "active_set_candidates": "超过候选组合上限" if lp.status == "candidate_limit" else str(lp.num_candidates_checked),
                    "lp_status": _status_zh(lp.status),
                    "lp_time_sec": _format_float(elapsed, 8),
                    "infeasible_detected": "是" if lp.status == "infeasible" else "否",
                }
            )
            rows.append(row)
    return rows


def version_specs(config: dict) -> list[dict]:
    main_gnn = config["main_gnn_strategy"]
    return [
        {
            "version_id": "raw_solver",
            "version": VERSION_ZH["raw_solver"],
            "lp_filtering": "未开启",
            "node_selection_name": "深度优先搜索",
            "node_selection": "dfs",
            "branching_strategy": STRATEGY_ZH["most_fractional"],
            "strategy_id": "most_fractional",
            "use_matrix_presolve": False,
        },
        {
            "version_id": "lp_presolve",
            "version": VERSION_ZH["lp_presolve"],
            "lp_filtering": "三项筛选",
            "node_selection_name": "深度优先搜索",
            "node_selection": "dfs",
            "branching_strategy": STRATEGY_ZH["most_fractional"],
            "strategy_id": "most_fractional",
            "use_matrix_presolve": True,
        },
        {
            "version_id": "best_bound",
            "version": VERSION_ZH["best_bound"],
            "lp_filtering": "三项筛选",
            "node_selection_name": "最优界优先队列",
            "node_selection": "best_bound",
            "branching_strategy": STRATEGY_ZH["most_fractional"],
            "strategy_id": "most_fractional",
            "use_matrix_presolve": True,
        },
        {
            "version_id": "gnn_enhanced",
            "version": VERSION_ZH["gnn_enhanced"],
            "lp_filtering": "三项筛选",
            "node_selection_name": "最优界优先队列",
            "node_selection": "best_bound",
            "branching_strategy": STRATEGY_ZH["gnn_main"],
            "strategy_id": main_gnn,
            "use_matrix_presolve": True,
        },
    ]


def run_queue_table(config: dict) -> list[dict]:
    rows = []
    for spec in config["instances"]:
        for solver_spec in version_specs(config)[1:3]:
            row = _median_run(spec, solver_spec, config)
            row.update(
                {
                    "section": "节点选择",
                    "search_strategy": solver_spec["node_selection_name"],
                }
            )
            rows.append(row)
    return rows


def run_branching_table(config: dict) -> list[dict]:
    rows = []
    strategies = ["most_fractional", config["main_gnn_strategy"], "strong_branching", *config["stability_gnn_strategies"]]
    seen = set()
    for strategy_id in strategies:
        if strategy_id in seen:
            continue
        seen.add(strategy_id)
        for spec in config["instances"]:
            solver_spec = {
                "strategy_id": strategy_id,
                "use_matrix_presolve": True,
                "node_selection": "best_bound",
            }
            row = _median_run(spec, solver_spec, config)
            display = STRATEGY_ZH["gnn_main"] if strategy_id == config["main_gnn_strategy"] else STRATEGY_ZH[strategy_id]
            stability_display = STRATEGY_ZH[strategy_id] if strategy_id.startswith("gnn_seed_") else display
            row.update(
                {
                    "strategy": display,
                    "strategy_detail": stability_display,
                }
            )
            rows.append(row)
    return rows


def run_unified_table(config: dict) -> list[dict]:
    rows = []
    for spec in config["instances"]:
        for solver_spec in version_specs(config):
            row = _median_run(spec, solver_spec, config)
            row.update(
                {
                    "solver_version": solver_spec["version"],
                    "lp_filtering": solver_spec["lp_filtering"],
                    "node_selection": solver_spec["node_selection_name"],
                    "branching_strategy": solver_spec["branching_strategy"],
                }
            )
            rows.append(row)
    return rows


def run_all(config: dict) -> None:
    presolve_rows = run_presolve_table(config)
    queue_rows = run_queue_table(config)
    presolve_queue_rows = presolve_rows + queue_rows
    _write_rows(PRESOLVE_QUEUE_CSV, presolve_queue_rows, sorted({k for row in presolve_queue_rows for k in row}))

    branching_rows = run_branching_table(config)
    _write_rows(BRANCHING_CSV, branching_rows, sorted({k for row in branching_rows for k in row}))

    unified_rows = run_unified_table(config)
    _write_rows(UNIFIED_CSV, unified_rows, sorted({k for row in unified_rows for k in row}))


def _append_row(path: Path, row: dict, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fieldnames})


def _done_keys(path: Path, key_fields: tuple[str, ...]) -> set[tuple[str, ...]]:
    return {tuple(row.get(field, "") for field in key_fields) for row in _read_csv(path)}


def _uc_instances(config: dict):
    stability_config = json.loads(Path(config["uc_stability_config"]).read_text(encoding="utf-8"))
    defaults = dict(stability_config.get("unit_commitment_generation", {}))
    generator = UnitCommitmentGenerator()
    for spec in stability_config["unit_commitment_instances"]:
        yield generator.generate(
            seed=int(spec["seed"]),
            units=int(spec["units"]),
            periods=int(spec["periods"]),
            split=str(spec["split"]),
            scale_group=str(spec["scale_group"]),
            demand_low=float(spec.get("demand_low", defaults.get("demand_low", 0.30))),
            demand_high=float(spec.get("demand_high", defaults.get("demand_high", 0.48))),
            reserve_low=float(spec.get("reserve_low", defaults.get("reserve_low", 0.02))),
            reserve_high=float(spec.get("reserve_high", defaults.get("reserve_high", 0.06))),
        )


def _solver_config(base_config: dict, settings_key: str) -> dict:
    config = dict(base_config)
    config["run_settings"] = dict(base_config[settings_key])
    return config


def _branching_solver_spec(strategy_id: str, lp_backend: str) -> dict:
    return {
        "strategy_id": strategy_id,
        "use_matrix_presolve": True,
        "node_selection": "best_bound",
        "lp_backend": lp_backend,
    }


def run_branching_uc(config: dict, resume: bool = True) -> None:
    run_config = _solver_config(config, "branching_settings")
    branching_fields = [
        "case",
        "family",
        "seed",
        "split",
        "scale_group",
        "units",
        "periods",
        "strategy",
        "status",
        "objective",
        "bb_nodes",
        "formal_lp_solved",
        "probe_lp_solved",
        "inference_time_sec",
        "probe_time_sec",
        "total_runtime_sec",
        "time_repeats_sec",
        "relative_gap",
        "limit_type",
        "lp_backend",
    ]
    stability_fields = list(branching_fields)
    if not resume:
        BRANCHING_CSV.unlink(missing_ok=True)
        STABILITY_CSV.unlink(missing_ok=True)

    branching_done = _done_keys(BRANCHING_CSV, ("case", "strategy")) if resume else set()
    stability_path = STABILITY_CSV
    stability_done = _done_keys(stability_path, ("case", "strategy")) if resume else set()

    main_gnn = config["main_gnn_strategy"]
    branching_strategies = ["most_fractional", main_gnn, "strong_branching"]
    stability_strategies = ["most_fractional", *config["stability_gnn_strategies"]]

    for instance in _uc_instances(config):
        spec = {
            "id": instance.instance_id,
            "family": instance.family_name,
            "seed": instance.seed,
            "problem_instance": instance,
        }
        for strategy_id in branching_strategies:
            label = STRATEGY_ZH["gnn_main"] if strategy_id == main_gnn else STRATEGY_ZH[strategy_id]
            key = (instance.instance_id, label)
            if key not in branching_done:
                row = _median_run_instance(instance, _branching_solver_spec(strategy_id, "scipy_highs"), run_config)
                row.update(
                    {
                        "strategy": label,
                        "split": instance.split,
                        "scale_group": instance.scale_group,
                        "units": str(instance.parameters.get("units", "")),
                        "periods": str(instance.parameters.get("periods", "")),
                    }
                )
                _append_row(BRANCHING_CSV, row, branching_fields)
                branching_done.add(key)
                print(f"branching {instance.instance_id} {label} {row['status']} nodes={row['bb_nodes']}")
        for strategy_id in stability_strategies:
            label = STRATEGY_ZH[strategy_id]
            key = (instance.instance_id, label)
            if key not in stability_done:
                row = _median_run_instance(instance, _branching_solver_spec(strategy_id, "scipy_highs"), run_config)
                row.update(
                    {
                        "strategy": label,
                        "split": instance.split,
                        "scale_group": instance.scale_group,
                        "units": str(instance.parameters.get("units", "")),
                        "periods": str(instance.parameters.get("periods", "")),
                    }
                )
                _append_row(stability_path, row, stability_fields)
                stability_done.add(key)
                print(f"stability {instance.instance_id} {label} {row['status']} nodes={row['bb_nodes']}")


def _median_run_instance(instance, solver_spec: dict, config: dict) -> dict:
    if config["run_settings"].get("warmup", True):
        _solve_once_instance(instance, solver_spec, config)
    rows = []
    for _ in range(int(config["run_settings"]["repeats"])):
        row, _ = _solve_once_instance(instance, solver_spec, config)
        rows.append(row)
    times = [float(row["total_runtime_sec"]) for row in rows]
    median_time = median(times)
    chosen_index = min(range(len(rows)), key=lambda i: (abs(times[i] - median_time), i))
    row = dict(rows[chosen_index])
    row["time_repeats_sec"] = ";".join(_format_float(t, 8) for t in times)
    row["total_runtime_sec"] = _format_float(median_time, 8)
    return row


def _solve_once_instance(instance, solver_spec: dict, config: dict) -> tuple[dict, object]:
    settings = config["run_settings"]
    policy = _policy(solver_spec["strategy_id"], config)
    start = perf_counter()
    result = solve_milp(
        instance.problem,
        tol=float(settings["tolerance"]),
        max_nodes=int(settings["max_nodes"]),
        branching_policy=policy,
        lp_backend=str(solver_spec.get("lp_backend", "active_set")),
        max_lp_candidates=int(settings["max_lp_candidates"]),
        use_matrix_presolve=True,
        matrix_presolve_options=full_presolve_options(),
        node_selection="best_bound",
        time_limit_sec=float(settings["time_limit_sec"]),
    )
    total_elapsed = perf_counter() - start
    row = {
        "case": instance.instance_id,
        "family": instance.family_name,
        "seed": str(instance.seed),
        "status": _status_zh(result.status),
        "objective": _format_float(result.objective_value),
        "bb_nodes": str(result.num_nodes),
        "formal_lp_solved": str(result.num_lp_solved),
        "probe_lp_solved": str(int(getattr(policy, "probe_lp_solved", 0))),
        "inference_time_sec": _format_float(getattr(policy, "inference_time_sec", 0.0), 8),
        "probe_time_sec": _format_float(getattr(policy, "probe_time_sec", 0.0), 8),
        "total_runtime_sec": _format_float(total_elapsed, 8),
        "relative_gap": _format_float(result.relative_gap),
        "limit_type": "" if result.status == "optimal" else _status_zh(result.status),
        "lp_backend": "SciPy-HiGHS" if str(solver_spec.get("lp_backend")) == "scipy_highs" else "自制active-set",
    }
    return row, result


def unified_2x2_specs(config: dict) -> list[dict]:
    main_gnn = config["main_gnn_strategy"]
    return [
        {
            "version": VERSION_ZH["active_original"],
            "node_lp": "自制active-set",
            "lp_backend": "active_set",
            "branching_method": STRATEGY_ZH["most_fractional"],
            "strategy_id": "most_fractional",
            "use_matrix_presolve": True,
            "node_selection": "best_bound",
        },
        {
            "version": VERSION_ZH["active_gnn"],
            "node_lp": "自制active-set",
            "lp_backend": "active_set",
            "branching_method": STRATEGY_ZH["gnn_main"],
            "strategy_id": main_gnn,
            "use_matrix_presolve": True,
            "node_selection": "best_bound",
        },
        {
            "version": VERSION_ZH["highs_original"],
            "node_lp": "SciPy-HiGHS",
            "lp_backend": "scipy_highs",
            "branching_method": STRATEGY_ZH["most_fractional"],
            "strategy_id": "most_fractional",
            "use_matrix_presolve": True,
            "node_selection": "best_bound",
        },
        {
            "version": VERSION_ZH["highs_gnn"],
            "node_lp": "SciPy-HiGHS",
            "lp_backend": "scipy_highs",
            "branching_method": STRATEGY_ZH["gnn_main"],
            "strategy_id": main_gnn,
            "use_matrix_presolve": True,
            "node_selection": "best_bound",
        },
    ]


def run_unified_2x2(config: dict, resume: bool = True) -> None:
    run_config = _solver_config(config, "unified_settings")
    fields = [
        "case",
        "family",
        "seed",
        "version",
        "node_lp",
        "branching_method",
        "status",
        "objective",
        "bb_nodes",
        "formal_lp_solved",
        "inference_time_sec",
        "total_runtime_sec",
        "time_repeats_sec",
        "relative_gap",
        "limit_type",
        "lp_backend",
    ]
    if not resume:
        UNIFIED_CSV.unlink(missing_ok=True)
    done = _done_keys(UNIFIED_CSV, ("case", "version")) if resume else set()
    for spec in config["unified_instances"]:
        for solver_spec in unified_2x2_specs(config):
            key = (str(spec["id"]), solver_spec["version"])
            if key in done:
                continue
            row = _median_run(spec, solver_spec, run_config)
            row.update(
                {
                    "version": solver_spec["version"],
                    "node_lp": solver_spec["node_lp"],
                    "branching_method": solver_spec["branching_method"],
                }
            )
            _append_row(UNIFIED_CSV, row, fields)
            done.add(key)
            print(f"unified {spec['id']} {solver_spec['version']} {row['status']} nodes={row['bb_nodes']}")


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_float(row: dict, key: str, default: float | None = None) -> float | None:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def _win_tie_loss(rows: list[dict], strategy_key: str, baseline: str, contender: str) -> tuple[int, int, int]:
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        grouped[row["case"]][row[strategy_key]] = row
    win = tie = loss = 0
    for mapping in grouped.values():
        if baseline not in mapping or contender not in mapping:
            continue
        base = mapping[baseline]
        other = mapping[contender]
        if base["status"] == other["status"] == "最优":
            b_nodes = int(base["bb_nodes"])
            o_nodes = int(other["bb_nodes"])
            if o_nodes < b_nodes:
                win += 1
            elif o_nodes == b_nodes:
                tie += 1
            else:
                loss += 1
        elif other["status"] == "最优" and base["status"] != "最优":
            win += 1
        elif other["status"] != "最优" and base["status"] == "最优":
            loss += 1
        else:
            b_gap = _as_float(base, "relative_gap", 1e9)
            o_gap = _as_float(other, "relative_gap", 1e9)
            if o_gap < b_gap - 1e-8:
                win += 1
            elif abs(o_gap - b_gap) <= 1e-8:
                tie += 1
            else:
                loss += 1
    return win, tie, loss


def _summary(rows: list[dict], key: str, labels: list[str], baseline: str | None = None) -> list[dict]:
    out = []
    for label in labels:
        part = [row for row in rows if row[key] == label]
        completed = [row for row in part if row["status"] == "最优"]
        nodes = [int(row["bb_nodes"]) for row in completed]
        lps = [int(row["formal_lp_solved"]) for row in completed]
        times = [float(row["total_runtime_sec"]) for row in completed]
        limits = [row for row in part if row["status"] != "最优"]
        if baseline is not None and label != baseline:
            w, t, l = _win_tie_loss(rows, key, baseline, label)
            wtl = f"{w}/{t}/{l}"
        else:
            wtl = "-"
        out.append(
            {
                "label": label,
                "completed": f"{len(completed)}/{len(part)}",
                "median_nodes": "" if not nodes else median(nodes),
                "median_lps": "" if not lps else median(lps),
                "median_time": "" if not times else median(times),
                "limits": len(limits),
                "wtl": wtl,
            }
        )
    return out


def write_charts(config: dict) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    try:
        from matplotlib import font_manager

        names = {font.name for font in font_manager.fontManager.ttflist}
        for name in ("PingFang SC", "Heiti SC", "Arial Unicode MS", "Songti SC"):
            if name in names:
                plt.rcParams["font.sans-serif"] = [name]
                break
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass
    charts = REPORT_DIR / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    presolve_rows = _read_csv(PRESOLVE_QUEUE_CSV)
    branching_rows = _read_csv(BRANCHING_CSV)

    cases = [spec["id"] for spec in config["instances"]]
    raw_values = []
    filtered_values = []
    limit_cases = set()
    for case in cases:
        raw = next(row for row in presolve_rows if row.get("section") == "三项筛选" and row["case"] == case and row["presolve_config"] == "未开启筛选")
        filtered = next(row for row in presolve_rows if row.get("section") == "三项筛选" and row["case"] == case and row["presolve_config"] == "三项筛选")
        def value(row):
            if row["active_set_candidates"] == "超过候选组合上限":
                limit_cases.add(case)
                return int(config["run_settings"]["max_lp_candidates"])
            return int(row["active_set_candidates"])
        raw_values.append(value(raw))
        filtered_values.append(value(filtered))
    x = np.arange(len(cases))
    plt.figure(figsize=(9, 4.8))
    plt.bar(x - 0.18, raw_values, width=0.36, label="未开启筛选")
    plt.bar(x + 0.18, filtered_values, width=0.36, label="三项筛选")
    plt.xticks(x, cases, rotation=20, ha="right")
    plt.ylabel("active-set候选组合数")
    plt.title("三项筛选前后active-set候选组合数")
    for i, case in enumerate(cases):
        if case in limit_cases:
            plt.text(i, max(raw_values[i], filtered_values[i]), "LIMIT", ha="center", va="bottom", fontsize=8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(charts / "统一实验_筛选候选组合数.png", dpi=180)
    plt.close()

    queue_rows = [row for row in presolve_rows if row.get("section") == "节点选择"]
    dfs = [int(next(row for row in queue_rows if row["case"] == case and row["search_strategy"] == "深度优先搜索")["bb_nodes"]) for case in cases]
    best = [int(next(row for row in queue_rows if row["case"] == case and row["search_strategy"] == "最优界优先队列")["bb_nodes"]) for case in cases]
    plt.figure(figsize=(9, 4.8))
    plt.plot(cases, dfs, marker="o", label="深度优先搜索")
    plt.plot(cases, best, marker="o", label="最优界优先队列")
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("B&B正式节点数")
    plt.title("DFS与最优界优先队列节点数")
    plt.legend()
    plt.tight_layout()
    plt.savefig(charts / "统一实验_队列节点数.png", dpi=180)
    plt.close()

    labels = ["最接近0.5分支（原规则）", "GNN图学习分支", "强分支（专家参考）"]
    grouped = defaultdict(dict)
    for row in branching_rows:
        if row["strategy"] in labels:
            grouped[row["case"]][row["strategy"]] = row
    plt.figure(figsize=(8, 5.2))
    base_nodes = []
    for label in labels[1:]:
        x_nodes = []
        y_nodes = []
        for case in cases:
            mapping = grouped[case]
            x_nodes.append(int(mapping[labels[0]]["bb_nodes"]))
            y_nodes.append(int(mapping[label]["bb_nodes"]))
        base_nodes.extend(x_nodes + y_nodes)
        plt.scatter(x_nodes, y_nodes, label=label, alpha=0.75)
    lim = max(base_nodes) * 1.08 if base_nodes else 1
    plt.plot([0, lim], [0, lim], color="#666666", linewidth=1)
    plt.xlabel("最接近0.5分支（原规则）节点数")
    plt.ylabel("对比策略节点数")
    plt.title("原规则、GNN、强分支节点数配对图")
    plt.xlim(0, lim)
    plt.ylim(0, lim)
    plt.legend()
    plt.tight_layout()
    plt.savefig(charts / "统一实验_分支策略节点配对.png", dpi=180)
    plt.close()


def write_report(config: dict) -> None:
    presolve_rows = _read_csv(PRESOLVE_QUEUE_CSV)
    branching_rows = _read_csv(BRANCHING_CSV)
    unified_rows = _read_csv(UNIFIED_CSV)
    write_charts(config)

    cases = [spec["id"] for spec in config["instances"]]
    table1_lines = [
        "| 案例 | 未筛选候选组合 | 三项筛选后候选组合 | 未筛选LP时间 | 筛选后LP时间 | 状态 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for case in cases:
        raw = next(row for row in presolve_rows if row.get("section") == "三项筛选" and row["case"] == case and row["presolve_config"] == "未开启筛选")
        full = next(row for row in presolve_rows if row.get("section") == "三项筛选" and row["case"] == case and row["presolve_config"] == "三项筛选")
        table1_lines.append(
            f"| {case} | {raw['active_set_candidates']} | {full['active_set_candidates']} | {raw['lp_time_sec']} | {full['lp_time_sec']} | {full['lp_status']} |"
        )

    table2_lines = [
        "| 案例 | 深度优先节点数 | 优先队列节点数 | 深度优先时间 | 优先队列时间 | 两者状态 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    queue_rows = [row for row in presolve_rows if row.get("section") == "节点选择"]
    for case in cases:
        dfs = next(row for row in queue_rows if row["case"] == case and row["search_strategy"] == "深度优先搜索")
        best = next(row for row in queue_rows if row["case"] == case and row["search_strategy"] == "最优界优先队列")
        table2_lines.append(
            f"| {case} | {dfs['bb_nodes']} | {best['bb_nodes']} | {dfs['total_runtime_sec']} | {best['total_runtime_sec']} | {dfs['status']} / {best['status']} |"
        )

    branch_summary = _summary(
        branching_rows,
        "strategy",
        ["最接近0.5分支（原规则）", "GNN图学习分支", "强分支（专家参考）"],
        baseline="最接近0.5分支（原规则）",
    )
    table3_lines = [
        "| 分支策略 | 完成实例数 | 节点数中位数 | 正式LP次数中位数 | 总时间中位数 | 相对原规则胜/平/负 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in branch_summary:
        table3_lines.append(
            f"| {row['label']} | {row['completed']} | {_format_float(row['median_nodes'])} | {_format_float(row['median_lps'])} | {_format_float(row['median_time'], 8)} | {row['wtl']} |"
        )

    gnn_labels = ["GNN图学习分支（种子1）", "GNN图学习分支（种子2）", "GNN图学习分支（种子3）"]
    gnn_summary = _summary(branching_rows, "strategy_detail", gnn_labels, baseline="最接近0.5分支（原规则）")
    gnn_lines = [
        "| GNN训练种子 | 完成实例数 | 节点数中位数 | 总时间中位数 | 相对原规则胜/平/负 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in gnn_summary:
        gnn_lines.append(
            f"| {row['label']} | {row['completed']} | {_format_float(row['median_nodes'])} | {_format_float(row['median_time'], 8)} | {row['wtl']} |"
        )

    unified_detail = [
        "| 案例 | 求解器版本 | LP筛选 | 节点选择 | 分支策略 | 状态 | B&B节点数 | 正式LP次数 | 总时间 | relative gap |",
        "|---|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in unified_rows:
        unified_detail.append(
            f"| {row['case']} | {row['solver_version']} | {row['lp_filtering']} | {row['node_selection']} | {row['branching_strategy']} | {row['status']} | {row['bb_nodes']} | {row['formal_lp_solved']} | {row['total_runtime_sec']} | {row['relative_gap']} |"
        )

    version_labels = [VERSION_ZH[key] for key in ("raw_solver", "lp_presolve", "best_bound", "gnn_enhanced")]
    version_summary = _summary(unified_rows, "solver_version", version_labels)
    version_lines = [
        "| 求解器版本 | 完成实例数 | 节点数中位数 | 正式LP次数中位数 | 总时间中位数 | LIMIT实例数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in version_summary:
        version_lines.append(
            f"| {row['label']} | {row['completed']} | {_format_float(row['median_nodes'])} | {_format_float(row['median_lps'])} | {_format_float(row['median_time'], 8)} | {row['limits']} |"
        )

    settings = config["run_settings"]
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# GNN稳定性与求解器阶段对比",
                "",
                "## 1. 历史优化路线",
                "",
                "本项目此前依次加入了三项LP筛选、最优界优先队列和GNN分支变量选择。旧的SciPy-HiGHS与Gurobi结果只作为历史参考，本轮主表全部重新使用自制active-set节点LP。",
                "",
                "## 2. 统一实验配置",
                "",
                f"- 统一实例数：{len(config['instances'])}",
                f"- 模型族：{', '.join(sorted({spec['family'] for spec in config['instances']}))}",
                "- 统一LP后端：自制active-set",
                f"- 节点上限：{settings['max_nodes']}",
                f"- 单次时间上限：{settings['time_limit_sec']} 秒",
                f"- active-set候选组合上限：{settings['max_lp_candidates']}",
                f"- 时间统计：预热1次，正式运行{settings['repeats']}次，取总时间中位数",
                "- 主GNN模型：GNN图学习分支（种子1），依据验证集指标固定选择，不按本轮测试结果挑选。",
                "",
                "四个求解器版本分别是：原始求解器、LP筛选优化版、优先队列优化版和GNN增强版。",
                "",
                "## 3. 三项LP筛选结果",
                "",
                *table1_lines,
                "",
                "## 4. 优先队列结果",
                "",
                *table2_lines,
                "",
                "## 5. 三种分支策略结果",
                "",
                "上一轮三seed稳定性验证的严格判据未通过；本轮不重新训练GNN，只冻结验证集指标最好的种子1作为主模型，并额外列出三个训练种子的结果范围。",
                "",
                *table3_lines,
                "",
                *gnn_lines,
                "",
                "## 6. 四代求解器总体结果",
                "",
                *version_lines,
                "",
                *unified_detail,
                "",
                "## 7. 当前结论",
                "",
                "- 三项筛选的作用主要体现在代表节点LP的候选组合数和LP时间变化，具体见表一；没有计算百分比。",
                "- 优先队列是否降低节点数要按同一批实例逐例看，不能用旧个位数节点样例外推。",
                "- GNN只改变分支变量选择；本轮没有使用SciPy-HiGHS、SCIP或Gurobi生成主表。",
                "- GNN推理时间已经计入总时间；若节点减少不足以抵消推理开销，表中会直接表现为总时间不占优。",
                "- LIMIT实例单独保留状态和gap，不把截断节点数当作完成求解的节点下降结论。",
                "",
                "## 8. 当前文件",
                "",
                "- `research/ml_branching/configs/unified_solver_comparison.json`：统一实例、上限、checkpoint配置。",
                "- `benchmarks/learning_branching/unified_solver_comparison.py`：统一active-set实验runner。",
                "- `research/ml_branching/reports/data/presolve_and_queue_comparison.csv`：三项筛选和队列对比原始数据。",
                "- `research/ml_branching/reports/data/branching_comparison.csv`：三种分支策略和三个GNN种子原始数据。",
                "- `research/ml_branching/reports/data/unified_comparison.csv`：LP后端与分支策略的2×2逐实例结果。",
                "- `research/ml_branching/reports/charts/`：三张中文图表。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _table_lines_presolve(presolve_rows: list[dict]) -> list[str]:
    cases = [
        row["case"]
        for row in presolve_rows
        if row.get("section") == "三项筛选" and row.get("presolve_config") == "未开启筛选"
    ]
    lines = [
        "| 案例 | 未筛选候选组合 | 三项筛选后候选组合 | 未筛选LP时间 | 筛选后LP时间 | 状态 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for case in cases:
        raw = next(row for row in presolve_rows if row.get("section") == "三项筛选" and row["case"] == case and row["presolve_config"] == "未开启筛选")
        full = next(row for row in presolve_rows if row.get("section") == "三项筛选" and row["case"] == case and row["presolve_config"] == "三项筛选")
        lines.append(f"| {case} | {raw['active_set_candidates']} | {full['active_set_candidates']} | {raw['lp_time_sec']} | {full['lp_time_sec']} | {full['lp_status']} |")
    return lines


def _table_lines_queue(presolve_rows: list[dict]) -> list[str]:
    queue_rows = [row for row in presolve_rows if row.get("section") == "节点选择"]
    cases = []
    for row in queue_rows:
        if row["case"] not in cases:
            cases.append(row["case"])
    lines = [
        "| 案例 | 深度优先节点数 | 优先队列节点数 | 深度优先时间 | 优先队列时间 | 两者状态 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for case in cases:
        dfs = next(row for row in queue_rows if row["case"] == case and row["search_strategy"] == "深度优先搜索")
        best = next(row for row in queue_rows if row["case"] == case and row["search_strategy"] == "最优界优先队列")
        lines.append(f"| {case} | {dfs['bb_nodes']} | {best['bb_nodes']} | {dfs['total_runtime_sec']} | {best['total_runtime_sec']} | {dfs['status']} / {best['status']} |")
    return lines


def _format_summary_table(rows: list[dict], key: str, labels: list[str], baseline: str | None, first_header: str) -> list[str]:
    summary = _summary(rows, key, labels, baseline=baseline)
    lines = [
        f"| {first_header} | 完成实例数 | 节点数中位数 | 正式LP次数中位数 | 总时间中位数 | 相对原规则胜/平/负 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(f"| {row['label']} | {row['completed']} | {_format_float(row['median_nodes'])} | {_format_float(row['median_lps'])} | {_format_float(row['median_time'], 8)} | {row['wtl']} |")
    return lines


def write_corrected_report(config: dict) -> None:
    presolve_rows = _read_csv(PRESOLVE_QUEUE_CSV)
    branching_rows = _read_csv(BRANCHING_CSV)
    stability_rows = _read_csv(STABILITY_CSV)
    unified_rows = _read_csv(UNIFIED_CSV)

    table1_lines = _table_lines_presolve(presolve_rows)
    table2_lines = _table_lines_queue(presolve_rows)
    table3_lines = _format_summary_table(
        branching_rows,
        "strategy",
        ["最接近0.5分支（原规则）", "GNN图学习分支", "强分支（专家参考）"],
        "最接近0.5分支（原规则）",
        "分支方法",
    )
    gnn_lines = _format_summary_table(
        stability_rows,
        "strategy",
        ["GNN图学习分支（种子1）", "GNN图学习分支（种子2）", "GNN图学习分支（种子3）"],
        "最接近0.5分支（原规则）",
        "GNN训练种子",
    )
    version_labels = [VERSION_ZH[key] for key in ("active_original", "active_gnn", "highs_original", "highs_gnn")]
    version_summary = _summary(unified_rows, "version", version_labels)
    version_lines = [
        "| 版本 | 节点LP | 分支方法 | 完成实例数 | 节点中位数 | 正式LP中位数 | 总时间中位数 | LIMIT实例数 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in version_summary:
        sample = next(item for item in unified_rows if item["version"] == row["label"])
        version_lines.append(
            f"| {row['label']} | {sample['node_lp']} | {sample['branching_method']} | {row['completed']} | {_format_float(row['median_nodes'])} | {_format_float(row['median_lps'])} | {_format_float(row['median_time'], 8)} | {row['limits']} |"
        )

    unified_detail = [
        "| 案例 | 版本 | 状态 | 节点数 | 正式LP数 | 推理时间 | 总时间 | relative gap |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in unified_rows:
        unified_detail.append(
            f"| {row['case']} | {row['version']} | {row['status']} | {row['bb_nodes']} | {row['formal_lp_solved']} | {row['inference_time_sec']} | {row['total_runtime_sec']} | {row['relative_gap']} |"
        )

    REPORT_PATH.write_text(
        "\n".join(
            [
                "# GNN稳定性与求解器对比",
                "",
                "## 1. 上次组会阶段进展",
                "",
                "上次组会阶段主要验证了两件事：三项LP筛选减少单节点active-set枚举，深度优先搜索改为最优界优先队列后可以改变开放节点访问顺序。下面两张表沿用上一轮已经跑完的数据，本轮没有重复执行耗时的active-set全量benchmark。",
                "",
                "### 三项筛选对active-set节点LP的影响",
                "",
                *table1_lines,
                "",
                "### 深度优先搜索与最优界优先队列对比",
                "",
                *table2_lines,
                "",
                "## 2. GNN分支方法",
                "",
                "- 最接近0.5分支（原规则）：在当前LP解中选择最接近0.5的二元变量分支。",
                "- GNN图学习分支：把当前节点LP、约束矩阵和候选变量构造成通用二部图，用冻结的GNN checkpoint给候选变量打分。",
                "- 强分支（专家参考）：对候选变量做试探分支并求子节点LP，用更昂贵的局部信息选择变量。",
                "",
                "## 3. 困难UC实例上的三种分支方法",
                "",
                "本表使用30个固定unit commitment实例，统一使用三项LP筛选、最优界优先队列和SciPy-HiGHS节点LP。没有重新训练GNN，也没有使用MLP或伪成本策略。",
                "",
                *table3_lines,
                "",
                *gnn_lines,
                "",
                "## 4. LP后端与GNN的2×2总体对比",
                "",
                "总表固定三项LP筛选、最优界优先队列、相同剪枝逻辑和相同上限，只切换节点LP后端与分支方法。",
                "",
                *version_lines,
                "",
                *unified_detail,
                "",
                "## 5. 结果解释",
                "",
                "- 三项LP筛选减少的是单个节点LP的active-set候选组合数和LP求解时间。",
                "- 最优界优先队列改变的是开放节点访问顺序，不改变模型、LP后端或分支规则。",
                "- GNN改变的是当前节点选择哪个二元变量分支。",
                "- SciPy-HiGHS替换的是节点LP后端，B&B主循环、剪枝和incumbent逻辑仍由本项目自写。",
                "- 因此，表三用于看分支策略，2×2总表用于区分LP后端和GNN分支的组合效果。",
                "",
                "## 6. 上一轮8个中小实例",
                "",
                "上一轮8个中小实例中，GNN相对原规则为1/7/0，强分支相对原规则也是1/7/0，说明那组实例对分支变量选择不敏感，只适合验证流程正确性，不适合作为GNN性能结论。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Unified active-set solver comparison.")
    parser.add_argument("command", choices=["run", "run-branching-uc", "run-unified", "report"])
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "run":
        run_branching_uc(config, resume=not args.no_resume)
        run_unified_2x2(config, resume=not args.no_resume)
    elif args.command == "run-branching-uc":
        run_branching_uc(config, resume=not args.no_resume)
    elif args.command == "run-unified":
        run_unified_2x2(config, resume=not args.no_resume)
    elif args.command == "report":
        write_corrected_report(config)


if __name__ == "__main__":
    main()
