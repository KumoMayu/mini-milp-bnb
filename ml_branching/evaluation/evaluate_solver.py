from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_branching.families import reconstruct_instance
from ml_branching.training.oracle.strong_branching import StrongBranchingPolicy
from solver import solve_milp
from solver.branch_and_bound import check_feasibility


FIELDNAMES = [
    "split",
    "family_name",
    "scale_group",
    "instance_id",
    "seed",
    "units",
    "num_variables",
    "num_binary_variables",
    "num_constraints",
    "strategy",
    "status",
    "objective",
    "solution_feasible",
    "formal_nodes",
    "formal_node_lp_solved",
    "probe_lp_solved",
    "solver_runtime_sec",
    "probe_runtime_sec",
    "inference_time_sec",
    "total_runtime_sec",
    "pruned_infeasible",
    "pruned_bound",
    "incumbent_value",
    "global_bound",
    "relative_gap",
    "limit_type",
    "matches_gurobi",
    "gurobi_status",
    "gurobi_objective",
    "gurobi_best_bound",
    "gurobi_mip_gap",
    "gurobi_runtime_sec",
    "gurobi_node_count",
    "gurobi_solution_count",
    "gurobi_time_limit_reached",
    "gurobi_optimal",
    "gurobi_threads",
    "gurobi_seed",
    "gurobi_time_limit",
]


def _instance_parameters(dataset_path: str | Path, splits: set[str] | None = None) -> list[dict]:
    manifest = json.loads((Path(dataset_path) / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    seen = set()
    for row in manifest.get("instances", []):
        if row.get("skipped") or not row.get("sample_paths"):
            continue
        if splits and row.get("split") not in splits:
            continue
        if row["instance_id"] in seen:
            continue
        seen.add(row["instance_id"])
        sample_path = Path(row["sample_paths"][0])
        if not sample_path.is_absolute():
            sample_path = ROOT / sample_path
        with np.load(sample_path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
        rows.append(metadata["instance_parameters"])
    return rows


def _gurobi_status_name(code: int) -> str:
    try:
        import gurobipy as gp

        mapping = {
            gp.GRB.OPTIMAL: "optimal",
            gp.GRB.TIME_LIMIT: "time_limit",
            gp.GRB.INFEASIBLE: "infeasible",
            gp.GRB.INF_OR_UNBD: "inf_or_unbd",
            gp.GRB.UNBOUNDED: "unbounded",
        }
        return mapping.get(code, f"gurobi_status_{code}")
    except Exception:
        return f"gurobi_status_{code}"


def _solve_gurobi(problem, time_limit: float, seed: int, threads: int) -> dict:
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as exc:
        return {
            "status": "not_available",
            "error": f"{type(exc).__name__}: {exc}",
            "objective": "",
            "best_bound": "",
            "mip_gap": "",
            "runtime_sec": "",
            "node_count": "",
            "solution_count": "",
            "time_limit_reached": "",
            "optimal": "False",
            "threads": str(threads),
            "seed": str(seed),
            "time_limit": str(time_limit),
        }
    model = gp.Model(problem.name)
    model.Params.OutputFlag = 0
    model.Params.Threads = int(threads)
    model.Params.Seed = int(seed)
    model.Params.TimeLimit = float(time_limit)
    variables = []
    for j, var_type in enumerate(problem.var_types):
        vtype = GRB.BINARY if var_type == "B" else GRB.CONTINUOUS
        variables.append(model.addVar(lb=float(problem.lb[j]), ub=float(problem.ub[j]), vtype=vtype, name=f"z{j}"))
    objective = gp.quicksum(float(problem.c[j]) * variables[j] for j in range(problem.num_vars))
    model.setObjective(objective, GRB.MINIMIZE if problem.sense == "min" else GRB.MAXIMIZE)
    for i in range(problem.num_constraints):
        model.addConstr(gp.quicksum(float(problem.G[i, j]) * variables[j] for j in range(problem.num_vars)) <= float(problem.h[i]))
    model.optimize()
    status = _gurobi_status_name(model.Status)
    solution_count = int(model.SolCount)
    objective_value = "" if solution_count <= 0 else f"{float(model.ObjVal):.10g}"
    best_bound = ""
    mip_gap = ""
    try:
        best_bound = f"{float(model.ObjBound):.10g}"
    except Exception:
        pass
    try:
        mip_gap = f"{float(model.MIPGap):.10g}"
    except Exception:
        pass
    return {
        "status": status,
        "error": "",
        "objective": objective_value,
        "best_bound": best_bound,
        "mip_gap": mip_gap,
        "runtime_sec": f"{float(model.Runtime):.6f}",
        "node_count": f"{float(model.NodeCount):.0f}",
        "solution_count": str(solution_count),
        "time_limit_reached": str(model.Status == GRB.TIME_LIMIT),
        "optimal": str(model.Status == GRB.OPTIMAL),
        "threads": str(threads),
        "seed": str(seed),
        "time_limit": str(time_limit),
    }


def _base_row(params: dict, problem) -> dict:
    stats = params.get("stats", {})
    return {
        "split": params.get("split", ""),
        "family_name": params["family_name"],
        "scale_group": params.get("scale_group", ""),
        "instance_id": params["instance_id"],
        "seed": str(params["seed"]),
        "units": str(params.get("units", params.get("size", ""))),
        "num_variables": str(stats.get("num_variables", problem.num_vars)),
        "num_binary_variables": str(stats.get("num_binary", len(problem.binary_indices))),
        "num_constraints": str(stats.get("num_constraints", problem.num_constraints)),
    }


def _solver_row(params: dict, strategy: str, policy, max_nodes: int, tolerance: float, gurobi: dict) -> dict:
    instance = reconstruct_instance(params)
    problem = instance.problem
    if hasattr(policy, "probe_lp_solved"):
        policy.probe_lp_solved = 0
        policy.probe_time_sec = 0.0
    if hasattr(policy, "inference_time_sec"):
        policy.inference_time_sec = 0.0
        policy.inference_calls = 0
    start = perf_counter()
    result = solve_milp(
        problem,
        lp_backend="scipy_highs",
        branching_policy=policy,
        max_nodes=max_nodes,
        use_matrix_presolve=True,
        tol=tolerance,
    )
    elapsed = perf_counter() - start
    feasible = False
    if result.x is not None:
        feasible = check_feasibility(problem, result.x, problem.lb, problem.ub, tolerance)
    matches = ""
    if result.objective_value is not None and gurobi.get("objective") and result.status == "optimal":
        matches = str(abs(float(result.objective_value) - float(gurobi["objective"])) <= 1e-7)
    row = _base_row(params, problem)
    row.update(
        {
            "strategy": strategy,
            "status": result.status,
            "objective": "" if result.objective_value is None else f"{float(result.objective_value):.10g}",
            "solution_feasible": str(bool(feasible)),
            "formal_nodes": str(result.num_nodes),
            "formal_node_lp_solved": str(result.num_lp_solved),
            "probe_lp_solved": str(int(getattr(policy, "probe_lp_solved", 0))),
            "solver_runtime_sec": f"{float(result.runtime_sec):.6f}",
            "probe_runtime_sec": f"{float(getattr(policy, 'probe_time_sec', 0.0)):.6f}",
            "inference_time_sec": f"{float(getattr(policy, 'inference_time_sec', 0.0)):.6f}",
            "total_runtime_sec": f"{elapsed:.6f}",
            "pruned_infeasible": str(result.num_pruned_infeasible),
            "pruned_bound": str(result.num_pruned_bound),
            "incumbent_value": "" if result.objective_value is None else f"{float(result.objective_value):.10g}",
            "global_bound": "" if result.global_bound is None else f"{float(result.global_bound):.10g}",
            "relative_gap": "" if result.relative_gap is None else f"{float(result.relative_gap):.10g}",
            "limit_type": "" if result.status == "optimal" else result.status,
            "matches_gurobi": matches,
            "gurobi_status": gurobi.get("status", ""),
            "gurobi_objective": gurobi.get("objective", ""),
            "gurobi_best_bound": gurobi.get("best_bound", ""),
            "gurobi_mip_gap": gurobi.get("mip_gap", ""),
            "gurobi_runtime_sec": gurobi.get("runtime_sec", ""),
            "gurobi_node_count": gurobi.get("node_count", ""),
            "gurobi_solution_count": gurobi.get("solution_count", ""),
            "gurobi_time_limit_reached": gurobi.get("time_limit_reached", ""),
            "gurobi_optimal": gurobi.get("optimal", ""),
            "gurobi_threads": gurobi.get("threads", ""),
            "gurobi_seed": gurobi.get("seed", ""),
            "gurobi_time_limit": gurobi.get("time_limit", ""),
        }
    )
    return row


def _gurobi_row(params: dict, problem, gurobi: dict) -> dict:
    row = _base_row(params, problem)
    row.update({field: "" for field in FIELDNAMES if field not in row})
    row.update(
        {
            "strategy": "gurobi",
            "status": gurobi.get("status", ""),
            "objective": gurobi.get("objective", ""),
            "formal_nodes": gurobi.get("node_count", ""),
            "formal_node_lp_solved": "",
            "total_runtime_sec": gurobi.get("runtime_sec", ""),
            "global_bound": gurobi.get("best_bound", ""),
            "relative_gap": gurobi.get("mip_gap", ""),
            "gurobi_status": gurobi.get("status", ""),
            "gurobi_objective": gurobi.get("objective", ""),
            "gurobi_best_bound": gurobi.get("best_bound", ""),
            "gurobi_mip_gap": gurobi.get("mip_gap", ""),
            "gurobi_runtime_sec": gurobi.get("runtime_sec", ""),
            "gurobi_node_count": gurobi.get("node_count", ""),
            "gurobi_solution_count": gurobi.get("solution_count", ""),
            "gurobi_time_limit_reached": gurobi.get("time_limit_reached", ""),
            "gurobi_optimal": gurobi.get("optimal", ""),
            "gurobi_threads": gurobi.get("threads", ""),
            "gurobi_seed": gurobi.get("seed", ""),
            "gurobi_time_limit": gurobi.get("time_limit", ""),
        }
    )
    return row


def _summary(rows: list[dict]) -> dict:
    out = {}
    for strategy in sorted({row["strategy"] for row in rows}):
        part = [row for row in rows if row["strategy"] == strategy]
        nodes = [float(row["formal_nodes"]) for row in part if row["formal_nodes"]]
        lps = [float(row["formal_node_lp_solved"]) for row in part if row["formal_node_lp_solved"]]
        times = [float(row["total_runtime_sec"]) for row in part if row["total_runtime_sec"]]
        completed = [row for row in part if row["status"] == "optimal"]
        matches = [row["matches_gurobi"] == "True" for row in part if row["matches_gurobi"]]
        out[strategy] = {
            "count": len(part),
            "completed_ratio": len(completed) / len(part) if part else 0.0,
            "objective_gurobi_match_ratio": None if not matches else mean(matches),
            "mean_nodes": mean(nodes) if nodes else None,
            "median_nodes": median(nodes) if nodes else None,
            "mean_formal_lp_solved": mean(lps) if lps else None,
            "mean_total_runtime_sec": mean(times) if times else None,
            "limit_ratio": sum(row["status"] != "optimal" for row in part) / len(part) if part else 0.0,
        }
    return out


def _by_family(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["family_name"]].append(row)
    return {family: _summary(part) for family, part in sorted(grouped.items())}


def write_reports(rows: list[dict], result: dict, report_dir: str | Path) -> None:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "solver_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    gurobi_csv = report_dir / "gurobi_comparison.csv"
    with gurobi_csv.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "split",
            "family_name",
            "scale_group",
            "instance_id",
            "num_variables",
            "num_binary_variables",
            "num_constraints",
            "status",
            "objective",
            "best_bound",
            "mip_gap",
            "runtime_sec",
            "node_count",
            "solution_count",
            "time_limit_reached",
            "optimal",
            "threads",
            "seed",
            "time_limit",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if row["strategy"] == "gurobi":
                writer.writerow(
                    {
                        "split": row["split"],
                        "family_name": row["family_name"],
                        "scale_group": row["scale_group"],
                        "instance_id": row["instance_id"],
                        "num_variables": row["num_variables"],
                        "num_binary_variables": row["num_binary_variables"],
                        "num_constraints": row["num_constraints"],
                        "status": row["gurobi_status"],
                        "objective": row["gurobi_objective"],
                        "best_bound": row["gurobi_best_bound"],
                        "mip_gap": row["gurobi_mip_gap"],
                        "runtime_sec": row["gurobi_runtime_sec"],
                        "node_count": row["gurobi_node_count"],
                        "solution_count": row["gurobi_solution_count"],
                        "time_limit_reached": row["gurobi_time_limit_reached"],
                        "optimal": row["gurobi_optimal"],
                        "threads": row["gurobi_threads"],
                        "seed": row["gurobi_seed"],
                        "time_limit": row["gurobi_time_limit"],
                    }
                )
    md_path = report_dir / "solver_comparison.md"
    md_path.write_text(
        "# Solver 级分支策略对比\n\n"
        f"- checkpoint: `{result['checkpoint']}`\n"
        f"- dataset: `{result['dataset']}`\n"
        f"- splits: `{result['splits']}`\n"
        f"- instances: `{result['instance_count']}`\n"
        f"- LP backend: `scipy_highs`\n"
        f"- node selection: `best-bound`\n\n"
        "Gurobi 使用完整 MILP 求解，节点数不与自写 B&B 节点数直接等价。\n\n"
        "```json\n"
        + json.dumps(result["summary"], indent=2, sort_keys=True, ensure_ascii=False)
        + "\n```\n",
        encoding="utf-8",
    )


def evaluate_solver(
    checkpoint: str | Path,
    dataset: str | Path,
    report_dir: str | Path,
    splits: list[str],
    max_nodes: int,
    tolerance: float,
    device: str,
    gurobi_time_limit: float,
    gurobi_seed: int,
    gurobi_threads: int,
) -> dict:
    from ml_branching.runtime.inference import LearnedBranchingPolicy

    learned_policy = LearnedBranchingPolicy.from_checkpoint(checkpoint, device=device)
    rows: list[dict] = []
    params_rows = _instance_parameters(dataset, set(splits))
    for params in params_rows:
        instance = reconstruct_instance(params)
        gurobi = _solve_gurobi(instance.problem, gurobi_time_limit, gurobi_seed, gurobi_threads)
        rows.append(_gurobi_row(params, instance.problem, gurobi))
        policies = [
            ("most_fractional", None),
            ("strong_branching", StrongBranchingPolicy(lp_backend="scipy_highs")),
            ("learned_mlp", learned_policy),
        ]
        for strategy, policy in policies:
            rows.append(_solver_row(params, strategy, policy, max_nodes, tolerance, gurobi))
    result = {
        "checkpoint": str(checkpoint),
        "dataset": str(dataset),
        "splits": splits,
        "instance_count": len(params_rows),
        "summary": _summary(rows),
        "by_family": _by_family(rows),
        "rows": rows,
    }
    write_reports(rows, result, report_dir)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return result


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Compare most_fractional, strong_branching, learned MLP, and Gurobi.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="ml_branching/data/generated/train")
    parser.add_argument("--report-dir", default="reports/learning_branching")
    parser.add_argument(
        "--splits",
        default="in_distribution_test,scale_extrapolation_test,family_holdout_test",
        help="Comma-separated instance splits to solve.",
    )
    parser.add_argument("--max-nodes", type=int, default=500)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gurobi-seed", type=int, default=20260715)
    parser.add_argument("--gurobi-threads", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    splits = [part.strip() for part in args.splits.split(",") if part.strip()]
    return evaluate_solver(
        checkpoint=args.checkpoint,
        dataset=args.dataset,
        report_dir=args.report_dir,
        splits=splits,
        max_nodes=args.max_nodes,
        tolerance=args.tolerance,
        device=args.device,
        gurobi_time_limit=args.time_limit,
        gurobi_seed=args.gurobi_seed,
        gurobi_threads=args.gurobi_threads,
    )


if __name__ == "__main__":
    main()
