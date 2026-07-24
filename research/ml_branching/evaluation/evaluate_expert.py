from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.ml_branching.families import reconstruct_instance
from research.ml_branching.training.oracle.strong_branching import StrongBranchingPolicy
from solver import solve_milp


FIELDNAMES = [
    "split",
    "family_name",
    "scale_group",
    "instance_id",
    "seed",
    "units",
    "strategy",
    "status",
    "objective",
    "formal_nodes",
    "formal_node_lp_solved",
    "probe_lp_solved",
    "solver_runtime_sec",
    "probe_runtime_sec",
    "total_runtime_sec",
    "pruned_infeasible",
    "pruned_bound",
    "global_bound",
    "relative_gap",
    "objective_match",
    "node_reduction_vs_most_fractional",
    "formal_lp_reduction_vs_most_fractional",
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
            sample_path = Path.cwd() / sample_path
        with sample_path.open("rb"):
            pass
        import numpy as np

        with np.load(sample_path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
        rows.append(metadata["instance_parameters"])
    return rows


def _solve_row(params: dict, strategy: str, policy, max_nodes: int, tolerance: float) -> tuple[dict, object]:
    instance = reconstruct_instance(params)
    if hasattr(policy, "probe_lp_solved"):
        policy.probe_lp_solved = 0
        policy.probe_time_sec = 0.0
    start = perf_counter()
    result = solve_milp(
        instance.problem,
        lp_backend="scipy_highs",
        branching_policy=policy,
        max_nodes=max_nodes,
        use_matrix_presolve=True,
        tol=tolerance,
    )
    elapsed = perf_counter() - start
    row = {
        "split": params.get("split", ""),
        "family_name": params["family_name"],
        "scale_group": params.get("scale_group", ""),
        "instance_id": params["instance_id"],
        "seed": str(params["seed"]),
        "units": str(params.get("units", params.get("size", ""))),
        "strategy": strategy,
        "status": result.status,
        "objective": "" if result.objective_value is None else f"{float(result.objective_value):.10g}",
        "formal_nodes": str(result.num_nodes),
        "formal_node_lp_solved": str(result.num_lp_solved),
        "probe_lp_solved": str(int(getattr(policy, "probe_lp_solved", 0))),
        "solver_runtime_sec": f"{float(result.runtime_sec):.6f}",
        "probe_runtime_sec": f"{float(getattr(policy, 'probe_time_sec', 0.0)):.6f}",
        "total_runtime_sec": f"{elapsed:.6f}",
        "pruned_infeasible": str(result.num_pruned_infeasible),
        "pruned_bound": str(result.num_pruned_bound),
        "global_bound": "" if result.global_bound is None else f"{float(result.global_bound):.10g}",
        "relative_gap": "" if result.relative_gap is None else f"{float(result.relative_gap):.10g}",
        "objective_match": "",
        "node_reduction_vs_most_fractional": "",
        "formal_lp_reduction_vs_most_fractional": "",
    }
    return row, result


def _summarize(rows: list[dict]) -> dict:
    summary = {}
    for strategy in sorted({row["strategy"] for row in rows}):
        part = [row for row in rows if row["strategy"] == strategy]
        nodes = [float(row["formal_nodes"]) for row in part]
        formal_lps = [float(row["formal_node_lp_solved"]) for row in part]
        total_times = [float(row["total_runtime_sec"]) for row in part]
        completed = [row for row in part if row["status"] == "optimal"]
        summary[strategy] = {
            "count": len(part),
            "completed_ratio": len(completed) / len(part) if part else 0.0,
            "mean_nodes": mean(nodes) if nodes else 0.0,
            "median_nodes": median(nodes) if nodes else 0.0,
            "mean_formal_lp_solved": mean(formal_lps) if formal_lps else 0.0,
            "mean_total_runtime_sec": mean(total_times) if total_times else 0.0,
            "objective_match_ratio": (
                None
                if not [row for row in part if row["objective_match"]]
                else mean([row["objective_match"] == "True" for row in part if row["objective_match"]])
            ),
        }
    return summary


def _by_family(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["family_name"]].append(row)
    return {family: _summarize(part) for family, part in sorted(grouped.items())}


def evaluate_expert(
    dataset_path: str | Path,
    report_dir: str | Path,
    splits: list[str] | None = None,
    max_nodes: int = 500,
    tolerance: float = 1e-8,
) -> dict:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    params_rows = _instance_parameters(dataset_path, set(splits) if splits else None)
    rows: list[dict] = []
    for params in params_rows:
        most_row, most_result = _solve_row(params, "most_fractional", None, max_nodes, tolerance)
        strong_policy = StrongBranchingPolicy(lp_backend="scipy_highs")
        strong_row, strong_result = _solve_row(params, "strong_branching", strong_policy, max_nodes, tolerance)
        if most_result.objective_value is not None and strong_result.objective_value is not None:
            match = abs(float(most_result.objective_value) - float(strong_result.objective_value)) <= 1e-7
            most_row["objective_match"] = str(match)
            strong_row["objective_match"] = str(match)
        strong_row["node_reduction_vs_most_fractional"] = str(most_result.num_nodes - strong_result.num_nodes)
        strong_row["formal_lp_reduction_vs_most_fractional"] = str(most_result.num_lp_solved - strong_result.num_lp_solved)
        rows.extend([most_row, strong_row])

    csv_path = report_dir / "expert_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "dataset": str(dataset_path),
        "instance_count": len(params_rows),
        "splits": splits or "all",
        "summary": _summarize(rows),
        "by_family": _by_family(rows),
        "rows": rows,
    }
    (report_dir / "expert_comparison_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    md = [
        "# Strong Branching 专家收益检查",
        "",
        f"- dataset: `{dataset_path}`",
        f"- instances: `{len(params_rows)}`",
        "",
        "```json",
        json.dumps(result["summary"], indent=2, sort_keys=True, ensure_ascii=False),
        "```",
    ]
    (report_dir / "expert_comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return result


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Compare most_fractional with strong branching before training.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--report-dir", default="research/ml_branching/reports")
    parser.add_argument("--splits", default=None, help="Comma-separated splits; default uses all accepted instances.")
    parser.add_argument("--max-nodes", type=int, default=500)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    splits = [part.strip() for part in args.splits.split(",") if part.strip()] if args.splits else None
    return evaluate_expert(args.dataset, args.report_dir, splits, args.max_nodes, args.tolerance)


if __name__ == "__main__":
    main()
