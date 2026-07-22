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

from ml_branching.training.data.dataset import BranchingDataset
from ml_branching.runtime.inference import LearnedBranchingPolicy, LearnedGNNBranchingPolicy
from ml_branching.training.oracle.strong_branching import StrongBranchingPolicy
from ml_branching.unit_commitment import generate_from_config, instance_from_parameters
from solver import solve_milp
from solver.branch_and_bound import check_feasibility
from solver.branching import MostFractionalPolicy


REPORT_DIR = Path("reports/learning_branching")
COMPARISON_CSV = REPORT_DIR / "gnn_solver_comparison.csv"
OFFLINE_CSV = REPORT_DIR / "gnn_offline_evaluation.csv"
MAIN_REPORT = REPORT_DIR / "GNN分支策略实验总览.md"


FIELDNAMES = [
    "split",
    "scale_group",
    "instance_id",
    "seed",
    "units",
    "periods",
    "budget",
    "strategy",
    "status",
    "objective",
    "solution_feasible",
    "formal_nodes",
    "formal_lp_solved",
    "probe_lp_solved",
    "probe_runtime_sec",
    "inference_time_sec",
    "solver_runtime_sec",
    "total_runtime_sec",
    "global_bound",
    "relative_gap",
    "limit_type",
]


def load_benchmark_config(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return dict(data.get("benchmark", data))


def _dataset_instance_parameters(dataset_path: str | Path, splits: set[str]) -> list[dict]:
    manifest = json.loads((Path(dataset_path) / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    seen = set()
    for row in manifest.get("instances", []):
        parameters = row.get("parameters")
        if parameters is None and row.get("sample_paths"):
            sample_path = Path(row["sample_paths"][0])
            if not sample_path.is_absolute():
                sample_path = ROOT / sample_path
            with np.load(sample_path, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata_json"].item()))
            parameters = metadata["instance_parameters"]
        if not parameters or parameters.get("split") not in splits:
            continue
        if parameters["instance_id"] in seen:
            continue
        seen.add(parameters["instance_id"])
        rows.append(parameters)
    return rows


def benchmark_instances(config: dict) -> list[dict]:
    if "dataset_path" in config:
        splits = set(config.get("test_splits", ["in_distribution_test", "scale_extrapolation_test"]))
        rows = _dataset_instance_parameters(config["dataset_path"], splits)
        max_instances = config.get("max_instances")
        return rows[: int(max_instances)] if max_instances is not None else rows
    master_seed = int(config.get("master_seed", 20260715))
    specs = config.get("benchmark_splits", {})
    rows = []
    for split, spec in specs.items():
        count = int(spec.get("instances", 1))
        for index in range(count):
            rows.append(generate_from_config(split, spec, index, master_seed).parameters)
    return rows


def _make_policy(name: str, config: dict):
    if name == "most_fractional":
        return MostFractionalPolicy()
    if name == "strong_branching":
        return StrongBranchingPolicy(lp_backend="scipy_highs", use_matrix_presolve=True)
    if name == "mlp":
        checkpoint = config.get("mlp_checkpoint")
        if not checkpoint:
            raise ValueError("mlp strategy requires mlp_checkpoint")
        return LearnedBranchingPolicy.from_checkpoint(checkpoint, device=config.get("device", "cpu"))
    if name.startswith("gnn"):
        checkpoints = dict(config.get("gnn_checkpoints", {}))
        checkpoint = checkpoints.get(name)
        if not checkpoint:
            raise ValueError(f"{name} strategy requires config.gnn_checkpoints.{name}")
        return LearnedGNNBranchingPolicy.from_checkpoint(checkpoint, device=config.get("device", "cpu"))
    raise ValueError(f"unknown strategy={name!r}")


def _already_done(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as f:
        return {(row["instance_id"], row["strategy"], row["budget"]) for row in csv.DictReader(f)}


def _append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in FIELDNAMES})


def solve_one(parameters: dict, strategy: str, budget: int, policy, tolerance: float) -> dict:
    instance = instance_from_parameters(parameters)
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
        branching_policy=policy,
        lp_backend="scipy_highs",
        use_matrix_presolve=True,
        max_nodes=int(budget),
        tol=float(tolerance),
    )
    elapsed = perf_counter() - start
    feasible = False
    if result.x is not None:
        feasible = check_feasibility(problem, result.x, problem.lb, problem.ub, tolerance)
    return {
        "split": parameters.get("split", ""),
        "scale_group": parameters.get("scale_group", ""),
        "instance_id": parameters["instance_id"],
        "seed": str(parameters["seed"]),
        "units": str(parameters.get("units", parameters.get("size", ""))),
        "periods": str(parameters.get("periods", "")),
        "budget": str(int(budget)),
        "strategy": strategy,
        "status": result.status,
        "objective": "" if result.objective_value is None else f"{float(result.objective_value):.10g}",
        "solution_feasible": str(bool(feasible)),
        "formal_nodes": str(int(result.num_nodes)),
        "formal_lp_solved": str(int(result.num_lp_solved)),
        "probe_lp_solved": str(int(getattr(policy, "probe_lp_solved", 0))),
        "probe_runtime_sec": f"{float(getattr(policy, 'probe_time_sec', 0.0)):.6f}",
        "inference_time_sec": f"{float(getattr(policy, 'inference_time_sec', 0.0)):.6f}",
        "solver_runtime_sec": f"{float(result.runtime_sec):.6f}",
        "total_runtime_sec": f"{elapsed:.6f}",
        "global_bound": "" if result.global_bound is None else f"{float(result.global_bound):.10g}",
        "relative_gap": "" if result.relative_gap is None else f"{float(result.relative_gap):.10g}",
        "limit_type": "" if result.status == "optimal" else result.status,
    }


def run_benchmark(config: dict, resume: bool = True) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if not resume and COMPARISON_CSV.exists():
        COMPARISON_CSV.unlink()
    done = _already_done(COMPARISON_CSV) if resume else set()
    rows = benchmark_instances(config)
    strategies = list(config.get("strategies", ["most_fractional", "strong_branching"]))
    budgets = [int(b) for b in config.get("budgets", [300])]
    tolerance = float(config.get("tolerance", 1e-8))
    for parameters in rows:
        for budget in budgets:
            for strategy in strategies:
                key = (str(parameters["instance_id"]), strategy, str(int(budget)))
                if key in done:
                    continue
                policy = _make_policy(strategy, config)
                row = solve_one(parameters, strategy, budget, policy, tolerance)
                _append_row(COMPARISON_CSV, row)
                done.add(key)
                print(
                    f"{row['instance_id']} budget={budget} strategy={strategy} "
                    f"status={row['status']} nodes={row['formal_nodes']} gap={row['relative_gap']}"
                )
    return COMPARISON_CSV


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float_or_none(text: str) -> float | None:
    if text in {"", "None", None}:
        return None
    return float(text)


def summarize_strategy_rows(rows: list[dict], budget: int) -> list[dict]:
    filtered = [row for row in rows if int(row["budget"]) == int(budget)]
    strategies = sorted({row["strategy"] for row in filtered})
    by_instance: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in filtered:
        by_instance[row["instance_id"]][row["strategy"]] = row
    summaries = []
    for strategy in strategies:
        strategy_rows = [row for row in filtered if row["strategy"] == strategy]
        completed = [row for row in strategy_rows if row["status"] == "optimal"]
        nodes = [int(row["formal_nodes"]) for row in completed]
        gaps = [_float_or_none(row["relative_gap"]) for row in strategy_rows]
        gaps = [gap for gap in gaps if gap is not None]
        wins = ties = losses = 0
        if strategy != "most_fractional":
            for instance_id, mapping in by_instance.items():
                if "most_fractional" not in mapping or strategy not in mapping:
                    continue
                base = mapping["most_fractional"]
                cur = mapping[strategy]
                base_gap = _float_or_none(base["relative_gap"])
                cur_gap = _float_or_none(cur["relative_gap"])
                if base["status"] == cur["status"] == "optimal":
                    b_nodes = int(base["formal_nodes"])
                    c_nodes = int(cur["formal_nodes"])
                    if c_nodes < b_nodes:
                        wins += 1
                    elif c_nodes == b_nodes:
                        ties += 1
                    else:
                        losses += 1
                elif cur["status"] == "optimal" and base["status"] != "optimal":
                    wins += 1
                elif cur["status"] != "optimal" and base["status"] == "optimal":
                    losses += 1
                elif cur_gap is not None and base_gap is not None:
                    if cur_gap < base_gap - 1e-8:
                        wins += 1
                    elif abs(cur_gap - base_gap) <= 1e-8:
                        ties += 1
                    else:
                        losses += 1
        summaries.append(
            {
                "strategy": strategy,
                "count": len(strategy_rows),
                "completion_rate": len(completed) / len(strategy_rows) if strategy_rows else 0.0,
                "median_nodes": median(nodes) if nodes else "",
                "mean_nodes": mean(nodes) if nodes else "",
                "median_gap": median(gaps) if gaps else "",
                "mean_gap": mean(gaps) if gaps else "",
                "win_tie_loss": "" if strategy == "most_fractional" else f"{wins}/{ties}/{losses}",
            }
        )
    return summaries


def _write_charts(rows: list[dict], dataset: BranchingDataset, report_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except Exception:
        return
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in ("PingFang SC", "Heiti SC", "Arial Unicode MS", "Songti SC"):
        if font_name in available_fonts:
            plt.rcParams["font.sans-serif"] = [font_name]
            break
    plt.rcParams["axes.unicode_minus"] = False
    charts = report_dir / "charts"
    charts.mkdir(parents=True, exist_ok=True)

    latest_budget = max({int(row["budget"]) for row in rows}) if rows else 0
    budget_rows = [row for row in rows if int(row["budget"]) == latest_budget]
    by_instance: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in budget_rows:
        by_instance[row["instance_id"]][row["strategy"]] = row

    plt.figure(figsize=(6.5, 4.2))
    for strategy in sorted({row["strategy"] for row in budget_rows if row["strategy"] != "most_fractional"}):
        x = []
        y = []
        for mapping in by_instance.values():
            if "most_fractional" in mapping and strategy in mapping:
                x.append(float(mapping["most_fractional"]["formal_nodes"]))
                y.append(float(mapping[strategy]["formal_nodes"]))
        if x:
            plt.scatter(x, y, label=strategy, alpha=0.75)
    if budget_rows:
        lim = max(float(row["formal_nodes"]) for row in budget_rows) * 1.05
        plt.plot([0, lim], [0, lim], color="#666666", linewidth=1)
        plt.xlim(0, lim)
        plt.ylim(0, lim)
    plt.xlabel("MostFractional 节点数")
    plt.ylabel("对比策略节点数")
    plt.title(f"节点数配对散点图 node_limit={latest_budget}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(charts / "paired_nodes_scatter.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.0, 4.2))
    strategies = sorted({row["strategy"] for row in rows})
    budgets = sorted({int(row["budget"]) for row in rows})
    width = 0.8 / max(1, len(strategies))
    x_base = np.arange(len(budgets))
    for offset, strategy in enumerate(strategies):
        values = []
        for budget in budgets:
            gaps = [_float_or_none(row["relative_gap"]) for row in rows if row["strategy"] == strategy and int(row["budget"]) == budget]
            gaps = [gap for gap in gaps if gap is not None]
            values.append(float(median(gaps)) if gaps else 0.0)
        plt.bar(x_base + offset * width, values, width=width, label=strategy)
    plt.xticks(x_base + width * (len(strategies) - 1) / 2, [str(b) for b in budgets])
    plt.xlabel("节点预算")
    plt.ylabel("relative gap 中位数")
    plt.title("不同预算下 gap 比较")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(charts / "gap_by_budget.png", dpi=180)
    plt.close()

    training_csv = report_dir / "gnn_training_summary.csv"
    if training_csv.exists():
        with training_csv.open(newline="", encoding="utf-8") as f:
            train_rows = list(csv.DictReader(f))
        plt.figure(figsize=(6.5, 4.0))
        labels = [row["run_name"] for row in train_rows]
        regrets = [float(row["validation_normalized_regret"]) for row in train_rows]
        plt.bar(np.arange(len(labels)), regrets, color="#335C81")
        plt.xticks(np.arange(len(labels)), labels, rotation=20, ha="right")
        plt.ylabel("validation normalized regret")
        plt.title("GNN / DAgger 训练进展")
        plt.tight_layout()
        plt.savefig(charts / "dagger_progress.png", dpi=180)
        plt.close()

    plt.figure(figsize=(7.0, 4.2))
    units = sorted({row["units"] for row in rows}, key=lambda x: int(x))
    for strategy in strategies:
        rates = []
        for unit in units:
            part = [row for row in budget_rows if row["strategy"] == strategy and row["units"] == unit]
            rates.append(sum(row["status"] == "optimal" for row in part) / len(part) if part else 0.0)
        plt.plot(units, rates, marker="o", label=strategy)
    plt.xlabel("机组数")
    plt.ylabel("完成率")
    plt.title(f"按规模分组完成率 node_limit={latest_budget}")
    plt.ylim(-0.02, 1.02)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(charts / "completion_by_units.png", dpi=180)
    plt.close()

    candidate_counts = [len(sample["arrays"]["candidate_indices"]) for sample in dataset.multi_candidate_samples(min_candidates=2)]
    plt.figure(figsize=(6.0, 4.0))
    plt.hist(candidate_counts, bins=range(min(candidate_counts), max(candidate_counts) + 2), color="#335C81", alpha=0.85)
    plt.xlabel("候选变量数量")
    plt.ylabel("决策样本数")
    plt.title("候选变量数量分布")
    plt.tight_layout()
    plt.savefig(charts / "candidate_count_distribution.png", dpi=180)
    plt.close()


def write_report(config: dict, comparison_csv: Path = COMPARISON_CSV) -> Path:
    rows = _read_rows(comparison_csv)
    dataset = BranchingDataset.from_dir(config.get("dataset_path", "ml_branching/data/generated/unit_commitment_round0"))
    dataset.assert_disjoint_splits()
    candidate_counts = [len(sample["arrays"]["candidate_indices"]) for sample in dataset.multi_candidate_samples(min_candidates=2)]
    train = dataset.multi_candidate_samples("train", 2)
    validation = dataset.multi_candidate_samples("validation", 2)
    test = dataset.multi_candidate_samples("in_distribution_test", 2) + dataset.multi_candidate_samples("scale_extrapolation_test", 2)
    budgets = sorted({int(row["budget"]) for row in rows}) or [int(b) for b in config.get("budgets", [300])]
    summary_by_budget = {budget: summarize_strategy_rows(rows, budget) for budget in budgets}
    lines = [
        "# GNN分支策略实验总览",
        "",
        "## 1. 这次模型到底在学什么",
        "",
        "每个 B&B 节点被表示为变量-约束二部图。变量节点包含所有当前 LP 变量，约束节点来自当前矩阵 $Gz \\le h$，非零矩阵元素对应图上的边。GNN 输出每个候选二元变量的分支分数，B&B 仍按原来的剪枝、分支和 best-bound 节点选择继续执行。",
        "",
        "## 2. 前两次尝试为什么不理想",
        "",
        "- 第一次固定启动成本数据中大量节点只有一个候选变量，离线准确率没有实际意义。",
        "- 第二次多模型 MLP 能模仿一部分 strong branching，但只看候选人工特征，solver 级节点数没有稳定下降。",
        "- 本轮只聚焦 unit commitment，并把完整变量-约束结构作为输入。",
        "",
        "## 3. Unit Commitment数据",
        "",
        "| split | 有效决策样本 |",
        "|---|---:|",
        f"| train | {len(train)} |",
        f"| validation | {len(validation)} |",
        f"| test | {len(test)} |",
        "",
        f"- 平均候选变量数：{(mean(candidate_counts) if candidate_counts else 0):.3f}",
        f"- 最大候选变量数：{max(candidate_counts) if candidate_counts else 0}",
        "",
        "## 4. GNN结构",
        "",
        "变量节点 -> 约束节点 -> 变量节点，重复 1-2 轮消息传递；最后只在候选二元变量上打分。",
        "",
        "## 5. 每轮训练和DAgger结果",
        "",
        "训练摘要见 `gnn_training_summary.csv`。DAgger 数据只从 train 实例采集，测试实例不进入聚合。",
        "",
        "## 6. 最终自己和自己比较",
        "",
    ]
    for budget in budgets:
        lines += [
            f"### node_limit={budget}",
            "",
            "| strategy | 完成率 | 节点中位数 | gap中位数 | 胜/平/负 |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in summary_by_budget[budget]:
            median_nodes = row["median_nodes"] if row["median_nodes"] != "" else "-"
            median_gap = row["median_gap"] if row["median_gap"] != "" else "-"
            if isinstance(median_gap, float):
                median_gap = f"{median_gap:.6g}"
            lines.append(
                f"| {row['strategy']} | {row['completion_rate']:.3f} | {median_nodes} | {median_gap} | {row['win_tie_loss'] or '-'} |"
            )
        lines.append("")
    lines += [
        "## 7. 是否达到明显突破",
        "",
        "判据根据 `gnn_solver_comparison.csv` 配对结果计算；如果某档预算或策略尚未跑满，当前只能标记为阶段性结果，不能宣称成功。",
        "",
        "## 8. 目前仍然是什么问题",
        "",
        "- active-set 不是本轮目标；本轮固定使用 SciPy-HiGHS 节点 LP。",
        "- strong branching 的 probe LP 时间单独记录，不能混入正式 B&B 节点。",
        "- 若 GNN 离线指标改善但 rollout 没改善，优先看轨迹分布偏移和 DAgger 数据，而不是盲目加大模型。",
        "",
        "## 9. 文件清理与当前结构",
        "",
        "- 保留上一轮最佳 MLP checkpoint：`trained_models/learning_branching/run_12_lr_0.001_h_64_seed_3.pt`。",
        "- 本轮有效入口：`ml_branching/train_gnn.py`、`ml_branching/training/dagger.py`、`ml_branching/unit_commitment/benchmark.py`。",
    ]
    MAIN_REPORT.parent.mkdir(parents=True, exist_ok=True)
    MAIN_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_charts(rows, dataset, MAIN_REPORT.parent)
    return MAIN_REPORT


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Run unit commitment branching-policy benchmark without commercial solver comparison.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--write-report-only", action="store_true")
    args = parser.parse_args(argv)
    config = load_benchmark_config(args.config)
    if not args.write_report_only:
        run_benchmark(config, resume=args.resume)
    report = write_report(config)
    print(f"wrote {report}")
    return report


if __name__ == "__main__":
    main()
