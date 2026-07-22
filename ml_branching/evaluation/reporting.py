from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(value, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def _font_name():
    import matplotlib.font_manager as fm

    preferred = ["PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", "Arial Unicode MS", "Heiti TC"]
    available = {font.name for font in fm.fontManager.ttflist}
    for name in preferred:
        if name in available:
            return name
    return "DejaVu Sans"


def _setup_matplotlib():
    import matplotlib.pyplot as plt

    font = _font_name()
    plt.rcParams["font.sans-serif"] = [font]
    plt.rcParams["axes.unicode_minus"] = False
    return plt, font


def _save_candidate_distribution(audit_rows: list[dict], chart_dir: Path) -> str | None:
    families = [row for row in audit_rows if row.get("group_type") == "family"]
    if not families:
        return None
    plt, _ = _setup_matplotlib()
    names = [row["group_name"] for row in families]
    single = [_float(row["single_candidate_samples"]) for row in families]
    multi = [_float(row["effective_multi_candidate_samples"]) for row in families]
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=180)
    ax.bar(names, single, label="单候选", color="#b8c2cc")
    ax.bar(names, multi, bottom=single, label="多候选", color="#2f5f9f")
    ax.set_title("候选变量数量分布（按模型族）")
    ax.set_ylabel("决策样本数")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = chart_dir / "candidate_distribution.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def _save_expert_signal(audit_rows: list[dict], chart_dir: Path) -> str | None:
    families = [row for row in audit_rows if row.get("group_type") == "family"]
    if not families:
        return None
    plt, _ = _setup_matplotlib()
    names = [row["group_name"] for row in families]
    divergence = [_float(row["divergence_rate"]) for row in families]
    margin = [_float(row["mean_expert_normalized_margin"]) for row in families]
    x = range(len(names))
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=180)
    ax.plot(x, divergence, marker="o", label="与 most_fractional 分歧率", color="#2f5f9f")
    ax.plot(x, margin, marker="s", label="平均 normalized margin", color="#7a4f01")
    ax.set_xticks(list(x), names, rotation=25, ha="right")
    ax.set_ylim(bottom=0)
    ax.set_title("strong branching 标签区分度")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = chart_dir / "expert_signal.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def _save_strategy_comparison(rows: list[dict], chart_dir: Path) -> str | None:
    data = [row for row in rows if row.get("strategy") in {"most_fractional", "strong_branching", "learned_mlp"}]
    if not data:
        return None
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in data:
        grouped[row["strategy"]].append(row)
    strategies = ["most_fractional", "strong_branching", "learned_mlp"]
    labels = {"most_fractional": "most_fractional", "strong_branching": "strong branching", "learned_mlp": "learned MLP"}
    nodes = [mean([_float(row["formal_nodes"]) for row in grouped[s]]) if grouped[s] else 0 for s in strategies]
    lps = [mean([_float(row["formal_node_lp_solved"]) for row in grouped[s]]) if grouped[s] else 0 for s in strategies]
    times = [mean([_float(row["total_runtime_sec"]) for row in grouped[s]]) if grouped[s] else 0 for s in strategies]
    plt, _ = _setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), dpi=180)
    for ax, values, title in zip(axes, [nodes, lps, times], ["平均节点数", "平均正式 LP", "平均总时间(s)"]):
        ax.bar([labels[s] for s in strategies], values, color=["#8a98a8", "#2f5f9f", "#406b3b"])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("三种自写分支策略对比")
    fig.tight_layout()
    path = chart_dir / "strategy_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def _save_gurobi_scale(rows: list[dict], chart_dir: Path) -> str | None:
    data = [row for row in rows if row.get("strategy") in {"learned_mlp", "gurobi"} and row.get("total_runtime_sec")]
    if not data:
        return None
    plt, _ = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=180)
    for strategy, color in (("learned_mlp", "#406b3b"), ("gurobi", "#2f5f9f")):
        part = sorted([row for row in data if row["strategy"] == strategy], key=lambda row: _float(row["num_binary_variables"]))
        if not part:
            continue
        ax.plot(
            [_float(row["num_binary_variables"]) for row in part],
            [_float(row["total_runtime_sec"]) for row in part],
            marker="o",
            linestyle="-",
            label=strategy,
            color=color,
        )
    ax.set_title("自写求解器与 Gurobi 时间曲线")
    ax.set_xlabel("二元变量数")
    ax.set_ylabel("运行时间(s)")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = chart_dir / "gurobi_scale_curve.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def _save_offline_generalization(rows: list[dict], chart_dir: Path) -> str | None:
    data = [row for row in rows if row.get("strategy") in {"learned_mlp", "linear_baseline"}]
    if not data:
        return None
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in data:
        grouped[(row["split"], row["strategy"])].append(row)
    splits = sorted({row["split"] for row in data})
    strategies = ["linear_baseline", "learned_mlp"]
    plt, _ = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=180)
    width = 0.35
    xs = list(range(len(splits)))
    for offset, strategy in zip([-width / 2, width / 2], strategies):
        values = []
        for split in splits:
            part = grouped.get((split, strategy), [])
            values.append(mean([_float(row["top1"]) for row in part]) if part else 0.0)
        ax.bar([x + offset for x in xs], values, width=width, label=strategy)
    ax.set_xticks(xs, splits, rotation=15)
    ax.set_ylim(0, 1.0)
    ax.set_title("离线泛化 top-1 准确率（只含多候选节点）")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = chart_dir / "offline_generalization.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def _write_training_csv(model_dir: Path, report_dir: Path) -> list[dict]:
    src = model_dir / "training_summary.csv"
    dst = report_dir / "training_summary.csv"
    if src.exists():
        shutil.copyfile(src, dst)
        return _read_csv(dst)
    summary_path = model_dir / "training_summary.json"
    if not summary_path.exists():
        return []
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = summary.get("runs", [])
    if rows:
        with dst.open("w", newline="", encoding="utf-8") as f:
            fields = sorted({key for row in rows for key in row})
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return rows


def _table_line(values: list[str]) -> str:
    return "| " + " | ".join(values) + " |"


def generate_report(
    dataset_path: str | Path = "ml_branching/data/generated/train",
    model_dir: str | Path = "trained_models/learning_branching",
    report_dir: str | Path = "reports/learning_branching",
) -> dict:
    dataset_path = Path(dataset_path)
    model_dir = Path(model_dir)
    report_dir = Path(report_dir)
    chart_dir = report_dir / "charts"
    report_dir.mkdir(parents=True, exist_ok=True)
    chart_dir.mkdir(parents=True, exist_ok=True)

    audit_summary_path = dataset_path / "audit_summary.json"
    audit_summary = json.loads(audit_summary_path.read_text(encoding="utf-8")) if audit_summary_path.exists() else {}
    audit_rows = _read_csv(report_dir / "dataset_audit.csv")
    expert_rows = _read_csv(report_dir / "expert_comparison.csv")
    training_rows = _write_training_csv(model_dir, report_dir)
    offline_rows = _read_csv(report_dir / "offline_evaluation.csv")
    solver_rows = _read_csv(report_dir / "solver_comparison.csv")

    charts = [
        _save_candidate_distribution(audit_rows, chart_dir),
        _save_expert_signal(audit_rows, chart_dir),
        _save_strategy_comparison(solver_rows, chart_dir),
        _save_gurobi_scale(solver_rows, chart_dir),
        _save_offline_generalization(offline_rows, chart_dir),
    ]
    charts = [chart for chart in charts if chart]

    overall = audit_summary.get("overall", {})
    by_family = audit_summary.get("by_family", {})
    best_summary = json.loads((model_dir / "training_summary.json").read_text(encoding="utf-8")) if (model_dir / "training_summary.json").exists() else {}
    best_run = best_summary.get("best_run", {})

    solver_by_strategy: dict[str, list[dict]] = defaultdict(list)
    for row in solver_rows:
        solver_by_strategy[row["strategy"]].append(row)

    lines = [
        "# 学习型分支策略重训总览",
        "",
        "## 1. 上一轮为什么无效",
        "",
        "上一轮固定启动成本单模型族数据中，每个 B&B 节点基本只有一个 fractional binary 候选变量；因此 loss=0、top-1=100% 只说明没有选择空间，不代表模型学到了分支变量比较能力。本轮已删除旧数据、旧 checkpoint、旧配置和旧报告，只保留 BranchingPolicy、strong branching 采集、特征提取、MLP/checkpoint 和 solver 评估框架。",
        "",
        "## 2. 本轮建立了哪些模型族",
        "",
        _table_line(["模型族", "连续变量", "二元变量", "主要约束", "进入训练", "holdout"]),
        _table_line(["---", "---", "---", "---", "---", "---"]),
        _table_line(["fixed_charge_multi_resource", "产量", "设备启用", "多需求/资源/容量联动", "是", "否"]),
        _table_line(["unit_commitment", "机组出力", "开停状态", "需求/备用/出力上下限/爬坡", "是", "否"]),
        _table_line(["capacity_expansion", "区域供给", "模块建设", "需求/容量/资源限制", "是", "否"]),
        _table_line(["facility_location", "运输量", "设施开启", "客户需求/设施容量/运输联动", "是", "否"]),
        _table_line(["activated_resource_allocation", "资源分配", "项目启用", "预算/最低需求/启用联动", "是", "否"]),
        _table_line(["random_sparse_block", "一般连续块", "一般二元块", "稀疏 $Ax+By\\le b$", "否", "是"]),
        "",
        "## 3. 数据是否真的可以训练",
        "",
        f"- Audit: `{audit_summary.get('audit_status', 'UNKNOWN')}`",
        f"- 有效多候选样本: `{overall.get('effective_multi_candidate_samples', '')}`",
        f"- 平均 candidate_count: `{overall.get('candidate_count_mean', '')}`",
        f"- 最大 candidate_count: `{overall.get('candidate_count_max', '')}`",
        f"- 专家与 most_fractional 分歧率: `{overall.get('divergence_rate', '')}`",
        "",
        _table_line(["family", "决策数", "多候选", "多候选比例", "平均候选", "最大候选", "分歧率"]),
        _table_line(["---", "---:", "---:", "---:", "---:", "---:", "---:"]),
    ]
    for family, stats in by_family.items():
        lines.append(
            _table_line(
                [
                    family,
                    str(stats["decision_samples"]),
                    str(stats["effective_multi_candidate_samples"]),
                    f"{stats['multi_candidate_ratio']:.3f}",
                    f"{stats['candidate_count_mean']:.3f}",
                    str(stats["candidate_count_max"]),
                    f"{stats['divergence_rate']:.3f}",
                ]
            )
        )
    for chart in charts[:2]:
        lines.extend(["", f"![]({Path(chart).relative_to(report_dir)})"])

    lines.extend(
        [
            "",
            "## 4. Strong branching是否值得学习",
            "",
            "strong branching 的作用不是保证总时间更短，而是提供更有信息量的分支变量标签。下表比较它与 most_fractional 的正式节点数和正式 LP 数。",
            "",
            _table_line(["strategy", "实例数", "平均节点", "平均正式 LP", "平均总时间(s)", "完成率"]),
            _table_line(["---", "---:", "---:", "---:", "---:", "---:"]),
        ]
    )
    expert_summary_path = report_dir / "expert_comparison_summary.json"
    expert_summary = json.loads(expert_summary_path.read_text(encoding="utf-8")) if expert_summary_path.exists() else {}
    for strategy, stats in expert_summary.get("summary", {}).items():
        lines.append(
            _table_line(
                [
                    strategy,
                    str(stats["count"]),
                    f"{stats['mean_nodes']:.3f}",
                    f"{stats['mean_formal_lp_solved']:.3f}",
                    f"{stats['mean_total_runtime_sec']:.6f}",
                    f"{stats['completed_ratio']:.3f}",
                ]
            )
        )

    lines.extend(
        [
            "",
            "## 5. MLP训练结果",
            "",
            f"- best checkpoint: `{best_summary.get('best_checkpoint', '')}`",
            f"- selection metric: `{best_summary.get('selection_metric', '')}`",
            f"- best validation normalized regret: `{best_run.get('validation_normalized_regret', '')}`",
            f"- best validation top-1: `{best_run.get('validation_top1_accuracy', '')}`",
            f"- effective train samples: `{best_run.get('effective_train_samples', '')}`",
            f"- effective validation samples: `{best_run.get('effective_validation_samples', '')}`",
            "",
            "训练只使用 `candidate_count>=2` 的节点；normalizer 只在 train split 上拟合；family_name 只用于分组和报告，不作为模型输入。",
            "",
            "## 6. 泛化能力",
            "",
        ]
    )
    if offline_rows:
        grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in offline_rows:
            grouped[(row["split"], row["strategy"])].append(row)
        lines.extend([
            _table_line(["split", "strategy", "样本数", "top-1", "top-3", "normalized regret"]),
            _table_line(["---", "---", "---:", "---:", "---:", "---:"]),
        ])
        for (split, strategy), rows in sorted(grouped.items()):
            lines.append(
                _table_line(
                    [
                        split,
                        strategy,
                        str(len(rows)),
                        f"{mean([_float(row['top1']) for row in rows]):.3f}",
                        f"{mean([_float(row['top3']) for row in rows]):.3f}",
                        f"{mean([_float(row['normalized_regret']) for row in rows]):.6f}",
                    ]
                )
            )
        chart = chart_dir / "offline_generalization.png"
        if chart.exists():
            lines.extend(["", f"![]({chart.relative_to(report_dir)})"])

    lines.extend(
        [
            "",
            "## 7. 三种自写分支策略对比",
            "",
            _table_line(["strategy", "行数", "平均节点", "平均正式 LP", "平均总时间(s)", "完成率", "Gurobi目标一致率"]),
            _table_line(["---", "---:", "---:", "---:", "---:", "---:", "---:"]),
        ]
    )
    for strategy in ["most_fractional", "strong_branching", "learned_mlp"]:
        rows = solver_by_strategy.get(strategy, [])
        if not rows:
            continue
        completed = [row for row in rows if row["status"] == "optimal"]
        matches = [row["matches_gurobi"] == "True" for row in rows if row["matches_gurobi"]]
        lines.append(
            _table_line(
                [
                    strategy,
                    str(len(rows)),
                    f"{mean([_float(row['formal_nodes']) for row in rows]):.3f}",
                    f"{mean([_float(row['formal_node_lp_solved']) for row in rows]):.3f}",
                    f"{mean([_float(row['total_runtime_sec']) for row in rows]):.6f}",
                    f"{len(completed) / len(rows):.3f}",
                    "" if not matches else f"{mean(matches):.3f}",
                ]
            )
        )
    chart = chart_dir / "strategy_comparison.png"
    if chart.exists():
        lines.extend(["", f"![]({chart.relative_to(report_dir)})"])

    lines.extend(
        [
            "",
            "## 8. 与Gurobi的真实差距",
            "",
            "Gurobi 使用默认工业算法；它的节点数与自写 B&B 节点数不能直接等价。这里主要看时间、status、gap 和规模变化。",
            "",
            _table_line(["family", "二元变量", "learned MLP时间", "Gurobi时间", "时间倍数", "learned status", "Gurobi status"]),
            _table_line(["---", "---:", "---:", "---:", "---:", "---", "---"]),
        ]
    )
    by_instance = defaultdict(dict)
    for row in solver_rows:
        by_instance[row["instance_id"]][row["strategy"]] = row
    for instance_id, strategy_rows in sorted(by_instance.items()):
        if "learned_mlp" not in strategy_rows or "gurobi" not in strategy_rows:
            continue
        learned = strategy_rows["learned_mlp"]
        gurobi = strategy_rows["gurobi"]
        gt = _float(gurobi["total_runtime_sec"])
        lt = _float(learned["total_runtime_sec"])
        ratio = "" if gt <= 0 else f"{lt / gt:.1f}x"
        lines.append(
            _table_line(
                [
                    learned["family_name"],
                    learned["num_binary_variables"],
                    learned["total_runtime_sec"],
                    gurobi["total_runtime_sec"],
                    ratio,
                    learned["status"],
                    gurobi["status"],
                ]
            )
        )
    chart = chart_dir / "gurobi_scale_curve.png"
    if chart.exists():
        lines.extend(["", f"![]({chart.relative_to(report_dir)})"])

    lines.extend(
        [
            "",
            "## 9. 本轮结论",
            "",
            "本轮结论是：MLP 已经不再是单候选假学习，能够在多候选节点上部分模仿 strong branching，并且整体离线结果优于线性基线；但它没有稳定转化为 solver 级节点数下降。当前更接近“能模仿专家，但求解器收益有限”。主要瓶颈仍在人工特征表达和自写 B&B/LP 后端，下一步若继续学习分支，更适合升级到利用矩阵图结构的模型，而不是单纯加大 MLP。",
            "",
            "## 10. 当前文件结构",
            "",
            "- `ml_branching/families/`: 多模型族生成与复现实例。",
            "- `ml_branching/audit_dataset.py`: 候选数量和 strong branching 标签审计。",
            "- `ml_branching/evaluate_expert.py`: most_fractional 与 strong branching 求解收益对比。",
            "- `ml_branching/train.py`: 审计通过后训练 MLP，并保留最佳 checkpoint。",
            "- `ml_branching/evaluate_offline.py`: 多 split 离线评估。",
            "- `ml_branching/evaluate_solver.py`: 三种自写策略与 Gurobi 对比。",
            "- `reports/learning_branching/`: 本轮报告、CSV 与图表。",
            "",
            "## 11. 删除和修改记录",
            "",
            "- 删除旧单候选数据集、旧 checkpoint、旧重复报告和旧 smoke/default 配置。",
            "- 删除旧 `ml_branching/data/instance_generator.py`，改为统一 family registry。",
            "- 保留 BranchingPolicy、StrongBranchingPolicy、特征、normalizer、MLP 和 checkpoint 框架。",
        ]
    )

    report_path = report_dir / "学习型分支策略重训总览.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"report": str(report_path), "charts": charts, "font": _font_name()}


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Generate the Chinese learning-branching retraining report.")
    parser.add_argument("--dataset", default="ml_branching/data/generated/train")
    parser.add_argument("--model-dir", default="trained_models/learning_branching")
    parser.add_argument("--report-dir", default="reports/learning_branching")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    result = generate_report(args.dataset, args.model_dir, args.report_dir)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
