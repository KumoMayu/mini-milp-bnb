from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_branching.families import get_family
from ml_branching.runtime.inference import LearnedBranchingPolicy, LearnedGNNBranchingPolicy
from ml_branching.training.oracle.strong_branching import StrongBranchingPolicy
from ml_branching.training.gnn_trainer import GNNTrainingConfig, train_gnn_one_config
from ml_branching.unit_commitment import UnitCommitmentGenerator
from solver import solve_milp
from solver.branch_and_bound import check_feasibility
from solver.branching import MostFractionalPolicy, PseudocostPolicy


REPORT_DIR = Path("reports/learning_branching")
STABILITY_REPORT = REPORT_DIR / "GNN稳定性与通用性验证.md"


RESULT_FIELDS = [
    "experiment",
    "family_name",
    "split",
    "scale_group",
    "instance_id",
    "seed",
    "units",
    "periods",
    "budget",
    "time_limit_sec",
    "strategy",
    "gnn_seed",
    "status",
    "completed",
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


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _font_setup(plt) -> None:
    try:
        from matplotlib import font_manager
    except Exception:
        return
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("PingFang SC", "Heiti SC", "Arial Unicode MS", "Songti SC"):
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def train_stability_seeds(config: dict, resume: bool = True) -> list[dict]:
    rows = []
    for seed in config["seeds"]:
        run_name = f"{config['run_prefix']}_{int(seed)}"
        checkpoint = Path(config["output_dir"]) / f"{run_name}.pt"
        if resume and checkpoint.exists():
            rows.append(
                {
                    "run_name": run_name,
                    "checkpoint_path": str(checkpoint),
                    "seed": int(seed),
                    "status": "reused",
                }
            )
            continue
        train_config = GNNTrainingConfig(
            dataset_path=config["dataset_path"],
            output_dir=config["output_dir"],
            report_dir=config["report_dir"],
            run_name=run_name,
            min_candidate_count=int(config["min_candidate_count"]),
            round_max=config.get("round_max"),
            seed=int(seed),
            learning_rate=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
            hidden_dim=int(config["hidden_dim"]),
            message_rounds=int(config["message_rounds"]),
            dropout=float(config["dropout"]),
            max_epochs=int(config["max_epochs"]),
            early_stopping_patience=int(config["early_stopping_patience"]),
            loss_type=str(config["loss_type"]),
            soft_temperature=float(config["soft_temperature"]),
            pairwise_weight=float(config["pairwise_weight"]),
            device=str(config["device"]),
        )
        rows.append(train_gnn_one_config(train_config))
    return rows


def _append_row(path: Path, row: dict, fields: list[str] = RESULT_FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def _read_rows(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _done_keys(path: Path) -> set[tuple[str, str, str]]:
    return {(row["instance_id"], row["strategy"], row["budget"]) for row in _read_rows(path)}


def _uc_instances(config: dict):
    generator = UnitCommitmentGenerator()
    defaults = dict(config.get("unit_commitment_generation", {}))
    for spec in config["unit_commitment_instances"]:
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


def _cross_instances(config: dict):
    raw = config["cross_family"]
    seeds = list(raw["seeds"])
    sizes = list(raw["sizes"])
    for family_name in raw["families"]:
        family = get_family(family_name)
        for index, seed in enumerate(seeds):
            size = int(sizes[index % len(sizes)])
            yield family.generate(
                seed=int(seed) + 100000 * (raw["families"].index(family_name) + 1),
                size=size,
                split="zero_shot_test",
                scale_group=family_name,
            )


def _make_policy(strategy: str, config: dict):
    if strategy == "most_fractional":
        return MostFractionalPolicy()
    if strategy == "pseudocost":
        return PseudocostPolicy()
    if strategy == "strong_branching":
        return StrongBranchingPolicy(lp_backend="scipy_highs", use_matrix_presolve=True)
    if strategy == "mlp":
        return LearnedBranchingPolicy.from_checkpoint(config["mlp_checkpoint"], device=config.get("device", "cpu"))
    if strategy.startswith("gnn_seed_"):
        return LearnedGNNBranchingPolicy.from_checkpoint(config["gnn_checkpoints"][strategy], device=config.get("device", "cpu"))
    raise ValueError(f"unknown strategy={strategy!r}")


def _base_row(instance, budget: int, time_limit_sec: float, strategy: str, experiment: str) -> dict:
    params = instance.parameters
    return {
        "experiment": experiment,
        "family_name": instance.family_name,
        "split": instance.split,
        "scale_group": instance.scale_group,
        "instance_id": instance.instance_id,
        "seed": str(instance.seed),
        "units": str(params.get("units", params.get("size", instance.size))),
        "periods": str(params.get("periods", "")),
        "budget": str(int(budget)),
        "time_limit_sec": f"{float(time_limit_sec):.6g}",
        "strategy": strategy,
        "gnn_seed": strategy.replace("gnn_seed_", "") if strategy.startswith("gnn_seed_") else "",
    }


def solve_instance(instance, strategy: str, policy, budget: int, time_limit_sec: float, tolerance: float, experiment: str) -> dict:
    if hasattr(policy, "probe_lp_solved"):
        policy.probe_lp_solved = 0
        policy.probe_time_sec = 0.0
    if hasattr(policy, "inference_time_sec"):
        policy.inference_time_sec = 0.0
        policy.inference_calls = 0
    start = perf_counter()
    result = solve_milp(
        instance.problem,
        branching_policy=policy,
        lp_backend="scipy_highs",
        use_matrix_presolve=True,
        max_nodes=int(budget),
        time_limit_sec=float(time_limit_sec),
        tol=float(tolerance),
    )
    elapsed = perf_counter() - start
    feasible = False
    if result.x is not None:
        feasible = check_feasibility(instance.problem, result.x, instance.problem.lb, instance.problem.ub, tolerance)
    row = _base_row(instance, budget, time_limit_sec, strategy, experiment)
    row.update(
        {
            "status": result.status,
            "completed": str(result.status == "optimal"),
            "objective": "" if result.objective_value is None else f"{float(result.objective_value):.10g}",
            "solution_feasible": str(bool(feasible)),
            "formal_nodes": str(int(result.num_nodes)),
            "formal_lp_solved": str(int(result.num_lp_solved)),
            "probe_lp_solved": str(int(getattr(policy, "probe_lp_solved", 0))),
            "probe_runtime_sec": f"{float(getattr(policy, 'probe_time_sec', 0.0)):.6f}",
            "inference_time_sec": f"{float(getattr(policy, 'inference_time_sec', 0.0)):.6f}",
            "solver_runtime_sec": f"{float(result.runtime_sec):.6f}",
            "total_runtime_sec": f"{float(elapsed):.6f}",
            "global_bound": "" if result.global_bound is None else f"{float(result.global_bound):.10g}",
            "relative_gap": "" if result.relative_gap is None else f"{float(result.relative_gap):.10g}",
            "limit_type": "" if result.status == "optimal" else result.status,
        }
    )
    return row


def run_uc_stability(config: dict, resume: bool = True) -> Path:
    path = Path(config["stability_csv"])
    if not resume and path.exists():
        path.unlink()
    done = _done_keys(path) if resume else set()
    strategies = ["most_fractional", "pseudocost", "mlp", *sorted(config["gnn_checkpoints"]), "strong_branching"]
    for instance in _uc_instances(config):
        for budget in config["budgets"]:
            for strategy in strategies:
                key = (instance.instance_id, strategy, str(int(budget)))
                if key in done:
                    continue
                row = solve_instance(
                    instance,
                    strategy,
                    _make_policy(strategy, config),
                    int(budget),
                    float(config["time_limit_sec"]),
                    float(config["tolerance"]),
                    "unit_commitment_stability",
                )
                _append_row(path, row)
                done.add(key)
                print(f"{instance.instance_id} budget={budget} {strategy} status={row['status']} nodes={row['formal_nodes']} gap={row['relative_gap']}")
    return path


def run_cross_family(config: dict, resume: bool = True) -> Path:
    path = Path(config["cross_family_csv"])
    if not resume and path.exists():
        path.unlink()
    done = _done_keys(path) if resume else set()
    strategies = ["most_fractional", "pseudocost", *sorted(config["gnn_checkpoints"])]
    budget = int(config["cross_family"]["budget"])
    for instance in _cross_instances(config):
        for strategy in strategies:
            key = (instance.instance_id, strategy, str(budget))
            if key in done:
                continue
            row = solve_instance(
                instance,
                strategy,
                _make_policy(strategy, config),
                budget,
                float(config["time_limit_sec"]),
                float(config["tolerance"]),
                "cross_family_zero_shot",
            )
            _append_row(path, row)
            done.add(key)
            print(f"{instance.instance_id} {strategy} status={row['status']} nodes={row['formal_nodes']} gap={row['relative_gap']}")
    return path


def _f(row: dict, key: str, default: float | None = None) -> float | None:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def _paired_maps(rows: list[dict], budget: int) -> dict[str, dict[str, dict]]:
    mapping: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        if int(row["budget"]) == int(budget):
            mapping[row["instance_id"]][row["strategy"]] = row
    return mapping


def _compare_pair(base: dict, other: dict) -> str:
    base_gap = _f(base, "relative_gap", 1e9)
    other_gap = _f(other, "relative_gap", 1e9)
    if base["status"] == other["status"] == "optimal":
        b_nodes = int(base["formal_nodes"])
        o_nodes = int(other["formal_nodes"])
        if o_nodes < b_nodes:
            return "win"
        if o_nodes == b_nodes:
            return "tie"
        return "loss"
    if other["status"] == "optimal" and base["status"] != "optimal":
        return "win"
    if other["status"] != "optimal" and base["status"] == "optimal":
        return "loss"
    if other_gap is not None and base_gap is not None:
        if other_gap < base_gap - 1e-8:
            return "win"
        if abs(other_gap - base_gap) <= 1e-8:
            return "tie"
    return "loss"


def _gnn_strategies(rows: list[dict]) -> list[str]:
    return sorted({row["strategy"] for row in rows if row["strategy"].startswith("gnn_seed_")})


def _wtl_against(maps: dict[str, dict[str, dict]], baseline: str, strategies: list[str]) -> list[int]:
    wtl = [0, 0, 0]
    for mapping in maps.values():
        for strategy in strategies:
            if baseline in mapping and strategy in mapping:
                result = _compare_pair(mapping[baseline], mapping[strategy])
                wtl[["win", "tie", "loss"].index(result)] += 1
    return wtl


def strategy_summary(rows: list[dict], budget: int, baseline: str = "most_fractional") -> list[dict]:
    budget_rows = [row for row in rows if int(row["budget"]) == int(budget)]
    strategies = sorted({row["strategy"] for row in budget_rows})
    paired = _paired_maps(rows, budget)
    out = []
    for strategy in strategies:
        part = [row for row in budget_rows if row["strategy"] == strategy]
        completed = [row for row in part if row["status"] == "optimal"]
        nodes = [int(row["formal_nodes"]) for row in completed]
        all_nodes = [int(row["formal_nodes"]) for row in part]
        gaps = [_f(row, "relative_gap") for row in part if row.get("relative_gap") != ""]
        wins = ties = losses = 0
        if strategy != baseline:
            for mapping in paired.values():
                if baseline in mapping and strategy in mapping:
                    result = _compare_pair(mapping[baseline], mapping[strategy])
                    wins += int(result == "win")
                    ties += int(result == "tie")
                    losses += int(result == "loss")
        geo_nodes = ""
        if nodes:
            geo_nodes = math.exp(mean(math.log(max(1.0, float(v))) for v in nodes))
        out.append(
            {
                "strategy": strategy,
                "count": len(part),
                "completion_rate": len(completed) / len(part) if part else 0.0,
                "median_nodes": median(nodes) if nodes else "",
                "mean_nodes": mean(nodes) if nodes else "",
                "geo_mean_nodes": geo_nodes,
                "median_all_nodes": median(all_nodes) if all_nodes else "",
                "median_gap": median(gaps) if gaps else "",
                "mean_inference_time": mean(float(row["inference_time_sec"]) for row in part) if part else 0.0,
                "win": wins,
                "tie": ties,
                "loss": losses,
            }
        )
    return out


def stability_decision(rows: list[dict], budget: int = 3000) -> dict:
    summaries = {row["strategy"]: row for row in strategy_summary(rows, budget)}
    base = summaries.get("most_fractional")
    if not base:
        return {"status": "FAIL", "reason": "missing most_fractional baseline"}
    passes = 0
    details = []
    for strategy in sorted(s for s in summaries if s.startswith("gnn_seed_")):
        row = summaries[strategy]
        if row["median_nodes"] == "" or base["median_nodes"] == "":
            reduction = 0.0
        else:
            reduction = (float(base["median_nodes"]) - float(row["median_nodes"])) / max(1.0, float(base["median_nodes"]))
        compared = row["win"] + row["tie"] + row["loss"]
        win_rate = row["win"] / compared if compared else 0.0
        ok = reduction >= 0.15 and win_rate > 0.60 and row["completion_rate"] >= base["completion_rate"]
        passes += int(ok)
        details.append(
            {
                "strategy": strategy,
                "median_reduction": reduction,
                "win_rate": win_rate,
                "completion_rate": row["completion_rate"],
                "pass": ok,
            }
        )
    return {"status": "PASS" if passes >= 2 else "FAIL", "seed_pass_count": passes, "details": details}


def _write_charts(uc_rows: list[dict], cross_rows: list[dict], report_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    _font_setup(plt)
    charts = report_dir / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    budget = max(int(row["budget"]) for row in uc_rows) if uc_rows else 0
    paired = _paired_maps(uc_rows, budget)
    seed_data = []
    labels = []
    for strategy in _gnn_strategies(uc_rows):
        values = [
            int(mapping[strategy]["formal_nodes"])
            for mapping in paired.values()
            if strategy in mapping and mapping[strategy]["status"] == "optimal"
        ]
        if values:
            seed_data.append(values)
            labels.append(strategy)
    if seed_data:
        plt.figure(figsize=(7, 4))
        plt.boxplot(seed_data, tick_labels=labels)
        plt.ylabel("完成实例节点数")
        plt.title(f"三个GNN seed节点数箱线图 node_limit={budget}")
        plt.tight_layout()
        plt.savefig(charts / "stability_seed_boxplot.png", dpi=180)
        plt.close()

    plt.figure(figsize=(6.5, 4.5))
    for strategy in labels:
        x, y = [], []
        for mapping in paired.values():
            if "most_fractional" in mapping and strategy in mapping:
                x.append(float(mapping["most_fractional"]["formal_nodes"]))
                y.append(float(mapping[strategy]["formal_nodes"]))
        if x:
            plt.scatter(x, y, label=strategy, alpha=0.75)
    if paired:
        lim = max(float(row["formal_nodes"]) for row in uc_rows if int(row["budget"]) == budget) * 1.05
        plt.plot([0, lim], [0, lim], color="#666666", linewidth=1)
        plt.xlim(0, lim)
        plt.ylim(0, lim)
    plt.xlabel("MostFractional 节点数")
    plt.ylabel("GNN 节点数")
    plt.title("GNN与MostFractional逐实例节点散点图")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(charts / "gnn_vs_most_fractional_scatter.png", dpi=180)
    plt.close()

    budgets = sorted({int(row["budget"]) for row in uc_rows})
    strategies = ["most_fractional", "pseudocost", *labels, "strong_branching"]
    plt.figure(figsize=(8, 4.5))
    width = 0.8 / max(1, len(strategies))
    xs = np.arange(len(budgets))
    for idx, strategy in enumerate(strategies):
        rates = []
        for b in budgets:
            part = [row for row in uc_rows if int(row["budget"]) == b and row["strategy"] == strategy]
            rates.append(sum(row["status"] == "optimal" for row in part) / len(part) if part else 0.0)
        plt.bar(xs + idx * width, rates, width=width, label=strategy)
    plt.xticks(xs + width * (len(strategies) - 1) / 2, [str(b) for b in budgets])
    plt.ylabel("完成率")
    plt.title("各节点预算完成率")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(charts / "budget_completion_rates.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    width = 0.8 / max(1, len(strategies))
    xs = np.arange(len(budgets))
    for idx, strategy in enumerate(strategies):
        gaps = []
        for b in budgets:
            values = [
                _f(row, "relative_gap")
                for row in uc_rows
                if int(row["budget"]) == b and row["strategy"] == strategy and row.get("relative_gap") != ""
            ]
            gaps.append(median(values) if values else 0.0)
        plt.bar(xs + idx * width, gaps, width=width, label=strategy)
    plt.xticks(xs + width * (len(strategies) - 1) / 2, [str(b) for b in budgets])
    plt.ylabel("relative gap 中位数")
    plt.title("各节点预算gap中位数")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(charts / "budget_gap_medians.png", dpi=180)
    plt.close()

    if cross_rows:
        families = sorted({row["family_name"] for row in cross_rows})
        wins = []
        losses = []
        ties = []
        for family in families:
            maps: dict[str, dict[str, dict]] = defaultdict(dict)
            for row in cross_rows:
                if row["family_name"] == family:
                    maps[row["instance_id"]][row["strategy"]] = row
            w, t, l = _wtl_against(maps, "most_fractional", _gnn_strategies(cross_rows))
            wins.append(w)
            ties.append(t)
            losses.append(l)
        x = np.arange(len(families))
        plt.figure(figsize=(8, 4.5))
        plt.bar(x, wins, label="win")
        plt.bar(x, ties, bottom=wins, label="tie")
        plt.bar(x, losses, bottom=np.array(wins) + np.array(ties), label="loss")
        plt.xticks(x, families, rotation=20, ha="right")
        plt.ylabel("实例数")
        plt.title("各模型族win/tie/loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(charts / "cross_family_win_tie_loss.png", dpi=180)
        plt.close()

    plt.figure(figsize=(6.5, 4.5))
    x, y = [], []
    for strategy in labels:
        sx, sy = [], []
        for mapping in paired.values():
            if "pseudocost" in mapping and strategy in mapping:
                sx.append(float(mapping["pseudocost"]["formal_nodes"]))
                sy.append(float(mapping[strategy]["formal_nodes"]))
        if sx:
            x.extend(sx)
            y.extend(sy)
            plt.scatter(sx, sy, alpha=0.65, label=strategy)
    if x:
        lim = max(max(x), max(y)) * 1.05
        plt.plot([0, lim], [0, lim], color="#666666", linewidth=1)
        plt.xlim(0, lim)
        plt.ylim(0, lim)
    plt.xlabel("Pseudocost 节点数")
    plt.ylabel("GNN 节点数")
    plt.title("Pseudocost与GNN对比")
    if x:
        plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(charts / "pseudocost_vs_gnn_scatter.png", dpi=180)
    plt.close()


def _format_num(value) -> str:
    if value == "" or value is None:
        return "-"
    return f"{float(value):.6g}"


def _checkpoint_training_rows(config: dict) -> list[dict]:
    try:
        import torch
    except Exception:
        return []
    rows = []
    for strategy, checkpoint_path in sorted(config.get("gnn_checkpoints", {}).items()):
        path = Path(checkpoint_path)
        if not path.exists():
            continue
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")
        log = checkpoint.get("training_log") or []
        best_epoch = checkpoint.get("best_epoch")
        best_row = next((row for row in log if row.get("epoch") == best_epoch), None)
        if best_row is None and log:
            best_row = min(log, key=lambda row: row.get("validation_loss", float("inf")))
            best_epoch = best_row.get("epoch")
        rows.append(
            {
                "strategy": strategy,
                "best_epoch": best_epoch if best_epoch is not None else "",
                "validation_top1": "" if best_row is None else best_row.get("validation_top1_accuracy", ""),
                "validation_top3": "" if best_row is None else best_row.get("validation_top3_accuracy", ""),
                "validation_normalized_regret": "" if best_row is None else best_row.get("validation_normalized_regret", ""),
                "validation_loss": "" if best_row is None else best_row.get("validation_loss", ""),
            }
        )
    return rows


def write_report(config: dict) -> Path:
    report_dir = Path(config["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    uc_rows = _read_rows(config["stability_csv"])
    cross_rows = _read_rows(config["cross_family_csv"])
    budgets = sorted({int(row["budget"]) for row in uc_rows})
    stability = stability_decision(uc_rows, budget=3000 if any(int(r["budget"]) == 3000 for r in uc_rows) else max(budgets or [0]))
    _write_charts(uc_rows, cross_rows, report_dir)

    lines = [
        "# GNN稳定性与通用性验证",
        "",
        "## 1. 当前研究对象",
        "",
        "本阶段研究的是一般 $Ax+By\\le b$、连续变量 $x$ 与二元变量 $y$ 下的分支变量选择策略，不是 unit commitment 专用求解器。",
        "",
        "## 2. GNN是否包含UC专用信息",
        "",
        "当前 GNN 输入只包含通用图特征：目标系数、LP 解、变量上下界、二元/候选标记、约束 RHS/activity/slack/行范数/非零数、矩阵非零系数边，以及节点深度和 incumbent gap。不包含机组编号、时段编号、需求/备用/爬坡标签、family id 或固定变量排列。",
        "",
        "## 3. 三个训练seed结果",
        "",
        "三个 seed 使用相同训练数据和超参数；checkpoint 只保留最终最佳模型。训练曲线保存在 checkpoint payload 的 `training_log` 中。",
        "",
        "| GNN seed | best epoch | validation top-1 | validation top-3 | normalized regret | validation loss |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in _checkpoint_training_rows(config):
        lines.append(
            f"| {row['strategy']} | {_format_num(row['best_epoch'])} | {_format_num(row['validation_top1'])} | {_format_num(row['validation_top3'])} | {_format_num(row['validation_normalized_regret'])} | {_format_num(row['validation_loss'])} |"
        )
    lines.extend(
        [
            "",
            "训练耗时未写入 checkpoint；本轮实际训练耗时以终端运行日志为准。推理耗时在下方策略表中按求解过程累计统计。",
            "",
        "## 4. UC稳定性结果",
        "",
        f"- 固定 UC 测试实例数：{len({row['instance_id'] for row in uc_rows})}",
        f"- 节点预算：{budgets}",
        f"- 保留的固定不可行实例数：{len({row['instance_id'] for row in uc_rows if row['status'] == 'infeasible'})}",
        f"- 稳定性结论：**STABILITY {stability['status']}**",
        "",
        ]
    )
    for budget in budgets:
        lines.extend(
            [
                f"### node_limit={budget}",
                "",
                "| 策略 | 完成率 | 节点中位数 | 节点几何均值 | gap中位数 | win/tie/loss | 推理时间均值(s) |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in strategy_summary(uc_rows, budget):
            lines.append(
                f"| {row['strategy']} | {row['completion_rate']:.3f} | {_format_num(row['median_nodes'])} | {_format_num(row['geo_mean_nodes'])} | {_format_num(row['median_gap'])} | {row['win']}/{row['tie']}/{row['loss']} | {row['mean_inference_time']:.6f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## 5. 稳定性判据逐项",
            "",
            "| GNN seed | 节点中位数下降 | win率 | 完成率 | 是否通过 |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for detail in stability.get("details", []):
        lines.append(
            f"| {detail['strategy']} | {detail['median_reduction']:.3f} | {detail['win_rate']:.3f} | {detail['completion_rate']:.3f} | {'PASS' if detail['pass'] else 'FAIL'} |"
        )
    lines.extend(["", "## 6. 跨模型族零样本结果", ""])
    if cross_rows:
        families = sorted({row["family_name"] for row in cross_rows})
        lines.extend(["| 模型族 | GNN相对原规则 | GNN相对pseudocost | 判断 |", "|---|---:|---:|---|"])
        gnn_strategies = _gnn_strategies(cross_rows)
        for family in families:
            maps: dict[str, dict[str, dict]] = defaultdict(dict)
            for row in cross_rows:
                if row["family_name"] == family:
                    maps[row["instance_id"]][row["strategy"]] = row
            gnn_wtl = _wtl_against(maps, "most_fractional", gnn_strategies)
            pseudo_wtl = _wtl_against(maps, "pseudocost", gnn_strategies)
            if gnn_wtl[0] > gnn_wtl[2] and pseudo_wtl[0] >= pseudo_wtl[2]:
                judgement = "有一定零样本泛化"
            elif gnn_wtl[2] > gnn_wtl[0] + gnn_wtl[1]:
                judgement = "明显退化"
            else:
                judgement = "基本持平"
            lines.append(f"| {family} | {gnn_wtl[0]}/{gnn_wtl[1]}/{gnn_wtl[2]} | {pseudo_wtl[0]}/{pseudo_wtl[1]}/{pseudo_wtl[2]} | {judgement} |")
    else:
        lines.append("TODO: cross-family benchmark not run.")
    lines.extend(
        [
            "",
            "## 7. 下一阶段判断",
            "",
            "根据稳定性与跨模型族结果判断是否进入多模型族联合训练、GNN+pseudocost 混合策略，或先修复训练稳定性。测试集 seed 已固定在 `stability_test.json`，不得按结果删改困难实例。",
            "",
            "## 8. 当前有效文件结构",
            "",
            "- 配置：`ml_branching/configs/stability_train.json`，`ml_branching/configs/stability_test.json`",
            "- CSV：`stability_results.csv`，`cross_family_results.csv`",
            "- checkpoint：`gnn_stability_seed_1.pt`，`gnn_stability_seed_2.pt`，`gnn_stability_seed_3.pt`，上一轮 MLP baseline",
            "- 图表：`reports/learning_branching/charts/`",
            "- 删除记录：旧 GNN 实验报告、旧 GNN CSV、旧配置、旧 checkpoint 和旧图表已由本报告与两份 CSV 替代。",
        ]
    )
    STABILITY_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return STABILITY_REPORT


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Stability and generality experiments for learned branching.")
    parser.add_argument("command", choices=["train", "benchmark-uc", "benchmark-cross", "report"])
    parser.add_argument("--train-config", default="ml_branching/configs/stability_train.json")
    parser.add_argument("--test-config", default="ml_branching/configs/stability_test.json")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "train":
        rows = train_stability_seeds(load_json(args.train_config), resume=not args.no_resume)
        print(json.dumps(rows, indent=2, sort_keys=True, ensure_ascii=False))
        return rows
    config = load_json(args.test_config)
    if args.command == "benchmark-uc":
        return run_uc_stability(config, resume=not args.no_resume)
    if args.command == "benchmark-cross":
        return run_cross_family(config, resume=not args.no_resume)
    return write_report(config)


if __name__ == "__main__":
    main()
