from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_branching.training.data import BranchingDataset


DEFAULT_THRESHOLDS = {
    "min_multi_candidate_ratio": 0.60,
    "min_mean_candidate_count": 2.0,
    "min_max_candidate_count": 4,
    "min_divergence_rate": 0.10,
    "min_effective_samples": 50,
    "min_effective_train_samples": 20,
    "min_effective_validation_samples": 10,
    "min_effective_test_samples": 10,
    "min_effective_families": 4,
    "min_family_multi_candidate_ratio": 0.20,
}


def _most_fractional_choice(sample: dict) -> int:
    candidates = [int(i) for i in sample["arrays"]["candidate_indices"]]
    lp_solution = sample["arrays"]["lp_solution"]
    return max((abs(float(lp_solution[i]) - round(float(lp_solution[i]))), int(i)) for i in candidates)[1]


def _expert_margin(sample: dict) -> tuple[float, float, bool]:
    scores = np.asarray(sample["arrays"]["expert_scores"], dtype=float)
    if len(scores) <= 1:
        return float(scores[0]) if len(scores) else 0.0, 0.0, True
    order = np.sort(scores)[::-1]
    margin = float(order[0] - order[1])
    normalized = margin / max(1.0, abs(float(order[0])))
    return margin, normalized, bool(normalized <= 1e-8)


def _sample_row(sample: dict) -> dict:
    metadata = sample["metadata"]
    candidate_count = len(sample["arrays"]["candidate_indices"])
    expert = int(metadata["expert_selected_variable"])
    most_fractional = _most_fractional_choice(sample)
    margin, normalized_margin, near_tie = _expert_margin(sample)
    params = metadata.get("instance_parameters", {})
    return {
        "dataset_id": metadata["dataset_id"],
        "split": metadata["split"],
        "family_name": metadata["family_name"],
        "scale_group": metadata["scale_group"],
        "instance_id": metadata["instance_id"],
        "seed": int(metadata["instance_seed"]),
        "units": int(params.get("units", params.get("size", 0))),
        "node_depth": int(metadata["node_depth"]),
        "candidate_count": int(candidate_count),
        "is_multi_candidate": int(candidate_count >= 2),
        "is_three_plus_candidate": int(candidate_count >= 3),
        "expert_variable": expert,
        "most_fractional_variable": most_fractional,
        "expert_differs_from_most_fractional": int(expert != most_fractional) if candidate_count >= 2 else 0,
        "expert_margin": margin,
        "expert_normalized_margin": normalized_margin,
        "near_tie": int(near_tie),
    }


def _aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {
            "decision_samples": 0,
            "single_candidate_samples": 0,
            "effective_multi_candidate_samples": 0,
            "three_plus_candidate_samples": 0,
            "multi_candidate_ratio": 0.0,
            "candidate_count_mean": 0.0,
            "candidate_count_median": 0.0,
            "candidate_count_max": 0,
            "divergence_rate": 0.0,
            "mean_expert_normalized_margin": 0.0,
            "near_tie_ratio": 0.0,
        }
    counts = [int(row["candidate_count"]) for row in rows]
    multi = [row for row in rows if int(row["candidate_count"]) >= 2]
    return {
        "decision_samples": len(rows),
        "single_candidate_samples": len(rows) - len(multi),
        "effective_multi_candidate_samples": len(multi),
        "three_plus_candidate_samples": sum(int(row["candidate_count"]) >= 3 for row in rows),
        "multi_candidate_ratio": len(multi) / len(rows),
        "candidate_count_mean": float(np.mean(counts)),
        "candidate_count_median": float(median(counts)),
        "candidate_count_max": max(counts),
        "divergence_rate": (
            0.0
            if not multi
            else sum(int(row["expert_differs_from_most_fractional"]) for row in multi) / len(multi)
        ),
        "mean_expert_normalized_margin": (
            0.0 if not multi else float(np.mean([float(row["expert_normalized_margin"]) for row in multi]))
        ),
        "near_tie_ratio": 0.0 if not multi else sum(int(row["near_tie"]) for row in multi) / len(multi),
    }


def _group(rows: list[dict], key: str) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return {name: _aggregate(part) for name, part in sorted(grouped.items())}


def _status(overall: dict, by_split: dict, by_family: dict, thresholds: dict) -> tuple[str, list[str]]:
    failures: list[str] = []
    if overall["multi_candidate_ratio"] < float(thresholds["min_multi_candidate_ratio"]):
        failures.append("overall multi-candidate ratio below threshold")
    if overall["candidate_count_mean"] < float(thresholds["min_mean_candidate_count"]):
        failures.append("overall mean candidate_count below threshold")
    if overall["candidate_count_max"] < int(thresholds["min_max_candidate_count"]):
        failures.append("overall max candidate_count below threshold")
    if overall["divergence_rate"] < float(thresholds["min_divergence_rate"]):
        failures.append("strong branching / most_fractional divergence below threshold")
    if overall["effective_multi_candidate_samples"] < int(thresholds["min_effective_samples"]):
        failures.append("effective multi-candidate sample count below threshold")

    split_minima = {
        "train": int(thresholds["min_effective_train_samples"]),
        "validation": int(thresholds["min_effective_validation_samples"]),
        "in_distribution_test": int(thresholds["min_effective_test_samples"]),
    }
    for split, min_count in split_minima.items():
        if split in by_split and by_split[split]["effective_multi_candidate_samples"] < min_count:
            failures.append(f"{split} effective sample count below threshold")

    effective_families = [
        family
        for family, stats in by_family.items()
        if stats["effective_multi_candidate_samples"] > 0
        and stats["multi_candidate_ratio"] >= float(thresholds["min_family_multi_candidate_ratio"])
    ]
    if len(effective_families) < int(thresholds["min_effective_families"]):
        failures.append("too few families have usable multi-candidate samples")
    return ("PASS" if not failures else "FAIL"), failures


def audit_dataset(dataset_path: str | Path, report_dir: str | Path, thresholds: dict | None = None) -> dict:
    dataset_path = Path(dataset_path)
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    dataset = BranchingDataset.from_dir(dataset_path)
    dataset.assert_disjoint_splits()
    rows = [_sample_row(sample) for sample in dataset]
    overall = _aggregate(rows)
    by_family = _group(rows, "family_name")
    by_split = _group(rows, "split")
    by_scale = _group(rows, "scale_group")
    by_depth = _group(rows, "node_depth")
    status, failures = _status(overall, by_split, by_family, thresholds)
    manifest_path = dataset_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    summary = {
        "dataset_id": manifest.get("dataset_id", rows[0]["dataset_id"] if rows else ""),
        "dataset_path": str(dataset_path),
        "audit_status": status,
        "failures": failures,
        "thresholds": thresholds,
        "overall": overall,
        "by_family": by_family,
        "by_split": by_split,
        "by_scale_group": by_scale,
        "by_node_depth": by_depth,
    }
    (dataset_path / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )

    csv_path = report_dir / "dataset_audit.csv"
    fields = [
        "group_type",
        "group_name",
        "decision_samples",
        "single_candidate_samples",
        "effective_multi_candidate_samples",
        "three_plus_candidate_samples",
        "multi_candidate_ratio",
        "candidate_count_mean",
        "candidate_count_median",
        "candidate_count_max",
        "divergence_rate",
        "mean_expert_normalized_margin",
        "near_tie_ratio",
        "audit_status",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for group_type, groups in (
            ("overall", {"all": overall}),
            ("family", by_family),
            ("split", by_split),
            ("scale_group", by_scale),
            ("node_depth", by_depth),
        ):
            for name, stats in groups.items():
                writer.writerow(
                    {
                        "group_type": group_type,
                        "group_name": name,
                        **stats,
                        "audit_status": status if group_type == "overall" else "",
                    }
                )

    md_path = report_dir / "dataset_audit.md"
    lines = [
        "# 数据可学习性审计",
        "",
        f"- dataset: `{summary['dataset_id']}`",
        f"- path: `{dataset_path}`",
        f"- audit: `{status}`",
        f"- failures: `{failures}`",
        "",
        "## 总体",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
    ]
    for key, value in overall.items():
        lines.append(f"| {key} | {value:.6g} |" if isinstance(value, float) else f"| {key} | {value} |")
    lines.extend(["", "## 按模型族", "", "| family | decisions | multi | ratio | mean | max | divergence |", "|---|---:|---:|---:|---:|---:|---:|"])
    for family, stats in by_family.items():
        lines.append(
            f"| {family} | {stats['decision_samples']} | {stats['effective_multi_candidate_samples']} | "
            f"{stats['multi_candidate_ratio']:.3f} | {stats['candidate_count_mean']:.3f} | "
            f"{stats['candidate_count_max']} | {stats['divergence_rate']:.3f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def _load_thresholds(config_path: str | Path | None) -> dict:
    if not config_path:
        return {}
    path = Path(config_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return dict(data.get("audit_thresholds", {}))


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Audit whether branching data has real multi-candidate signal.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", default=None, help="Optional dataset config containing audit_thresholds.")
    parser.add_argument("--report-dir", default="reports/learning_branching")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    summary = audit_dataset(args.dataset, args.report_dir, _load_thresholds(args.config))
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    main()
