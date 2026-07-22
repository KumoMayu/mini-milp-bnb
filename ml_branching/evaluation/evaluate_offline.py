from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_branching.training.data import BranchingDataset
from ml_branching.runtime.features import FeatureNormalizer, candidate_feature_matrix
from ml_branching.runtime.graph import batch_graph_states, sample_to_bipartite_state
from ml_branching.runtime.models import BipartiteGNN, MLPRanker
from ml_branching.training.gnn_trainer import GNN_CHECKPOINT_VERSION
from ml_branching.training.trainer import evaluate_samples, normalized_regret, target_position


def _linear_scores(path: Path, normalized_features: np.ndarray) -> np.ndarray | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "ok":
        return None
    weights = np.asarray(payload["weights"], dtype=float)
    return normalized_features @ weights + float(payload["bias"])


def _rows_for_samples(model, normalizer, samples, device: str, strategy: str = "learned_mlp", linear_path: Path | None = None) -> list[dict]:
    rows = []
    model.eval()
    with torch.no_grad():
        for sample in samples:
            features, _ = candidate_feature_matrix(sample)
            normalized = normalizer.transform(features)
            if strategy == "linear_baseline":
                scores = _linear_scores(linear_path or Path(), normalized)
                if scores is None:
                    continue
            else:
                tensor = torch.as_tensor(normalized, dtype=torch.float32, device=device)
                scores = model(tensor).detach().cpu().numpy()
            order = list(np.argsort(-scores))
            target = target_position(sample)
            expert_scores = np.asarray(sample["arrays"]["expert_scores"], dtype=float)
            metadata = sample["metadata"]
            params = metadata["instance_parameters"]
            rows.append(
                {
                    "strategy": strategy,
                    "split": metadata["split"],
                    "family_name": metadata["family_name"],
                    "scale_group": metadata["scale_group"],
                    "instance_id": metadata["instance_id"],
                    "seed": int(metadata["instance_seed"]),
                    "units": int(params.get("units", params.get("size", 0))),
                    "node_depth": int(metadata["node_depth"]),
                    "candidate_count": len(order),
                    "top1": int(order[0] == target),
                    "top3": int(target in order[: min(3, len(order))]),
                    "normalized_regret": normalized_regret(sample, order[0]),
                    "predicted_expert_score": float(expert_scores[order[0]]),
                    "best_expert_score": float(np.max(expert_scores)),
                    "predicted_variable": int(sample["arrays"]["candidate_indices"][order[0]]),
                    "expert_variable": int(metadata["expert_selected_variable"]),
                }
            )
    return rows


def _rows_for_gnn_samples(model, samples, device: str, strategy: str = "learned_gnn") -> list[dict]:
    rows = []
    model.eval()
    with torch.no_grad():
        for sample in samples:
            state = sample_to_bipartite_state(sample)
            batch = batch_graph_states([state]).to(device)
            scores = model(batch).detach().cpu().numpy()
            order = list(np.argsort(-scores))
            target = target_position(sample)
            expert_scores = np.asarray(sample["arrays"]["expert_scores"], dtype=float)
            metadata = sample["metadata"]
            params = metadata["instance_parameters"]
            rows.append(
                {
                    "strategy": strategy,
                    "split": metadata["split"],
                    "family_name": metadata["family_name"],
                    "scale_group": metadata["scale_group"],
                    "instance_id": metadata["instance_id"],
                    "seed": int(metadata["instance_seed"]),
                    "units": int(params.get("units", params.get("size", 0))),
                    "node_depth": int(metadata["node_depth"]),
                    "candidate_count": len(order),
                    "top1": int(order[0] == target),
                    "top3": int(target in order[: min(3, len(order))]),
                    "normalized_regret": normalized_regret(sample, order[0]),
                    "predicted_expert_score": float(expert_scores[order[0]]),
                    "best_expert_score": float(np.max(expert_scores)),
                    "predicted_variable": int(sample["arrays"]["candidate_indices"][order[0]]),
                    "expert_variable": int(metadata["expert_selected_variable"]),
                }
            )
    return rows


def _aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {
            "count": 0,
            "top1_accuracy": 0.0,
            "top3_accuracy": 0.0,
            "mean_normalized_regret": float("nan"),
            "median_normalized_regret": float("nan"),
            "mean_predicted_expert_score": float("nan"),
            "mean_best_expert_score": float("nan"),
            "std_normalized_regret": float("nan"),
        }
    regrets = [float(row["normalized_regret"]) for row in rows]
    return {
        "count": len(rows),
        "top1_accuracy": float(np.mean([row["top1"] for row in rows])),
        "top3_accuracy": float(np.mean([row["top3"] for row in rows])),
        "mean_normalized_regret": float(np.mean(regrets)),
        "median_normalized_regret": float(np.median(regrets)),
        "std_normalized_regret": float(np.std(regrets)),
        "mean_predicted_expert_score": float(np.mean([row["predicted_expert_score"] for row in rows])),
        "mean_best_expert_score": float(np.mean([row["best_expert_score"] for row in rows])),
    }


def _group(rows: list[dict], key: str) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return {name: _aggregate(part) for name, part in sorted(grouped.items())}


def _nested_group(rows: list[dict], primary: str, secondary: str = "strategy") -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row[primary])].append(row)
    return {name: _group(part, secondary) for name, part in sorted(grouped.items())}


def write_reports(rows: list[dict], result: dict, report_dir: str | Path) -> None:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "gnn_offline_evaluation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["empty"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Evaluate learned branching checkpoint on held-out test decisions.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="ml_branching/data/generated/train")
    parser.add_argument("--report-dir", default="reports/learning_branching")
    parser.add_argument(
        "--splits",
        default="in_distribution_test,scale_extrapolation_test,family_holdout_test",
        help="Comma-separated split names. Only candidate_count>=2 samples are evaluated.",
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    dataset = BranchingDataset.from_dir(args.dataset)
    dataset.assert_disjoint_splits()
    splits = [part.strip() for part in args.splits.split(",") if part.strip()]
    test_samples = []
    for split in splits:
        test_samples.extend(dataset.multi_candidate_samples(split, min_candidates=2))
    if payload.get("checkpoint_version") == GNN_CHECKPOINT_VERSION:
        model = BipartiteGNN.from_config_dict(payload["model_config"])
        model.load_state_dict(payload["model_state_dict"])
        model.to(args.device)
        rows = _rows_for_gnn_samples(model, test_samples, args.device, "learned_gnn")
    else:
        model = MLPRanker.from_config_dict(payload["model_config"])
        model.load_state_dict(payload["model_state_dict"])
        model.to(args.device)
        normalizer = FeatureNormalizer.from_dict(payload["normalizer"])
        linear_path = Path(args.checkpoint).parent / "linear_baseline.json"
        rows = _rows_for_samples(model, normalizer, test_samples, args.device, "learned_mlp")
        rows.extend(_rows_for_samples(model, normalizer, test_samples, args.device, "linear_baseline", linear_path))
    result = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "splits": splits,
        "overall": _aggregate(rows),
        "by_strategy": _group(rows, "strategy"),
        "by_candidate_count": _group(rows, "candidate_count"),
        "by_units": _group(rows, "units"),
        "by_split": _nested_group(rows, "split"),
        "by_family": _nested_group(rows, "family_name"),
        "by_node_depth": _group(rows, "node_depth"),
        "by_seed": _group(rows, "seed"),
    }
    write_reports(rows, result, args.report_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
