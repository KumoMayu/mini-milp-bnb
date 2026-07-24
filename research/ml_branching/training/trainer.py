from __future__ import annotations

import hashlib
import csv
import json
import platform
import random
import sys
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from research.ml_branching.training.data.dataset import BranchingDataset
from research.ml_branching.runtime.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, FeatureNormalizer, candidate_feature_matrix
from research.ml_branching.runtime.models import MLPConfig, MLPRanker
from research.ml_branching.training.config import TrainingConfig
from research.ml_branching.runtime.checkpoints import CHECKPOINT_VERSION
from research.ml_branching.training.losses import node_cross_entropy



def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def manifest_hash(dataset_path: str | Path) -> str:
    path = Path(dataset_path) / "manifest.json"
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def target_position(sample: dict) -> int:
    candidates = list(map(int, sample["arrays"]["candidate_indices"]))
    expert = int(sample["metadata"].get("expert_selected_variable", sample["metadata"].get("expert_choice")))
    return candidates.index(expert)


def normalized_regret(sample: dict, predicted_position: int) -> float:
    scores = np.asarray(sample["arrays"].get("expert_scores", sample["metadata"].get("strong_scores")), dtype=float)
    best = float(np.max(scores))
    chosen = float(scores[int(predicted_position)])
    return (best - chosen) / max(1.0, abs(best))


def evaluate_samples(model: MLPRanker, samples: list[dict], normalizer: FeatureNormalizer, device: str = "cpu") -> dict:
    model.eval()
    losses: list[float] = []
    top1 = 0
    top3 = 0
    regrets: list[float] = []
    predicted_scores: list[float] = []
    best_scores: list[float] = []
    with torch.no_grad():
        for sample in samples:
            features, _ = candidate_feature_matrix(sample)
            features = normalizer.transform(features)
            tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
            target = target_position(sample)
            scores = model(tensor)
            loss = node_cross_entropy(scores, target)
            order = torch.argsort(scores, descending=True).detach().cpu().numpy().tolist()
            expert_scores = np.asarray(sample["arrays"].get("expert_scores", sample["metadata"].get("strong_scores")), dtype=float)
            losses.append(float(loss.detach().cpu()))
            top1 += int(order[0] == target)
            top3 += int(target in order[: min(3, len(order))])
            regrets.append(normalized_regret(sample, order[0]))
            predicted_scores.append(float(expert_scores[order[0]]))
            best_scores.append(float(np.max(expert_scores)))
    count = len(samples)
    return {
        "count": count,
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "top1_accuracy": float(top1 / count) if count else 0.0,
        "top3_accuracy": float(top3 / count) if count else 0.0,
        "mean_normalized_regret": float(np.mean(regrets)) if regrets else float("nan"),
        "median_normalized_regret": float(np.median(regrets)) if regrets else float("nan"),
        "mean_predicted_expert_score": float(np.mean(predicted_scores)) if predicted_scores else float("nan"),
        "mean_best_expert_score": float(np.mean(best_scores)) if best_scores else float("nan"),
    }


def _effective_samples(samples: list[dict], min_candidate_count: int) -> list[dict]:
    return [
        sample
        for sample in samples
        if len(sample["arrays"]["candidate_indices"]) >= int(min_candidate_count)
    ]


def _check_audit_gate(config: TrainingConfig) -> dict | None:
    if not config.require_audit_pass:
        return None
    audit_path = Path(config.dataset_path) / "audit_summary.json"
    if not audit_path.exists():
        raise ValueError(f"audit_summary.json is required before training: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("audit_status") != "PASS":
        raise ValueError(f"dataset audit did not pass; status={audit.get('audit_status')} failures={audit.get('failures')}")
    if config.required_dataset_id and audit.get("dataset_id") != config.required_dataset_id:
        raise ValueError(
            f"dataset_id mismatch: required={config.required_dataset_id!r}, audit={audit.get('dataset_id')!r}"
        )
    return audit


def _balanced_epoch_indices(samples: list[dict], rng: np.random.Generator, balance_by_family: bool) -> list[int]:
    if not balance_by_family:
        return rng.permutation(len(samples)).astype(int).tolist()
    family_to_indices: dict[str, list[int]] = {}
    for idx, sample in enumerate(samples):
        family_to_indices.setdefault(sample["metadata"].get("family_name", "unknown"), []).append(idx)
    if not family_to_indices:
        return []
    target = max(len(indices) for indices in family_to_indices.values())
    indices: list[int] = []
    for family_indices in family_to_indices.values():
        chosen = rng.choice(family_indices, size=target, replace=len(family_indices) < target)
        indices.extend(int(i) for i in chosen)
    rng.shuffle(indices)
    return indices


def _fit_linear_baseline(train_samples: list[dict], normalizer: FeatureNormalizer, output_dir: Path) -> dict:
    feature_rows = []
    score_rows = []
    for sample in train_samples:
        features, _ = candidate_feature_matrix(sample)
        feature_rows.append(normalizer.transform(features))
        score_rows.append(np.asarray(sample["arrays"]["expert_scores"], dtype=float))
    if not feature_rows:
        payload = {"status": "skipped", "reason": "no train samples"}
    else:
        X = np.vstack(feature_rows)
        y = np.concatenate(score_rows)
        X_aug = np.column_stack([X, np.ones(len(X))])
        ridge = 1e-6
        lhs = X_aug.T @ X_aug + ridge * np.eye(X_aug.shape[1])
        rhs = X_aug.T @ y
        weights = np.linalg.solve(lhs, rhs)
        payload = {
            "status": "ok",
            "feature_names": FEATURE_NAMES,
            "weights": weights[:-1].tolist(),
            "bias": float(weights[-1]),
            "train_candidate_rows": int(len(X)),
            "method": "ridge least-squares scorer on expert_scores",
        }
    (output_dir / "linear_baseline.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def _device(name: str) -> str:
    if name == "mps" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _checkpoint_payload(
    config: TrainingConfig,
    normalizer: FeatureNormalizer,
    dataset: BranchingDataset,
    dataset_path: str,
    best_epoch: int,
    validation_metrics: dict,
    training_log: list[dict],
    training_time_sec: float,
) -> dict:
    manifest_path = Path(dataset_path) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "feature_names": FEATURE_NAMES,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "normalizer": normalizer.to_dict(),
        "dataset_id": manifest.get("dataset_id", ""),
        "dataset_manifest_hash": manifest_hash(dataset_path),
        "training_seed": int(config.seed),
        "training_config": asdict(config),
        "best_epoch": int(best_epoch),
        "validation_metrics": validation_metrics,
        "training_log": training_log,
        "training_time_sec": float(training_time_sec),
        "torch_version": torch.__version__,
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "determinism_note": "torch.use_deterministic_algorithms(True, warn_only=True); CPU is the default device.",
        "train_instance_ids": dataset.by_split("train").instance_ids(),
        "validation_instance_ids": dataset.by_split("validation").instance_ids(),
        "split_instance_ids": {split: dataset.by_split(split).instance_ids() for split in dataset.split_names()},
    }


def train_one_config(config: TrainingConfig, run_name: str | None = None) -> dict:
    set_reproducible_seed(int(config.seed))
    device = _device(config.device)
    audit = _check_audit_gate(config)
    dataset = BranchingDataset.from_dir(config.dataset_path)
    dataset.assert_disjoint_splits()
    train_samples = _effective_samples(dataset.by_split("train").samples, config.min_candidate_count)
    validation_samples = _effective_samples(
        dataset.by_split("validation").samples or dataset.by_split("valid").samples,
        config.min_candidate_count,
    )
    if not train_samples or not validation_samples:
        raise ValueError(
            f"training requires non-empty effective train and validation splits with candidate_count>={config.min_candidate_count}"
        )

    train_features = [candidate_feature_matrix(sample)[0] for sample in train_samples]
    normalizer = FeatureNormalizer.fit(train_features, FEATURE_NAMES)
    model = MLPRanker(
        MLPConfig(
            input_dim=len(FEATURE_NAMES),
            hidden_dim=int(config.hidden_dim),
            num_layers=int(config.num_layers),
            dropout=float(config.dropout),
        )
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.learning_rate), weight_decay=float(config.weight_decay))

    best_metric = float("inf")
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_epoch = 0
    best_validation_metrics: dict = {}
    stale = 0
    log: list[dict] = []
    rng = np.random.default_rng(int(config.seed))
    start = perf_counter()

    for epoch in range(1, int(config.max_epochs) + 1):
        model.train()
        losses: list[float] = []
        for idx in _balanced_epoch_indices(train_samples, rng, bool(config.balance_train_by_family)):
            sample = train_samples[int(idx)]
            features, _ = candidate_feature_matrix(sample)
            features = normalizer.transform(features)
            tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
            target = target_position(sample)
            optimizer.zero_grad()
            scores = model(tensor)
            loss = node_cross_entropy(scores, target)
            loss.backward()
            if config.gradient_clip and config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.gradient_clip))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        train_metrics = evaluate_samples(model, train_samples, normalizer, device)
        validation_metrics = evaluate_samples(model, validation_samples, normalizer, device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "train_eval_loss": train_metrics["loss"],
            "train_top1_accuracy": train_metrics["top1_accuracy"],
            "train_normalized_regret": train_metrics["mean_normalized_regret"],
            "validation_loss": validation_metrics["loss"],
            "validation_top1_accuracy": validation_metrics["top1_accuracy"],
            "validation_normalized_regret": validation_metrics["mean_normalized_regret"],
        }
        log.append(row)
        metric = validation_metrics["mean_normalized_regret"]
        if metric < best_metric - float(config.min_delta):
            best_metric = metric
            best_epoch = epoch
            best_validation_metrics = validation_metrics
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(config.early_stopping_patience):
                break

    model.load_state_dict(best_state)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = run_name or f"lr_{config.learning_rate:g}_h_{config.hidden_dim}_seed_{config.seed}"
    checkpoint_path = output_dir / f"{name}.pt"
    payload = _checkpoint_payload(
        config=config,
        normalizer=normalizer,
        dataset=dataset,
        dataset_path=config.dataset_path,
        best_epoch=best_epoch,
        validation_metrics=best_validation_metrics,
        training_log=log,
        training_time_sec=perf_counter() - start,
    )
    model.save_checkpoint(checkpoint_path, payload)
    linear_baseline = _fit_linear_baseline(train_samples, normalizer, output_dir)
    return {
        "checkpoint_path": str(checkpoint_path),
        "config": asdict(config),
        "seed": int(config.seed),
        "train_loss": log[-1]["train_loss"] if log else float("nan"),
        "validation_loss": best_validation_metrics.get("loss", float("nan")),
        "validation_top1_accuracy": best_validation_metrics.get("top1_accuracy", 0.0),
        "validation_normalized_regret": best_validation_metrics.get("mean_normalized_regret", float("nan")),
        "best_epoch": int(best_epoch),
        "training_runtime_sec": float(payload["training_time_sec"]),
        "effective_train_samples": len(train_samples),
        "effective_validation_samples": len(validation_samples),
        "audit_status": None if audit is None else audit.get("audit_status"),
        "linear_baseline_path": str(output_dir / "linear_baseline.json"),
        "linear_baseline_status": linear_baseline.get("status"),
    }


def _grid_values(config: TrainingConfig) -> list[TrainingConfig]:
    grid = config.hyperparameter_grid or {}
    learning_rates = grid.get("learning_rate", [config.learning_rate])
    hidden_dims = grid.get("hidden_dim", [config.hidden_dim])
    seeds = grid.get("seed", [config.seed])
    configs = []
    for lr in learning_rates:
        for hidden_dim in hidden_dims:
            for seed in seeds:
                configs.append(
                    replace(
                        config,
                        learning_rate=float(lr),
                        hidden_dim=int(hidden_dim),
                        seed=int(seed),
                        hyperparameter_grid=None,
                    )
                )
    return configs


def run_training_grid(config: TrainingConfig) -> dict:
    rows = []
    for index, run_config in enumerate(_grid_values(config), start=1):
        run_name = f"run_{index:02d}_lr_{run_config.learning_rate:g}_h_{run_config.hidden_dim}_seed_{run_config.seed}"
        rows.append(train_one_config(run_config, run_name=run_name))
    best = min(rows, key=lambda row: (row["validation_normalized_regret"], row["validation_loss"], row["seed"]))
    output_dir = Path(config.output_dir)
    summary = {
        "runs": rows,
        "best_checkpoint": best["checkpoint_path"],
        "selection_metric": "validation_normalized_regret",
        "best_run": best,
    }
    for row in rows:
        path = Path(row["checkpoint_path"])
        if str(path) != best["checkpoint_path"] and path.exists():
            path.unlink()
            row["checkpoint_removed_after_selection"] = True
        else:
            row["checkpoint_removed_after_selection"] = False
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    csv_path = output_dir / "training_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return summary
