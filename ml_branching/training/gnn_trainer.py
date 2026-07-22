from __future__ import annotations

import csv
import json
import platform
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from ml_branching.training.data.dataset import BranchingDataset
from ml_branching.runtime.graph import (
    GLOBAL_FEATURE_NAMES,
    GRAPH_SCHEMA_VERSION,
    VARIABLE_FEATURE_NAMES,
    CONSTRAINT_FEATURE_NAMES,
    EDGE_FEATURE_NAMES,
    batch_graph_states,
    sample_to_bipartite_state,
)
from ml_branching.runtime.models import BipartiteGNN, BipartiteGNNConfig
from ml_branching.runtime.checkpoints import GNN_CHECKPOINT_VERSION
from ml_branching.training.losses import (
    expert_margin_weight,
    node_cross_entropy,
    pairwise_ranking_loss,
    soft_cross_entropy,
    soft_targets_from_expert_scores,
)



@dataclass(frozen=True)
class GNNTrainingConfig:
    dataset_path: str
    output_dir: str = "trained_models/learning_branching"
    report_dir: str = "reports/learning_branching"
    run_name: str = "gnn_round0"
    min_candidate_count: int = 2
    round_max: int | None = None
    seed: int = 0
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    hidden_dim: int = 64
    message_rounds: int = 2
    dropout: float = 0.0
    max_epochs: int = 20
    early_stopping_patience: int = 5
    min_delta: float = 1e-7
    gradient_clip: float = 1.0
    loss_type: str = "soft_margin"
    soft_temperature: float = 1.0
    pairwise_weight: float = 0.0
    device: str = "cpu"


def load_gnn_training_config(path: str | Path) -> GNNTrainingConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed = set(GNNTrainingConfig.__dataclass_fields__)
    return GNNTrainingConfig(**{key: value for key, value in data.items() if key in allowed})


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def select_device(name: str) -> str:
    if name == "cuda" and torch.cuda.is_available():
        return "cuda"
    if name == "mps" and torch.backends.mps.is_available():
        return "mps"
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    return "cpu"


def _effective_samples(dataset: BranchingDataset, split: str, min_candidates: int, round_max: int | None) -> list[dict]:
    samples = dataset.multi_candidate_samples(split, min_candidates)
    if round_max is None:
        return samples
    return [
        sample
        for sample in samples
        if int(sample["metadata"].get("round_id", 0)) <= int(round_max)
    ]


def _target_position(sample: dict) -> int:
    candidates = list(map(int, sample["arrays"]["candidate_indices"]))
    expert = int(sample["metadata"].get("expert_selected_variable", sample["metadata"].get("expert_choice")))
    return candidates.index(expert)


def _normalized_regret(sample: dict, predicted_position: int) -> float:
    scores = np.asarray(sample["arrays"]["expert_scores"], dtype=float)
    best = float(np.max(scores))
    chosen = float(scores[int(predicted_position)])
    return (best - chosen) / max(1.0, abs(best))


def _candidate_counts(samples: list[dict]) -> list[int]:
    return [int(len(sample["arrays"]["candidate_indices"])) for sample in samples]


def _loss_for_sample(model: BipartiteGNN, sample: dict, device: str, loss_type: str, temperature: float, pairwise_weight: float):
    state = sample_to_bipartite_state(sample)
    batch = batch_graph_states([state]).to(device)
    scores = model(batch)
    expert_position = _target_position(sample)
    expert_scores = torch.as_tensor(sample["arrays"]["expert_scores"], dtype=torch.float32, device=device)
    if loss_type == "hard_ce":
        loss = node_cross_entropy(scores, expert_position)
    elif loss_type in {"soft_ce", "soft_margin"}:
        target = soft_targets_from_expert_scores(expert_scores, temperature=temperature)
        loss = soft_cross_entropy(scores, target)
        if loss_type == "soft_margin":
            loss = loss * expert_margin_weight(expert_scores)
    else:
        raise ValueError('loss_type must be "hard_ce", "soft_ce", or "soft_margin"')
    if pairwise_weight > 0:
        loss = loss + float(pairwise_weight) * pairwise_ranking_loss(scores, expert_scores)
    return loss, scores


def evaluate_gnn_samples(
    model: BipartiteGNN,
    samples: list[dict],
    device: str = "cpu",
    loss_type: str = "soft_margin",
    temperature: float = 1.0,
    pairwise_weight: float = 0.0,
) -> dict:
    model.eval()
    losses: list[float] = []
    top1 = 0
    top3 = 0
    regrets: list[float] = []
    margins: list[float] = []
    with torch.no_grad():
        for sample in samples:
            loss, scores = _loss_for_sample(model, sample, device, loss_type, temperature, pairwise_weight)
            order = torch.argsort(scores, descending=True).detach().cpu().numpy().astype(int).tolist()
            target = _target_position(sample)
            expert_scores = np.asarray(sample["arrays"]["expert_scores"], dtype=float)
            sorted_expert = np.sort(expert_scores)[::-1]
            margin = 0.0 if len(sorted_expert) < 2 else (float(sorted_expert[0]) - float(sorted_expert[1])) / max(1.0, abs(float(sorted_expert[0])))
            margins.append(margin)
            losses.append(float(loss.detach().cpu()))
            top1 += int(order[0] == target)
            top3 += int(target in order[: min(3, len(order))])
            regrets.append(_normalized_regret(sample, order[0]))
    count = len(samples)
    candidate_counts = _candidate_counts(samples)
    return {
        "count": int(count),
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "top1_accuracy": float(top1 / count) if count else 0.0,
        "top3_accuracy": float(top3 / count) if count else 0.0,
        "mean_normalized_regret": float(np.mean(regrets)) if regrets else float("nan"),
        "median_normalized_regret": float(np.median(regrets)) if regrets else float("nan"),
        "mean_candidate_count": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
        "max_candidate_count": int(max(candidate_counts)) if candidate_counts else 0,
        "mean_expert_margin": float(np.mean(margins)) if margins else float("nan"),
    }


def _checkpoint_payload(config: GNNTrainingConfig, dataset: BranchingDataset, best_epoch: int, metrics: dict, log: list[dict], runtime: float) -> dict:
    manifest_path = Path(config.dataset_path) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    return {
        "checkpoint_version": GNN_CHECKPOINT_VERSION,
        "graph_schema_version": GRAPH_SCHEMA_VERSION,
        "variable_feature_names": VARIABLE_FEATURE_NAMES,
        "constraint_feature_names": CONSTRAINT_FEATURE_NAMES,
        "edge_feature_names": EDGE_FEATURE_NAMES,
        "global_feature_names": GLOBAL_FEATURE_NAMES,
        "dataset_id": manifest.get("dataset_id", ""),
        "training_config": asdict(config),
        "training_seed": int(config.seed),
        "best_epoch": int(best_epoch),
        "validation_metrics": metrics,
        "training_log": log,
        "training_time_sec": float(runtime),
        "split_instance_ids": {split: dataset.by_split(split).instance_ids() for split in dataset.split_names()},
        "torch_version": torch.__version__,
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
    }


def train_gnn_one_config(config: GNNTrainingConfig) -> dict:
    set_reproducible_seed(int(config.seed))
    device = select_device(config.device)
    dataset = BranchingDataset.from_dir(config.dataset_path)
    dataset.assert_disjoint_splits()
    train_samples = _effective_samples(dataset, "train", config.min_candidate_count, config.round_max)
    validation_samples = _effective_samples(dataset, "validation", config.min_candidate_count, config.round_max)
    if not train_samples or not validation_samples:
        raise ValueError("GNN training requires non-empty effective train and validation splits")

    first_state = sample_to_bipartite_state(train_samples[0])
    model = BipartiteGNN(
        BipartiteGNNConfig(
            variable_dim=int(first_state.variable_features.shape[1]),
            constraint_dim=int(first_state.constraint_features.shape[1]),
            edge_dim=int(first_state.edge_features.shape[1]),
            global_dim=int(first_state.global_features.shape[0]),
            hidden_dim=int(config.hidden_dim),
            message_rounds=int(config.message_rounds),
            dropout=float(config.dropout),
        )
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.learning_rate), weight_decay=float(config.weight_decay))
    rng = np.random.default_rng(int(config.seed))
    best_metric = float("inf")
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_epoch = 0
    best_metrics: dict = {}
    stale = 0
    log: list[dict] = []
    start = perf_counter()

    for epoch in range(1, int(config.max_epochs) + 1):
        model.train()
        losses: list[float] = []
        for idx in rng.permutation(len(train_samples)).astype(int).tolist():
            sample = train_samples[int(idx)]
            optimizer.zero_grad()
            loss, _ = _loss_for_sample(
                model,
                sample,
                device,
                config.loss_type,
                float(config.soft_temperature),
                float(config.pairwise_weight),
            )
            loss.backward()
            if config.gradient_clip and config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.gradient_clip))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        train_metrics = evaluate_gnn_samples(model, train_samples, device, config.loss_type, config.soft_temperature, config.pairwise_weight)
        validation_metrics = evaluate_gnn_samples(model, validation_samples, device, config.loss_type, config.soft_temperature, config.pairwise_weight)
        row = {
            "epoch": int(epoch),
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "train_top1_accuracy": train_metrics["top1_accuracy"],
            "train_normalized_regret": train_metrics["mean_normalized_regret"],
            "validation_loss": validation_metrics["loss"],
            "validation_top1_accuracy": validation_metrics["top1_accuracy"],
            "validation_top3_accuracy": validation_metrics["top3_accuracy"],
            "validation_normalized_regret": validation_metrics["mean_normalized_regret"],
        }
        log.append(row)
        metric = validation_metrics["mean_normalized_regret"]
        if metric < best_metric - float(config.min_delta):
            best_metric = metric
            best_epoch = epoch
            best_metrics = validation_metrics
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(config.early_stopping_patience):
                break

    model.load_state_dict(best_state)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{config.run_name}.pt"
    payload = _checkpoint_payload(config, dataset, best_epoch, best_metrics, log, perf_counter() - start)
    model.save_checkpoint(checkpoint_path, payload)
    row = {
        "run_name": config.run_name,
        "checkpoint_path": str(checkpoint_path),
        "seed": int(config.seed),
        "loss_type": config.loss_type,
        "hidden_dim": int(config.hidden_dim),
        "message_rounds": int(config.message_rounds),
        "train_samples": int(len(train_samples)),
        "validation_samples": int(len(validation_samples)),
        "best_epoch": int(best_epoch),
        "validation_loss": best_metrics.get("loss", float("nan")),
        "validation_top1_accuracy": best_metrics.get("top1_accuracy", 0.0),
        "validation_top3_accuracy": best_metrics.get("top3_accuracy", 0.0),
        "validation_normalized_regret": best_metrics.get("mean_normalized_regret", float("nan")),
        "training_runtime_sec": float(payload["training_time_sec"]),
        "device": device,
    }
    return row


def append_training_summary(row: dict, report_dir: str | Path) -> Path:
    path = Path(report_dir) / "gnn_training_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
    rows = [old for old in existing if old.get("run_name") != str(row.get("run_name"))]
    rows.append({key: str(value) for key, value in row.items()})
    fieldnames = sorted({key for item in rows for key in item})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def train_gnn_from_config(config_path: str | Path) -> dict:
    config = load_gnn_training_config(config_path)
    row = train_gnn_one_config(config)
    append_training_summary(row, config.report_dir)
    return row


__all__ = [
    "GNN_CHECKPOINT_VERSION",
    "GNNTrainingConfig",
    "append_training_summary",
    "evaluate_gnn_samples",
    "load_gnn_training_config",
    "select_device",
    "set_reproducible_seed",
    "train_gnn_from_config",
    "train_gnn_one_config",
]
