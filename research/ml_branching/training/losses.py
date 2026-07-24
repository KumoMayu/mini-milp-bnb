from __future__ import annotations

import torch
import torch.nn.functional as F


def node_cross_entropy(scores: torch.Tensor, expert_position: int) -> torch.Tensor:
    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional for one decision node")
    if not 0 <= int(expert_position) < scores.shape[0]:
        raise ValueError("expert_position is outside candidate score vector")
    target = torch.tensor([int(expert_position)], dtype=torch.long, device=scores.device)
    return F.cross_entropy(scores.unsqueeze(0), target)


def soft_targets_from_expert_scores(expert_scores: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if expert_scores.ndim != 1:
        raise ValueError("expert_scores must be one-dimensional")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    centered = expert_scores - torch.max(expert_scores)
    target = torch.softmax(centered / float(temperature), dim=0)
    if not torch.all(torch.isfinite(target)):
        raise ValueError("soft target contains nan or inf")
    return target


def soft_cross_entropy(scores: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    if scores.ndim != 1 or target_probs.ndim != 1:
        raise ValueError("scores and target_probs must be one-dimensional")
    if scores.shape != target_probs.shape:
        raise ValueError("scores and target_probs must have the same shape")
    return -(target_probs * F.log_softmax(scores, dim=0)).sum()


def expert_margin_weight(expert_scores: torch.Tensor, min_weight: float = 0.25, max_weight: float = 2.0) -> torch.Tensor:
    if expert_scores.numel() <= 1:
        return torch.tensor(float(min_weight), dtype=expert_scores.dtype, device=expert_scores.device)
    top2 = torch.topk(expert_scores, k=2).values
    margin = (top2[0] - top2[1]) / torch.clamp(torch.abs(top2[0]), min=1.0)
    return torch.clamp(min_weight + margin, min=min_weight, max=max_weight)


def pairwise_ranking_loss(scores: torch.Tensor, expert_scores: torch.Tensor, margin: float = 0.05) -> torch.Tensor:
    if scores.shape != expert_scores.shape:
        raise ValueError("scores and expert_scores must have the same shape")
    losses = []
    for i in range(len(scores)):
        for j in range(len(scores)):
            if float(expert_scores[i]) <= float(expert_scores[j]):
                continue
            desired_gap = torch.as_tensor(float(margin), dtype=scores.dtype, device=scores.device)
            losses.append(torch.relu(desired_gap - (scores[i] - scores[j])))
    if not losses:
        return torch.zeros((), dtype=scores.dtype, device=scores.device)
    return torch.stack(losses).mean()
