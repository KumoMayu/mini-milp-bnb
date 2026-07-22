from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from ml_branching.runtime.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, FeatureNormalizer, candidate_feature_matrix
from ml_branching.runtime.graph import GRAPH_SCHEMA_VERSION, runtime_context_to_sample, sample_to_bipartite_state, single_graph_batch
from ml_branching.runtime.models import BipartiteGNN
from ml_branching.runtime.models import MLPRanker
from ml_branching.runtime.checkpoints import CHECKPOINT_VERSION, GNN_CHECKPOINT_VERSION
from solver.branching import BranchingContext


class LearnedBranchingPolicy:
    def __init__(
        self,
        model: MLPRanker,
        normalizer: FeatureNormalizer,
        checkpoint_payload: dict,
        checkpoint: str,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.normalizer = normalizer
        self.payload = checkpoint_payload
        self.checkpoint = checkpoint
        self.device = device
        self.inference_calls = 0
        self.inference_time_sec = 0.0

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "cpu") -> "LearnedBranchingPolicy":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"learned branching checkpoint not found: {path}")
        payload = MLPRanker.load_checkpoint(path, map_location=device)
        required = {
            "checkpoint_version",
            "model_state_dict",
            "model_config",
            "feature_names",
            "feature_schema_version",
            "normalizer",
            "dataset_id",
            "training_config",
            "validation_metrics",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"checkpoint missing required fields: {missing}")
        if payload["checkpoint_version"] != CHECKPOINT_VERSION:
            raise ValueError(f"unsupported checkpoint_version={payload['checkpoint_version']!r}")
        if list(payload["feature_names"]) != FEATURE_NAMES:
            raise ValueError("checkpoint feature_names do not match current FEATURE_NAMES")
        if payload["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
            raise ValueError("checkpoint feature_schema_version does not match current feature schema")
        normalizer = FeatureNormalizer.from_dict(payload["normalizer"])
        model = MLPRanker.from_config_dict(payload["model_config"])
        model.load_state_dict(payload["model_state_dict"])
        return cls(model=model, normalizer=normalizer, checkpoint_payload=payload, checkpoint=str(path), device=device)

    def _runtime_sample(self, context: BranchingContext) -> dict:
        if context.problem is None:
            raise ValueError("learned branching requires a MILPProblem in BranchingContext")
        if any(context.problem.var_types[index] != "B" for index in context.candidate_indices):
            raise ValueError("learned branching currently supports binary branching candidates only")
        return {
            "arrays": {
                "internal_objective_coefficients": context.problem.internal_c,
                "G": context.problem.G,
                "h": context.problem.h,
                "node_lb": context.node_lb,
                "node_ub": context.node_ub,
                "lp_solution": context.lp_result.x,
                "candidate_indices": np.asarray(context.candidate_indices, dtype=int),
                "binary_variable_indices": np.asarray(context.problem.binary_indices, dtype=int),
            },
            "metadata": {
                "node_depth": context.node_depth,
                "parent_internal_lp_bound": float(context.current_node_internal_bound),
                "incumbent_internal_value": context.incumbent_internal_value,
                "instance_parameters": {"units": len(context.problem.binary_indices)},
            },
        }

    def select_variable(self, context: BranchingContext) -> int:
        if not context.candidate_indices:
            raise ValueError("no fractional binary branching candidates")
        start = perf_counter()
        sample = self._runtime_sample(context)
        features, names = candidate_feature_matrix(sample)
        if names != FEATURE_NAMES:
            raise ValueError("runtime feature names do not match checkpoint")
        normalized = self.normalizer.transform(features)
        tensor = torch.as_tensor(normalized, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            scores = self.model(tensor).detach().cpu().numpy()
        candidates = list(map(int, context.candidate_indices))
        best_local = max(range(len(candidates)), key=lambda i: (float(scores[i]), -candidates[i]))
        choice = candidates[best_local]
        if choice not in context.candidate_indices:
            raise ValueError("learned policy selected a non-candidate variable")
        self.inference_calls += 1
        self.inference_time_sec += perf_counter() - start
        return int(choice)


class LearnedGNNBranchingPolicy:
    def __init__(
        self,
        model: BipartiteGNN,
        checkpoint_payload: dict,
        checkpoint: str,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.payload = checkpoint_payload
        self.checkpoint = checkpoint
        self.device = device
        self.inference_calls = 0
        self.inference_time_sec = 0.0

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "cpu") -> "LearnedGNNBranchingPolicy":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"learned GNN branching checkpoint not found: {path}")
        payload = BipartiteGNN.load_checkpoint(path, map_location=device)
        required = {
            "checkpoint_version",
            "model_state_dict",
            "model_config",
            "graph_schema_version",
            "training_config",
            "validation_metrics",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"GNN checkpoint missing required fields: {missing}")
        if payload["checkpoint_version"] != GNN_CHECKPOINT_VERSION:
            raise ValueError(f"unsupported GNN checkpoint_version={payload['checkpoint_version']!r}")
        if payload["graph_schema_version"] != GRAPH_SCHEMA_VERSION:
            raise ValueError("checkpoint graph_schema_version does not match current graph schema")
        model = BipartiteGNN.from_config_dict(payload["model_config"])
        model.load_state_dict(payload["model_state_dict"])
        return cls(model=model, checkpoint_payload=payload, checkpoint=str(path), device=device)

    def select_variable(self, context: BranchingContext) -> int:
        if not context.candidate_indices:
            raise ValueError("no fractional binary branching candidates")
        start = perf_counter()
        sample = runtime_context_to_sample(context)
        state = sample_to_bipartite_state(sample, tolerance=context.tolerance)
        batch = single_graph_batch(state).to(self.device)
        self.model.eval()
        with torch.no_grad():
            scores = self.model(batch).detach().cpu().numpy()
        candidates = list(map(int, context.candidate_indices))
        best_local = max(range(len(candidates)), key=lambda i: (float(scores[i]), -candidates[i]))
        choice = candidates[best_local]
        if choice not in context.candidate_indices:
            raise ValueError("learned GNN selected a non-candidate variable")
        self.inference_calls += 1
        self.inference_time_sec += perf_counter() - start
        return int(choice)
