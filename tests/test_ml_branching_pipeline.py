import json

import numpy as np
import pytest

from ml_branching.evaluation.audit_dataset import audit_dataset
from ml_branching.training.data.collector import collect_dataset
from ml_branching.training.data.dataset import BranchingDataset
from ml_branching.evaluation.evaluate_solver import _solve_gurobi
from ml_branching.families import FAMILY_REGISTRY, get_family, reconstruct_instance
from ml_branching.runtime.features import FEATURE_NAMES, FeatureNormalizer, candidate_feature_matrix
from ml_branching.training.oracle.strong_branching import StrongBranchingPolicy
from solver import solve_milp


def _tiny_multifamily_config(tmp_path, dataset_id="test_multifamily"):
    return {
        "dataset_id": dataset_id,
        "output_dir": str(tmp_path / "generated"),
        "master_seed": 300,
        "families": ["unit_commitment", "capacity_expansion"],
        "splits": {
            "train": {"instances_per_family": 1, "sizes": [4], "scale_group": "small"},
            "validation": {"instances_per_family": 1, "sizes": [4], "scale_group": "small"},
            "in_distribution_test": {"instances_per_family": 1, "sizes": [4], "scale_group": "small"},
            "family_holdout_test": {
                "instances_per_family": 1,
                "families": ["random_sparse_block"],
                "sizes": [5],
                "scale_group": "holdout",
            },
        },
        "max_nodes_per_instance": 50,
        "max_decisions_per_instance": 5,
        "time_limit_per_instance": 10.0,
        "max_generation_attempts": 30,
        "lp_backend": "scipy_highs",
        "scoring_mode": "product",
        "epsilon": 1e-6,
        "mu": 0.25,
        "max_lp_candidates": None,
        "use_matrix_presolve": True,
        "tolerance": 1e-8,
    }


def test_family_generators_are_seed_reproducible_and_solvable():
    pytest.importorskip("scipy")
    for family_name in sorted(FAMILY_REGISTRY):
        family = get_family(family_name)
        a = family.generate(seed=10, size=5, split="train", scale_group="small")
        b = family.generate(seed=10, size=5, split="train", scale_group="small")
        assert a.instance_id == b.instance_id
        assert np.allclose(a.problem.G, b.problem.G)
        assert np.allclose(a.problem.c, b.problem.c)
        assert np.allclose(a.problem.h, b.problem.h)
        result = solve_milp(a.problem, lp_backend="scipy_highs", max_nodes=120)
        assert result.status in {"optimal", "node_limit"}
        if result.status == "optimal":
            assert result.objective_value is not None


def test_collect_dataset_has_multi_candidate_samples_and_split_isolation(tmp_path):
    pytest.importorskip("scipy")
    config = _tiny_multifamily_config(tmp_path)
    manifest = collect_dataset(config, overwrite=True)
    dataset = BranchingDataset.from_dir(config["output_dir"])
    dataset.assert_disjoint_splits()
    counts = [len(sample["arrays"]["candidate_indices"]) for sample in dataset]
    assert max(counts) >= 2
    assert any(count >= 2 for count in counts)
    train_families = {
        sample["metadata"]["family_name"]
        for sample in dataset.by_split("train").samples
    }
    holdout_families = {
        sample["metadata"]["family_name"]
        for sample in dataset.by_split("family_holdout_test").samples
    }
    assert "random_sparse_block" not in train_families
    assert holdout_families <= {"random_sparse_block"}
    assert manifest["total_samples"] == len(dataset.samples)


def test_candidate_count_one_is_filtered_from_training_metrics(tmp_path):
    pytest.importorskip("scipy")
    config = _tiny_multifamily_config(tmp_path)
    collect_dataset(config, overwrite=True)
    dataset = BranchingDataset.from_dir(config["output_dir"])
    effective = dataset.multi_candidate_samples(min_candidates=2)
    assert effective
    assert all(len(sample["arrays"]["candidate_indices"]) >= 2 for sample in effective)


def test_features_do_not_include_family_or_labels(tmp_path):
    pytest.importorskip("scipy")
    config = _tiny_multifamily_config(tmp_path)
    collect_dataset(config, overwrite=True)
    dataset = BranchingDataset.from_dir(config["output_dir"])
    sample = dataset.multi_candidate_samples(min_candidates=2)[0]
    features, names = candidate_feature_matrix(sample)
    assert features.shape[0] == len(sample["arrays"]["candidate_indices"])
    forbidden = {"family_name", "delta_0", "delta_1", "expert_scores", "child_0_bound", "child_1_bound"}
    assert not forbidden & set(names)
    assert FEATURE_NAMES == names
    normalizer = FeatureNormalizer.fit([features])
    assert normalizer.transform(features).shape == features.shape


def test_audit_fail_blocks_training(tmp_path):
    pytest.importorskip("torch")
    from ml_branching.training.config import TrainingConfig
    from ml_branching.training.trainer import train_one_config

    data_dir = tmp_path / "bad_data"
    data_dir.mkdir()
    (data_dir / "audit_summary.json").write_text(
        json.dumps({"audit_status": "FAIL", "dataset_id": "bad", "failures": ["candidate_count=1"]}),
        encoding="utf-8",
    )
    config = TrainingConfig(dataset_path=str(data_dir), output_dir=str(tmp_path / "models"), required_dataset_id="bad")
    with pytest.raises(ValueError, match="audit did not pass"):
        train_one_config(config)


def test_training_checkpoint_and_learned_policy_are_valid(tmp_path):
    pytest.importorskip("scipy")
    pytest.importorskip("torch")
    from ml_branching.runtime.inference import LearnedBranchingPolicy
    from ml_branching.training.train import main as train_main

    config = _tiny_multifamily_config(tmp_path, dataset_id="trainable_test")
    collect_dataset(config, overwrite=True)
    audit_dataset(
        config["output_dir"],
        tmp_path / "reports",
        thresholds={
            "min_multi_candidate_ratio": 0.1,
            "min_mean_candidate_count": 1.0,
            "min_max_candidate_count": 2,
            "min_divergence_rate": 0.0,
            "min_effective_samples": 1,
            "min_effective_train_samples": 1,
            "min_effective_validation_samples": 1,
            "min_effective_test_samples": 1,
            "min_effective_families": 1,
        },
    )
    train_config = tmp_path / "train_config.json"
    train_config.write_text(
        json.dumps(
            {
                "dataset_path": config["output_dir"],
                "output_dir": str(tmp_path / "models"),
                "require_audit_pass": True,
                "required_dataset_id": "trainable_test",
                "min_candidate_count": 2,
                "seed": 3,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "hidden_dim": 8,
                "num_layers": 2,
                "dropout": 0.0,
                "max_epochs": 2,
                "early_stopping_patience": 2,
                "device": "cpu",
                "hyperparameter_grid": {"learning_rate": [0.001], "hidden_dim": [8], "seed": [3]},
            }
        ),
        encoding="utf-8",
    )
    train_main(["--config", str(train_config)])
    summary = json.loads((tmp_path / "models" / "training_summary.json").read_text())
    checkpoint = summary["best_checkpoint"]
    policy = LearnedBranchingPolicy.from_checkpoint(checkpoint)
    params = BranchingDataset.from_dir(config["output_dir"]).multi_candidate_samples("in_distribution_test", 2)[0]["metadata"]["instance_parameters"]
    result = solve_milp(reconstruct_instance(params).problem, branching_policy=policy, lp_backend="scipy_highs", max_nodes=80)
    assert result.status in {"optimal", "node_limit"}


def test_strong_branching_probe_does_not_pollute_formal_node_count():
    pytest.importorskip("scipy")
    instance = get_family("unit_commitment").generate(seed=40, size=4, split="eval", scale_group="small")
    most = solve_milp(instance.problem, lp_backend="scipy_highs", max_nodes=100)
    policy = StrongBranchingPolicy(lp_backend="scipy_highs")
    strong = solve_milp(instance.problem, lp_backend="scipy_highs", branching_policy=policy, max_nodes=100)
    assert policy.probe_lp_solved >= 0
    assert strong.num_lp_solved <= strong.num_nodes + 3
    if most.status == strong.status == "optimal":
        assert most.objective_value == pytest.approx(strong.objective_value)


def test_gurobi_metrics_are_read_when_available():
    pytest.importorskip("gurobipy")
    instance = get_family("capacity_expansion").generate(seed=55, size=4, split="eval", scale_group="small")
    metrics = _solve_gurobi(instance.problem, time_limit=5.0, seed=1, threads=1)
    assert {"status", "objective", "best_bound", "mip_gap", "runtime_sec", "node_count"} <= set(metrics)
    assert metrics["threads"] == "1"
    assert metrics["time_limit"] == "5.0"
