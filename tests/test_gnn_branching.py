from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("torch")

from ml_branching.training.data.dataset import BranchingDataset
from ml_branching.runtime.graph import batch_graph_states, sample_to_bipartite_state
from ml_branching.unit_commitment import instance_from_parameters


DATASET = "ml_branching/data/generated/unit_commitment_round0"


def _sample():
    dataset = BranchingDataset.from_dir(DATASET)
    return dataset.multi_candidate_samples("train", 2)[0]


def test_graph_edges_match_nonzero_g_matrix():
    sample = _sample()
    state = sample_to_bipartite_state(sample)
    G = np.asarray(sample["arrays"]["G"], dtype=float)
    expected = set(zip(*np.nonzero(np.abs(G) > 1e-12)))
    actual = {(int(row), int(col)) for col, row in state.edge_indices.T}
    assert actual == expected


def test_candidate_mask_and_positions_are_legal():
    sample = _sample()
    state = sample_to_bipartite_state(sample)
    candidates = set(map(int, sample["arrays"]["candidate_indices"]))
    assert set(map(int, state.candidate_indices)) == candidates
    assert set(map(int, state.candidate_positions)) == candidates
    candidate_mask = state.variable_features[:, 6]
    for index in range(len(candidate_mask)):
        assert bool(candidate_mask[index]) == (index in candidates)


def test_graph_batch_keeps_candidate_offsets_separate():
    sample = _sample()
    state_a = sample_to_bipartite_state(sample)
    state_b = sample_to_bipartite_state(sample)
    batch = batch_graph_states([state_a, state_b])
    assert batch.num_graphs == 2
    assert batch.graph_variable_offsets.tolist()[0] == 0
    assert batch.graph_variable_offsets.tolist()[1] == state_a.variable_features.shape[0]
    assert batch.candidate_graph_ids[: len(state_a.candidate_positions)].eq(0).all()
    assert batch.candidate_graph_ids[len(state_a.candidate_positions) :].eq(1).all()


def test_bipartite_gnn_supports_variable_graph_sizes_and_permutation_equivariance():
    torch = pytest.importorskip("torch")
    from ml_branching.runtime.models import BipartiteGNN, BipartiteGNNConfig

    sample = _sample()
    state = sample_to_bipartite_state(sample)
    model = BipartiteGNN(
        BipartiteGNNConfig(
            variable_dim=state.variable_features.shape[1],
            constraint_dim=state.constraint_features.shape[1],
            edge_dim=state.edge_features.shape[1],
            global_dim=state.global_features.shape[0],
            hidden_dim=8,
            message_rounds=1,
        )
    )
    model.eval()
    with torch.no_grad():
        scores = model(batch_graph_states([state]))

    var_perm = np.arange(state.variable_features.shape[0])
    con_perm = np.arange(state.constraint_features.shape[0])
    rng = np.random.default_rng(3)
    rng.shuffle(var_perm)
    rng.shuffle(con_perm)
    inv_var = np.empty_like(var_perm)
    inv_var[var_perm] = np.arange(len(var_perm))
    inv_con = np.empty_like(con_perm)
    inv_con[con_perm] = np.arange(len(con_perm))
    permuted = type(state)(
        variable_features=state.variable_features[var_perm],
        constraint_features=state.constraint_features[con_perm],
        edge_indices=np.vstack([inv_var[state.edge_indices[0]], inv_con[state.edge_indices[1]]]),
        edge_features=state.edge_features.copy(),
        global_features=state.global_features.copy(),
        candidate_indices=state.candidate_indices.copy(),
        candidate_positions=inv_var[state.candidate_positions],
        metadata=state.metadata,
    )
    with torch.no_grad():
        permuted_scores = model(batch_graph_states([permuted]))
    assert torch.allclose(scores, permuted_scores, atol=1e-6)


def test_soft_target_and_margin_are_finite():
    torch = pytest.importorskip("torch")
    from ml_branching.training.losses import expert_margin_weight, soft_cross_entropy, soft_targets_from_expert_scores

    expert = torch.tensor([1.0, 1.01, -2.0])
    target = soft_targets_from_expert_scores(expert, temperature=0.7)
    loss = soft_cross_entropy(torch.zeros(3), target)
    weight = expert_margin_weight(expert)
    assert torch.isfinite(target).all()
    assert torch.isfinite(loss)
    assert torch.isfinite(weight)


def test_dagger_uses_only_train_instances():
    from ml_branching.training.dagger import train_instance_parameters

    params = train_instance_parameters(DATASET)
    assert params
    assert {p["split"] for p in params} == {"train"}


def test_unit_commitment_reconstruction_is_exact_enough():
    sample = _sample()
    params = sample["metadata"]["instance_parameters"]
    instance = instance_from_parameters(params)
    assert np.allclose(instance.problem.G, sample["arrays"]["G"])
    assert np.allclose(instance.problem.h, sample["arrays"]["h"])
    assert np.allclose(instance.problem.internal_c, sample["arrays"]["internal_objective_coefficients"])


def test_gnn_checkpoint_loads_and_policy_returns_legal_candidate(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("scipy")
    from ml_branching.runtime.inference import LearnedGNNBranchingPolicy
    from ml_branching.runtime.models import BipartiteGNN, BipartiteGNNConfig
    from ml_branching.training.gnn_trainer import GNN_CHECKPOINT_VERSION
    from solver.branching import BranchingContext
    from solver.lp_backends import get_lp_relaxation_solver

    sample = _sample()
    state = sample_to_bipartite_state(sample)
    model = BipartiteGNN(
        BipartiteGNNConfig(
            variable_dim=state.variable_features.shape[1],
            constraint_dim=state.constraint_features.shape[1],
            edge_dim=state.edge_features.shape[1],
            global_dim=state.global_features.shape[0],
            hidden_dim=8,
            message_rounds=1,
        )
    )
    checkpoint = tmp_path / "gnn.pt"
    model.save_checkpoint(
        checkpoint,
        {
            "checkpoint_version": GNN_CHECKPOINT_VERSION,
            "graph_schema_version": state.metadata["schema_version"],
            "training_config": {},
            "validation_metrics": {},
        },
    )
    policy = LearnedGNNBranchingPolicy.from_checkpoint(checkpoint, device="cpu")
    instance = instance_from_parameters(sample["metadata"]["instance_parameters"])
    lp = get_lp_relaxation_solver("scipy_highs")(instance.problem, sample["arrays"]["node_lb"], sample["arrays"]["node_ub"])
    context = BranchingContext(
        problem=instance.problem,
        node_id=0,
        node_depth=0,
        node_lb=sample["arrays"]["node_lb"],
        node_ub=sample["arrays"]["node_ub"],
        lp_result=lp,
        candidate_indices=tuple(map(int, sample["arrays"]["candidate_indices"])),
        incumbent_internal_value=None,
        current_node_internal_bound=float(lp.objective_value),
        tolerance=1e-8,
    )
    choice = policy.select_variable(context)
    assert choice in context.candidate_indices


def test_no_invalid_old_report_references():
    allowed = {
        "GNN稳定性与通用性验证.md",
        "GNN分支学习过程说明.md",
        "presolve_and_queue_comparison.csv",
        "branching_comparison.csv",
        "stability_results.csv",
        "unified_comparison.csv",
    }
    report_dir = __import__("pathlib").Path("reports/learning_branching")
    if report_dir.exists():
        files = {p.name for p in report_dir.iterdir() if p.is_file()}
        assert files <= allowed
