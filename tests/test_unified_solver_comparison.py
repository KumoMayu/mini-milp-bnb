import csv
import json
from pathlib import Path

import pytest

from benchmarks.learning_branching import unified_solver_comparison as unified
from solver import MatrixPresolveOptions, solve_milp
from solver.lp_active_set import solve_lp_relaxation


def test_matrix_presolve_filters_can_be_toggled_independently():
    problem = unified.build_problem(
        {
            "id": "toggle_case",
            "family": "unit_commitment",
            "seed": 1,
            "units": 3,
            "demand_fraction": 0.58,
        }
    )
    node_lb = problem.lb.copy()
    node_ub = problem.ub.copy()
    node_ub[problem.binary_indices[0]] = 0.0

    no_filter = solve_lp_relaxation(problem, node_lb, node_ub, use_matrix_presolve=False)
    fixed_only = solve_lp_relaxation(
        problem,
        node_lb,
        node_ub,
        use_matrix_presolve=True,
        matrix_presolve_options=MatrixPresolveOptions(True, False, False, 1),
    )
    full = solve_lp_relaxation(
        problem,
        node_lb,
        node_ub,
        use_matrix_presolve=True,
        matrix_presolve_options=unified.full_presolve_options(),
    )

    assert no_filter.num_fixed_vars == 0
    assert fixed_only.num_fixed_vars >= 1
    assert full.num_fixed_vars >= fixed_only.num_fixed_vars
    assert full.num_candidates_checked <= fixed_only.num_candidates_checked <= no_filter.num_candidates_checked


def test_four_solver_versions_have_expected_configuration():
    config = unified.load_config()
    versions = {row["version"]: row for row in unified.unified_2x2_specs(config)}

    assert set(versions) == {
        "自制求解器基础版",
        "自制求解器GNN增强版",
        "开源LP增强版",
        "开源LP＋GNN完整版",
    }
    assert versions["自制求解器基础版"]["lp_backend"] == "active_set"
    assert versions["自制求解器基础版"]["strategy_id"] == "most_fractional"
    assert versions["自制求解器GNN增强版"]["lp_backend"] == "active_set"
    assert versions["自制求解器GNN增强版"]["strategy_id"] == "gnn_seed_1"
    assert versions["开源LP增强版"]["lp_backend"] == "scipy_highs"
    assert versions["开源LP增强版"]["strategy_id"] == "most_fractional"
    assert versions["开源LP＋GNN完整版"]["lp_backend"] == "scipy_highs"
    assert versions["开源LP＋GNN完整版"]["strategy_id"] == "gnn_seed_1"
    assert all(row["use_matrix_presolve"] is True for row in versions.values())
    assert all(row["node_selection"] == "best_bound" for row in versions.values())


def test_base_2x2_version_uses_active_set_with_presolve():
    config = unified.load_config()
    spec = config["unified_instances"][0]
    base = unified.unified_2x2_specs(config)[0]

    row, _ = unified._solve_once(spec, base, config)

    assert row["lp_backend"] == "自制active-set"
    assert base["use_matrix_presolve"] is True
    assert int(row["fixed_variables"]) >= 0
    assert int(row["removed_rows"]) >= 0
    assert int(row["tightened_bounds"]) >= 0


def test_2x2_lp_backend_pairs_only_change_lp_backend():
    config = unified.load_config()
    versions = {row["version"]: row for row in unified.unified_2x2_specs(config)}
    active = versions["自制求解器基础版"]
    highs = versions["开源LP增强版"]

    assert active["lp_backend"] == "active_set"
    assert highs["lp_backend"] == "scipy_highs"
    assert active["strategy_id"] == highs["strategy_id"] == "most_fractional"
    assert active["use_matrix_presolve"] == highs["use_matrix_presolve"] is True
    assert active["node_selection"] == highs["node_selection"] == "best_bound"


def test_gnn_version_only_changes_branching_strategy():
    config = unified.load_config()
    versions = {row["version"]: row for row in unified.unified_2x2_specs(config)}
    base = versions["开源LP增强版"]
    gnn = versions["开源LP＋GNN完整版"]

    assert base["lp_backend"] == gnn["lp_backend"] == "scipy_highs"
    assert base["use_matrix_presolve"] == gnn["use_matrix_presolve"] is True
    assert base["node_selection"] == gnn["node_selection"] == "best_bound"
    assert base["strategy_id"] == "most_fractional"
    assert gnn["strategy_id"] == config["main_gnn_strategy"]


def test_unified_config_freezes_seed1_checkpoint_without_test_selection():
    config = unified.load_config()

    assert config["main_gnn_strategy"] == "gnn_seed_1"
    assert config["gnn_checkpoints"]["gnn_seed_1"].endswith("gnn_stability_seed_1.pt")
    selection_text = json.dumps(config["selection_notes"], ensure_ascii=False)
    assert "GNN" in selection_text
    assert "not used for instance selection" in selection_text


def test_chinese_mappings_are_complete_for_reported_names():
    for key in ["most_fractional", "gnn_main", "gnn_seed_1", "gnn_seed_2", "gnn_seed_3", "strong_branching"]:
        assert key in unified.STRATEGY_ZH
    for key in ["optimal", "infeasible", "candidate_limit", "node_limit", "time_limit"]:
        assert key in unified.STATUS_ZH


def test_timed_run_uses_median_of_three_repeats(monkeypatch):
    calls = iter([9.0, 1.0, 5.0])

    def fake_solve_once(spec, solver_spec, config):
        value = next(calls)
        return (
            {
                "total_runtime_sec": str(value),
                "status": "最优",
                "bb_nodes": "1",
                "formal_lp_solved": "1",
            },
            object(),
        )

    monkeypatch.setattr(unified, "_solve_once", fake_solve_once)
    config = {"run_settings": {"warmup": False, "repeats": 3}}
    row = unified._median_run({}, {}, config)

    assert row["total_runtime_sec"] == "5"
    assert row["time_repeats_sec"] == "9;1;5"


def test_generated_csvs_do_not_expose_internal_strategy_or_status_ids():
    for path in [
        Path("reports/learning_branching/data/presolve_and_queue_comparison.csv"),
        Path("reports/learning_branching/data/branching_comparison.csv"),
        Path("reports/learning_branching/data/stability_results.csv"),
        Path("reports/learning_branching/data/unified_comparison.csv"),
    ]:
        if not path.exists():
            pytest.skip("unified comparison CSVs have not been generated")
        text = path.read_text(encoding="utf-8")
        assert "most_fractional" not in text
        assert "gnn_seed_" not in text
        assert "strong_branching" not in text
        assert "optimal" not in text
        assert "node_limit" not in text
        assert "candidate_limit" not in text
        assert "time_limit" not in text


def test_completed_objectives_are_consistent_in_generated_csvs():
    path = Path("reports/learning_branching/data/unified_comparison.csv")
    if not path.exists():
        pytest.skip("unified comparison CSV has not been generated")
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    by_case = {}
    for row in rows:
        if row["status"] == "最优":
            by_case.setdefault(row["case"], set()).add(round(float(row["objective"]), 8))
    assert by_case
    assert all(len(values) == 1 for values in by_case.values())


def test_limit_status_and_gap_columns_are_recorded():
    path = Path("reports/learning_branching/data/unified_comparison.csv")
    if not path.exists():
        pytest.skip("unified comparison CSV has not been generated")
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    assert all("limit_type" in row for row in rows)
    assert all("relative_gap" in row for row in rows)


def test_strategy_policies_return_legal_candidates_when_torch_available():
    pytest.importorskip("torch")
    config = unified.load_config()
    spec = config["instances"][0]
    problem = unified.build_problem(spec)
    lp = solve_lp_relaxation(
        problem,
        problem.lb,
        problem.ub,
        use_matrix_presolve=True,
        matrix_presolve_options=unified.full_presolve_options(),
    )
    candidates = tuple(problem.binary_indices[:2])
    context = type(
        "_Context",
        (),
        {
            "problem": problem,
            "node_id": 0,
            "node_depth": 0,
            "node_lb": problem.lb,
            "node_ub": problem.ub,
            "lp_result": lp,
            "candidate_indices": candidates,
            "incumbent_internal_value": None,
            "current_node_internal_bound": lp.objective_value,
            "tolerance": 1e-8,
        },
    )()

    for strategy in ["most_fractional", "gnn_seed_1"]:
        policy = unified._policy(strategy, config)
        assert policy.select_variable(context) in candidates


def test_solver_runs_with_dfs_and_best_bound_active_set():
    config = unified.load_config()
    spec = config["instances"][0]
    problem = unified.build_problem(spec)
    dfs = solve_milp(
        problem,
        lp_backend="active_set",
        use_matrix_presolve=True,
        matrix_presolve_options=unified.full_presolve_options(),
        node_selection="dfs",
        max_nodes=50,
    )
    best = solve_milp(
        problem,
        lp_backend="active_set",
        use_matrix_presolve=True,
        matrix_presolve_options=unified.full_presolve_options(),
        node_selection="best_bound",
        max_nodes=50,
    )

    assert dfs.objective_value == pytest.approx(best.objective_value)
