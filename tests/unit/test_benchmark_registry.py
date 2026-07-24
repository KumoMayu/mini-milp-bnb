from __future__ import annotations

import numpy as np

from benchmarks.cases import available_families, build_case, iter_cases


def test_registry_contains_required_lp_and_milp_families():
    assert available_families("lp") == (
        "general_lp",
        "dense_lp",
        "numerical_lp",
    )
    assert available_families("milp") == (
        "knapsack",
        "set_cover",
        "facility_location",
        "unit_commitment",
        "lot_sizing",
    )


def test_small_and_large_use_expected_fixed_seeds():
    assert {case.seed for case in iter_cases("small")} == {0, 1, 2}
    assert {case.seed for case in iter_cases("large")} == {0}


def test_case_generation_is_deterministic():
    first = build_case("facility_location", "small", 2)
    second = build_case("facility_location", "small", 2)

    assert first.case_id == second.case_id
    assert np.array_equal(first.problem.c, second.problem.c)
    assert np.array_equal(first.problem.G, second.problem.G)
    assert np.array_equal(first.problem.h, second.problem.h)


def test_numerical_family_has_explicit_statuses():
    statuses = {
        build_case("numerical_lp", "small", seed).expected_status
        for seed in (0, 1, 2)
    }
    assert statuses == {"optimal", "infeasible", "unbounded"}


def test_large_cases_are_larger_than_small_cases():
    for family in available_families():
        small = build_case(family, "small", 0)
        large = build_case(family, "large", 0)
        assert large.metadata["num_variables"] > small.metadata["num_variables"]
        assert large.metadata["num_constraints"] >= small.metadata["num_constraints"]
