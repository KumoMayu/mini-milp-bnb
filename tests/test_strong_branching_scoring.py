import pytest

from ml_branching.training.oracle.scoring import child_delta, infeasible_improvement_from_parent, score_from_deltas


def test_product_score_prefers_two_sided_improvement():
    two_sided=score_from_deltas(2.0,2.0)
    one_sided=score_from_deltas(4.0,0.0)

    assert two_sided>one_sided


def test_infeasible_child_delta_is_configurable():
    assert child_delta(10.0,None,"infeasible",infeasible_improvement=7.5)==pytest.approx(7.5)
    assert child_delta(-12.0,None,"infeasible")==pytest.approx(12.0)
    assert infeasible_improvement_from_parent(0.25)==pytest.approx(1.0)
    assert child_delta(10.0,8.0,"optimal")==pytest.approx(2.0)
    assert child_delta(10.0,None,"candidate_limit")==pytest.approx(0.0)


def test_scoring_is_deterministic():
    assert score_from_deltas(1.25,0.5)==pytest.approx(score_from_deltas(1.25,0.5))


def test_weighted_score_matches_definition():
    assert score_from_deltas(2.0,5.0,mode="weighted",mu=0.25)==pytest.approx(2.75)
