import numpy as np

from examples.fixed_charge_block import build_problem as build_fixed_charge
from examples.general_integer_block import build_problem as build_general_integer
from examples.unit_commitment_tiny import build_problem as build_unit_commitment
from solver import solve_milp


def test_core_examples_build_valid_problems():
    for builder in (build_fixed_charge,build_general_integer,build_unit_commitment):
        problem=builder()
        assert problem.num_constraints>0
        assert problem.num_vars>0
        assert len(problem.integer_indices)>0
        assert np.all(np.isfinite(problem.lb))
        assert np.all(np.isfinite(problem.ub))


def test_core_examples_solve_optimal():
    for builder in (build_fixed_charge,build_general_integer,build_unit_commitment):
        result=solve_milp(builder())
        assert result.status=="optimal"
        assert result.objective_value is not None
