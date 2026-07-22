import numpy as np

from examples.fixed_charge_block import build_problem as build_fixed_charge
from examples.production_expansion_binary import build_problem as build_production_expansion
from examples.unit_commitment_tiny import build_problem as build_unit_commitment
from solver import solve_milp


CORE_BUILDERS=(build_fixed_charge,build_unit_commitment,build_production_expansion)


def test_core_examples_build_valid_binary_block_problems():
    for builder in CORE_BUILDERS:
        problem=builder()
        assert problem.num_constraints>0
        assert problem.num_vars>0
        assert problem.num_continuous>0
        assert len(problem.binary_indices)>0
        assert len(problem.integer_indices)==len(problem.binary_indices)
        assert np.all(np.isfinite(problem.lb))
        assert np.all(np.isfinite(problem.ub))


def test_core_examples_solve_optimal():
    for builder in CORE_BUILDERS:
        result=solve_milp(builder())
        assert result.status=="optimal"
        assert result.objective_value is not None
