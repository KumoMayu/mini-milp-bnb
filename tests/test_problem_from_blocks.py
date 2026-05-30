import numpy as np
import pytest

from solver import MILPProblem


def test_from_blocks_concatenates_standard_matrix():
    problem=MILPProblem.from_blocks(
        c_x=[1,2],
        c_y=[3,4],
        A=[[1,0],[0,1]],
        B=[[2,0],[0,3]],
        b=[5,6],
        x_lb=[0,0],
        x_ub=[10,10],
        y_lb=[0,0],
        y_ub=[5,1],
        y_types=["I","B"],
        sense="max",
    )

    assert problem.num_vars==4
    assert problem.num_constraints==2
    assert problem.num_continuous==2
    assert problem.num_integer==2
    assert problem.integer_indices==[2,3]
    assert problem.binary_indices==[3]
    assert np.allclose(problem.G,[[1,0,2,0],[0,1,0,3]])
    assert np.allclose(problem.c,[1,2,3,4])
    assert np.allclose(problem.lb,[0,0,0,0])
    assert np.allclose(problem.ub,[10,10,5,1])


def test_from_blocks_rejects_bad_binary_bounds():
    with pytest.raises(ValueError,match="binary"):
        MILPProblem.from_blocks(
            c_x=[1],
            c_y=[2],
            A=[[1]],
            B=[[1]],
            b=[3],
            x_lb=[0],
            x_ub=[5],
            y_lb=[0],
            y_ub=[2],
            y_types=["B"],
        )


def test_from_blocks_rejects_continuous_y_type():
    with pytest.raises(ValueError,match="y_types"):
        MILPProblem.from_blocks(
            c_x=[1],
            c_y=[2],
            A=[[1]],
            B=[[1]],
            b=[3],
            x_lb=[0],
            x_ub=[5],
            y_lb=[0],
            y_ub=[5],
            y_types=["C"],
        )
