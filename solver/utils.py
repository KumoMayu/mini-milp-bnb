from __future__ import annotations

import numpy as np


def is_integer_value(value: float,tol: float = 1e-8) -> bool:
    return abs(float(value)-round(float(value)))<=tol


def is_integral_solution(x,var_types,tol: float = 1e-8) -> bool:
    for i,t in enumerate(var_types):
        if t in ("I","B") and not is_integer_value(float(x[i]),tol):
            return False
    return True


def choose_branch_variable(x,var_types,tol: float = 1e-8,rule: str = "most_fractional") -> int | None:
    candidates=[]
    for i,t in enumerate(var_types):
        if t in ("I","B") and not is_integer_value(float(x[i]),tol):
            fractional_distance=abs(float(x[i])-round(float(x[i])))
            candidates.append((fractional_distance,i))

    if not candidates:
        return None
    if rule=="first_fractional":
        return min(i for _,i in candidates)
    if rule=="most_fractional":
        return max(candidates)[1]
    raise ValueError('branching_rule must be "most_fractional" or "first_fractional"')


def check_feasibility(problem,x,lb=None,ub=None,tol: float = 1e-8) -> bool:
    x=np.asarray(x,dtype=float)
    lb=np.asarray(problem.lb if lb is None else lb,dtype=float)
    ub=np.asarray(problem.ub if ub is None else ub,dtype=float)
    if np.any(x<lb-tol) or np.any(x>ub+tol):
        return False
    if np.any(problem.G@x>problem.h+tol):
        return False
    return is_integral_solution(x,problem.var_types,tol)


def format_vector(x,digits: int = 6) -> str:
    return np.array2string(
        np.asarray(x,dtype=float),
        precision=digits,
        suppress_small=True,
        separator=", ",
    )
