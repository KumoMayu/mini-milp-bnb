from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    print("gurobipy is not installed. This optional comparison was not executed.")
    raise SystemExit(0)

from examples.fixed_charge_block import build_problem as build_fixed_charge
from examples.general_integer_block import build_problem as build_general_integer
from examples.unit_commitment_tiny import build_problem as build_unit_commitment
from solver import solve_milp


CASES=[
    ("fixed_charge_block",build_fixed_charge),
    ("general_integer_block",build_general_integer),
    ("unit_commitment_tiny",build_unit_commitment),
]


def solve_with_gurobi(problem):
    model=gp.Model(problem.name)
    model.Params.OutputFlag=0

    variables=[]
    for j,var_type in enumerate(problem.var_types):
        if var_type=="B":
            vtype=GRB.BINARY
        elif var_type=="I":
            vtype=GRB.INTEGER
        else:
            vtype=GRB.CONTINUOUS
        variables.append(
            model.addVar(
                lb=float(problem.lb[j]),
                ub=float(problem.ub[j]),
                vtype=vtype,
                name=f"z{j}",
            )
        )

    objective=gp.quicksum(float(problem.c[j])*variables[j] for j in range(problem.num_vars))
    model.setObjective(objective,GRB.MAXIMIZE if problem.sense=="max" else GRB.MINIMIZE)

    for i in range(problem.num_constraints):
        model.addConstr(
            gp.quicksum(float(problem.G[i,j])*variables[j] for j in range(problem.num_vars))
            <= float(problem.h[i]),
            name=f"c{i}",
        )

    start=perf_counter()
    model.optimize()
    runtime=perf_counter()-start

    if model.Status!=GRB.OPTIMAL:
        return None,runtime,model.Status
    return float(model.ObjVal),runtime,model.Status


def format_value(value) -> str:
    if value is None:
        return "None"
    return f"{float(value):.8g}"


def main() -> None:
    rows=[]
    for case_name,builder in CASES:
        problem=builder()
        mini_result=solve_milp(problem)
        gurobi_obj,gurobi_time,gurobi_status=solve_with_gurobi(problem)
        match=(
            mini_result.objective_value is not None
            and gurobi_obj is not None
            and abs(float(mini_result.objective_value)-float(gurobi_obj))<=1e-7
        )
        rows.append(
            [
                case_name,
                format_value(mini_result.objective_value),
                format_value(gurobi_obj),
                str(match),
                f"{mini_result.runtime_sec:.6f}",
                f"{gurobi_time:.6f}",
                str(mini_result.num_nodes),
                str(mini_result.num_lp_solved),
                str(gurobi_status),
            ]
        )

    headers=[
        "case",
        "mini_obj",
        "gurobi_obj",
        "match",
        "mini_time",
        "gurobi_time",
        "mini_nodes",
        "mini_lp_solved",
        "gurobi_status",
    ]
    widths=[max(len(headers[i]),*(len(row[i]) for row in rows)) for i in range(len(headers))]
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-"*width for width in widths))
    for row in rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(row))))
    print()
    print("Gurobi is expected to be faster; this script checks same-sample objectives and rough timing only.")


if __name__=="__main__":
    main()
