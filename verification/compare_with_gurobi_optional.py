from __future__ import annotations

import importlib.util
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

from benchmarks.solver.cases import CORE_CASES, SCALING_CASES
from solver import solve_milp


CASES=CORE_CASES+SCALING_CASES
MAX_LP_CANDIDATES=250000
TOL=1e-7


def scipy_available() -> bool:
    if importlib.util.find_spec("scipy") is None:
        return False
    from scipy.optimize import linprog

    return bool(linprog(c=[1.0],bounds=[(0.0,1.0)],method="highs").success)


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
        variables.append(model.addVar(lb=float(problem.lb[j]),ub=float(problem.ub[j]),vtype=vtype,name=f"z{j}"))

    objective=gp.quicksum(float(problem.c[j])*variables[j] for j in range(problem.num_vars))
    model.setObjective(objective,GRB.MAXIMIZE if problem.sense=="max" else GRB.MINIMIZE)

    for i in range(problem.num_constraints):
        model.addConstr(
            gp.quicksum(float(problem.G[i,j])*variables[j] for j in range(problem.num_vars))<=float(problem.h[i]),
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


def display_status(status: str) -> str:
    if status in {"candidate_limit","node_limit","time_limit"}:
        return "LIMIT"
    return status


def make_table(headers,rows) -> str:
    widths=[max(len(headers[i]),*(len(row[i]) for row in rows)) for i in range(len(headers))]
    lines=[
        " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))),
        "-+-".join("-"*width for width in widths),
    ]
    lines.extend(" | ".join(row[i].ljust(widths[i]) for i in range(len(row))) for row in rows)
    return "\n".join(lines)


def mini_rows(case_name: str,problem,gurobi_obj,gurobi_time,include_scipy: bool):
    rows=[]
    for backend,use_presolve in [("active_set",True),("scipy_highs",False)]:
        if backend=="scipy_highs" and not include_scipy:
            continue
        result=solve_milp(
            problem,
            lp_backend=backend,
            use_matrix_presolve=use_presolve,
            max_lp_candidates=MAX_LP_CANDIDATES,
        )
        match=(
            result.status=="optimal"
            and result.objective_value is not None
            and gurobi_obj is not None
            and abs(float(result.objective_value)-float(gurobi_obj))<=TOL
        )
        rows.append(
            [
                case_name,
                backend,
                display_status(result.status),
                format_value(result.objective_value),
                format_value(gurobi_obj),
                str(match),
                f"{result.runtime_sec:.6f}",
                f"{gurobi_time:.6f}",
                str(result.num_nodes),
                str(result.num_lp_solved),
            ]
        )
    return rows


def main() -> None:
    include_scipy=scipy_available()
    rows=[]
    for case_name,builder in CASES:
        problem=builder()
        gurobi_obj,gurobi_time,gurobi_status=solve_with_gurobi(problem)
        if gurobi_status!=GRB.OPTIMAL:
            rows.append([case_name,"gurobi_reference",f"gurobi_status_{gurobi_status}","None","None","False","0",f"{gurobi_time:.6f}","",""])
            continue
        rows.extend(mini_rows(case_name,problem,gurobi_obj,gurobi_time,include_scipy))

    headers=[
        "case",
        "mini_backend",
        "mini_status",
        "mini_obj",
        "gurobi_obj",
        "match",
        "mini_time",
        "gurobi_time",
        "mini_nodes",
        "mini_lp_solved",
    ]
    print(make_table(headers,rows))
    print()
    print("Rows with mini_status=LIMIT are not proven optimal by the mini active-set backend.")


if __name__=="__main__":
    main()
