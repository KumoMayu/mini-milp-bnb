from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path
from time import perf_counter

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from benchmarks.solver.cases import ALL_CASES, BATCH_CASES, CORE_CASES, SCALING_CASES
from solver import solve_milp


FIELDNAMES=[
    "suite",
    "run_type",
    "case",
    "seed",
    "units",
    "solver",
    "backend",
    "presolve",
    "status",
    "objective",
    "global_bound",
    "relative_gap",
    "nodes",
    "lp_solved",
    "prune_infeasible",
    "prune_bound",
    "prune_integral",
    "removed_rows",
    "tightened_bounds",
    "candidates_checked",
    "time_sec",
    "match_reference",
    "note",
]

MAX_NODES=200
MAX_LP_CANDIDATES=250000
TOL=1e-7


def optional_module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def scipy_highs_available() -> bool:
    try:
        from scipy.optimize import linprog
    except ImportError:
        return False
    result=linprog(c=[1.0],bounds=[(0.0,1.0)],method="highs")
    return bool(result.success)


def display_status(status: str) -> str:
    if status in {"candidate_limit","node_limit","time_limit"}:
        return "LIMIT"
    return status


def format_value(value) -> str:
    if value is None:
        return ""
    return f"{float(value):.8g}"


def solver_name(backend: str) -> str:
    if backend=="active_set":
        return "mini_active_set"
    if backend=="scipy_highs":
        return "mini_scipy_highs"
    if backend=="gurobi":
        return "gurobi"
    return backend


def blank_row(
    suite: str,
    run_type: str,
    case_name: str,
    backend: str,
    presolve: str,
    status: str,
    note: str,
    seed: int | None = None,
    units: int | None = None,
) -> dict[str,str]:
    row={field:"" for field in FIELDNAMES}
    row.update(
        {
            "suite": suite,
            "run_type": run_type,
            "case": case_name,
            "seed": "" if seed is None else str(seed),
            "units": "" if units is None else str(units),
            "solver": solver_name(backend),
            "backend": backend,
            "presolve": presolve,
            "status": status,
            "note": note,
        }
    )
    return row


def result_row(
    suite: str,
    run_type: str,
    case_name: str,
    backend: str,
    presolve: str,
    result,
    elapsed: float,
    note: str = "",
    seed: int | None = None,
    units: int | None = None,
) -> dict[str,str]:
    candidates="-" if backend=="scipy_highs" else str(result.num_lp_candidates_checked)
    return {
        "suite": suite,
        "run_type": run_type,
        "case": case_name,
        "seed": "" if seed is None else str(seed),
        "units": "" if units is None else str(units),
        "solver": solver_name(backend),
        "backend": backend,
        "presolve": presolve,
        "status": display_status(result.status),
        "objective": format_value(result.objective_value),
        "global_bound": format_value(result.global_bound),
        "relative_gap": "" if result.relative_gap is None else f"{float(result.relative_gap):.8g}",
        "nodes": str(result.num_nodes),
        "lp_solved": str(result.num_lp_solved),
        "prune_infeasible": str(result.num_pruned_infeasible),
        "prune_bound": str(result.num_pruned_bound),
        "prune_integral": str(result.num_pruned_optimality),
        "removed_rows": str(result.num_removed_rows),
        "tightened_bounds": str(result.num_tightened_bounds),
        "candidates_checked": candidates,
        "time_sec": f"{elapsed:.6f}",
        "match_reference": "",
        "note": note if result.status not in {"candidate_limit","node_limit","time_limit"} else f"raw_status={result.status}",
    }


def solve_mini_case(
    suite: str,
    run_type: str,
    case_name: str,
    builder,
    backend: str,
    presolve: bool | None,
    seed: int | None = None,
    units: int | None = None,
) -> dict[str,str]:
    problem=builder()
    start=perf_counter()
    result=solve_milp(
        problem,
        lp_backend=backend,
        use_matrix_presolve=bool(presolve),
        max_nodes=MAX_NODES,
        max_lp_candidates=MAX_LP_CANDIDATES,
    )
    elapsed=perf_counter()-start
    return result_row(
        suite=suite,
        run_type=run_type,
        case_name=case_name,
        backend=backend,
        presolve="n/a" if presolve is None else ("on" if presolve else "off"),
        result=result,
        elapsed=elapsed,
        seed=seed,
        units=units,
    )


def solve_gurobi_case(suite: str,case_name: str,builder,seed: int | None = None,units: int | None = None) -> dict[str,str]:
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        return blank_row(suite,"gurobi",case_name,"gurobi","n/a","SKIPPED","gurobipy not installed",seed,units)

    problem=builder()
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
    elapsed=perf_counter()-start
    status="optimal" if model.Status==GRB.OPTIMAL else f"gurobi_status_{model.Status}"
    row=blank_row(suite,"gurobi",case_name,"gurobi","n/a",status,"optional full MIP reference",seed,units)
    row["objective"]=format_value(float(model.ObjVal) if model.Status==GRB.OPTIMAL else None)
    row["global_bound"]=row["objective"]
    row["relative_gap"]="0" if model.Status==GRB.OPTIMAL else ""
    row["time_sec"]=f"{elapsed:.6f}"
    row["candidates_checked"]="-"
    return row


def case_builders_for_suite(suite: str):
    if suite=="core":
        return CORE_CASES
    if suite=="scaling":
        return SCALING_CASES
    return ALL_CASES


def run_suite(suite: str) -> list[dict[str,str]]:
    rows=[]
    if suite=="core":
        for case_name,builder in CORE_CASES:
            rows.append(solve_mini_case("core","active_set_presolve_on",case_name,builder,"active_set",True))
    elif suite=="scaling":
        for case_name,builder in SCALING_CASES:
            rows.append(solve_mini_case("scaling","active_set_presolve_on",case_name,builder,"active_set",True))
    elif suite=="presolve":
        for case_name,builder in ALL_CASES:
            rows.append(solve_mini_case("presolve","active_set_presolve_off",case_name,builder,"active_set",False))
            rows.append(solve_mini_case("presolve","active_set_presolve_on",case_name,builder,"active_set",True))
    elif suite=="backends":
        scipy_ok=scipy_highs_available()
        for case_name,builder in ALL_CASES:
            rows.append(solve_mini_case("backends","active_set_presolve_on",case_name,builder,"active_set",True))
            if scipy_ok:
                rows.append(solve_mini_case("backends","scipy_highs",case_name,builder,"scipy_highs",None))
            else:
                rows.append(blank_row("backends","scipy_highs",case_name,"scipy_highs","n/a","SKIPPED","scipy not installed"))
            rows.append(solve_gurobi_case("backends",case_name,builder))
    elif suite=="batch":
        scipy_ok=scipy_highs_available()
        for case_name,builder,units,seed in BATCH_CASES:
            rows.append(
                solve_mini_case(
                    "batch",
                    "active_set_presolve_on",
                    case_name,
                    builder,
                    "active_set",
                    True,
                    seed,
                    units,
                )
            )
            if scipy_ok:
                rows.append(solve_mini_case("batch","scipy_highs",case_name,builder,"scipy_highs",None,seed,units))
            else:
                rows.append(blank_row("batch","scipy_highs",case_name,"scipy_highs","n/a","SKIPPED","scipy not installed",seed,units))
            rows.append(solve_gurobi_case("batch",case_name,builder,seed,units))
    else:
        raise ValueError(f"unknown suite: {suite}")
    add_reference_matches(rows)
    return rows


def add_reference_matches(rows: list[dict[str,str]]) -> None:
    references={}
    for row in rows:
        if row["backend"]=="gurobi" and row["status"]=="optimal" and row["objective"]:
            references[row["case"]]=float(row["objective"])
    for row in rows:
        if row["case"] not in references and row["backend"]=="scipy_highs" and row["status"]=="optimal" and row["objective"]:
            references[row["case"]]=float(row["objective"])
    for row in rows:
        if row["case"] not in references and row["run_type"]=="active_set_presolve_on" and row["status"]=="optimal" and row["objective"]:
            references[row["case"]]=float(row["objective"])

    for row in rows:
        if not row["objective"] or row["status"]!="optimal" or row["case"] not in references:
            continue
        row["match_reference"]=str(abs(float(row["objective"])-references[row["case"]])<=TOL)


def make_markdown_table(rows: list[dict[str,str]]) -> str:
    headers=FIELDNAMES
    widths=[max(len(header),*(len(row[header]) for row in rows)) for header in headers]
    lines=[
        " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))),
        "-+-".join("-"*width for width in widths),
    ]
    lines.extend(" | ".join(row[headers[i]].ljust(widths[i]) for i in range(len(headers))) for row in rows)
    return "\n".join(lines)


def write_outputs(rows: list[dict[str,str]],suite: str) -> None:
    reports_dir=ROOT/"reports"
    reports_dir.mkdir(exist_ok=True)

    csv_path=reports_dir/"benchmark_latest.csv"
    with csv_path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    md_path=reports_dir/"benchmark_latest.md"
    md_path.write_text(
        "# Benchmark Latest\n\n"
        f"Suite: `{suite}`\n\n"
        f"Limits: `max_nodes={MAX_NODES}`, `max_lp_candidates={MAX_LP_CANDIDATES}`.\n\n"
        "Status `LIMIT` means the mini solver reached a configured node/candidate/time limit; "
        "the objective is the incumbent value when available, not a proven optimum.\n\n"
        "```text\n"
        +make_markdown_table(rows)
        +"\n```\n",
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None):
    parser=argparse.ArgumentParser(description="Run mini MILP benchmark suites.")
    parser.add_argument("--suite",choices=["core","scaling","presolve","backends","batch","all"],default="core")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> list[dict[str,str]]:
    args=parse_args(argv)
    suites=["core","scaling","presolve","backends","batch"] if args.suite=="all" else [args.suite]
    rows=[]
    for suite in suites:
        rows.extend(run_suite(suite))
    add_reference_matches(rows)
    print(make_markdown_table(rows))
    write_outputs(rows,args.suite)
    return rows


if __name__=="__main__":
    main()
