from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from examples.fixed_charge_block import build_problem as build_fixed_charge
from examples.general_integer_block import build_problem as build_general_integer
from examples.unit_commitment_tiny import build_problem as build_unit_commitment
from solver import solve_milp


CASES=[
    ("fixed_charge_block",build_fixed_charge),
    ("general_integer_block",build_general_integer),
    ("unit_commitment_tiny",build_unit_commitment),
]


def format_value(value) -> str:
    if value is None:
        return "None"
    return f"{float(value):.8g}"


def main() -> None:
    headers=[
        "case",
        "status",
        "objective",
        "nodes",
        "lp_solved",
        "prune_inf",
        "prune_bound",
        "prune_opt",
        "time_sec",
    ]
    rows=[]

    for case_name,builder in CASES:
        problem=builder()
        start=perf_counter()
        result=solve_milp(problem)
        elapsed=perf_counter()-start
        rows.append(
            [
                case_name,
                result.status,
                format_value(result.objective_value),
                str(result.num_nodes),
                str(result.num_lp_solved),
                str(result.num_pruned_infeasible),
                str(result.num_pruned_bound),
                str(result.num_pruned_optimality),
                f"{elapsed:.6f}",
            ]
        )

    widths=[max(len(headers[i]),*(len(row[i]) for row in rows)) for i in range(len(headers))]
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-"*width for width in widths))
    for row in rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(row))))


if __name__=="__main__":
    main()
