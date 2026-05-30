from __future__ import annotations

import math
from time import perf_counter

import numpy as np

from .lp_active_set import solve_lp_relaxation
from .node import BBNode
from .result import MILPResult
from .utils import check_feasibility, choose_branch_variable, format_vector, is_integral_solution


class BranchAndBoundSolver:
    def __init__(
        self,
        problem,
        tol: float = 1e-8,
        max_nodes: int = 10000,
        node_selection: str = "dfs",
        branching_rule: str = "most_fractional",
        verbose: bool = False,
    ):
        if node_selection!="dfs":
            raise ValueError('node_selection currently supports only "dfs"')
        if branching_rule not in ("most_fractional","first_fractional"):
            raise ValueError('branching_rule must be "most_fractional" or "first_fractional"')
        self.problem=problem
        self.tol=tol
        self.max_nodes=max_nodes
        self.node_selection=node_selection
        self.branching_rule=branching_rule
        self.verbose=verbose

    def solve(self) -> MILPResult:
        start=perf_counter()
        problem=self.problem
        tol=self.tol

        pending_nodes=[
            BBNode(
                node_id=0,
                depth=0,
                lb=problem.lb.copy(),
                ub=problem.ub.copy(),
                parent_id=None,
                branch_direction="root",
            )
        ]
        next_node_id=1

        incumbent_x=None
        incumbent_value=-np.inf
        best_upper_bound=None

        num_nodes=0
        num_lp_solved=0
        num_pruned_infeasible=0
        num_pruned_bound=0
        num_pruned_optimality=0
        num_integer_solutions=0
        logs=[]

        def add_log(text: str) -> None:
            logs.append(text)
            if self.verbose:
                print(text)

        def make_result(status: str) -> MILPResult:
            elapsed=perf_counter()-start
            objective=None if incumbent_x is None else problem.recover_objective_value(incumbent_value)
            if best_upper_bound is None:
                bound=None
            else:
                bound=problem.recover_objective_value(best_upper_bound)
            if incumbent_x is None:
                x_continuous=None
                y_integer=None
            else:
                x_continuous,y_integer=problem.split_solution(incumbent_x)
            return MILPResult(
                status=status,
                objective_value=objective,
                x=None if incumbent_x is None else incumbent_x.copy(),
                x_continuous=x_continuous,
                y_integer=y_integer,
                internal_objective_value=None if incumbent_x is None else float(incumbent_value),
                best_bound=bound,
                num_nodes=num_nodes,
                num_lp_solved=num_lp_solved,
                num_pruned_infeasible=num_pruned_infeasible,
                num_pruned_bound=num_pruned_bound,
                num_pruned_optimality=num_pruned_optimality,
                num_integer_solutions=num_integer_solutions,
                runtime_sec=elapsed,
                log=logs,
            )

        while pending_nodes:
            if num_nodes>=self.max_nodes:
                add_log(f"node_limit reached after {num_nodes} nodes")
                return make_result("node_limit")

            node=pending_nodes.pop()
            num_nodes+=1

            lp=solve_lp_relaxation(problem,node.lb,node.ub,tol)
            num_lp_solved+=1

            if lp.status!="optimal":
                num_pruned_infeasible+=1
                add_log(
                    f"node={node.node_id} depth={node.depth} LP={lp.status} "
                    f"prune=infeasibility"
                )
                continue

            node_upper_bound=float(lp.objective_value)
            node.lp_bound=node_upper_bound
            if best_upper_bound is None or node_upper_bound>best_upper_bound:
                best_upper_bound=node_upper_bound

            add_log(
                f"node={node.node_id} depth={node.depth} LP=optimal "
                f"upper_bound={node_upper_bound:.6g} x={format_vector(lp.x)}"
            )

            if incumbent_x is not None and node_upper_bound<=incumbent_value+tol:
                num_pruned_bound+=1
                add_log(
                    f"node={node.node_id} depth={node.depth} prune=bound "
                    f"upper_bound={node_upper_bound:.6g} incumbent={incumbent_value:.6g}"
                )
                continue

            if is_integral_solution(lp.x,problem.var_types,tol):
                candidate=lp.x.copy()
                for index in problem.integer_indices:
                    candidate[index]=round(candidate[index])

                if check_feasibility(problem,candidate,node.lb,node.ub,tol):
                    candidate_value=float(problem.internal_c@candidate)
                    num_pruned_optimality+=1
                    if candidate_value>incumbent_value+tol:
                        incumbent_x=candidate
                        incumbent_value=candidate_value
                        num_integer_solutions+=1
                        add_log(
                            f"node={node.node_id} depth={node.depth} incumbent updated "
                            f"internal_obj={candidate_value:.6g} "
                            f"obj={problem.recover_objective_value(candidate_value):.6g} "
                            f"x={format_vector(candidate)}"
                        )
                    else:
                        add_log(
                            f"node={node.node_id} depth={node.depth} prune=optimality "
                            f"integer LP solution not better"
                        )
                else:
                    num_pruned_infeasible+=1
                    add_log(
                        f"node={node.node_id} depth={node.depth} integer LP point "
                        f"failed feasibility check after rounding"
                    )
                continue

            branch_var=choose_branch_variable(lp.x,problem.var_types,tol,self.branching_rule)
            if branch_var is None:
                add_log(f"node={node.node_id} no fractional integer variable found")
                continue

            value=float(lp.x[branch_var])
            floor_value=math.floor(value)
            ceil_value=math.ceil(value)

            left_lb=node.lb.copy()
            left_ub=node.ub.copy()
            left_ub[branch_var]=min(left_ub[branch_var],floor_value)

            right_lb=node.lb.copy()
            right_ub=node.ub.copy()
            right_lb[branch_var]=max(right_lb[branch_var],ceil_value)

            add_log(
                f"node={node.node_id} depth={node.depth} no_pruning_branch "
                f"z[{branch_var}]={value:.6g}: left <= {floor_value}, right >= {ceil_value}"
            )

            if np.all(right_lb<=right_ub+tol):
                pending_nodes.append(
                    BBNode(
                        node_id=next_node_id,
                        depth=node.depth+1,
                        lb=right_lb,
                        ub=right_ub,
                        parent_id=node.node_id,
                        branch_var=branch_var,
                        branch_value=value,
                        branch_direction=">=",
                    )
                )
                next_node_id+=1
            else:
                add_log(f"node={node.node_id} right child skipped because lb>ub")

            if np.all(left_lb<=left_ub+tol):
                pending_nodes.append(
                    BBNode(
                        node_id=next_node_id,
                        depth=node.depth+1,
                        lb=left_lb,
                        ub=left_ub,
                        parent_id=node.node_id,
                        branch_var=branch_var,
                        branch_value=value,
                        branch_direction="<=",
                    )
                )
                next_node_id+=1
            else:
                add_log(f"node={node.node_id} left child skipped because lb>ub")

        if incumbent_x is None:
            return make_result("infeasible")
        return make_result("optimal")
