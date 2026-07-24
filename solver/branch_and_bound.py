from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .lp_backends import get_lp_relaxation_solver
from .result import MILPResult
from .search_strategy import BestBoundNodePool, DepthFirstNodePool
from .branching import BranchingContext, policy_from_rule


@dataclass
class BBNode:
    node_id: int
    depth: int
    lb: np.ndarray
    ub: np.ndarray
    parent_id: int | None = None
    branch_var: int | None = None
    branch_var_group: str | None = None
    branch_value: float | None = None
    branch_direction: str | None = None
    lp_bound: float | None = None


def is_integer_value(value: float,tol: float = 1e-8) -> bool:
    return abs(float(value)-round(float(value)))<=tol


def is_binary_solution(x,binary_indices,tol: float = 1e-8) -> bool:
    for i in binary_indices:
        value=float(x[i])
        if abs(value-round(value))>tol:
            return False
    return True


def fractional_binary_candidates(x,binary_indices,tol: float = 1e-8) -> tuple[int, ...]:
    candidates=[]
    for i in binary_indices:
        if not is_integer_value(float(x[i]),tol):
            candidates.append(int(i))
    return tuple(sorted(candidates))


def choose_binary_branch_variable(x,binary_indices,tol: float = 1e-8,rule: str = "most_fractional") -> int | None:
    candidate_indices=fractional_binary_candidates(x,binary_indices,tol)
    if not candidate_indices:
        return None

    class _LP:
        pass

    lp=_LP()
    lp.x=x
    context=BranchingContext(
        problem=None,
        node_id=-1,
        node_depth=-1,
        node_lb=np.array([]),
        node_ub=np.array([]),
        lp_result=lp,
        candidate_indices=candidate_indices,
        incumbent_internal_value=None,
        current_node_internal_bound=None,
        tolerance=tol,
    )
    return policy_from_rule(rule).select_variable(context)


def check_feasibility(problem,x,lb=None,ub=None,tol: float = 1e-8) -> bool:
    x=np.asarray(x,dtype=float)
    lb=np.asarray(problem.lb if lb is None else lb,dtype=float)
    ub=np.asarray(problem.ub if ub is None else ub,dtype=float)
    if np.any(x<lb-tol) or np.any(x>ub+tol):
        return False
    if np.any(problem.G@x>problem.h+tol):
        return False
    return is_binary_solution(x,problem.binary_indices,tol)


def format_vector(x,digits: int = 6) -> str:
    return np.array2string(
        np.asarray(x,dtype=float),
        precision=digits,
        suppress_small=True,
        separator=", ",
    )


def find_initial_incumbent(
    problem,
    lp_solver,
    tol: float = 1e-8,
    max_lp_candidates: int | None = None,
    use_matrix_presolve: bool = True,
    matrix_presolve_options=None,
    root_lp=None,
    max_lp_iterations: int | None = None,
):
    """Try simple binary-y assignments before the best-bound search starts."""
    stats={
        "num_lp_solved": 0,
        "num_candidates_checked": 0,
        "num_fixed_vars": 0,
        "num_removed_rows": 0,
        "num_tightened_bounds": 0,
        "num_free_vars": 0,
        "num_simplex_iterations": 0,
        "lp_runtime_sec": 0.0,
        "logs": [],
    }
    binary_indices=problem.binary_indices
    if not binary_indices:
        return None,-np.inf,stats

    y_size=len(binary_indices)
    candidates=[
        ("all_on",np.ones(y_size,dtype=float)),
        ("all_off",np.zeros(y_size,dtype=float)),
    ]

    if root_lp is None:
        lp_start=perf_counter()
        root_lp=lp_solver(
            problem,
            problem.lb,
            problem.ub,
            tol,
            max_lp_candidates,
            use_matrix_presolve,
            matrix_presolve_options,
            max_lp_iterations,
        )
        stats["lp_runtime_sec"]+=perf_counter()-lp_start
        stats["num_lp_solved"]+=1
        stats["num_candidates_checked"]+=root_lp.num_candidates_checked
        stats["num_simplex_iterations"]+=root_lp.num_iterations
        stats["num_fixed_vars"]+=root_lp.num_fixed_vars
        stats["num_removed_rows"]+=root_lp.num_removed_rows
        stats["num_tightened_bounds"]+=root_lp.num_tightened_bounds
        stats["num_free_vars"]+=root_lp.num_free_vars
    if root_lp.status=="optimal" and root_lp.x is not None:
        rounded=np.array([1.0 if root_lp.x[i]>=0.5 else 0.0 for i in binary_indices])
        candidates.append(("rounded_root_lp",rounded))

    best_x=None
    best_value=-np.inf
    seen=set()

    for label,y_values in candidates:
        key=tuple(float(v) for v in y_values)
        if key in seen:
            continue
        seen.add(key)

        node_lb=problem.lb.copy()
        node_ub=problem.ub.copy()
        for index,value in zip(binary_indices,y_values):
            node_lb[index]=value
            node_ub[index]=value

        lp_start=perf_counter()
        lp=lp_solver(
            problem,
            node_lb,
            node_ub,
            tol,
            max_lp_candidates,
            use_matrix_presolve,
            matrix_presolve_options,
            max_lp_iterations,
        )
        stats["lp_runtime_sec"]+=perf_counter()-lp_start
        stats["num_lp_solved"]+=1
        stats["num_candidates_checked"]+=lp.num_candidates_checked
        stats["num_simplex_iterations"]+=lp.num_iterations
        stats["num_fixed_vars"]+=lp.num_fixed_vars
        stats["num_removed_rows"]+=lp.num_removed_rows
        stats["num_tightened_bounds"]+=lp.num_tightened_bounds
        stats["num_free_vars"]+=lp.num_free_vars

        if lp.status!="optimal" or lp.x is None:
            stats["logs"].append(f"initial_incumbent {label}: LP={lp.status}")
            continue

        candidate=lp.x.copy()
        for index,value in zip(binary_indices,y_values):
            candidate[index]=value

        if not check_feasibility(problem,candidate,node_lb,node_ub,tol):
            stats["logs"].append(f"initial_incumbent {label}: infeasible after check")
            continue

        candidate_value=float(problem.internal_c@candidate)
        stats["logs"].append(
            f"initial_incumbent {label}: feasible internal_obj={candidate_value:.6g} "
            f"obj={problem.recover_objective_value(candidate_value):.6g}"
        )
        if candidate_value>best_value+tol:
            best_x=candidate
            best_value=candidate_value

    return best_x,best_value,stats


class BranchAndBoundSolver:
    def __init__(
        self,
        problem,
        tol: float = 1e-8,
        max_nodes: int = 10000,
        branching_rule: str = "most_fractional",
        branching_policy=None,
        lp_backend: str = "active_set",
        max_lp_candidates: int | None = None,
        use_matrix_presolve: bool = True,
        matrix_presolve_options=None,
        node_selection: str = "best_bound",
        time_limit_sec: float | None = None,
        verbose: bool = False,
        max_lp_iterations: int | None = None,
    ):
        if branching_policy is not None and branching_rule!="most_fractional":
            raise ValueError("branching_policy cannot be combined with a non-default branching_rule")
        self.problem=problem
        self.tol=tol
        self.max_nodes=max_nodes
        self.branching_rule=branching_rule
        self.branching_policy=branching_policy if branching_policy is not None else policy_from_rule(branching_rule)
        self.lp_backend=str(lp_backend).lower()
        self.lp_solver=get_lp_relaxation_solver(self.lp_backend)
        self.max_lp_candidates=max_lp_candidates
        self.max_lp_iterations=max_lp_iterations
        self.use_matrix_presolve=use_matrix_presolve
        self.matrix_presolve_options=matrix_presolve_options
        self.node_selection=str(node_selection)
        if self.node_selection not in ("best_bound","dfs"):
            raise ValueError('node_selection must be "best_bound" or "dfs"')
        self.time_limit_sec=None if time_limit_sec is None else float(time_limit_sec)
        self.verbose=verbose

    def solve(self) -> MILPResult:
        start=perf_counter()
        problem=self.problem
        tol=self.tol
        node_pool=BestBoundNodePool() if self.node_selection=="best_bound" else DepthFirstNodePool()

        next_node_id=1
        incumbent_x=None
        incumbent_value=-np.inf
        num_nodes=0
        num_lp_solved=0
        num_pruned_infeasible=0
        num_pruned_bound=0
        num_pruned_optimality=0
        num_integer_solutions=0
        num_lp_candidates_checked=0
        num_simplex_iterations=0
        lp_runtime_sec=0.0
        num_fixed_vars_eliminated=0
        num_removed_rows=0
        num_tightened_bounds=0
        num_free_vars_total=0
        num_heuristic_lp_solved=0
        initial_incumbent_found=False
        logs=[]

        def add_log(text: str) -> None:
            logs.append(text)
            if self.verbose:
                print(text)

        def time_limit_reached() -> bool:
            return self.time_limit_sec is not None and perf_counter()-start>=self.time_limit_sec

        def solve_node_lp(node: BBNode):
            nonlocal num_nodes,num_lp_solved,num_lp_candidates_checked
            nonlocal num_simplex_iterations,lp_runtime_sec
            nonlocal num_fixed_vars_eliminated,num_removed_rows,num_tightened_bounds
            nonlocal num_free_vars_total
            if num_nodes>=self.max_nodes:
                return None
            lp_start=perf_counter()
            lp=self.lp_solver(
                problem,
                node.lb,
                node.ub,
                tol,
                self.max_lp_candidates,
                self.use_matrix_presolve,
                self.matrix_presolve_options,
                self.max_lp_iterations,
            )
            lp_runtime_sec+=perf_counter()-lp_start
            num_nodes+=1
            num_lp_solved+=1
            num_lp_candidates_checked+=lp.num_candidates_checked
            num_simplex_iterations+=lp.num_iterations
            num_fixed_vars_eliminated+=lp.num_fixed_vars
            num_removed_rows+=lp.num_removed_rows
            num_tightened_bounds+=lp.num_tightened_bounds
            num_free_vars_total+=lp.num_free_vars
            return lp

        def recover_bound(internal_bound: float | None) -> float | None:
            if internal_bound is None:
                return None
            return problem.recover_objective_value(float(internal_bound))

        def current_global_bound(status: str) -> float | None:
            if status=="optimal" and incumbent_x is not None:
                return problem.recover_objective_value(float(incumbent_value))
            return recover_bound(node_pool.best_bound())

        def current_relative_gap(status: str,global_bound: float | None) -> float | None:
            if incumbent_x is None or global_bound is None:
                return None
            incumbent_obj=problem.recover_objective_value(float(incumbent_value))
            if status=="optimal" and len(node_pool)==0:
                return 0.0
            if problem.sense=="min":
                raw_gap=incumbent_obj-float(global_bound)
            else:
                raw_gap=float(global_bound)-incumbent_obj
            return max(0.0,float(raw_gap))/max(1.0,abs(float(incumbent_obj)))

        def make_result(status: str) -> MILPResult:
            elapsed=perf_counter()-start
            objective=None if incumbent_x is None else problem.recover_objective_value(incumbent_value)
            global_bound=current_global_bound(status)
            relative_gap=current_relative_gap(status,global_bound)
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
                best_bound=global_bound,
                global_bound=global_bound,
                relative_gap=relative_gap,
                num_nodes=num_nodes,
                num_lp_solved=num_lp_solved,
                num_pruned_infeasible=num_pruned_infeasible,
                num_pruned_bound=num_pruned_bound,
                num_pruned_optimality=num_pruned_optimality,
                num_integer_solutions=num_integer_solutions,
                num_lp_candidates_checked=num_lp_candidates_checked,
                num_fixed_vars_eliminated=num_fixed_vars_eliminated,
                num_removed_rows=num_removed_rows,
                num_tightened_bounds=num_tightened_bounds,
                num_free_vars_total=num_free_vars_total,
                num_heuristic_lp_solved=num_heuristic_lp_solved,
                initial_incumbent_found=initial_incumbent_found,
                runtime_sec=elapsed,
                log=logs,
                lp_backend=self.lp_backend,
                num_simplex_iterations=num_simplex_iterations,
                lp_runtime_sec=lp_runtime_sec,
            )

        def classify_solved_node(node: BBNode,lp) -> str:
            nonlocal incumbent_x,incumbent_value,num_pruned_infeasible
            nonlocal num_pruned_bound,num_pruned_optimality,num_integer_solutions

            if lp is None:
                return "node_limit"

            if lp.status=="candidate_limit":
                add_log(
                    f"node={node.node_id} depth={node.depth} LP=candidate_limit "
                    f"checked={lp.num_candidates_checked}"
                )
                return "candidate_limit"

            if lp.status=="infeasible":
                num_pruned_infeasible+=1
                add_log(
                    f"node={node.node_id} depth={node.depth} LP={lp.status} "
                    f"prune=infeasibility"
                )
                return "closed"

            if lp.status!="optimal":
                status=lp.status if lp.status in {
                    "iteration_limit",
                    "numerical_error",
                    "unsupported",
                    "unbounded",
                    "lp_error",
                } else "lp_error"
                add_log(
                    f"node={node.node_id} depth={node.depth} LP={lp.status} "
                    f"terminate={status} message={lp.message}"
                )
                return status

            node_bound=float(lp.objective_value)
            node.lp_bound=node_bound
            add_log(
                f"node={node.node_id} depth={node.depth} LP=optimal "
                f"bound={node_bound:.6g} free_vars={lp.num_free_vars} "
                f"fixed_vars={lp.num_fixed_vars} removed_rows={lp.num_removed_rows} "
                f"tightened_bounds={lp.num_tightened_bounds} candidates={lp.num_candidates_checked} "
                f"iterations={lp.num_iterations} backend={lp.backend} "
                f"x={format_vector(lp.x)}"
            )

            if incumbent_x is not None and node_bound<=incumbent_value+tol:
                num_pruned_bound+=1
                add_log(
                    f"node={node.node_id} depth={node.depth} prune=bound "
                    f"bound={node_bound:.6g} incumbent={incumbent_value:.6g}"
                )
                return "closed"

            if is_binary_solution(lp.x,problem.binary_indices,tol):
                candidate=lp.x.copy()
                for index in problem.binary_indices:
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
                return "closed"

            node_pool.push(node,lp,node_bound)
            return "open"

        root_node=BBNode(
            node_id=0,
            depth=0,
            lb=problem.lb.copy(),
            ub=problem.ub.copy(),
            parent_id=None,
            branch_direction="root",
        )
        root_lp=solve_node_lp(root_node)

        incumbent_x,incumbent_value,heuristic_stats=find_initial_incumbent(
            problem,
            self.lp_solver,
            tol,
            self.max_lp_candidates,
            self.use_matrix_presolve,
            self.matrix_presolve_options,
            root_lp=root_lp,
            max_lp_iterations=self.max_lp_iterations,
        )
        num_lp_solved+=int(heuristic_stats["num_lp_solved"])
        num_lp_candidates_checked+=int(heuristic_stats["num_candidates_checked"])
        num_simplex_iterations+=int(heuristic_stats["num_simplex_iterations"])
        lp_runtime_sec+=float(heuristic_stats["lp_runtime_sec"])
        num_fixed_vars_eliminated+=int(heuristic_stats["num_fixed_vars"])
        num_removed_rows+=int(heuristic_stats["num_removed_rows"])
        num_tightened_bounds+=int(heuristic_stats["num_tightened_bounds"])
        num_free_vars_total+=int(heuristic_stats["num_free_vars"])
        num_heuristic_lp_solved=int(heuristic_stats["num_lp_solved"])
        initial_incumbent_found=incumbent_x is not None
        num_integer_solutions=1 if incumbent_x is not None else 0
        logs.extend(heuristic_stats["logs"])

        root_status=classify_solved_node(root_node,root_lp)
        terminal_lp_statuses={
            "candidate_limit",
            "node_limit",
            "iteration_limit",
            "numerical_error",
            "unsupported",
            "unbounded",
            "lp_error",
        }
        if root_status in terminal_lp_statuses:
            return make_result(root_status)

        while len(node_pool)>0:
            if time_limit_reached():
                add_log(f"time_limit reached after {perf_counter()-start:.6g} sec")
                return make_result("time_limit")
            if num_nodes>=self.max_nodes:
                add_log(f"node_limit reached after {num_nodes} LP-solved nodes")
                return make_result("node_limit")

            node,lp,node_bound=node_pool.pop()
            if incumbent_x is not None and node_bound<=incumbent_value+tol:
                num_pruned_bound+=1
                add_log(
                    f"node={node.node_id} depth={node.depth} prune=bound_after_queue "
                    f"bound={node_bound:.6g} incumbent={incumbent_value:.6g}"
                )
                continue

            candidate_indices=fractional_binary_candidates(lp.x,problem.binary_indices,tol)
            if not candidate_indices:
                add_log(f"node={node.node_id} no fractional binary variable found")
                continue
            context=BranchingContext(
                problem=problem,
                node_id=node.node_id,
                node_depth=node.depth,
                node_lb=node.lb.copy(),
                node_ub=node.ub.copy(),
                lp_result=lp,
                candidate_indices=candidate_indices,
                incumbent_internal_value=None if incumbent_x is None else float(incumbent_value),
                current_node_internal_bound=float(node_bound),
                tolerance=tol,
            )
            branch_var=self.branching_policy.select_variable(context)
            if branch_var not in candidate_indices:
                raise ValueError("branching policy returned an index outside candidate_indices")
            if branch_var is None:
                add_log(f"node={node.node_id} no fractional binary variable found")
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
                f"branch_var={branch_var} branch_var_group=y value={value:.6g} "
                f"left_ub={floor_value} right_lb={ceil_value}"
            )

            for child_lb,child_ub,direction in (
                (right_lb,right_ub,">="),
                (left_lb,left_ub,"<="),
            ):
                if time_limit_reached():
                    add_log(f"time_limit reached after {perf_counter()-start:.6g} sec")
                    return make_result("time_limit")
                branch_value_for_policy=ceil_value if direction==">=" else floor_value
                if np.any(child_lb>child_ub+tol):
                    add_log(f"node={node.node_id} {direction} child skipped because lb>ub")
                    if hasattr(self.branching_policy,"observe_branch_result"):
                        self.branching_policy.observe_branch_result(
                            context,
                            int(branch_var),
                            int(branch_value_for_policy),
                            "infeasible",
                            None,
                        )
                    continue
                child=BBNode(
                    node_id=next_node_id,
                    depth=node.depth+1,
                    lb=child_lb,
                    ub=child_ub,
                    parent_id=node.node_id,
                    branch_var=branch_var,
                    branch_var_group="y",
                    branch_value=value,
                    branch_direction=direction,
                )
                next_node_id+=1
                child_lp=solve_node_lp(child)
                if hasattr(self.branching_policy,"observe_branch_result"):
                    self.branching_policy.observe_branch_result(
                        context,
                        int(branch_var),
                        int(branch_value_for_policy),
                        "node_limit" if child_lp is None else str(child_lp.status),
                        None if child_lp is None else child_lp.objective_value,
                    )
                child_status=classify_solved_node(child,child_lp)
                if child_status in terminal_lp_statuses:
                    return make_result(child_status)

        if incumbent_x is None:
            return make_result("infeasible")
        return make_result("optimal")


def solve_milp(
    problem,
    tol: float = 1e-8,
    max_nodes: int = 10000,
    branching_rule: str = "most_fractional",
    branching_policy=None,
    lp_backend: str = "active_set",
    max_lp_candidates: int | None = None,
    use_matrix_presolve: bool = True,
    matrix_presolve_options=None,
    node_selection: str = "best_bound",
    time_limit_sec: float | None = None,
    verbose: bool = False,
    max_lp_iterations: int | None = None,
):
    """Solve a MILPProblem with best-bound Branch-and-Bound."""
    solver=BranchAndBoundSolver(
        problem=problem,
        tol=tol,
        max_nodes=max_nodes,
        branching_rule=branching_rule,
        branching_policy=branching_policy,
        lp_backend=lp_backend,
        max_lp_candidates=max_lp_candidates,
        max_lp_iterations=max_lp_iterations,
        use_matrix_presolve=use_matrix_presolve,
        matrix_presolve_options=matrix_presolve_options,
        node_selection=node_selection,
        time_limit_sec=time_limit_sec,
        verbose=verbose,
    )
    return solver.solve()
