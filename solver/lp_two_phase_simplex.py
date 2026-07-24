from __future__ import annotations

from time import perf_counter

import numpy as np

from .lp_standard_form import (
    GeneralLPStandardForm,
    InfeasibleBoundsError,
    UnsupportedStandardFormError,
    standardize_general_lp,
)
from .lp_tableau_simplex import (
    ITERATION_LIMIT,
    NUMERICAL_ERROR,
    OPTIMAL,
    UNBOUNDED,
    TableauSimplexSolver,
)
from .matrix_presolve import presolve_node_matrix, reconstruct_solution
from .result import LPResult


INFEASIBLE = "infeasible"
UNSUPPORTED = "unsupported"


class TwoPhaseTableauSimplexSolver(TableauSimplexSolver):
    """Primal two-phase tableau simplex for bounded-below general LPs."""

    def solve(
        self,
        c,
        A,
        b,
        constraint_senses=None,
        lb=None,
        ub=None,
        sense: str = "max",
    ) -> LPResult:
        start = perf_counter()
        try:
            standard = standardize_general_lp(
                c=c,
                A=A,
                b=b,
                constraint_senses=constraint_senses,
                lb=lb,
                ub=ub,
                sense=sense,
                tolerance=self.tolerance,
            )
        except UnsupportedStandardFormError as exc:
            return self._empty_result(UNSUPPORTED, str(exc), start)
        except InfeasibleBoundsError as exc:
            return self._empty_result(INFEASIBLE, str(exc), start)

        tableau = self._build_equality_tableau(standard.A_eq, standard.b_eq)
        basis = list(standard.initial_basis)
        column_names = list(standard.column_names)
        artificial = set(standard.artificial_indices)
        iteration_log: list[str] = []
        phase_one_iterations = 0
        phase_two_iterations = 0

        if self.verbose:
            print("phase | iteration | entering | leaving | objective | basis")

        if artificial:
            self._set_objective_row(
                tableau,
                standard.phase_one_objective,
                basis,
                objective_constant=0.0,
            )
            phase_status, iterations, message = self._run_phase(
                tableau=tableau,
                basis=basis,
                column_names=column_names,
                phase="I",
                max_iterations=self.max_iterations,
                objective_display_multiplier=1.0,
                iteration_log=iteration_log,
            )
            phase_one_iterations += iterations

            if phase_status != OPTIMAL:
                if phase_status == UNBOUNDED:
                    phase_status = NUMERICAL_ERROR
                    message = "Phase I auxiliary problem was unexpectedly unbounded"
                return self._terminal_result(
                    status=phase_status,
                    message=message,
                    standard=standard,
                    tableau=tableau,
                    basis=basis,
                    start=start,
                    iteration_log=iteration_log,
                    phase_one_iterations=phase_one_iterations,
                    phase_two_iterations=0,
                    include_solution=False,
                )

            tableau_values = self._extract_tableau_values(tableau, basis)
            artificial_sum = float(
                sum(max(0.0, tableau_values[index]) for index in artificial)
            )
            phase_one_tolerance = self._comparison_tolerance(
                artificial_sum,
                float(np.max(np.abs(standard.b_eq), initial=0.0)),
            )
            if artificial_sum > phase_one_tolerance:
                return self._terminal_result(
                    status=INFEASIBLE,
                    message=(
                        "Phase I optimum retains positive artificial-variable "
                        f"sum {artificial_sum:.10g}"
                    ),
                    standard=standard,
                    tableau=tableau,
                    basis=basis,
                    start=start,
                    iteration_log=iteration_log,
                    phase_one_iterations=phase_one_iterations,
                    phase_two_iterations=0,
                    include_solution=False,
                )

            cleanup_budget = self.max_iterations - phase_one_iterations
            (
                cleanup_status,
                cleanup_iterations,
                cleanup_message,
                tableau,
            ) = self._remove_artificial_basics(
                tableau=tableau,
                basis=basis,
                artificial_indices=artificial,
                column_names=column_names,
                max_iterations=cleanup_budget,
                iteration_log=iteration_log,
            )
            phase_one_iterations += cleanup_iterations
            if cleanup_status != OPTIMAL:
                return self._terminal_result(
                    status=cleanup_status,
                    message=cleanup_message,
                    standard=standard,
                    tableau=tableau,
                    basis=basis,
                    start=start,
                    iteration_log=iteration_log,
                    phase_one_iterations=phase_one_iterations,
                    phase_two_iterations=0,
                    include_solution=False,
                )

        try:
            (
                tableau,
                basis,
                column_names,
                phase_two_objective,
            ) = self._drop_artificial_columns(
                tableau=tableau,
                basis=basis,
                column_names=column_names,
                phase_two_objective=standard.phase_two_objective,
                artificial_indices=artificial,
            )
        except FloatingPointError as exc:
            return self._terminal_result(
                status=NUMERICAL_ERROR,
                message=str(exc),
                standard=standard,
                tableau=tableau,
                basis=basis,
                start=start,
                iteration_log=iteration_log,
                phase_one_iterations=phase_one_iterations,
                phase_two_iterations=0,
                include_solution=False,
            )

        # Phase I's objective row is discarded here. The original internal
        # objective is rebuilt and canonicalized against the feasible basis.
        self._set_objective_row(
            tableau,
            phase_two_objective,
            basis,
            objective_constant=standard.objective_constant_internal,
        )
        phase_two_budget = self.max_iterations - phase_one_iterations
        phase_status, phase_two_iterations, message = self._run_phase(
            tableau=tableau,
            basis=basis,
            column_names=column_names,
            phase="II",
            max_iterations=phase_two_budget,
            objective_display_multiplier=standard.objective_sign,
            iteration_log=iteration_log,
        )

        return self._terminal_result(
            status=phase_status,
            message=message,
            standard=standard,
            tableau=tableau,
            basis=basis,
            start=start,
            iteration_log=iteration_log,
            phase_one_iterations=phase_one_iterations,
            phase_two_iterations=phase_two_iterations,
            include_solution=phase_status in {OPTIMAL, ITERATION_LIMIT},
        )

    def _build_equality_tableau(
        self,
        A_eq: np.ndarray,
        b_eq: np.ndarray,
    ) -> np.ndarray:
        tableau = np.zeros((A_eq.shape[0] + 1, A_eq.shape[1] + 1), dtype=float)
        tableau[:-1, :-1] = A_eq
        tableau[:-1, -1] = b_eq
        return tableau

    def _set_objective_row(
        self,
        tableau: np.ndarray,
        objective: np.ndarray,
        basis: list[int],
        objective_constant: float,
    ) -> None:
        if objective.shape != (tableau.shape[1] - 1,):
            raise ValueError("objective dimension does not match tableau columns")
        tableau[-1, :] = 0.0
        tableau[-1, :-1] = -objective
        tableau[-1, -1] = objective_constant

        # Remove every basic-variable coefficient from the objective row.
        # This yields the reduced costs for the current basis, not a new basis.
        for row, basic_variable in enumerate(basis):
            factor = float(tableau[-1, basic_variable])
            if abs(factor) > self.tolerance:
                tableau[-1, :] -= factor * tableau[row, :]
        self._clean_small_values(tableau)

    def _run_phase(
        self,
        *,
        tableau: np.ndarray,
        basis: list[int],
        column_names: list[str],
        phase: str,
        max_iterations: int,
        objective_display_multiplier: float,
        iteration_log: list[str],
    ) -> tuple[str, int, str]:
        iterations = 0
        while True:
            if not np.all(np.isfinite(tableau)):
                return NUMERICAL_ERROR, iterations, f"Phase {phase} tableau is non-finite"

            entering = self.choose_entering_variable(tableau, basis)
            if entering is None:
                return OPTIMAL, iterations, f"Phase {phase} reached an optimal tableau"

            try:
                leaving_row = self.choose_leaving_row(tableau, basis, entering)
            except FloatingPointError as exc:
                return NUMERICAL_ERROR, iterations, f"Phase {phase}: {exc}"
            if leaving_row is None:
                return (
                    UNBOUNDED,
                    iterations,
                    f"Phase {phase} entering variable {entering} has no leaving row",
                )
            if iterations >= max_iterations:
                return (
                    ITERATION_LIMIT,
                    iterations,
                    f"iteration limit reached during Phase {phase}",
                )

            leaving_variable = basis[leaving_row]
            try:
                self.pivot(tableau, leaving_row, entering)
            except FloatingPointError as exc:
                return NUMERICAL_ERROR, iterations, f"Phase {phase}: {exc}"
            basis[leaving_row] = entering
            iterations += 1
            line = self._phase_iteration_line(
                phase=phase,
                iteration=iterations,
                entering=entering,
                leaving=leaving_variable,
                tableau=tableau,
                basis=basis,
                column_names=column_names,
                objective_display_multiplier=objective_display_multiplier,
            )
            iteration_log.append(line)
            if self.verbose:
                print(line)

    def _remove_artificial_basics(
        self,
        *,
        tableau: np.ndarray,
        basis: list[int],
        artificial_indices: set[int],
        column_names: list[str],
        max_iterations: int,
        iteration_log: list[str],
    ) -> tuple[str, int, str, np.ndarray]:
        iterations = 0
        row = 0
        while row < len(basis):
            if basis[row] not in artificial_indices:
                row += 1
                continue

            rhs = float(tableau[row, -1])
            if abs(rhs) > self._comparison_tolerance(rhs):
                return (
                    NUMERICAL_ERROR,
                    iterations,
                    "positive artificial basic value remained after feasible Phase I",
                    tableau,
                )

            current_basis = set(basis)
            candidates = [
                column
                for column in range(tableau.shape[1] - 1)
                if column not in artificial_indices
                and column not in current_basis
                and abs(tableau[row, column]) > self.tolerance
            ]
            if candidates:
                if iterations >= max_iterations:
                    return (
                        ITERATION_LIMIT,
                        iterations,
                        "iteration limit reached while removing artificial basics",
                        tableau,
                    )
                entering = min(candidates)
                leaving = basis[row]
                try:
                    self.pivot(tableau, row, entering)
                except FloatingPointError as exc:
                    return NUMERICAL_ERROR, iterations, str(exc), tableau
                basis[row] = entering
                iterations += 1
                line = self._phase_iteration_line(
                    phase="I-cleanup",
                    iteration=iterations,
                    entering=entering,
                    leaving=leaving,
                    tableau=tableau,
                    basis=basis,
                    column_names=column_names,
                    objective_display_multiplier=1.0,
                )
                iteration_log.append(line)
                if self.verbose:
                    print(line)
                row += 1
                continue

            # With RHS zero and no non-artificial pivot column, removing the
            # artificial variable leaves 0 = 0. The row is redundant.
            tableau = np.delete(tableau, row, axis=0)
            basis.pop(row)

        return OPTIMAL, iterations, "artificial basic variables removed", tableau

    def _drop_artificial_columns(
        self,
        *,
        tableau: np.ndarray,
        basis: list[int],
        column_names: list[str],
        phase_two_objective: np.ndarray,
        artificial_indices: set[int],
    ) -> tuple[np.ndarray, list[int], list[str], np.ndarray]:
        if any(index in artificial_indices for index in basis):
            raise FloatingPointError("cannot drop an artificial column still in the basis")

        keep = [
            column
            for column in range(tableau.shape[1] - 1)
            if column not in artificial_indices
        ]
        old_to_new = {old: new for new, old in enumerate(keep)}
        remapped_basis = [old_to_new[index] for index in basis]
        reduced_tableau = np.column_stack([tableau[:, keep], tableau[:, -1]])
        reduced_names = [column_names[index] for index in keep]
        reduced_objective = phase_two_objective[np.asarray(keep, dtype=int)]
        return reduced_tableau, remapped_basis, reduced_names, reduced_objective

    def _extract_tableau_values(
        self,
        tableau: np.ndarray,
        basis: list[int],
    ) -> np.ndarray:
        values = np.zeros(tableau.shape[1] - 1, dtype=float)
        for row, basic_variable in enumerate(basis):
            values[basic_variable] = tableau[row, -1]
        values[np.abs(values) <= self.tolerance] = 0.0
        return values

    def _terminal_result(
        self,
        *,
        status: str,
        message: str,
        standard: GeneralLPStandardForm,
        tableau: np.ndarray,
        basis: list[int],
        start: float,
        iteration_log: list[str],
        phase_one_iterations: int,
        phase_two_iterations: int,
        include_solution: bool,
    ) -> LPResult:
        final_status = status
        final_message = message
        x = None
        objective = None

        if include_solution:
            tableau_values = self._extract_tableau_values(tableau, basis)
            candidate = standard.recover_original_solution(tableau_values)
            if self._is_original_solution_feasible(standard, candidate):
                original_objective = float(standard.original_c @ candidate)
                internal_objective = float(tableau[-1, -1])
                expected_internal = float(
                    standard.objective_sign * original_objective
                )
                if abs(internal_objective - expected_internal) <= self._comparison_tolerance(
                    internal_objective,
                    expected_internal,
                ):
                    x = candidate
                    objective = original_objective
                else:
                    final_status = NUMERICAL_ERROR
                    final_message = (
                        "Phase II objective disagrees with recovered original solution"
                    )
            else:
                final_status = NUMERICAL_ERROR
                final_message = "final basis does not recover a feasible original solution"

        runtime = perf_counter() - start
        total_iterations = phase_one_iterations + phase_two_iterations
        if self.verbose:
            objective_text = "-" if objective is None else f"{objective:.10g}"
            print(
                f"status={final_status} phase_I={phase_one_iterations} "
                f"phase_II={phase_two_iterations} objective={objective_text} "
                f"basis={list(basis)}"
            )

        return LPResult(
            status=final_status,
            objective_value=objective,
            x=x,
            num_candidates_checked=0,
            message=final_message,
            num_free_vars=standard.num_transformed_variables,
            num_fixed_vars=int(standard.recovery.fixed_indices.size),
            backend="tableau_simplex_two_phase",
            num_iterations=total_iterations,
            basis_indices=tuple(int(index) for index in basis),
            runtime_sec=runtime,
            iteration_log=tuple(iteration_log),
            phase_one_iterations=phase_one_iterations,
            phase_two_iterations=phase_two_iterations,
        )

    def _empty_result(self, status: str, message: str, start: float) -> LPResult:
        return LPResult(
            status=status,
            objective_value=None,
            x=None,
            num_candidates_checked=0,
            message=message,
            backend="tableau_simplex_two_phase",
            runtime_sec=perf_counter() - start,
        )

    def _is_original_solution_feasible(
        self,
        standard: GeneralLPStandardForm,
        x: np.ndarray,
    ) -> bool:
        if not np.all(np.isfinite(x)):
            return False
        for index, value in enumerate(x):
            bound_tolerance = self._comparison_tolerance(
                value,
                standard.original_lb[index],
                standard.original_ub[index]
                if np.isfinite(standard.original_ub[index])
                else 0.0,
            )
            if value < standard.original_lb[index] - bound_tolerance:
                return False
            if (
                np.isfinite(standard.original_ub[index])
                and value > standard.original_ub[index] + bound_tolerance
            ):
                return False

        activities = standard.original_A @ x
        for activity, rhs, row_sense in zip(
            activities,
            standard.original_b,
            standard.original_constraint_senses,
        ):
            row_tolerance = self._comparison_tolerance(activity, rhs)
            if row_sense == "<=" and activity > rhs + row_tolerance:
                return False
            if row_sense == ">=" and activity < rhs - row_tolerance:
                return False
            if row_sense == "=" and abs(activity - rhs) > row_tolerance:
                return False
        return True

    def _comparison_tolerance(self, *values: float) -> float:
        finite_values = [abs(float(value)) for value in values if np.isfinite(value)]
        return self.tolerance * max([1.0, *finite_values])

    def _clean_small_values(self, tableau: np.ndarray) -> None:
        tableau[np.abs(tableau) <= self.tolerance] = 0.0

    def _phase_iteration_line(
        self,
        *,
        phase: str,
        iteration: int,
        entering: int,
        leaving: int,
        tableau: np.ndarray,
        basis: list[int],
        column_names: list[str],
        objective_display_multiplier: float,
    ) -> str:
        objective = objective_display_multiplier * float(tableau[-1, -1])
        return (
            f"{phase:9s} | {iteration:9d} | {column_names[entering]:12s} | "
            f"{column_names[leaving]:11s} | {objective:10.6g} | "
            f"[{', '.join(column_names[index] for index in basis)}]"
        )


def solve_lp_relaxation_two_phase(
    problem,
    node_lb,
    node_ub,
    tol: float = 1e-8,
    max_candidates: int | None = None,
    use_matrix_presolve: bool = True,
    matrix_presolve_options=None,
    max_iterations: int | None = None,
) -> LPResult:
    """Solve one B&B node with the self-written two-phase tableau simplex.

    B&B uses an internal maximization model ``max internal_c^T z`` with
    ``G z <= h``. Current node bounds are passed through unchanged. Optional
    matrix presolve may first remove fixed variables and redundant rows; the
    general-LP standardizer then shifts the remaining finite lower bounds and
    adds upper-bound rows. Any reduced solution is reconstructed in the
    original variable space before it is returned to B&B.
    """
    del max_candidates
    start = perf_counter()
    c = np.asarray(problem.internal_c, dtype=float)
    lb = np.asarray(node_lb, dtype=float)
    ub = np.asarray(node_ub, dtype=float)

    if lb.shape != c.shape or ub.shape != c.shape:
        raise ValueError("node bounds must match the problem variable dimension")
    if np.any(lb > ub + tol):
        return LPResult(
            status=INFEASIBLE,
            objective_value=None,
            x=None,
            num_candidates_checked=0,
            message="node lower bound exceeds upper bound",
            backend="two_phase_simplex",
            runtime_sec=perf_counter() - start,
        )

    presolve = None
    if use_matrix_presolve:
        presolve = presolve_node_matrix(
            c,
            problem.G,
            problem.h,
            lb,
            ub,
            tol,
            options=matrix_presolve_options,
        )
        if presolve.status != "ok":
            return LPResult(
                status=INFEASIBLE,
                objective_value=None,
                x=None,
                num_candidates_checked=0,
                message=(
                    presolve.infeasible_reason
                    or "node matrix presolve detected an infeasible node"
                ),
                num_free_vars=len(presolve.free_indices),
                num_fixed_vars=len(presolve.fixed_indices),
                num_removed_rows=presolve.removed_rows,
                num_tightened_bounds=presolve.tightened_bounds,
                backend="two_phase_simplex",
                runtime_sec=perf_counter() - start,
            )
        if len(presolve.c_reduced) == 0:
            return LPResult(
                status=OPTIMAL,
                objective_value=float(presolve.objective_constant),
                x=presolve.fixed_values.copy(),
                num_candidates_checked=0,
                message="all variables fixed, fixed point feasible",
                num_free_vars=0,
                num_fixed_vars=len(presolve.fixed_indices),
                num_removed_rows=presolve.removed_rows,
                num_tightened_bounds=presolve.tightened_bounds,
                backend="two_phase_simplex",
                runtime_sec=perf_counter() - start,
            )
        c_lp = presolve.c_reduced
        A_lp = presolve.G_reduced
        b_lp = presolve.h_reduced
        lb_lp = presolve.lb_reduced
        ub_lp = presolve.ub_reduced
    else:
        c_lp = c
        A_lp = np.asarray(problem.G, dtype=float)
        b_lp = np.asarray(problem.h, dtype=float)
        lb_lp = lb
        ub_lp = ub

    solver = TwoPhaseTableauSimplexSolver(
        tolerance=tol,
        max_iterations=1000 if max_iterations is None else max_iterations,
    )
    result = solver.solve(
        c=c_lp,
        A=A_lp,
        b=b_lp,
        constraint_senses=["<="] * len(b_lp),
        lb=lb_lp,
        ub=ub_lp,
        sense="max",
    )
    result.backend = "two_phase_simplex"

    if presolve is not None:
        result.num_fixed_vars += len(presolve.fixed_indices)
        result.num_removed_rows = presolve.removed_rows
        result.num_tightened_bounds = presolve.tightened_bounds
        if result.x is not None:
            result.x = reconstruct_solution(presolve, result.x)
            result.objective_value = float(c @ result.x)

    result.runtime_sec = perf_counter() - start
    return result
