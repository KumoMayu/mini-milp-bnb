from __future__ import annotations

from time import perf_counter

import numpy as np

from .lp_standard_form import prepare_standard_form
from .result import LPResult


OPTIMAL = "optimal"
UNBOUNDED = "unbounded"
ITERATION_LIMIT = "iteration_limit"
NUMERICAL_ERROR = "numerical_error"


class TableauSimplexSolver:
    """Teaching-oriented primal tableau simplex for a nonnegative <= LP.

    The core tableau solves ``max c^T x`` subject to ``A x <= b``, ``x >= 0``
    and ``b >= 0``. Slack variables form the initial basis. The final row
    stores the negative reduced costs, so a negative entry identifies a
    nonbasic variable that can improve the maximization objective.
    """

    def __init__(
        self,
        tolerance: float = 1e-9,
        max_iterations: int = 1000,
        verbose: bool = False,
    ) -> None:
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be a positive finite number")
        if isinstance(max_iterations, bool) or int(max_iterations) != max_iterations:
            raise ValueError("max_iterations must be a nonnegative integer")
        if max_iterations < 0:
            raise ValueError("max_iterations must be a nonnegative integer")
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)
        self.verbose = bool(verbose)

    def solve(self, c, A, b, sense: str = "max") -> LPResult:
        """Solve one LP and return only original variables in ``result.x``.

        ``sense='min'`` is handled only by negating the objective before the
        same maximization tableau is built. Constraints and variable domains
        remain within the phase-1 supported form.
        """
        start = perf_counter()
        normalized_sense = str(sense).lower()
        if normalized_sense not in {"max", "min"}:
            raise ValueError('sense must be "max" or "min"')

        standard = prepare_standard_form(c, A, b, self.tolerance)
        objective_sign = 1.0 if normalized_sense == "max" else -1.0
        internal_c = objective_sign * standard.c
        tableau, basis = self._build_initial_tableau(
            internal_c,
            standard.A,
            standard.b,
        )
        num_original = standard.num_variables
        iterations = 0
        log: list[str] = []

        if self.verbose:
            print("iteration | entering | leaving | objective | basis")

        while True:
            if not np.all(np.isfinite(tableau)):
                return self._make_result(
                    status=NUMERICAL_ERROR,
                    message="tableau contains a non-finite value",
                    tableau=tableau,
                    basis=basis,
                    c=standard.c,
                    A=standard.A,
                    b=standard.b,
                    objective_sign=objective_sign,
                    num_original=num_original,
                    iterations=iterations,
                    start=start,
                    log=log,
                    include_current_solution=False,
                )

            entering = self.choose_entering_variable(tableau, basis)
            if entering is None:
                return self._make_result(
                    status=OPTIMAL,
                    message="optimal tableau reached",
                    tableau=tableau,
                    basis=basis,
                    c=standard.c,
                    A=standard.A,
                    b=standard.b,
                    objective_sign=objective_sign,
                    num_original=num_original,
                    iterations=iterations,
                    start=start,
                    log=log,
                    include_current_solution=True,
                )

            try:
                leaving_row = self.choose_leaving_row(tableau, basis, entering)
            except FloatingPointError as exc:
                return self._make_result(
                    status=NUMERICAL_ERROR,
                    message=str(exc),
                    tableau=tableau,
                    basis=basis,
                    c=standard.c,
                    A=standard.A,
                    b=standard.b,
                    objective_sign=objective_sign,
                    num_original=num_original,
                    iterations=iterations,
                    start=start,
                    log=log,
                    include_current_solution=False,
                )

            if leaving_row is None:
                return self._make_result(
                    status=UNBOUNDED,
                    message=f"entering variable {entering} has no eligible leaving row",
                    tableau=tableau,
                    basis=basis,
                    c=standard.c,
                    A=standard.A,
                    b=standard.b,
                    objective_sign=objective_sign,
                    num_original=num_original,
                    iterations=iterations,
                    start=start,
                    log=log,
                    include_current_solution=False,
                )

            if iterations >= self.max_iterations:
                return self._make_result(
                    status=ITERATION_LIMIT,
                    message=f"iteration limit {self.max_iterations} reached before optimality",
                    tableau=tableau,
                    basis=basis,
                    c=standard.c,
                    A=standard.A,
                    b=standard.b,
                    objective_sign=objective_sign,
                    num_original=num_original,
                    iterations=iterations,
                    start=start,
                    log=log,
                    include_current_solution=True,
                )

            leaving_variable = basis[leaving_row]
            try:
                self.pivot(tableau, leaving_row, entering)
            except FloatingPointError as exc:
                return self._make_result(
                    status=NUMERICAL_ERROR,
                    message=str(exc),
                    tableau=tableau,
                    basis=basis,
                    c=standard.c,
                    A=standard.A,
                    b=standard.b,
                    objective_sign=objective_sign,
                    num_original=num_original,
                    iterations=iterations,
                    start=start,
                    log=log,
                    include_current_solution=False,
                )

            basis[leaving_row] = entering
            iterations += 1
            line = self._iteration_line(
                iteration=iterations,
                entering=entering,
                leaving=leaving_variable,
                tableau=tableau,
                basis=basis,
                num_original=num_original,
                objective_sign=objective_sign,
            )
            log.append(line)
            if self.verbose:
                print(line)

    def _build_initial_tableau(
        self,
        c: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
    ) -> tuple[np.ndarray, list[int]]:
        """Add slacks; their columns are the initial feasible basis."""
        m, n = A.shape
        tableau = np.zeros((m + 1, n + m + 1), dtype=float)
        tableau[:m, :n] = A
        tableau[:m, n : n + m] = np.eye(m, dtype=float)
        tableau[:m, -1] = b

        # The objective row represents z - c^T x = 0. Its entries are the
        # negatives of reduced costs; negative entries can enter the basis.
        tableau[-1, :n] = -c
        basis = list(range(n, n + m))
        return tableau, basis

    def choose_entering_variable(
        self,
        tableau: np.ndarray,
        basis: list[int],
    ) -> int | None:
        """Apply Bland's rule: smallest-index improving nonbasic variable."""
        basic = set(basis)
        for column in range(tableau.shape[1] - 1):
            if column not in basic and tableau[-1, column] < -self.tolerance:
                return column
        return None

    def choose_leaving_row(
        self,
        tableau: np.ndarray,
        basis: list[int],
        entering: int,
    ) -> int | None:
        """Use the minimum-ratio test, excluding zero and negative directions."""
        ratios: list[tuple[float, int, int]] = []
        for row in range(tableau.shape[0] - 1):
            direction = float(tableau[row, entering])
            if direction <= self.tolerance:
                continue
            rhs = float(tableau[row, -1])
            if rhs < -self.tolerance:
                raise FloatingPointError(
                    "current basis lost primal feasibility during ratio test"
                )
            ratio = max(0.0, rhs) / direction
            ratios.append((ratio, int(basis[row]), row))

        if not ratios:
            return None

        minimum = min(item[0] for item in ratios)
        tied = [item for item in ratios if abs(item[0] - minimum) <= self.tolerance]
        # Bland's tie-break chooses the row whose current basic variable has
        # the smallest global column index.
        return min(tied, key=lambda item: (item[1], item[2]))[2]

    def pivot(self, tableau: np.ndarray, pivot_row: int, pivot_column: int) -> None:
        """Make one entering column basic and eliminate it from all other rows."""
        pivot_value = float(tableau[pivot_row, pivot_column])
        if not np.isfinite(pivot_value) or abs(pivot_value) <= self.tolerance:
            raise FloatingPointError("pivot element is zero, tiny, or non-finite")

        tableau[pivot_row, :] = tableau[pivot_row, :] / pivot_value
        for row in range(tableau.shape[0]):
            if row == pivot_row:
                continue
            factor = float(tableau[row, pivot_column])
            if abs(factor) <= self.tolerance:
                tableau[row, pivot_column] = 0.0
                continue
            tableau[row, :] = tableau[row, :] - factor * tableau[pivot_row, :]

        tableau[np.abs(tableau) <= self.tolerance] = 0.0
        if not np.all(np.isfinite(tableau)):
            raise FloatingPointError("pivot produced a non-finite tableau")

    def _extract_solution(
        self,
        tableau: np.ndarray,
        basis: list[int],
        num_original: int,
    ) -> np.ndarray:
        values = np.zeros(tableau.shape[1] - 1, dtype=float)
        for row, basic_variable in enumerate(basis):
            values[basic_variable] = tableau[row, -1]
        x = values[:num_original].copy()
        x[np.abs(x) <= self.tolerance] = 0.0
        return x

    def _make_result(
        self,
        *,
        status: str,
        message: str,
        tableau: np.ndarray,
        basis: list[int],
        c: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        objective_sign: float,
        num_original: int,
        iterations: int,
        start: float,
        log: list[str],
        include_current_solution: bool,
    ) -> LPResult:
        x = None
        objective = None
        final_status = status
        final_message = message

        if include_current_solution:
            candidate = self._extract_solution(tableau, basis, num_original)
            feasible = bool(
                np.all(np.isfinite(candidate))
                and np.all(candidate >= -self.tolerance)
                and np.all(A @ candidate <= b + self.tolerance)
            )
            if feasible:
                x = candidate
                objective = float(c @ candidate)
                internal_tableau_objective = float(tableau[-1, -1])
                internal_recomputed = float(objective_sign * objective)
                objective_tolerance = self.tolerance * max(
                    1.0,
                    abs(internal_tableau_objective),
                    abs(internal_recomputed),
                )
                if abs(internal_tableau_objective - internal_recomputed) > objective_tolerance:
                    final_status = NUMERICAL_ERROR
                    final_message = "tableau objective disagrees with reconstructed primal solution"
                    x = None
                    objective = None
            else:
                final_status = NUMERICAL_ERROR
                final_message = "tableau basis does not reconstruct a feasible primal solution"

        elapsed = perf_counter() - start
        if self.verbose:
            objective_text = "-" if objective is None else f"{objective:.10g}"
            print(
                f"status={final_status} iterations={iterations} "
                f"objective={objective_text} basis={list(basis)}"
            )

        return LPResult(
            status=final_status,
            objective_value=objective,
            x=x,
            num_candidates_checked=0,
            message=final_message,
            num_free_vars=num_original,
            num_fixed_vars=0,
            backend="tableau_simplex",
            num_iterations=iterations,
            basis_indices=tuple(int(index) for index in basis),
            runtime_sec=elapsed,
            iteration_log=tuple(log),
        )

    def _iteration_line(
        self,
        *,
        iteration: int,
        entering: int,
        leaving: int,
        tableau: np.ndarray,
        basis: list[int],
        num_original: int,
        objective_sign: float,
    ) -> str:
        objective = objective_sign * float(tableau[-1, -1])
        return (
            f"{iteration:9d} | {self._variable_name(entering, num_original):8s} | "
            f"{self._variable_name(leaving, num_original):7s} | "
            f"{objective:9.6g} | "
            f"[{', '.join(self._variable_name(index, num_original) for index in basis)}]"
        )

    @staticmethod
    def _variable_name(index: int, num_original: int) -> str:
        if index < num_original:
            return f"x{index + 1}"
        return f"s{index - num_original + 1}"


def solve_lp_relaxation_tableau(
    problem,
    node_lb,
    node_ub,
    tol: float = 1e-8,
    max_candidates: int | None = None,
    use_matrix_presolve: bool = True,
    matrix_presolve_options=None,
) -> LPResult:
    """Compatibility adapter for explicitly selected simple standard LPs.

    Phase 1 is intentionally not ready for general B&B nodes. This adapter
    accepts only zero lower bounds. Finite nonnegative upper bounds are written
    as additional ``x_i <= ub_i`` rows before calling the standalone tableau
    solver. Negative right-hand sides remain unsupported.
    """
    del max_candidates, use_matrix_presolve, matrix_presolve_options
    lb = np.asarray(node_lb, dtype=float)
    ub = np.asarray(node_ub, dtype=float)
    if lb.shape != problem.c.shape or ub.shape != problem.c.shape:
        raise ValueError("node bounds must match the problem variable dimension")
    if np.any(np.abs(lb) > tol):
        raise ValueError(
            "tableau_simplex phase 1 only supports zero lower bounds and is not "
            "ready for branched B&B nodes"
        )
    if np.any(~np.isfinite(ub)) or np.any(ub < -tol):
        raise ValueError("tableau_simplex adapter requires finite nonnegative upper bounds")

    A = np.vstack([problem.G, np.eye(problem.num_vars, dtype=float)])
    b = np.concatenate([problem.h, ub])
    return TableauSimplexSolver(tolerance=tol).solve(
        c=problem.internal_c,
        A=A,
        b=b,
        sense="max",
    )
