from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class UnsupportedStandardFormError(ValueError):
    """Raised when an LP needs transformations not implemented in phase 1."""


class InfeasibleBoundsError(ValueError):
    """Raised when variable bounds contradict each other."""


@dataclass(frozen=True)
class StandardFormLP:
    """Validated phase-1 input for max c^T x subject to A x <= b, x >= 0."""

    c: np.ndarray
    A: np.ndarray
    b: np.ndarray

    @property
    def num_variables(self) -> int:
        return int(self.c.size)

    @property
    def num_constraints(self) -> int:
        return int(self.b.size)


def prepare_standard_form(c, A, b, tolerance: float = 1e-9) -> StandardFormLP:
    """Validate and copy the phase-1 tableau simplex input.

    Variables are implicitly nonnegative. Because this phase has no artificial
    variables or Phase I, every right-hand side must be nonnegative so the
    slack-variable basis is an initial basic feasible solution.
    """
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be a positive finite number")

    c_array = np.asarray(c, dtype=float)
    A_array = np.asarray(A, dtype=float)
    b_array = np.asarray(b, dtype=float)

    if c_array.ndim != 1:
        raise ValueError("c must be a one-dimensional vector")
    if A_array.ndim != 2:
        raise ValueError("A must be a two-dimensional matrix")
    if b_array.ndim != 1:
        raise ValueError("b must be a one-dimensional vector")
    if A_array.shape != (b_array.size, c_array.size):
        raise ValueError("A must have shape (len(b), len(c))")
    if not np.all(np.isfinite(c_array)):
        raise ValueError("c must contain only finite values")
    if not np.all(np.isfinite(A_array)):
        raise ValueError("A must contain only finite values")
    if not np.all(np.isfinite(b_array)):
        raise ValueError("b must contain only finite values")
    if np.any(b_array < -tolerance):
        raise UnsupportedStandardFormError(
            "tableau simplex phase 1 requires b >= 0; negative right-hand "
            "sides require constraint conversion and a Phase I method"
        )

    cleaned_b = b_array.copy()
    cleaned_b[np.abs(cleaned_b) <= tolerance] = 0.0
    return StandardFormLP(c=c_array.copy(), A=A_array.copy(), b=cleaned_b)


@dataclass(frozen=True)
class VariableRecovery:
    """Map nonnegative transformed variables back to the original variables."""

    shift: np.ndarray
    free_indices: np.ndarray
    fixed_indices: np.ndarray

    @property
    def original_size(self) -> int:
        return int(self.shift.size)

    def recover(self, transformed_x) -> np.ndarray:
        transformed = np.asarray(transformed_x, dtype=float)
        if transformed.shape != (self.free_indices.size,):
            raise ValueError("transformed solution has the wrong dimension")
        original = self.shift.copy()
        original[self.free_indices] += transformed
        return original


@dataclass(frozen=True)
class GeneralLPStandardForm:
    """Equality-form tableau data plus an explicit original-variable mapping."""

    A_eq: np.ndarray
    b_eq: np.ndarray
    initial_basis: tuple[int, ...]
    artificial_indices: tuple[int, ...]
    phase_one_objective: np.ndarray
    phase_two_objective: np.ndarray
    objective_constant_internal: float
    objective_sign: float
    column_names: tuple[str, ...]
    num_transformed_variables: int
    recovery: VariableRecovery
    original_c: np.ndarray
    original_A: np.ndarray
    original_b: np.ndarray
    original_constraint_senses: tuple[str, ...]
    original_lb: np.ndarray
    original_ub: np.ndarray
    original_sense: str
    normalized_constraint_senses: tuple[str, ...]

    @property
    def num_tableau_variables(self) -> int:
        return int(self.A_eq.shape[1])

    def recover_original_solution(self, tableau_values) -> np.ndarray:
        values = np.asarray(tableau_values, dtype=float)
        if values.ndim != 1 or values.size < self.num_transformed_variables:
            raise ValueError("tableau solution is too short for variable recovery")
        return self.recovery.recover(values[: self.num_transformed_variables])


def standardize_general_lp(
    c,
    A,
    b,
    constraint_senses=None,
    lb=None,
    ub=None,
    sense: str = "max",
    tolerance: float = 1e-9,
) -> GeneralLPStandardForm:
    """Convert a bounded-below general LP into equality tableau form.

    Nonfixed variables are shifted as ``x = lb + x_prime``. Fixed variables
    are removed first. Finite transformed upper bounds become additional
    ``x_prime <= ub - lb`` rows. Constraint rows with negative RHS are
    multiplied by -1 and their inequality direction is reversed.
    """
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be a positive finite number")

    c_array = np.asarray(c, dtype=float)
    A_array = np.asarray(A, dtype=float)
    b_array = np.asarray(b, dtype=float)
    normalized_objective_sense = str(sense).lower()

    if c_array.ndim != 1:
        raise ValueError("c must be a one-dimensional vector")
    if A_array.ndim != 2:
        raise ValueError("A must be a two-dimensional matrix")
    if b_array.ndim != 1:
        raise ValueError("b must be a one-dimensional vector")
    if A_array.shape != (b_array.size, c_array.size):
        raise ValueError("A must have shape (len(b), len(c))")
    if normalized_objective_sense not in {"max", "min"}:
        raise ValueError('sense must be "max" or "min"')
    if not np.all(np.isfinite(c_array)):
        raise ValueError("c must contain only finite values")
    if not np.all(np.isfinite(A_array)):
        raise ValueError("A must contain only finite values")
    if not np.all(np.isfinite(b_array)):
        raise ValueError("b must contain only finite values")

    m, n = A_array.shape
    senses = _normalize_constraint_senses(constraint_senses, m)
    lb_array = np.zeros(n, dtype=float) if lb is None else np.asarray(lb, dtype=float)
    ub_array = np.full(n, np.inf, dtype=float) if ub is None else np.asarray(ub, dtype=float)
    if lb_array.shape != (n,) or ub_array.shape != (n,):
        raise ValueError("lb and ub must be one-dimensional vectors matching len(c)")
    if np.any(np.isnan(lb_array)) or np.any(np.isnan(ub_array)):
        raise ValueError("lb and ub must not contain NaN")
    if np.any(~np.isfinite(lb_array)):
        raise UnsupportedStandardFormError(
            "two-phase tableau simplex requires a finite lower bound for every "
            "variable; free and upper-only variables are unsupported"
        )
    if np.any(np.isneginf(ub_array)):
        raise InfeasibleBoundsError("an upper bound of -infinity is infeasible")
    if np.any(lb_array > ub_array + tolerance):
        raise InfeasibleBoundsError("a variable lower bound exceeds its upper bound")

    finite_ub = np.isfinite(ub_array)
    fixed_mask = finite_ub & (np.abs(ub_array - lb_array) <= tolerance)
    fixed_indices = np.flatnonzero(fixed_mask)
    free_indices = np.flatnonzero(~fixed_mask)

    shift = lb_array.copy()
    if fixed_indices.size:
        shift[fixed_indices] = 0.5 * (
            lb_array[fixed_indices] + ub_array[fixed_indices]
        )

    reduced_A = A_array[:, free_indices]
    shifted_b = b_array - A_array @ shift
    reduced_c = c_array[free_indices]
    row_coefficients = [reduced_A[row].copy() for row in range(m)]
    row_rhs = [float(value) for value in shifted_b]
    row_senses = list(senses)

    for transformed_index, original_index in enumerate(free_indices):
        if not np.isfinite(ub_array[original_index]):
            continue
        width = float(ub_array[original_index] - lb_array[original_index])
        if width < -tolerance:
            raise InfeasibleBoundsError(
                "a transformed upper-bound width is negative"
            )
        bound_row = np.zeros(free_indices.size, dtype=float)
        bound_row[transformed_index] = 1.0
        row_coefficients.append(bound_row)
        row_rhs.append(max(0.0, width))
        row_senses.append("<=")

    normalized_rows: list[np.ndarray] = []
    normalized_rhs: list[float] = []
    normalized_senses: list[str] = []
    for coefficients, rhs, row_sense in zip(
        row_coefficients,
        row_rhs,
        row_senses,
    ):
        if rhs < -tolerance:
            coefficients = -coefficients
            rhs = -rhs
            row_sense = _flip_constraint_sense(row_sense)
        elif abs(rhs) <= tolerance:
            rhs = 0.0
        normalized_rows.append(np.asarray(coefficients, dtype=float))
        normalized_rhs.append(float(rhs))
        normalized_senses.append(row_sense)

    num_rows = len(normalized_rows)
    if num_rows:
        equality_matrix = np.vstack(normalized_rows)
    else:
        equality_matrix = np.zeros((0, free_indices.size), dtype=float)
    column_names = [f"x{int(index) + 1}_shift" for index in free_indices]
    initial_basis: list[int] = []
    artificial_indices: list[int] = []

    def append_column(row_index: int, coefficient: float, name: str) -> int:
        nonlocal equality_matrix
        column = np.zeros(num_rows, dtype=float)
        column[row_index] = coefficient
        equality_matrix = np.column_stack([equality_matrix, column])
        column_names.append(name)
        return equality_matrix.shape[1] - 1

    for row_index, row_sense in enumerate(normalized_senses):
        row_number = row_index + 1
        if row_sense == "<=":
            slack = append_column(row_index, 1.0, f"s{row_number}")
            initial_basis.append(slack)
        elif row_sense == ">=":
            append_column(row_index, -1.0, f"r{row_number}")
            artificial = append_column(row_index, 1.0, f"a{row_number}")
            artificial_indices.append(artificial)
            initial_basis.append(artificial)
        else:
            artificial = append_column(row_index, 1.0, f"a{row_number}")
            artificial_indices.append(artificial)
            initial_basis.append(artificial)

    objective_sign = 1.0 if normalized_objective_sense == "max" else -1.0
    phase_two_objective = np.zeros(equality_matrix.shape[1], dtype=float)
    phase_two_objective[: free_indices.size] = objective_sign * reduced_c
    phase_one_objective = np.zeros(equality_matrix.shape[1], dtype=float)
    if artificial_indices:
        phase_one_objective[np.asarray(artificial_indices, dtype=int)] = -1.0

    return GeneralLPStandardForm(
        A_eq=equality_matrix,
        b_eq=np.asarray(normalized_rhs, dtype=float),
        initial_basis=tuple(initial_basis),
        artificial_indices=tuple(artificial_indices),
        phase_one_objective=phase_one_objective,
        phase_two_objective=phase_two_objective,
        objective_constant_internal=float(objective_sign * (c_array @ shift)),
        objective_sign=objective_sign,
        column_names=tuple(column_names),
        num_transformed_variables=int(free_indices.size),
        recovery=VariableRecovery(
            shift=shift,
            free_indices=free_indices,
            fixed_indices=fixed_indices,
        ),
        original_c=c_array.copy(),
        original_A=A_array.copy(),
        original_b=b_array.copy(),
        original_constraint_senses=senses,
        original_lb=lb_array.copy(),
        original_ub=ub_array.copy(),
        original_sense=normalized_objective_sense,
        normalized_constraint_senses=tuple(normalized_senses),
    )


def _normalize_constraint_senses(constraint_senses, num_rows: int) -> tuple[str, ...]:
    if constraint_senses is None:
        return tuple("<=" for _ in range(num_rows))
    senses = tuple(str(value).strip() for value in constraint_senses)
    if len(senses) != num_rows:
        raise ValueError("constraint_senses length must equal len(b)")
    invalid = [value for value in senses if value not in {"<=", ">=", "="}]
    if invalid:
        raise ValueError('constraint senses must be "<=", ">=", or "="')
    return senses


def _flip_constraint_sense(sense: str) -> str:
    if sense == "<=":
        return ">="
    if sense == ">=":
        return "<="
    return "="
