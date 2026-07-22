from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MatrixPresolveResult:
    status: str
    c_reduced: np.ndarray
    G_reduced: np.ndarray
    h_reduced: np.ndarray
    lb_reduced: np.ndarray
    ub_reduced: np.ndarray
    free_indices: np.ndarray
    fixed_indices: np.ndarray
    fixed_values: np.ndarray
    objective_constant: float
    removed_rows: int
    tightened_bounds: int
    fixed_variables: int
    infeasible_reason: str | None = None


@dataclass(frozen=True)
class MatrixPresolveOptions:
    eliminate_fixed_variables: bool = True
    remove_redundant_rows: bool = True
    tighten_bounds: bool = True
    max_rounds: int = 3


def _activity_bounds(row: np.ndarray,lb: np.ndarray,ub: np.ndarray) -> tuple[float,float,np.ndarray]:
    min_terms=np.where(row>=0.0,row*lb,row*ub)
    max_terms=np.where(row>=0.0,row*ub,row*lb)
    return float(np.sum(min_terms)),float(np.sum(max_terms)),min_terms


def _compress(c,G,h,lb,ub,tol: float,eliminate_fixed_variables: bool = True):
    if eliminate_fixed_variables:
        fixed_mask=np.abs(ub-lb)<=tol
    else:
        fixed_mask=np.zeros(len(c),dtype=bool)
    fixed_indices=np.flatnonzero(fixed_mask)
    free_indices=np.flatnonzero(~fixed_mask)

    fixed_values=np.zeros(len(c),dtype=float)
    if len(fixed_indices)>0:
        fixed_values[fixed_indices]=0.5*(lb[fixed_indices]+ub[fixed_indices])

    c_reduced=c[free_indices]
    G_reduced=G[:,free_indices]
    if len(fixed_indices)>0:
        h_reduced=h-G[:,fixed_indices]@fixed_values[fixed_indices]
        objective_constant=float(c[fixed_indices]@fixed_values[fixed_indices])
    else:
        h_reduced=h.copy()
        objective_constant=0.0

    return (
        c_reduced,
        G_reduced,
        h_reduced,
        lb[free_indices],
        ub[free_indices],
        free_indices,
        fixed_indices,
        fixed_values,
        objective_constant,
    )


def _remove_rows(G,h,lb,ub,tol: float):
    kept_rows=[]
    kept_rhs=[]
    removed=0

    for row,rhs in zip(G,h):
        if np.all(np.abs(row)<=tol):
            if rhs<-tol:
                return None,None,removed,"zero row violates RHS"
            removed+=1
            continue

        lower,upper,_=_activity_bounds(row,lb,ub)
        if lower>rhs+tol:
            return None,None,removed,"row lower activity exceeds RHS"
        if upper<=rhs+tol:
            removed+=1
            continue

        duplicate_index=None
        for i,existing_row in enumerate(kept_rows):
            if np.allclose(row,existing_row,atol=tol,rtol=0.0):
                duplicate_index=i
                break

        if duplicate_index is None:
            kept_rows.append(row.copy())
            kept_rhs.append(float(rhs))
            continue

        removed+=1
        if rhs<kept_rhs[duplicate_index]-tol:
            kept_rows[duplicate_index]=row.copy()
            kept_rhs[duplicate_index]=float(rhs)

    if kept_rows:
        return np.vstack(kept_rows),np.asarray(kept_rhs,dtype=float),removed,None
    return np.zeros((0,G.shape[1]),dtype=float),np.zeros(0,dtype=float),removed,None


def reconstruct_solution(presolve_result: MatrixPresolveResult,x_reduced) -> np.ndarray:
    x_full=presolve_result.fixed_values.copy()
    x_full[presolve_result.free_indices]=np.asarray(x_reduced,dtype=float)
    return x_full


def presolve_node_matrix(
    c,
    G,
    h,
    lb,
    ub,
    tol: float = 1e-8,
    max_rounds: int | None = None,
    options: MatrixPresolveOptions | None = None,
) -> MatrixPresolveResult:
    if options is None:
        options=MatrixPresolveOptions(max_rounds=3 if max_rounds is None else int(max_rounds))
    elif max_rounds is not None and int(max_rounds)!=int(options.max_rounds):
        raise ValueError("max_rounds cannot conflict with options.max_rounds")
    c=np.asarray(c,dtype=float)
    G=np.asarray(G,dtype=float)
    h=np.asarray(h,dtype=float)
    lb=np.asarray(lb,dtype=float).copy()
    ub=np.asarray(ub,dtype=float).copy()

    tightened_bounds=0

    if np.any(lb>ub+tol):
        return MatrixPresolveResult(
            status="infeasible",
            c_reduced=np.zeros(0),
            G_reduced=np.zeros((0,0)),
            h_reduced=np.zeros(0),
            lb_reduced=np.zeros(0),
            ub_reduced=np.zeros(0),
            free_indices=np.zeros(0,dtype=int),
            fixed_indices=np.arange(len(c)),
            fixed_values=np.zeros(len(c)),
            objective_constant=0.0,
            removed_rows=0,
            tightened_bounds=0,
            fixed_variables=0,
            infeasible_reason="lower bound exceeds upper bound",
        )

    final_removed_rows=0

    for _ in range(int(options.max_rounds)):
        (
            c_reduced,
            G_reduced,
            h_reduced,
            lb_reduced,
            ub_reduced,
            free_indices,
            fixed_indices,
            fixed_values,
            objective_constant,
        )=_compress(c,G,h,lb,ub,tol,options.eliminate_fixed_variables)

        if len(free_indices)==0:
            if np.any(G@fixed_values>h+tol):
                return MatrixPresolveResult(
                    status="infeasible",
                    c_reduced=np.zeros(0),
                    G_reduced=np.zeros((0,0)),
                    h_reduced=np.zeros(0),
                    lb_reduced=np.zeros(0),
                    ub_reduced=np.zeros(0),
                    free_indices=free_indices,
                    fixed_indices=fixed_indices,
                    fixed_values=fixed_values,
                    objective_constant=objective_constant,
                    removed_rows=0,
                    tightened_bounds=tightened_bounds,
                    fixed_variables=len(fixed_indices),
                    infeasible_reason="all variables fixed but constraints violated",
                )
            return MatrixPresolveResult(
                status="ok",
                c_reduced=np.zeros(0),
                G_reduced=np.zeros((0,0)),
                h_reduced=np.zeros(0),
                lb_reduced=np.zeros(0),
                ub_reduced=np.zeros(0),
                free_indices=free_indices,
                fixed_indices=fixed_indices,
                fixed_values=fixed_values,
                objective_constant=objective_constant,
                removed_rows=len(h),
                tightened_bounds=tightened_bounds,
                fixed_variables=len(fixed_indices),
            )

        if options.remove_redundant_rows or options.tighten_bounds:
            G_kept,h_kept,removed_rows,reason=_remove_rows(G_reduced,h_reduced,lb_reduced,ub_reduced,tol)
            final_removed_rows=removed_rows
            if reason is not None:
                return MatrixPresolveResult(
                    status="infeasible",
                    c_reduced=c_reduced,
                    G_reduced=np.zeros((0,len(c_reduced))),
                    h_reduced=np.zeros(0),
                    lb_reduced=lb_reduced,
                    ub_reduced=ub_reduced,
                    free_indices=free_indices,
                    fixed_indices=fixed_indices,
                    fixed_values=fixed_values,
                    objective_constant=objective_constant,
                    removed_rows=removed_rows,
                    tightened_bounds=tightened_bounds,
                    fixed_variables=len(fixed_indices),
                    infeasible_reason=reason,
                )
        else:
            G_kept,h_kept=G_reduced,h_reduced
            removed_rows=0

        changed=False
        if not options.tighten_bounds:
            break
        for row,rhs in zip(G_kept,h_kept):
            _,_,min_terms=_activity_bounds(row,lb_reduced,ub_reduced)
            total_min=float(np.sum(min_terms))
            for local_index,coef in enumerate(row):
                if abs(coef)<=tol:
                    continue
                min_without=total_min-float(min_terms[local_index])
                global_index=free_indices[local_index]
                candidate=(rhs-min_without)/coef
                if coef>0.0:
                    if candidate<ub[global_index]-tol:
                        ub[global_index]=max(lb[global_index],min(ub[global_index],candidate))
                        tightened_bounds+=1
                        changed=True
                else:
                    if candidate>lb[global_index]+tol:
                        lb[global_index]=min(ub[global_index],max(lb[global_index],candidate))
                        tightened_bounds+=1
                        changed=True
                if lb[global_index]>ub[global_index]+tol:
                    return MatrixPresolveResult(
                        status="infeasible",
                        c_reduced=c_reduced,
                        G_reduced=G_kept,
                        h_reduced=h_kept,
                        lb_reduced=lb_reduced,
                        ub_reduced=ub_reduced,
                        free_indices=free_indices,
                        fixed_indices=fixed_indices,
                        fixed_values=fixed_values,
                        objective_constant=objective_constant,
                        removed_rows=removed_rows,
                        tightened_bounds=tightened_bounds,
                        fixed_variables=len(fixed_indices),
                        infeasible_reason="bound tightening crossed bounds",
                    )

        if not changed:
            break

    (
        c_reduced,
        G_reduced,
        h_reduced,
        lb_reduced,
        ub_reduced,
        free_indices,
        fixed_indices,
        fixed_values,
        objective_constant,
    )=_compress(c,G,h,lb,ub,tol,options.eliminate_fixed_variables)

    if len(free_indices)==0:
        if np.any(G@fixed_values>h+tol):
            return MatrixPresolveResult(
                status="infeasible",
                c_reduced=np.zeros(0),
                G_reduced=np.zeros((0,0)),
                h_reduced=np.zeros(0),
                lb_reduced=np.zeros(0),
                ub_reduced=np.zeros(0),
                free_indices=free_indices,
                fixed_indices=fixed_indices,
                fixed_values=fixed_values,
                objective_constant=objective_constant,
                removed_rows=final_removed_rows,
                tightened_bounds=tightened_bounds,
                fixed_variables=len(fixed_indices),
                infeasible_reason="all variables fixed but constraints violated",
            )
        return MatrixPresolveResult(
            status="ok",
            c_reduced=np.zeros(0),
            G_reduced=np.zeros((0,0)),
            h_reduced=np.zeros(0),
            lb_reduced=np.zeros(0),
            ub_reduced=np.zeros(0),
            free_indices=free_indices,
            fixed_indices=fixed_indices,
            fixed_values=fixed_values,
            objective_constant=objective_constant,
            removed_rows=len(h),
            tightened_bounds=tightened_bounds,
            fixed_variables=len(fixed_indices),
        )

    if options.remove_redundant_rows:
        G_reduced,h_reduced,removed_rows,reason=_remove_rows(G_reduced,h_reduced,lb_reduced,ub_reduced,tol)
        if reason is not None:
            return MatrixPresolveResult(
                status="infeasible",
                c_reduced=c_reduced,
                G_reduced=np.zeros((0,len(c_reduced))),
                h_reduced=np.zeros(0),
                lb_reduced=lb_reduced,
                ub_reduced=ub_reduced,
                free_indices=free_indices,
                fixed_indices=fixed_indices,
                fixed_values=fixed_values,
                objective_constant=objective_constant,
                removed_rows=removed_rows,
                tightened_bounds=tightened_bounds,
                fixed_variables=len(fixed_indices),
                infeasible_reason=reason,
            )
    else:
        removed_rows=final_removed_rows

    return MatrixPresolveResult(
        status="ok",
        c_reduced=c_reduced,
        G_reduced=G_reduced,
        h_reduced=h_reduced,
        lb_reduced=lb_reduced,
        ub_reduced=ub_reduced,
        free_indices=free_indices,
        fixed_indices=fixed_indices,
        fixed_values=fixed_values,
        objective_constant=objective_constant,
        removed_rows=removed_rows,
        tightened_bounds=tightened_bounds,
        fixed_variables=len(fixed_indices),
    )
