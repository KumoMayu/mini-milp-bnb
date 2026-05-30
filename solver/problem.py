from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MILPProblem:
    c: object
    A: object
    b: object
    sense: str = "max"
    lb: object | None = None
    ub: object | None = None
    var_types: object | None = None
    name: str = "standard_milp"
    c_x: object | None = None
    c_y: object | None = None
    block_A: object | None = None
    block_B: object | None = None
    x_lb: object | None = None
    x_ub: object | None = None
    y_lb: object | None = None
    y_ub: object | None = None
    y_types: object | None = None
    num_continuous: int | None = None

    @classmethod
    def from_blocks(
        cls,
        c_x,
        c_y,
        A,
        B,
        b,
        x_lb,
        x_ub,
        y_lb,
        y_ub,
        y_types,
        sense: str = "max",
        name: str = "block_milp",
    ) -> "MILPProblem":
        """Create the public block MILP form A x + B y <= b.

        The public modeling interface separates continuous variables x from
        integer/binary variables y:

            min/max c_x^T x + c_y^T y
            s.t.    A x + B y <= b
                    x_lb <= x <= x_ub
                    y_lb <= y <= y_ub

        Internally this method concatenates z=[x;y], G=[A B], c=[c_x;c_y],
        and creates the unified matrix form G z <= b used by the solver.
        """
        c_x=np.asarray(c_x,dtype=float)
        c_y=np.asarray(c_y,dtype=float)
        A=np.asarray(A,dtype=float)
        B=np.asarray(B,dtype=float)
        b=np.asarray(b,dtype=float)
        x_lb=np.asarray(x_lb,dtype=float)
        x_ub=np.asarray(x_ub,dtype=float)
        y_lb=np.asarray(y_lb,dtype=float)
        y_ub=np.asarray(y_ub,dtype=float)
        y_types=[str(t).upper() for t in y_types]

        if c_x.ndim!=1:
            raise ValueError("c_x must be a one-dimensional vector")
        if c_y.ndim!=1:
            raise ValueError("c_y must be a one-dimensional vector")
        if A.ndim!=2:
            raise ValueError("A must be a two-dimensional matrix")
        if B.ndim!=2:
            raise ValueError("B must be a two-dimensional matrix")
        if b.ndim!=1:
            raise ValueError("b must be a one-dimensional vector")
        if A.shape[0]!=B.shape[0]:
            raise ValueError("A and B must have the same number of rows")
        if A.shape[0]!=b.shape[0]:
            raise ValueError("len(b) must equal the number of rows in A and B")
        if A.shape[1]!=len(c_x):
            raise ValueError("len(c_x) must equal A.shape[1]")
        if B.shape[1]!=len(c_y):
            raise ValueError("len(c_y) must equal B.shape[1]")
        if x_lb.ndim!=1 or x_ub.ndim!=1:
            raise ValueError("x_lb and x_ub must be one-dimensional vectors")
        if y_lb.ndim!=1 or y_ub.ndim!=1:
            raise ValueError("y_lb and y_ub must be one-dimensional vectors")
        if len(x_lb)!=len(c_x) or len(x_ub)!=len(c_x):
            raise ValueError("x_lb and x_ub length must equal len(c_x)")
        if len(y_lb)!=len(c_y) or len(y_ub)!=len(c_y):
            raise ValueError("y_lb and y_ub length must equal len(c_y)")
        if len(y_types)!=len(c_y):
            raise ValueError("y_types length must equal len(c_y)")
        bad=[t for t in y_types if t not in ("I","B")]
        if bad:
            raise ValueError('y_types may only contain "I" and "B"')
        for i,t in enumerate(y_types):
            if t=="B" and (abs(y_lb[i])>1e-12 or abs(y_ub[i]-1.0)>1e-12):
                raise ValueError("binary y variables must have y_lb=0 and y_ub=1")

        c=np.concatenate([c_x,c_y])
        G=np.concatenate([A,B],axis=1)
        lb=np.concatenate([x_lb,y_lb])
        ub=np.concatenate([x_ub,y_ub])
        var_types=["C"]*len(c_x)+y_types

        return cls(
            c=c,
            A=G,
            b=b,
            sense=sense,
            lb=lb,
            ub=ub,
            var_types=var_types,
            name=name,
            c_x=c_x,
            c_y=c_y,
            block_A=A,
            block_B=B,
            x_lb=x_lb,
            x_ub=x_ub,
            y_lb=y_lb,
            y_ub=y_ub,
            y_types=y_types,
            num_continuous=len(c_x),
        )

    @classmethod
    def from_standard(
        cls,
        c,
        G,
        h,
        lb,
        ub,
        var_types,
        sense: str = "max",
        name: str = "standard_milp",
    ) -> "MILPProblem":
        return cls(c=c,A=G,b=h,sense=sense,lb=lb,ub=ub,var_types=var_types,name=name)

    def __post_init__(self) -> None:
        self.c=np.asarray(self.c,dtype=float)
        self.A=np.asarray(self.A,dtype=float)
        self.b=np.asarray(self.b,dtype=float)
        self.sense=str(self.sense).lower()

        if self.c.ndim!=1:
            raise ValueError("c must be a one-dimensional vector")
        if self.A.ndim!=2:
            raise ValueError("A/G must be a two-dimensional matrix")
        if self.b.ndim!=1:
            raise ValueError("b/h must be a one-dimensional vector")
        if self.sense not in ("min","max"):
            raise ValueError('sense must be "min" or "max"')

        n=len(self.c)
        if self.A.shape[1]!=n:
            raise ValueError("A.shape[1] must equal len(c)")
        if self.A.shape[0]!=len(self.b):
            raise ValueError("A.shape[0] must equal len(b)")
        if self.lb is None or self.ub is None:
            raise ValueError("finite lb and ub must be provided")

        self.lb=np.asarray(self.lb,dtype=float)
        self.ub=np.asarray(self.ub,dtype=float)
        if self.lb.ndim!=1 or self.ub.ndim!=1:
            raise ValueError("lb and ub must be one-dimensional vectors")
        if len(self.lb)!=n or len(self.ub)!=n:
            raise ValueError("lb and ub length must equal number of variables")
        if not np.all(np.isfinite(self.lb)) or not np.all(np.isfinite(self.ub)):
            raise ValueError("lb and ub must be finite")
        if np.any(self.lb>self.ub):
            raise ValueError("each lower bound must be <= upper bound")

        if self.var_types is None:
            self.var_types=["C"]*n
        else:
            self.var_types=[str(t).upper() for t in self.var_types]
        if len(self.var_types)!=n:
            raise ValueError("var_types length must equal number of variables")
        bad=[t for t in self.var_types if t not in ("C","I","B")]
        if bad:
            raise ValueError('var_types may only contain "C", "I", and "B"')
        for i,t in enumerate(self.var_types):
            if t=="B" and (abs(self.lb[i])>1e-12 or abs(self.ub[i]-1.0)>1e-12):
                raise ValueError("binary variables must have lb=0 and ub=1")

        if self.num_continuous is None:
            self.num_continuous=sum(1 for t in self.var_types if t=="C")
        self.G=self.A
        self.h=self.b

    @property
    def num_vars(self) -> int:
        return len(self.c)

    @property
    def num_constraints(self) -> int:
        return len(self.b)

    @property
    def num_integer(self) -> int:
        return len(self.integer_indices)

    @property
    def integer_indices(self) -> list[int]:
        return [i for i,t in enumerate(self.var_types) if t in ("I","B")]

    @property
    def binary_indices(self) -> list[int]:
        return [i for i,t in enumerate(self.var_types) if t=="B"]

    @property
    def continuous_indices(self) -> list[int]:
        return [i for i,t in enumerate(self.var_types) if t=="C"]

    @property
    def internal_c(self) -> np.ndarray:
        if self.sense=="max":
            return self.c.copy()
        return -self.c.copy()

    def recover_objective_value(self,internal_objective_value: float) -> float:
        value=float(internal_objective_value)
        if self.sense=="max":
            return value
        return -value

    def split_solution(self,z: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        split=int(self.num_continuous)
        return z[:split].copy(),z[split:].copy()
