# 两阶段表格单纯形：一般 LP 标准化

## 1. 独立入口

`TableauSimplexSolver` 保留第一阶段接口和行为。一般 LP 使用：

```python
from solver import TwoPhaseTableauSimplexSolver

result = TwoPhaseTableauSimplexSolver().solve(
    c=c,
    A=A,
    b=b,
    constraint_senses=["<=", ">=", "="],
    lb=lb,
    ub=ub,
    sense="min",
)
```

该入口没有成为 B&B 默认后端，也没有改变 `active_set`。

## 2. 变量和约束标准化

固定变量先消元。其余变量按有限下界平移：

$$
x=lb+x',\qquad x'\ge0.
$$

因此约束右端项变为 $b-A\,lb$，目标常数增加 $c^Tlb$。有限上界变为 $x'_i\le ub_i-lb_i$。`VariableRecovery` 保存平移量、保留变量索引和固定变量索引，最终解由映射恢复，不由求解器猜测。

负 RHS 行整体乘以 $-1$，同时交换 `<=` 与 `>=`。`<=` 行加入松弛变量；`>=` 行加入系数为 $-1$ 的剩余变量和人工变量；`=` 行加入人工变量。

## 3. Phase I 与人工变量

Phase I 在内部最大化人工变量和的相反数：

$$
\max -\sum_j a_j.
$$

人工变量为对应 `>=` 和 `=` 行提供初始基。Phase I 最优后，若人工变量之和仍大于可行性容差，则原问题不可行。

值为零但仍在基中的人工变量会优先用索引最小的合法非人工列 pivot 换出。若该行移除人工列后只剩 $0=0$，则删除冗余行。随后删除全部人工列，并同步重映射 `basis_indices`。

## 4. Phase II

Phase II 不重新初始化基。程序清空 Phase I 目标行，用原目标系数和变量平移产生的目标常数重建目标行，再针对 Phase I 得到的可行基计算约化成本。

最终 `LPResult.x` 位于原变量空间，`objective_value` 按原始 `min/max` 方向计算。结果还记录 `phase_one_iterations`、`phase_two_iterations`、总迭代数、最终基、运行时间和可选迭代日志。

## 5. 运行

```zsh
.venv/bin/python -m pytest tests/test_lp_tableau_simplex.py tests/test_lp_two_phase_simplex.py
.venv/bin/python examples/two_phase_simplex_demo.py
```

当前仍不支持下界为负无穷的变量，包括真正自由变量和仅有上界的变量。修正单纯形、对偶单纯形、warm start、basis 继承、稀疏矩阵、Devex 和 scaling 不在本阶段范围内。
