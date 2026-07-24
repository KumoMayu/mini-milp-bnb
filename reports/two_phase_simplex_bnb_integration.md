# Two-Phase Tableau Simplex 接入 B&B

## 1. 接口

B&B 将原问题统一为内部最大化节点 LP：

$$
\max\ c_{\mathrm{internal}}^Tz
\quad\text{s.t.}\quad Gz\le h,\quad lb^N\le z\le ub^N.
$$

`lp_backend="two_phase_simplex"` 通过统一 LP backend 接口调用自写
`TwoPhaseTableauSimplexSolver`。默认后端仍为 `active_set`；分支规则、
best-bound 节点队列、incumbent 更新和剪枝条件未改变。

## 2. 节点上下界与变量恢复

当前节点的 `node_lb/node_ub` 先进入可选矩阵预处理。预处理可删除固定变量、
冗余行并收紧界；剩余界交给一般 LP 标准化模块，按
$z=lb^N+z'$ 平移并把有限上界转为约束。分支固定的 $y=0$ 或 $y=1$ 会被消元，
LP 结束后按映射恢复完整原变量向量。

若关闭矩阵预处理，固定变量仍由 Phase 2 标准化模块直接消元和恢复。

## 3. 状态映射

| LP 状态 | B&B 处理 |
|---|---|
| `optimal` | 继续 bound、整数性和分支判断 |
| `infeasible` | 按不可行节点剪枝 |
| `candidate_limit` | 保留 active-set 限制状态并终止 |
| `iteration_limit` | 原样终止，不标为 optimal 或 infeasible |
| `numerical_error` | 原样终止，不标为 optimal 或 infeasible |
| `unsupported` / `lp_error` / `unbounded` | 原样终止，不作不可行剪枝 |

`MILPResult` 新增 LP backend 名称、累计 simplex 迭代数和节点 LP 累计时间。

## 4. 与 Active-Set 的区别

Active-set 枚举可能成为顶点的活跃约束组合；表格单纯形从一个基本可行解出发，
通过 pivot 在相邻基之间移动。CSV 将 `candidates_checked` 与
`simplex_iterations` 分列，两者不是同一工作量指标。

## 5. 实验设置

- 案例：3 个 core、`scaling_units_2..5`、20 个固定 seed batch 案例；
- 后端：active-set、two-phase tableau simplex、SciPy-HiGHS、Gurobi；
- B&B：`max_nodes=200`，矩阵预处理开启；
- active-set：`candidate_limit=250000`；
- two-phase：每个节点 `iteration_limit=10000`；
- 时间：每个案例/后端预热 1 次，正式运行 3 次，报告中位数；
- Gurobi：单线程，仅作完整 MIP 参考；
- 原始三次时间见 `reports/two_phase_simplex_bnb_results.csv`。

## 6. Core 与 Scaling 结果

时间均为 3 次正式运行的总时间中位数。`LP work` 对 active-set 表示候选数，
对 two-phase 表示 simplex 迭代数；其他后端不填该列。

| case | backend | status | objective | nodes | LP solved | LP work | LP time (s) | total time (s) | match Gurobi |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_charge_block | active_set | optimal | 28.5 | 7 | 10 | 9936 | 0.070606 | 0.071159 | True |
| fixed_charge_block | two_phase_simplex | optimal | 28.5 | 7 | 10 | 30 | 0.003878 | 0.004262 | True |
| fixed_charge_block | scipy_highs | optimal | 28.5 | 7 | 10 | - | 0.006709 | 0.007105 | True |
| fixed_charge_block | gurobi | optimal | 28.5 | 0 | - | - | - | 0.000207 | True |
| unit_commitment_tiny | active_set | optimal | 26 | 5 | 8 | 406 | 0.005099 | 0.005295 | True |
| unit_commitment_tiny | two_phase_simplex | optimal | 26 | 5 | 8 | 7 | 0.002610 | 0.002814 | True |
| unit_commitment_tiny | scipy_highs | optimal | 26 | 5 | 8 | - | 0.004683 | 0.004928 | True |
| unit_commitment_tiny | gurobi | optimal | 26 | 0 | - | - | - | 0.000139 | True |
| production_expansion_binary | active_set | optimal | 31.5 | 9 | 12 | 9946 | 0.066718 | 0.067247 | True |
| production_expansion_binary | two_phase_simplex | optimal | 31.5 | 9 | 12 | 29 | 0.004000 | 0.004424 | True |
| production_expansion_binary | scipy_highs | optimal | 31.5 | 9 | 12 | - | 0.007341 | 0.007819 | True |
| production_expansion_binary | gurobi | optimal | 31.5 | 1 | - | - | - | 0.000219 | True |
| scaling_units_2 | active_set | optimal | 18.399 | 5 | 8 | 406 | 0.004270 | 0.004447 | True |
| scaling_units_2 | two_phase_simplex | optimal | 18.399 | 5 | 8 | 7 | 0.001876 | 0.002054 | True |
| scaling_units_2 | scipy_highs | optimal | 18.399 | 5 | 8 | - | 0.003668 | 0.003871 | True |
| scaling_units_2 | gurobi | optimal | 18.399 | 0 | - | - | - | 0.000126 | True |
| scaling_units_3 | active_set | optimal | 28.488 | 5 | 8 | 9596 | 0.064334 | 0.064672 | True |
| scaling_units_3 | two_phase_simplex | optimal | 28.488 | 5 | 8 | 16 | 0.002937 | 0.003226 | True |
| scaling_units_3 | scipy_highs | optimal | 28.488 | 5 | 8 | - | 0.005092 | 0.005409 | True |
| scaling_units_3 | gurobi | optimal | 28.488 | 1 | - | - | - | 0.000314 | True |
| scaling_units_4 | active_set | optimal | 35.665 | 13 | 16 | 252274 | 1.647458 | 1.648321 | True |
| scaling_units_4 | two_phase_simplex | optimal | 35.665 | 13 | 16 | 44 | 0.006676 | 0.007274 | True |
| scaling_units_4 | scipy_highs | optimal | 35.665 | 13 | 16 | - | 0.011081 | 0.011792 | True |
| scaling_units_4 | gurobi | optimal | 35.665 | 1 | - | - | - | 0.000388 | True |
| scaling_units_5 | active_set | candidate_limit | 50.506 | 1 | 3 | 250462 | 1.698344 | 1.698419 | - |
| scaling_units_5 | two_phase_simplex | optimal | 42.656 | 9 | 12 | 69 | 0.007643 | 0.008210 | True |
| scaling_units_5 | scipy_highs | optimal | 42.656 | 9 | 12 | - | 0.010695 | 0.011336 | True |
| scaling_units_5 | gurobi | optimal | 42.656 | 1 | - | - | - | 0.000647 | True |

`scaling_units_5` 中 active-set 的 `50.506` 是触发候选上限时的 incumbent，
不是最优性证明。two-phase、SciPy-HiGHS 与 Gurobi 的已证明目标值均为 `42.656`。

## 7. 全部 27 个案例汇总

| backend | optimal | LIMIT | 与 Gurobi 一致 | 总时间中位数 (s) | LP 时间中位数 (s) |
|---|---:|---:|---:|---:|---:|
| active_set | 21 | 6 | 21/21 completed | 0.067247 | 0.066718 |
| two_phase_simplex | 27 | 0 | 27/27 | 0.004227 | 0.003878 |
| scipy_highs | 27 | 0 | 27/27 | 0.005796 | 0.005504 |
| gurobi | 27 | 0 | 27/27 | 0.000258 | - |

active-set 的 6 个限制为 `scaling_units_5` 和 5 个 `batch_units_5_seed_*`。
two-phase 没有触发 `iteration_limit` 或 `numerical_error`。本组极小模型中
two-phase 的总时间中位数低于 SciPy-HiGHS，但这不构成一般性能结论；HiGHS 的
固定调用开销在极小 LP 上占比较高。

## 8. 已知问题与下一步

- 每个 B&B 节点重新构建稠密 tableau；
- 没有 warm start 或父子节点 basis 继承；
- 没有 scaling、稀疏矩阵、Devex、对偶单纯形；
- 数值稳定性仍弱于成熟 LP 后端；
- 当前案例规模小，不能外推到大型 MILP。

当前接入已经摆脱 active-set 组合枚举，并在 27 个案例上完成统一正确性验证。
下一步适合进入 revised simplex 的受控实现；不应据此宣称接近工业求解器性能。
