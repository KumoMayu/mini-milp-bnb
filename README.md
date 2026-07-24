# 小型 MILP 分支定界求解器

这是一个用于学习求解器内部机制的小型 MILP 原型。项目包含自写 Branch-and-Bound、多个节点 LP 后端、矩阵预处理、搜索策略实验和学习型分支实验；它不以替代工业求解器为目标。

## 当前结构

- `solver/`：MILP 数据结构、自写 Branch-and-Bound、节点选择、分支策略和 LP 后端。
- `solver/lp_active_set.py`：枚举活跃约束组合的教学型 LP 后端。
- `solver/lp_tableau_simplex.py`：Phase 1 基础表格单纯形。
- `solver/lp_standard_form.py`、`solver/lp_two_phase_simplex.py`：一般 LP 标准化、Phase I / Phase II，以及可选 B&B 节点 LP 后端。
- `solver/lp_scipy_highs.py`：可选 SciPy-HiGHS 节点 LP 参考后端。
- `ml_branching/`：学习型分支的数据、训练、推理和评估代码。
- `examples/`、`benchmarks/`、`tests/`：案例、可重复实验和测试。
- `verification/`：可选 Gurobi 结果验证与性能对照；Gurobi 不参与自写 LP 实际求解。
- `reports/`：正式实验结果和方法说明。

## MILP 主接口

主建模形式为：

```text
min / max c_x^T x + c_y^T y
s.t.      A x + B y <= b
          lb_x <= x <= ub_x
          y in {0,1}
```

内部拼接为 `z=[x;y]`、`G=[A B]`，统一处理 `Gz<=b`。

```python
from solver import MILPProblem, solve_milp

problem = MILPProblem.from_blocks(
    c_x=c_x,
    c_y=c_y,
    A=A,
    B=B,
    b=b,
    x_lb=x_lb,
    x_ub=x_ub,
    sense="min",
)

result = solve_milp(problem)
print(result.simple_summary())
```

当前 B&B 默认 LP 后端仍是 `active_set`。`two_phase_simplex` 与
`scipy_highs` 均为显式选择的节点 LP 后端；前者完全使用自写 pivot，后者用于参考对照。

```python
result = solve_milp(
    problem,
    lp_backend="two_phase_simplex",
    max_lp_iterations=10_000,
)
```

## 自写 LP 求解器演进

```text
active-set 候选顶点枚举
  -> 基础表格单纯形
  -> 一般 LP 标准化
  -> Phase I / Phase II
  -> 作为可选节点 LP 后端接入 B&B
  -> 后续研究修正单纯形
```

Phase 1 求解 `Ax<=b, x>=0, b>=0`。Phase 2 使用独立的 `TwoPhaseTableauSimplexSolver`，不会调用 SciPy、HiGHS 或 Gurobi完成实际 pivot。

## Phase 2 支持范围

- `<=`、`>=`、`=` 约束和正负 RHS；
- 有限非零下界、有限上界和固定变量；
- 最大化与最小化；
- Phase I / Phase II 和原变量恢复；
- `optimal`、`infeasible`、`unbounded`、`iteration_limit`、`numerical_error`；
- Bland rule、统一 tolerance 和确定性运行。

尚未支持：

- `lb=-inf`、真正自由变量、仅有上界而无有限下界的变量；
- 修正单纯形、对偶单纯形、warm start、父子节点 basis 继承；
- 稀疏矩阵、scaling、Devex；
- 工业级数值稳定性和大规模稀疏 LP。

方法说明见 `reports/tableau_simplex_phase1.md`、`reports/tableau_simplex_phase2.md`
和 `reports/two_phase_simplex_bnb_integration.md`。

## 安装

核心依赖只有 NumPy 和 pytest：

```zsh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

可选 SciPy-HiGHS：

```zsh
python -m pip install -r requirements-optional.txt
```

学习型分支环境依赖见 `requirements-ml.txt`。`gurobipy` 仅用于可选验证，不写入核心依赖。

## 运行

核心案例：

```zsh
.venv/bin/python examples/fixed_charge_block.py
.venv/bin/python examples/unit_commitment_tiny.py
.venv/bin/python examples/production_expansion_binary.py
```

Phase 1、Phase 2 和全部测试：

```zsh
.venv/bin/python -m pytest tests/test_lp_tableau_simplex.py
.venv/bin/python -m pytest tests/test_lp_two_phase_simplex.py
.venv/bin/python -m pytest
.venv-ml/bin/python -m pytest
```

两个 pivot 日志演示：

```zsh
.venv/bin/python examples/tableau_simplex_demo.py
.venv/bin/python examples/two_phase_simplex_demo.py
```

Benchmark 入口：

```zsh
.venv/bin/python -m benchmarks.solver.run --suite core
.venv/bin/python -m benchmarks.solver.run --suite backends
.venv/bin/python -m benchmarks.solver.run --suite batch
.venv-ml/bin/python -m benchmarks.solver.compare_two_phase_bnb
```

后一个命令比较 active-set、two-phase tableau simplex、SciPy-HiGHS 与
optional Gurobi，使用预热后 3 次正式运行的中位数，输出到
`reports/two_phase_simplex_bnb_results.csv`。active-set candidates 与 simplex
iterations 是不同指标，报告中分列记录。

## 已验证结果

- Phase 1 + Phase 2：`32 passed`；
- 核心环境：`107 passed, 4 skipped`；
- ML/Gurobi 环境：`119 passed`；
- 随机一般 LP 与 SciPy-HiGHS：`20/20` 目标值一致。
- B&B 统一对比：two-phase、SciPy-HiGHS 与 Gurobi 在 `27/27` 个案例上目标值一致；
- `scaling_units_5`：active-set 达到候选数限制，two-phase、SciPy-HiGHS 与 Gurobi 均为 `42.656`。

这些结果用于当前小规模案例的回归验证，不代表工业规模性能或完整数值鲁棒性。
当前 tableau 每个节点都重新构造稠密表格，也没有 warm start 或 basis 继承；
下一步是受控实现 revised simplex，而不是继续扩展 tableau 功能面。
