# 小型 MILP 分支定界求解器

这是一个用于理解求解器机制的小型 MILP 原型。项目保留自写
Branch-and-Bound、矩阵预处理、分支与节点选择，以及多种节点 LP 后端；它不以
替代工业求解器为目标。

## 目录

- `solver/`：MILP 数据结构、B&B 主循环、预处理和 LP 后端。
- `benchmarks/`：统一的 LP/MILP 案例 registry 与 small/large 运行入口。
- `tests/unit/`、`tests/integration/`：模块测试和端到端一致性测试。
- `examples/`：tableau、two-phase、B&B 和机组组合示例。
- `reports/`：最新 benchmark 和关键阶段结果。
- `research/ml_branching/`：学习型分支研究代码、配置、结果和本地模型目录。
- `private/`：本地材料，保持 Git 忽略。

## 求解路径

MILP 主接口使用

```text
min / max c_x^T x + c_y^T y
s.t.      A x + B y <= b
```

其中 `x` 为连续变量，`y` 为整数或二元变量；内部统一为 `Gz <= b`。

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
result = solve_milp(problem, lp_backend="two_phase_simplex")
```

当前保留三条核心比较路径：

1. 自写 B&B + 自写 two-phase tableau simplex。
2. 自写 B&B + SciPy-HiGHS 节点 LP。
3. Gurobi 完整 LP/MILP，仅用于可选结果与性能对照。

`active_set` 仍是 B&B 的兼容默认后端和教学实现，但不参加 large 主性能表。

## 安装

Mac/Linux：

```zsh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-optional.txt
```

Windows：

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-optional.txt
```

`requirements-optional.txt` 只提供 SciPy-HiGHS。Gurobi 验证要求环境中已有可用的
`gurobipy` 和许可证；学习型分支环境另见 `requirements-ml.txt`。

## 运行

示例：

```zsh
.venv/bin/python examples/tableau_simplex_demo.py
.venv/bin/python examples/two_phase_simplex_demo.py
.venv/bin/python examples/branch_and_bound_demo.py
.venv/bin/python examples/unit_commitment_tiny.py
```

测试：

```zsh
.venv/bin/python -m pytest
.\.venv\Scripts\python.exe -m pytest
```

统一 benchmark：

```zsh
.venv/bin/python -m benchmarks.run small
.venv/bin/python -m benchmarks.run large
```

`small` 对全部模型族运行 seed 0、1、2，用于快速回归；`large` 对每族运行
seed 0，并启用 60 秒、节点数、simplex 迭代数和 tableau 内存限制。可用
`--family unit_commitment` 等参数筛选模型族。结果覆盖写入
`reports/benchmark_latest.csv` 和 `reports/benchmark_latest.md`。

## 当前边界

- tableau 使用稠密矩阵；
- 每个 B&B 节点冷启动，没有 warm start 或 basis 继承；
- 没有利用大规模稀疏结构；
- 数值稳定性有限；
- 不应把极小案例上的耗时外推为普遍快于 HiGHS 或 Gurobi。

算法阶段说明见 `reports/tableau_simplex_phase1.md`、
`reports/tableau_simplex_phase2.md` 和
`reports/two_phase_simplex_bnb_integration.md`。
