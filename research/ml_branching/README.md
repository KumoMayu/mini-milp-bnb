# 学习型分支模块

`research/ml_branching/` 是自写 MILP Branch-and-Bound 求解器的可选研究模块。普通求解主线仍在 `solver/`，默认 `from solver import MILPProblem, solve_milp` 不依赖 Torch，也不会自动加载 GNN。

## 目录职责

| 目录 | 职责 |
|---|---|
| `runtime/` | GNN 求解时需要的代码，包括变量—约束图构造、图特征、GNN 模型、checkpoint 加载和 `BranchingPolicy` 适配器。 |
| `training/` | 训练和数据采集代码，包括强分支专家标签、数据集读取、loss、训练循环和 DAgger 数据聚合工具。普通求解不需要导入这里。 |
| `evaluation/` | 离线 top-1/top-3/regret 评估、solver 级策略比较、稳定性实验和报告汇总。 |
| `configs/` | 数据生成、GNN 训练、稳定性测试和 2×2 求解器对比配置，路径统一使用项目根目录相对路径。 |
| `data/generated/` | 已生成的数据集样本。当前主数据集是 `unit_commitment_round0`。 |

当前 checkpoint 统一放在：

```text
research/ml_branching/trained_models/learning_branching/
```

默认 GNN 推理模型是 `gnn_stability_seed_1.pt`，因为它在三个稳定性 checkpoint 中验证集 regret 和 validation loss 最低。`gnn_stability_seed_2.pt` 与 `gnn_stability_seed_3.pt` 用于稳定性对比；`run_12_lr_0.001_h_64_seed_3.pt` 是旧 MLP 基线，不是当前默认 GNN。

## 常用命令

普通求解不需要 ML 环境：

```zsh
.venv/bin/python examples/unit_commitment_tiny.py
.venv/bin/python -m pytest tests/test_branch_and_bound.py tests/test_lp_backends.py
```

GNN 求解需要 ML 环境和 checkpoint；下面是一个小型 smoke：

```zsh
.venv-ml/bin/python - <<'PY'
from examples.unit_commitment_tiny import build_problem
from solver import solve_milp
from research.ml_branching.runtime.inference import LearnedGNNBranchingPolicy

policy = LearnedGNNBranchingPolicy.from_checkpoint(
    "research/ml_branching/trained_models/learning_branching/gnn_stability_seed_1.pt"
)
result = solve_milp(build_problem(), branching_policy=policy, lp_backend="scipy_highs")
print(result.status, result.objective_value)
PY
```

训练入口保留在：

```zsh
.venv-ml/bin/python -m research.ml_branching.training.train_gnn \
  --config research/ml_branching/configs/stability_train.json
```

本阶段不继续训练模型，不重跑耗时 benchmark。现有结果见
`research/ml_branching/reports/GNN稳定性与通用性验证.md`。
