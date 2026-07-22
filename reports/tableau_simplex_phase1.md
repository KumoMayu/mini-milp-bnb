# 表格单纯形 LP 后端：第一阶段

## 1. 与 active-set 的区别

`active-set` 枚举足够多的活跃约束组合，逐个求交点并检查可行性。表格单纯形从一个基本可行解出发，每次通过一次 pivot 移动到相邻基本可行解，不枚举全部候选顶点。

本阶段只求解：

$$
\max c^T x
$$

$$
Ax \le b,\qquad x \ge 0,\qquad b \ge 0.
$$

`sense="min"` 只通过目标函数变号复用同一张最大化表，返回值仍按调用者原始方向计算。

## 2. 表格约定

加入松弛变量后有 $Ax+s=b$。初始时原变量为非基变量且取零，松弛变量为基变量且取 $b$，因此 $b\ge0$ 直接给出初始基本可行解。

表格最后一行保存“约化成本的相反数”。若某个非基变量对应项小于 `-tolerance`，增大它可以改善最大化目标。默认用 Bland rule 选择索引最小的入基变量；出基变量由最小比值检验确定，只考虑 pivot 列中严格大于 `tolerance` 的方向，并用当前基变量索引处理并列。

一次 pivot 包含：

1. 用 pivot 元素归一化出基行；
2. 消去该列在其余约束行和目标行中的系数；
3. 将该行的基变量索引替换为入基变量索引。

`LPResult.x` 只包含原变量，不包含松弛变量；`basis_indices` 使用整张表的零基列索引，其中原变量在前、松弛变量在后。

## 3. 运行

单元测试：

```zsh
.venv/bin/python -m pytest tests/test_lp_tableau_simplex.py
```

逐次 pivot 演示：

```zsh
.venv/bin/python examples/tableau_simplex_demo.py
```

默认 `verbose=False`，测试与现有 benchmark 不打印 pivot 日志。

## 4. 当前边界

尚未支持负右端项、`>=` 约束、等式、自由变量、人工变量、Phase I / Phase II、一般上下界、warm start、对偶单纯形和稀疏矩阵。`lp_backends.py` 已登记 `tableau_simplex` 名称，但第一阶段只建议直接调用 `TableauSimplexSolver`；它没有成为 B&B 默认后端。

一般 LP 与两阶段法现由独立的 `TwoPhaseTableauSimplexSolver` 提供，参见 `reports/tableau_simplex_phase2.md`。第一阶段接口仍保持原行为。修正单纯形和 B&B 接入不在本阶段范围内。
