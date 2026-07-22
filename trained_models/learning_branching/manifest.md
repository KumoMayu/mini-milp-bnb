# Learning Branching Checkpoint Manifest

本目录是学习型分支策略的唯一 checkpoint 目录。本轮只记录用途，不移动、不复制、不删除模型文件。

| checkpoint | 类型 | seed | 主要验证指标 | 当前用途 |
|---|---|---:|---|---|
| `gnn_stability_seed_1.pt` | GNN | 1 | validation loss 0.932592；top-1 0.533333；top-3 0.900000；mean regret 0.165067 | 当前默认 GNN 推理模型；验证 regret 和 loss 最低 |
| `gnn_stability_seed_2.pt` | GNN | 2 | validation loss 0.991460；top-1 0.506667；top-3 0.853333；mean regret 0.212429 | GNN 稳定性对比 |
| `gnn_stability_seed_3.pt` | GNN | 3 | validation loss 1.019468；top-1 0.480000；top-3 0.873333；mean regret 0.208595 | GNN 稳定性对比 |
| `run_12_lr_0.001_h_64_seed_3.pt` | MLP | 3 | validation loss 3.478496；top-1 0.502890；top-3 0.832370；mean regret 0.224432 | 旧 MLP 基线；仍被 `stability_test.json` 记录，但不是当前默认 GNN |

当前 GNN checkpoint 的共同配置：

- checkpoint version: `learning_branching_gnn_checkpoint_v1`
- graph schema: `bipartite_graph_state_v1`
- variable feature dim: 11
- constraint feature dim: 7
- edge feature dim: 1
- global feature dim: 5
- hidden dim: 32
- message rounds: 2
- dataset: `ml_branching/data/generated/unit_commitment_round0`

默认使用规则：

```text
GNN runtime 默认选择 gnn_stability_seed_1.pt。
seed 2 和 seed 3 只用于稳定性结果复核。
旧 MLP checkpoint 保留为历史基线，不作为当前 GNN 分支策略入口。
```
