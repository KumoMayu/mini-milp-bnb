# Benchmark Latest

- 测试时间：`2026-07-24T20:49:50+08:00`
- 模式：`all`
- 环境：Python `3.14.5`，`macOS-26.5.1-arm64-arm-64bit-Mach-O`
- 完成状态与预期一致：`95/96`
- 与 Gurobi 状态/目标一致：`95/96`
- 首个限制/失败：`numerical_lp_large_seed_0` / `two_phase_simplex`: `numerical_error`，预期 `optimal`

```text
case                           | scale | backend               | status          | objective    | solve_time   | nodes | LP iter | gap | match
-------------------------------+-------+-----------------------+-----------------+--------------+--------------+-------+---------+-----+------
general_lp_small_seed_0        | small | two_phase_simplex     | optimal         | 74.03774273  | 0.076376708  | -     | 353     | -   | True
general_lp_small_seed_0        | small | scipy_highs           | optimal         | 74.03774273  | 0.319530458  | -     | 49      | -   | True
general_lp_small_seed_0        | small | gurobi                | optimal         | 74.03774273  | 0.004226042  | -     | 64      | 0   | True
general_lp_small_seed_1        | small | two_phase_simplex     | optimal         | 95.35974168  | 0.064841042  | -     | 297     | -   | True
general_lp_small_seed_1        | small | scipy_highs           | optimal         | 95.35974168  | 0.001795917  | -     | 68      | -   | True
general_lp_small_seed_1        | small | gurobi                | optimal         | 95.35974168  | 0.000714083  | -     | 70      | 0   | True
general_lp_small_seed_2        | small | two_phase_simplex     | optimal         | 73.1200852   | 0.089694667  | -     | 417     | -   | True
general_lp_small_seed_2        | small | scipy_highs           | optimal         | 73.1200852   | 0.001902917  | -     | 58      | -   | True
general_lp_small_seed_2        | small | gurobi                | optimal         | 73.1200852   | 0.001250084  | -     | 66      | 0   | True
dense_lp_small_seed_0          | small | two_phase_simplex     | optimal         | 102.0261474  | 0.098514167  | -     | 444     | -   | True
dense_lp_small_seed_0          | small | scipy_highs           | optimal         | 102.0261474  | 0.002150458  | -     | 35      | -   | True
dense_lp_small_seed_0          | small | gurobi                | optimal         | 102.0261474  | 0.000737250  | -     | 44      | 0   | True
dense_lp_small_seed_1          | small | two_phase_simplex     | optimal         | 86.67323218  | 0.056884125  | -     | 258     | -   | True
dense_lp_small_seed_1          | small | scipy_highs           | optimal         | 86.67323218  | 0.001939708  | -     | 28      | -   | True
dense_lp_small_seed_1          | small | gurobi                | optimal         | 86.67323218  | 0.000610917  | -     | 34      | 0   | True
dense_lp_small_seed_2          | small | two_phase_simplex     | optimal         | 98.48830401  | 0.084442708  | -     | 375     | -   | True
dense_lp_small_seed_2          | small | scipy_highs           | optimal         | 98.48830401  | 0.002111458  | -     | 43      | -   | True
dense_lp_small_seed_2          | small | gurobi                | optimal         | 98.48830401  | 0.000716208  | -     | 42      | 0   | True
numerical_lp_small_seed_0      | small | two_phase_simplex     | optimal         | -4.248206408 | 0.007406791  | -     | 84      | -   | True
numerical_lp_small_seed_0      | small | scipy_highs           | optimal         | -4.248206408 | 0.001310000  | -     | 22      | -   | True
numerical_lp_small_seed_0      | small | gurobi                | optimal         | -4.248206408 | 0.000311542  | -     | 22      | 0   | True
numerical_lp_small_seed_1      | small | two_phase_simplex     | infeasible      | -            | 0.000202542  | -     | 1       | -   | True
numerical_lp_small_seed_1      | small | scipy_highs           | infeasible      | -            | 0.000614750  | -     | 0       | -   | True
numerical_lp_small_seed_1      | small | gurobi                | infeasible      | -            | 0.000030500  | -     | 0       | 0   | True
numerical_lp_small_seed_2      | small | two_phase_simplex     | unbounded       | -            | 0.000102333  | -     | 0       | -   | True
numerical_lp_small_seed_2      | small | scipy_highs           | unbounded       | -            | 0.000651625  | -     | 0       | -   | True
numerical_lp_small_seed_2      | small | gurobi                | unbounded       | -            | 0.000042709  | -     | 0       | 0   | True
knapsack_small_seed_0          | small | bnb_two_phase_simplex | optimal         | 172          | 0.016274542  | 35    | 337     | 0   | True
knapsack_small_seed_0          | small | bnb_scipy_highs       | optimal         | 172          | 0.026624625  | 35    | -       | 0   | True
knapsack_small_seed_0          | small | gurobi                | optimal         | 172          | 0.005137541  | 1     | 1       | 0   | True
knapsack_small_seed_1          | small | bnb_two_phase_simplex | optimal         | 127          | 0.029506250  | 57    | 737     | 0   | True
knapsack_small_seed_1          | small | bnb_scipy_highs       | optimal         | 127          | 0.046437125  | 57    | -       | 0   | True
knapsack_small_seed_1          | small | gurobi                | optimal         | 127          | 0.000251834  | 1     | 1       | 0   | True
knapsack_small_seed_2          | small | bnb_two_phase_simplex | optimal         | 127          | 0.020857417  | 47    | 470     | 0   | True
knapsack_small_seed_2          | small | bnb_scipy_highs       | optimal         | 127          | 0.037141083  | 47    | -       | 0   | True
knapsack_small_seed_2          | small | gurobi                | optimal         | 127          | 0.000234166  | 1     | 1       | 0   | True
set_cover_small_seed_0         | small | bnb_two_phase_simplex | optimal         | 14.694       | 0.002647084  | 1     | 26      | 0   | True
set_cover_small_seed_0         | small | bnb_scipy_highs       | optimal         | 14.694       | 0.002312500  | 1     | -       | 0   | True
set_cover_small_seed_0         | small | gurobi                | optimal         | 14.694       | 0.000177958  | 1     | 4       | 0   | True
set_cover_small_seed_1         | small | bnb_two_phase_simplex | optimal         | 16.754       | 0.002625417  | 1     | 25      | 0   | True
set_cover_small_seed_1         | small | bnb_scipy_highs       | optimal         | 16.754       | 0.002293833  | 1     | -       | 0   | True
set_cover_small_seed_1         | small | gurobi                | optimal         | 16.754       | 0.000251042  | 1     | 5       | 0   | True
set_cover_small_seed_2         | small | bnb_two_phase_simplex | optimal         | 15.857       | 0.002401125  | 1     | 13      | 0   | True
set_cover_small_seed_2         | small | bnb_scipy_highs       | optimal         | 15.857       | 0.002574917  | 1     | -       | 0   | True
set_cover_small_seed_2         | small | gurobi                | optimal         | 15.857       | 0.000220042  | 1     | 7       | 0   | True
facility_location_small_seed_0 | small | bnb_two_phase_simplex | optimal         | 42.50750239  | 0.202718875  | 1     | 214     | 0   | True
facility_location_small_seed_0 | small | bnb_scipy_highs       | optimal         | 42.50750239  | 0.169451375  | 1     | -       | 0   | True
facility_location_small_seed_0 | small | gurobi                | optimal         | 42.50750239  | 0.000450917  | 1     | 52      | 0   | True
facility_location_small_seed_1 | small | bnb_two_phase_simplex | optimal         | 49.81120372  | 0.419543250  | 3     | 561     | 0   | True
facility_location_small_seed_1 | small | bnb_scipy_highs       | optimal         | 49.81120372  | 0.339417083  | 3     | -       | 0   | True
facility_location_small_seed_1 | small | gurobi                | optimal         | 49.81120372  | 0.004331959  | 1     | 99      | 0   | True
facility_location_small_seed_2 | small | bnb_two_phase_simplex | optimal         | 42.42159752  | 0.192492417  | 1     | 218     | 0   | True
facility_location_small_seed_2 | small | bnb_scipy_highs       | optimal         | 42.42159752  | 0.163150292  | 1     | -       | 0   | True
facility_location_small_seed_2 | small | gurobi                | optimal         | 42.42159752  | 0.000502708  | 1     | 58      | 0   | True
unit_commitment_small_seed_0   | small | bnb_two_phase_simplex | optimal         | 165.1462485  | 0.154248750  | 17    | 527     | 0   | True
unit_commitment_small_seed_0   | small | bnb_scipy_highs       | optimal         | 165.1462485  | 0.145010542  | 17    | -       | 0   | True
unit_commitment_small_seed_0   | small | gurobi                | optimal         | 165.1462485  | 0.000667875  | 1     | 14      | 0   | True
unit_commitment_small_seed_1   | small | bnb_two_phase_simplex | optimal         | 156.2981489  | 0.071577625  | 7     | 224     | 0   | True
unit_commitment_small_seed_1   | small | bnb_scipy_highs       | optimal         | 156.2981489  | 0.067535167  | 7     | -       | 0   | True
unit_commitment_small_seed_1   | small | gurobi                | optimal         | 156.2981489  | 0.000455000  | 1     | 12      | 0   | True
unit_commitment_small_seed_2   | small | bnb_two_phase_simplex | optimal         | 138.674215   | 0.250642125  | 33    | 837     | 0   | True
unit_commitment_small_seed_2   | small | bnb_scipy_highs       | optimal         | 138.674215   | 0.237166458  | 33    | -       | 0   | True
unit_commitment_small_seed_2   | small | gurobi                | optimal         | 138.674215   | 0.000821917  | 1     | 17      | 0   | True
lot_sizing_small_seed_0        | small | bnb_two_phase_simplex | optimal         | 154.853557   | 0.226453542  | 11    | 504     | 0   | True
lot_sizing_small_seed_0        | small | bnb_scipy_highs       | optimal         | 154.853557   | 0.209488334  | 11    | -       | 0   | True
lot_sizing_small_seed_0        | small | gurobi                | optimal         | 154.853557   | 0.000776709  | 1     | 23      | 0   | True
lot_sizing_small_seed_1        | small | bnb_two_phase_simplex | optimal         | 154.484297   | 0.375869083  | 21    | 756     | 0   | True
lot_sizing_small_seed_1        | small | bnb_scipy_highs       | optimal         | 154.484297   | 0.351902667  | 21    | -       | 0   | True
lot_sizing_small_seed_1        | small | gurobi                | optimal         | 154.484297   | 0.000874208  | 1     | 27      | 0   | True
lot_sizing_small_seed_2        | small | bnb_two_phase_simplex | optimal         | 128.003247   | 0.602342625  | 33    | 1387    | 0   | True
lot_sizing_small_seed_2        | small | bnb_scipy_highs       | optimal         | 128.003247   | 0.542898792  | 33    | -       | 0   | True
lot_sizing_small_seed_2        | small | gurobi                | optimal         | 128.003247   | 0.000856167  | 1     | 31      | 0   | True
general_lp_large_seed_0        | large | two_phase_simplex     | optimal         | 232.7841166  | 5.606694375  | -     | 6792    | -   | True
general_lp_large_seed_0        | large | scipy_highs           | optimal         | 232.7841166  | 0.005254916  | -     | 210     | -   | True
general_lp_large_seed_0        | large | gurobi                | optimal         | 232.7841166  | 0.005118708  | -     | 245     | 0   | True
dense_lp_large_seed_0          | large | two_phase_simplex     | optimal         | 244.9809904  | 1.646416167  | -     | 2556    | -   | True
dense_lp_large_seed_0          | large | scipy_highs           | optimal         | 244.9809904  | 0.006861333  | -     | 56      | -   | True
dense_lp_large_seed_0          | large | gurobi                | optimal         | 244.9809904  | 0.002757541  | -     | 64      | 0   | True
numerical_lp_large_seed_0      | large | two_phase_simplex     | numerical_error | -            | 0.477125416  | -     | 1735    | -   | False
numerical_lp_large_seed_0      | large | scipy_highs           | optimal         | -1.856269696 | 0.003888041  | -     | 79      | -   | True
numerical_lp_large_seed_0      | large | gurobi                | optimal         | -1.856269696 | 0.002368875  | -     | 76      | 0   | True
knapsack_large_seed_0          | large | bnb_two_phase_simplex | optimal         | 329          | 0.127774709  | 107   | 3636    | 0   | True
knapsack_large_seed_0          | large | bnb_scipy_highs       | optimal         | 329          | 0.093808708  | 107   | -       | 0   | True
knapsack_large_seed_0          | large | gurobi                | optimal         | 329          | 0.000317833  | 1     | 1       | 0   | True
set_cover_large_seed_0         | large | bnb_two_phase_simplex | optimal         | 15.144       | 0.126939584  | 11    | 1125    | 0   | True
set_cover_large_seed_0         | large | bnb_scipy_highs       | optimal         | 15.144       | 0.027519417  | 5     | -       | 0   | True
set_cover_large_seed_0         | large | gurobi                | optimal         | 15.144       | 0.002298708  | 1     | 18      | 0   | True
facility_location_large_seed_0 | large | bnb_two_phase_simplex | optimal         | 62.33122497  | 1.851971375  | 1     | 658     | 0   | True
facility_location_large_seed_0 | large | bnb_scipy_highs       | optimal         | 62.33122497  | 1.319294500  | 1     | -       | 0   | True
facility_location_large_seed_0 | large | gurobi                | optimal         | 62.33122497  | 0.000964958  | 1     | 118     | 0   | True
unit_commitment_large_seed_0   | large | bnb_two_phase_simplex | optimal         | 362.8789379  | 1.259805500  | 25    | 2188    | 0   | True
unit_commitment_large_seed_0   | large | bnb_scipy_highs       | optimal         | 362.8789379  | 1.096134666  | 25    | -       | 0   | True
unit_commitment_large_seed_0   | large | gurobi                | optimal         | 362.8789379  | 0.000755667  | 1     | 30      | 0   | True
lot_sizing_large_seed_0        | large | bnb_two_phase_simplex | optimal         | 282.56806    | 28.494588584 | 269   | 38152   | 0   | True
lot_sizing_large_seed_0        | large | bnb_scipy_highs       | optimal         | 282.56806    | 24.299470542 | 269   | -       | 0   | True
lot_sizing_large_seed_0        | large | gurobi                | optimal         | 282.56806    | 0.002583292  | 1     | 70      | 0   | True
```

## 按模型族汇总

```text
family            | backend               | expected_ok | LIMIT | median_sec
------------------+-----------------------+-------------+-------+-----------
dense_lp          | gurobi                | 4/4         | 0     | 0.000727
dense_lp          | scipy_highs           | 4/4         | 0     | 0.002131
dense_lp          | two_phase_simplex     | 4/4         | 0     | 0.091478
facility_location | bnb_scipy_highs       | 4/4         | 0     | 0.254434
facility_location | bnb_two_phase_simplex | 4/4         | 0     | 0.311131
facility_location | gurobi                | 4/4         | 0     | 0.000734
general_lp        | gurobi                | 4/4         | 0     | 0.002738
general_lp        | scipy_highs           | 4/4         | 0     | 0.003579
general_lp        | two_phase_simplex     | 4/4         | 0     | 0.083036
knapsack          | bnb_scipy_highs       | 4/4         | 0     | 0.041789
knapsack          | bnb_two_phase_simplex | 4/4         | 0     | 0.025182
knapsack          | gurobi                | 4/4         | 0     | 0.000285
lot_sizing        | bnb_scipy_highs       | 4/4         | 0     | 0.447401
lot_sizing        | bnb_two_phase_simplex | 4/4         | 0     | 0.489106
lot_sizing        | gurobi                | 4/4         | 0     | 0.000865
numerical_lp      | gurobi                | 4/4         | 0     | 0.000177
numerical_lp      | scipy_highs           | 4/4         | 0     | 0.000981
numerical_lp      | two_phase_simplex     | 3/4         | 0     | 0.000203
set_cover         | bnb_scipy_highs       | 4/4         | 0     | 0.002444
set_cover         | bnb_two_phase_simplex | 4/4         | 0     | 0.002636
set_cover         | gurobi                | 4/4         | 0     | 0.000236
unit_commitment   | bnb_scipy_highs       | 4/4         | 0     | 0.191088
unit_commitment   | bnb_two_phase_simplex | 4/4         | 0     | 0.202445
unit_commitment   | gurobi                | 4/4         | 0     | 0.000712
```

## LIMIT / Resource / Failure

- `numerical_lp_large_seed_0` / `two_phase_simplex`: `numerical_error`，预期 `optimal`

## 当前判断

当前首先暴露的是数值稳定性：自写 tableau 出现 numerical_error。

详细 build、LP、剪枝、残差和迭代字段见 `reports/benchmark_latest.csv`。
