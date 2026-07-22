# Benchmark Latest

Suite: `core`

Limits: `max_nodes=200`, `max_lp_candidates=250000`.

Status `LIMIT` means the mini solver reached a configured node/candidate/time limit; the objective is the incumbent value when available, not a proven optimum.

```text
suite | run_type               | case                        | seed | units | solver          | backend    | presolve | status  | objective | global_bound | relative_gap | nodes | lp_solved | prune_infeasible | prune_bound | prune_integral | removed_rows | tightened_bounds | candidates_checked | time_sec | match_reference | note
------+------------------------+-----------------------------+------+-------+-----------------+------------+----------+---------+-----------+--------------+--------------+-------+-----------+------------------+-------------+----------------+--------------+------------------+--------------------+----------+-----------------+-----
core  | active_set_presolve_on | fixed_charge_block          |      |       | mini_active_set | active_set | on       | optimal | 28.5      | 28.5         | 0            | 7     | 10        | 1                | 3           | 0              | 16           | 21               | 9936               | 0.077268 | True            |
core  | active_set_presolve_on | unit_commitment_tiny        |      |       | mini_active_set | active_set | on       | optimal | 26        | 26           | 0            | 5     | 8         | 2                | 1           | 0              | 20           | 24               | 406                | 0.006296 | True            |
core  | active_set_presolve_on | production_expansion_binary |      |       | mini_active_set | active_set | on       | optimal | 31.5      | 31.5         | 0            | 9     | 12        | 2                | 2           | 1              | 19           | 26               | 9946               | 0.074918 | True            |
```
