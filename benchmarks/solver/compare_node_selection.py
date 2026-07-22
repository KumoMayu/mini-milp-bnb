from __future__ import annotations

import csv
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]
DFS_CSV=ROOT/"legacy2/before_best_bound_20260711/outputs/benchmark_latest_dfs.csv"
BEST_BOUND_CSV=ROOT/"reports/benchmark_latest.csv"
OUT_CSV=ROOT/"reports/节点选择算法对比.csv"
OUT_MD=ROOT/"reports/节点选择算法对比.md"

HEADERS=[
    "数据集",
    "案例",
    "seed",
    "units",
    "算法",
    "状态",
    "目标值",
    "节点数",
    "LP求解次数",
    "不可行剪枝数",
    "界剪枝数",
    "整数剪枝数",
    "全局界",
    "相对gap",
    "运行时间",
    "是否与Gurobi一致",
    "备注",
]


def read_rows(path: Path) -> list[dict[str,str]]:
    with path.open(newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))


def algorithm_name(prefix: str,row: dict[str,str]) -> str:
    backend=row.get("backend","")
    if backend=="gurobi":
        return "Gurobi参考"
    if backend=="active_set":
        return f"{prefix}+active-set"
    if backend=="scipy_highs":
        return f"{prefix}+SciPy-HiGHS"
    return f"{prefix}+{backend}"


def convert(prefix: str,row: dict[str,str]) -> dict[str,str]:
    objective=row.get("objective","")
    is_gurobi=row.get("backend")=="gurobi" and row.get("status")=="optimal"
    return {
        "数据集": row.get("suite",""),
        "案例": row.get("case",""),
        "seed": row.get("seed",""),
        "units": row.get("units",""),
        "算法": algorithm_name(prefix,row),
        "状态": row.get("status",""),
        "目标值": objective,
        "节点数": row.get("nodes",""),
        "LP求解次数": row.get("lp_solved",""),
        "不可行剪枝数": row.get("prune_infeasible",""),
        "界剪枝数": row.get("prune_bound",""),
        "整数剪枝数": row.get("prune_integral",""),
        "全局界": row.get("global_bound","") or (objective if is_gurobi else ""),
        "相对gap": row.get("relative_gap","") or ("0" if is_gurobi else ""),
        "运行时间": row.get("time_sec",""),
        "是否与Gurobi一致": row.get("match_reference",""),
        "备注": row.get("note",""),
    }


def make_rows() -> list[dict[str,str]]:
    rows=[]
    for row in read_rows(DFS_CSV):
        if row.get("backend")!="gurobi":
            rows.append(convert("DFS",row))
    for row in read_rows(BEST_BOUND_CSV):
        if row.get("backend")=="gurobi":
            rows.append(convert("",row))
        else:
            rows.append(convert("best-bound",row))
    return rows


def write_csv(rows: list[dict[str,str]]) -> None:
    with OUT_CSV.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str,str]]) -> None:
    widths={header:max(len(header),*(len(row[header]) for row in rows)) for header in HEADERS}
    lines=[
        "# 节点选择算法对比",
        "",
        "本报告由真实运行结果生成。DFS 数据来自修改前基线，best-bound 数据来自本轮运行后的 benchmark。",
        "",
        "说明：LIMIT 行的目标值只是已有 incumbent，不代表已证明最优。active-set 在 candidate limit 触发时没有有效 LP 全局界，因此全局界和 gap 留空。",
        "",
        "```text",
        " | ".join(header.ljust(widths[header]) for header in HEADERS),
        "-+-".join("-"*widths[header] for header in HEADERS),
    ]
    for row in rows:
        lines.append(" | ".join(row[header].ljust(widths[header]) for header in HEADERS))
    lines.extend(["```",""])

    dfs=[row for row in rows if row["算法"].startswith("DFS")]
    best=[row for row in rows if row["算法"].startswith("best-bound")]
    dfs_done=[row for row in dfs if row["状态"]=="optimal" and row["是否与Gurobi一致"]]
    best_done=[row for row in best if row["状态"]=="optimal" and row["是否与Gurobi一致"]]
    lines.extend(
        [
            "## 汇总",
            "",
            f"- DFS 已完成且可比对行数：{len(dfs_done)}，其中与 Gurobi 一致：{sum(row['是否与Gurobi一致']=='True' for row in dfs_done)}。",
            f"- best-bound 已完成且可比对行数：{len(best_done)}，其中与 Gurobi 一致：{sum(row['是否与Gurobi一致']=='True' for row in best_done)}。",
            f"- DFS LIMIT 行数：{sum(row['状态']=='LIMIT' for row in dfs)}。",
            f"- best-bound LIMIT 行数：{sum(row['状态']=='LIMIT' for row in best)}。",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines),encoding="utf-8")


def main() -> None:
    rows=make_rows()
    write_csv(rows)
    write_markdown(rows)
    print(OUT_CSV)
    print(OUT_MD)
    print(f"rows={len(rows)}")


if __name__=="__main__":
    main()
