from __future__ import annotations

import csv

from benchmarks.run import main


def test_unified_benchmark_writes_latest_reports(tmp_path):
    rows = main(
        [
            "small",
            "--family",
            "numerical_lp",
            "--backend",
            "custom",
            "--report-dir",
            str(tmp_path),
        ]
    )

    assert len(rows) == 3
    assert {row["seed"] for row in rows} == {"0", "1", "2"}
    assert {row["backend"] for row in rows} == {"two_phase_simplex"}
    assert all(row["status"] for row in rows)

    csv_path = tmp_path / "benchmark_latest.csv"
    markdown_path = tmp_path / "benchmark_latest.md"
    assert csv_path.is_file()
    assert markdown_path.is_file()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        written_rows = list(csv.DictReader(handle))
    assert len(written_rows) == 3
    assert "按模型族汇总" in markdown_path.read_text(encoding="utf-8")
