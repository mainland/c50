#!/usr/bin/env python3
"""Run the benchmark smoke suite and check its result and report.

The suite builds the command-line program from this checkout and trains
with the installed or ``PYTHONPATH`` c50 package, so it needs CMake, a C++
compiler, NumPy, and pandas. It runs on POSIX systems only.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="c50-benchmark-smoke-") as scratch:
        scratch_path = Path(scratch)
        output = scratch_path / "smoke.json"
        subprocess.run(
            [
                sys.executable,
                str(REPOSITORY / "benchmarks" / "run.py"),
                "--suite",
                "smoke",
                "--installed-python",
                "--allow-dirty",
                "--jobs",
                "2",
                "--cache",
                str(scratch_path / "cache"),
                "--work",
                str(scratch_path / "work"),
                "--output",
                str(output),
            ],
            check=True,
        )
        result = json.loads(output.read_text())
        report = subprocess.run(
            [sys.executable, str(REPOSITORY / "benchmarks" / "report.py"), str(output)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    assert result["format_version"] == 1
    assert result["suite"] == "smoke"
    assert [c["classifier"] for c in result["comparisons"]] == ["tree", "rules", "boost"]
    for comparison in result["comparisons"]:
        assert comparison["same_classifier"], comparison["classifier"]
        (sample,) = comparison["samples"]["c50"]
        assert sample["wall_seconds"] > 0
        assert sample["peak_rss_bytes"] > 0
    assert [w["ties"] for w in result["workers"]] == ["reference", "stable"]
    for scaling in result["workers"]:
        assert scaling["same_classifier"], scaling["ties"]
        assert sorted(scaling["samples"]) == ["1", "2"]
    assert "| synthetic-2000x12-c2 (2,000 rows, 12 attributes, 3 classes) | tree |" in report
    assert "### Split workers and tie order" in report


if __name__ == "__main__":
    main()
