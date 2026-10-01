#!/usr/bin/env python3
"""Run the benchmark smoke suite and check its result.

The suite builds the command-line program from this checkout, so it needs
CMake, a C++ compiler, NumPy, and pandas. It runs on POSIX systems only.
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
        output = scratch_path / "results" / "smoke.json"
        subprocess.run(
            [
                sys.executable,
                str(REPOSITORY / "benchmarks" / "run.py"),
                "--suite",
                "smoke",
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

    assert result["format_version"] == 1
    assert result["suite"] == "smoke"
    assert [c["classifier"] for c in result["comparisons"]] == ["tree", "rules", "boost"]
    for comparison in result["comparisons"]:
        assert comparison["same_classifier"], comparison["classifier"]
        (sample,) = comparison["samples"]["c50"]
        assert sample["wall_seconds"] > 0
        assert sample["peak_rss_bytes"] > 0
    assert len(result["source"]["commit"]) == 40


if __name__ == "__main__":
    main()
