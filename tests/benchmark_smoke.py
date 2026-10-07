#!/usr/bin/env python3
"""Run the benchmark smoke suite and check its result and report.

The suite builds the command-line program from this checkout and trains
with the installed or ``PYTHONPATH`` c50 package. With ``--build-python``, it
also builds the package's private Python core. It needs CMake, a C++ compiler,
NumPy, pandas, and scikit-learn, plus pybind11 and an LTO-capable compiler
for the Python build. It runs on POSIX systems only.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build-python", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="c50-benchmark-smoke-") as scratch:
        scratch_path = Path(scratch)
        output = scratch_path / "results" / "smoke.json"
        subprocess.run(
            [
                sys.executable,
                str(REPOSITORY / "benchmarks" / "run.py"),
                "--suite",
                "smoke",
                *([] if args.build_python else ["--installed-python"]),
                "--allow-dirty",
                "--jobs",
                "2",
                "--cache",
                str(scratch_path / "cache"),
                "--work",
                os.path.relpath(scratch_path / "work"),
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
        build = result["builds"]["c50"]
        assert not build["private_core"]
        assert "-flto" not in build["learner_compile_command"]
        if args.build_python:
            python_build = build["python_build"]
            assert python_build["private_core"]
            assert "-flto" in python_build["learner_compile_command"]
            assert "-fvisibility=hidden" in python_build["learner_compile_command"]
            assert "The Python extension's core was built separately" in report
            assert "-fvisibility=hidden" in report
            assert "installed_python_package" not in build
        else:
            assert "python_build" not in build
            assert "Python workloads used the installed `c50` package" in report
        # Results retained before the separate Python build remain readable.
        legacy = json.loads(output.read_text())
        legacy["builds"]["c50"].pop("python_build", None)
        legacy["builds"]["c50"].pop("private_core")
        legacy_path = scratch_path / "legacy.json"
        legacy_path.write_text(json.dumps(legacy))
        subprocess.run(
            [sys.executable, str(REPOSITORY / "benchmarks/report.py"), str(legacy_path)],
            check=True, capture_output=True, text=True,
        )
        # CI has no upstream archive, so check the reference columns with this
        # library's samples standing in for both reference builds.
        referenced = json.loads(output.read_text())
        for name, command in (("reference", "gcc -ffloat-store -O3"),
                              ("reference_o3", "gcc -O3")):
            referenced["builds"][name] = {
                "archive_sha256": "", "compiler": "gcc", "command": command,
                "program_sha256": "",
            }
            for row in referenced["comparisons"]:
                row["samples"][name] = row["samples"]["c50"]
        for plain in (True, False):
            if not plain:
                del referenced["builds"]["reference_o3"]
            referenced_path = scratch_path / "referenced.json"
            referenced_path.write_text(json.dumps(referenced))
            table = [
                line for line in subprocess.run(
                    [sys.executable, str(REPOSITORY / "benchmarks/report.py"),
                     str(referenced_path)],
                    check=True, capture_output=True, text=True,
                ).stdout.split("\n### ")[1].splitlines()
                if line.startswith("|")
            ]
            assert ("| C5.0 2.07 -O3, s |" in table[0]) == plain
            assert ("| Time ratio to -O3 |" in table[0]) == plain
            assert "| C5.0 2.07, s |" in table[0]
            assert len({line.count("|") for line in table}) == 1, table
            assert len(table) == 2 + len(referenced["comparisons"])
        portable = scratch_path / "portable.json"
        result["machine"].update(cpu=None, cpu_governor=None, memory_bytes=None)
        # A development version names its commit, which the report must omit.
        result["builds"]["c50"].pop("python_build", None)
        result["builds"]["c50"]["installed_python_package"] = (
            f"1.0.0a2.dev1+g{result['source']['commit'][:9]}.d20261007"
        )
        portable.write_text(json.dumps(result))
        portable_report = subprocess.run(
            [sys.executable, str(REPOSITORY / "benchmarks/report.py"), str(portable)],
            check=True, capture_output=True, text=True,
        ).stdout
        assert "an unreported CPU" in portable_report
        assert "frequency governor" not in portable_report
        assert "GiB of memory" not in portable_report
        assert "version `1.0.0a2.dev1`" in portable_report
        assert result["source"]["commit"][:7] not in portable_report
        pending = scratch_path / "pending.md"
        pending.write_text(
            "<!-- begin generated benchmark results -->\n\n"
            "No release result has been retained yet.\n\n"
            "<!-- end generated benchmark results -->\n"
        )
        pending_check = [sys.executable, str(REPOSITORY / "benchmarks/report.py"),
                         str(scratch_path / "missing.json"), "--check", str(pending)]
        subprocess.run(pending_check, check=True, capture_output=True, text=True)
        pending.write_text(pending.read_text().replace(
            "No release result has been retained yet.", "Measured results are here."
        ))
        stale = subprocess.run(pending_check, capture_output=True, text=True)
        assert stale.returncode == 1
        assert "is missing" in stale.stderr

    assert result["format_version"] == 2
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
    assert [e["classifier"] for e in result["estimator"]] == ["tree", "rules", "boost"]
    for row in result["estimator"]:
        assert row["same_predictions"], row["classifier"]
        for interface in ("native", "estimator"):
            (sample,) = row["samples"][interface]
            assert sample["fit_seconds"] > 0
            assert sample["predict_seconds"] > 0
    assert "### Python estimator and low-level interface" in report
    assert len(result["source"]["commit"]) == 40
    assert result["source"]["commit"][:7] not in report


if __name__ == "__main__":
    main()
