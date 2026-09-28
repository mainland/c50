#!/usr/bin/env python3
"""Validate the benchmark schema and deterministic workload digests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def run_smoke_matrix(repository: Path) -> dict[str, Any]:
    """Run and parse the isolated smoke matrix.

    Args:
        repository: Repository root containing the benchmark scripts.

    Returns:
        Parsed matrix result.
    """
    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "benchmarks" / "run_matrix.py"),
            "--profile",
            "smoke",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result: dict[str, Any] = json.loads(completed.stdout)
    return result


def main() -> None:
    """Require stable digests and ordered resource checkpoints."""
    repository = Path(__file__).resolve().parents[1]
    first = run_smoke_matrix(repository)
    second = run_smoke_matrix(repository)

    assert first["matrix_profile"] == "smoke"
    assert len(first["workloads"]) == 1
    first_workload = first["workloads"][0]
    second_workload = second["workloads"][0]
    assert first_workload["model"]["stable_serialized_sha256"] == (
        second_workload["model"]["stable_serialized_sha256"]
    )
    assert first_workload["workload"]["prediction_sha256"] == (
        second_workload["workload"]["prediction_sha256"]
    )

    samples = first_workload["operation_wall_seconds"]
    assert {
        "model_load", "serialized_string_copy", "empty_prediction",
        "prediction_1_rows", "prediction_500_rows",
    } <= samples.keys()
    for values in samples.values():
        assert len(values) == first_workload["workload"]["operation_repetitions"]
        assert all(value >= 0 for value in values)

    checkpoints = first_workload["resources"]["peak_rss_checkpoints_bytes"]
    if all(value is not None for value in checkpoints.values()):
        assert checkpoints["after_data_generation"] <= checkpoints["after_training"]
        assert checkpoints["after_training"] <= checkpoints["after_prediction"]
        assert checkpoints["after_prediction"] <= checkpoints["after_operations"]
        assert first_workload["resources"]["peak_rss_bytes"] == checkpoints[
            "after_operations"
        ]


if __name__ == "__main__":
    main()
