#!/usr/bin/env python3
"""Run a repeatable matrix of isolated large-dataset benchmarks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Workload:
    """One named invocation of ``large_dataset.py``."""

    name: str
    arguments: tuple[str, ...]


SMOKE_WORKLOADS = (
    Workload(
        "smoke-mixed",
        (
            "--rows",
            "2000",
            "--features",
            "12",
            "--categorical-features",
            "2",
            "--prediction-rows",
            "500",
        ),
    ),
)

STANDARD_WORKLOADS = SMOKE_WORKLOADS + (
    Workload(
        "medium-continuous",
        (
            "--rows",
            "25000",
            "--features",
            "30",
            "--categorical-features",
            "0",
        ),
    ),
    Workload(
        "medium-mixed",
        (
            "--rows",
            "100000",
            "--features",
            "50",
            "--categorical-features",
            "5",
        ),
    ),
)

LARGE_WORKLOADS = STANDARD_WORKLOADS + (
    Workload(
        "target-envelope",
        (
            "--rows",
            "1000000",
            "--features",
            "100",
            "--categorical-features",
            "10",
        ),
    ),
)

PROFILES = {
    "smoke": SMOKE_WORKLOADS,
    "standard": STANDARD_WORKLOADS,
    "large": LARGE_WORKLOADS,
}


def parse_args() -> argparse.Namespace:
    """Parse matrix-runner arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILES),
        default="smoke",
        help="workload set to run",
    )
    return parser.parse_args()


def run_workload(script: Path, workload: Workload) -> dict[str, Any]:
    """Run one workload in a fresh process.

    Args:
        script: Path to ``large_dataset.py``.
        workload: Workload name and command-line arguments.

    Returns:
        Parsed benchmark result with its matrix case name.
    """
    completed = subprocess.run(
        [sys.executable, str(script), *workload.arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    result: dict[str, Any] = json.loads(completed.stdout)
    result["matrix_case"] = workload.name
    return result


def main() -> None:
    """Run the selected workload profile and emit one JSON document."""
    args = parse_args()
    script = Path(__file__).with_name("large_dataset.py")
    results = [
        run_workload(script, workload) for workload in PROFILES[args.profile]
    ]
    print(
        json.dumps(
            {"matrix_profile": args.profile, "workloads": results},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
