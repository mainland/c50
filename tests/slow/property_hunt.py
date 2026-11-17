# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Run the dense property test on many Hypothesis examples.

The hunt draws examples from the strategy of `test_dense_properties.py` and
checks each with the same function as that test. It writes each example to a
log, flushed and synced, before it trains, so that a crash leaves the crashing
input as the last record. Under pytest, a segmentation fault kills the process
before Hypothesis prints its example. The hunt does not use the Hypothesis
database or `conftest.py`, and it draws the same examples again from the same
seed.

Run it through the `property-hunt` CMake target, or through
`tests/run_built_python.py`, which loads the extension of a build. The script
takes `--log`, `--examples`, and `--seed`, or `--replay LOG` to train the last
example in a log, which is the input of a crash or the failing example of a
failed check.

    python tests/run_built_python.py BUILD/python SCRIPT [--replay LOG]

The options default to the environment variables `C50_PROPERTY_HUNT_LOG`,
`C50_PROPERTY_HUNT_EXAMPLES`, and `C50_PROPERTY_HUNT_SEED`, and then to fixed
values. `C50_PROPERTY_HUNT_REPLAY` defaults `--replay`, so that the CMake
target, which sets the environment that a sanitizer build needs, can replay a
log. The target runs in the build directory, so name the log by an absolute
path or one relative to that directory.

The script refuses to run under `python -O`, `-OO`, or `PYTHONOPTIMIZE`, and
exits with status 2 before it opens a log. Optimized execution removes the
`assert` statements of the checks, so a hunt or a replay would pass on an
example that the test rejects.
"""

from __future__ import annotations

import argparse
import dataclasses
import faulthandler
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import IO, Any

import c50
import numpy as np
from hypothesis import HealthCheck, given, seed, settings

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from test_dense_properties import (
    Dataset,
    check_text_and_dense_equivalence,
    datasets,
)


def record(log: IO[str], dataset: Dataset) -> None:
    """Append one example to the log and force it to disk.

    Args:
        log: Open log, one JSON object per line.
        dataset: Example about to be trained.
    """
    fields: dict[str, Any] = {}
    for field in dataclasses.fields(dataset):
        value = getattr(dataset, field.name)
        if isinstance(value, np.ndarray):
            value = value.tolist()
            if value and field.name.startswith("dense_"):
                value = [
                    [
                        None if element != element else element
                        for element in row
                    ]
                    for row in value
                ]
        elif isinstance(value, c50.ModelKind):
            value = value.name
        fields[field.name] = value
    log.write(json.dumps(fields, allow_nan=False) + "\n")
    log.flush()
    os.fsync(log.fileno())


def load_last(path: Path) -> Dataset:
    """Read the last example of a log.

    Args:
        path: Log written by `record`.

    Returns:
        The last example, with the types that `datasets` draws. A JSON `null`
        in a dense array reads as NaN, as does the `NaN` token of older logs.
    """
    lines = path.read_text().splitlines()
    if not lines:
        raise SystemExit(f"{path} holds no example")
    fields = json.loads(lines[-1])
    for name in ("dense_training", "dense_cases"):
        fields[name] = np.asarray(
            [
                [np.nan if element is None else element for element in row]
                for row in fields[name]
            ],
            dtype=np.float64,
        )
    fields["class_indices"] = np.asarray(
        fields["class_indices"], dtype=np.uintp
    )
    fields["kind"] = c50.ModelKind[fields["kind"]]
    return Dataset(**fields)


def hunt(log_path: Path, examples: int, seed_value: int) -> int:
    """Check generated examples, logging each before it trains.

    Args:
        log_path: File that receives the examples, replacing its contents.
        examples: Number of examples to draw.
        seed_value: Hypothesis seed, which fixes the examples.

    Returns:
        The number of examples checked, which the log also holds. It is less
        than `examples` when Hypothesis exhausts the distinct examples.
    """
    checked = 0
    with log_path.open("w") as log:

        @seed(seed_value)
        @settings(
            deadline=None,
            max_examples=examples,
            database=None,
            suppress_health_check=list(HealthCheck),
        )
        @given(dataset=datasets())
        def run(dataset: Dataset) -> None:
            nonlocal checked
            record(log, dataset)
            checked += 1
            check_text_and_dense_equivalence(dataset)

        run()
    return checked


def main(argv: Sequence[str] | None = None) -> None:
    """Run the hunt, or replay the last example of a log.

    Args:
        argv: Command-line arguments, or None for `sys.argv`.
    """
    if sys.flags.optimize:
        print(
            "property_hunt.py: refusing to run with optimized Python "
            "(-O, -OO, or PYTHONOPTIMIZE), which removes the assertions "
            "of the checks",
            file=sys.stderr,
        )
        raise SystemExit(2)
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--log",
        type=Path,
        default=Path(
            os.environ.get("C50_PROPERTY_HUNT_LOG", "property-hunt.jsonl")
        ),
        help="log of examples, replaced by each hunt",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=int(os.environ.get("C50_PROPERTY_HUNT_EXAMPLES", "1000")),
        help="number of examples to draw",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("C50_PROPERTY_HUNT_SEED", "1")),
        help="Hypothesis seed",
    )
    parser.add_argument(
        "--replay",
        type=Path,
        metavar="LOG",
        default=(
            Path(os.environ["C50_PROPERTY_HUNT_REPLAY"])
            if os.environ.get("C50_PROPERTY_HUNT_REPLAY")
            else None
        ),
        help="train the last example of LOG and exit",
    )
    args = parser.parse_args(argv)

    faulthandler.enable()
    if args.replay is not None:
        check_text_and_dense_equivalence(load_last(args.replay))
        print(f"replayed the last example of {args.replay}", flush=True)
        return
    checked = hunt(args.log, args.examples, args.seed)
    print(
        f"checked {checked} examples of seed {args.seed}, "
        f"logged to {args.log}",
        flush=True,
    )


if __name__ == "__main__":
    main()
