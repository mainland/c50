#!/usr/bin/env python3
# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Compare the library's c5.0 with C5.0 Release 2.07 on generated data.

Each case trains one generated dataset with the same options in both programs,
in separate empty directories. The comparison covers the exit status, the
console output, and the written `.tree` or `.rules` file. It normalizes only
what `tests/test_cli.sh` normalizes: the start time, the elapsed time, the date
of the model `id`, and trailing white space. Any other difference, including a
different exit status, fails the case.

usage: compare.py --reference REFC50 --candidate C50 [--only TEXT]
                  [--keep DIRECTORY] [--jobs N]
"""

from __future__ import annotations

import argparse
import difflib
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import generate

TIMEOUT_SECONDS = 300


@dataclass(frozen=True)
class Case:
    """One training run, repeated with each program."""

    dataset: str
    options: tuple[str, ...]
    costs: bool = False

    @property
    def name(self) -> str:
        text = " ".join(self.options) or "default"
        return f"{self.dataset}: {text}" + (" +costs" if self.costs else "")

    @property
    def model_extension(self) -> str | None:
        """The extension of the model file, or None without one."""
        if "-X" in self.options:
            return None
        return "rules" if "-r" in self.options else "tree"


def _cases() -> list[Case]:
    options_by_dataset: dict[str, list[tuple[str, ...]]] = {
        "mixed": [
            (),
            ("-r",),
            ("-b",),
            ("-t", "4"),
            ("-r", "-b", "-t", "3"),
            ("-s",),
            ("-s", "-r"),
            ("-s", "-b", "-t", "3"),
            ("-g",),
            ("-g", "-r"),
            ("-m", "1"),
            ("-m", "12"),
            ("-m", "30", "-r"),
            ("-c", "5"),
            ("-c", "25", "-r"),
            ("-c", "90"),
            ("-p",),
            ("-p", "-b", "-t", "3"),
            ("-S", "60", "-I", "7"),
            ("-S", "60", "-I", "7", "-r"),
            ("-S", "70", "-I", "9", "-t", "3"),
            ("-X", "4", "-I", "3"),
            ("-X", "3", "-I", "5", "-r"),
            ("-X", "3", "-I", "5", "-b", "-t", "2"),
            ("-r", "-u", "5"),
            ("-s", "-g", "-p"),
        ],
        "plain": [
            (),
            ("-r",),
            ("-w",),
            ("-w", "-r"),
            ("-w", "-b", "-t", "3"),
            ("-r", "-u", "3", "-w"),
            ("-s", "-w", "-m", "3", "-c", "40", "-r"),
            ("-s",),
            ("-b", "-t", "3"),
        ],
        "binary": [
            (),
            ("-r",),
            ("-b",),
            ("-t", "5", "-r"),
            ("-w",),
            ("-w", "-b", "-t", "3"),
            ("-s",),
            ("-g", "-m", "5"),
            ("-c", "10"),
            ("-c", "60", "-r"),
            ("-p",),
            ("-p", "-r"),
            ("-S", "50", "-I", "11"),
            ("-S", "80", "-I", "13", "-b", "-t", "3"),
            ("-X", "5", "-I", "17"),
            ("-X", "3", "-I", "2", "-b", "-t", "3"),
            ("-r", "-u", "4"),
        ],
        "discrete": [
            (),
            ("-r",),
            ("-b",),
            ("-t", "6", "-r"),
            ("-w",),
            ("-s",),
            ("-s", "-r"),
            ("-g",),
            ("-m", "2"),
            ("-c", "5"),
            ("-p",),
            ("-S", "60", "-I", "5"),
            ("-X", "4", "-I", "19"),
        ],
        # Bad data values. The three bad values of `fewbad` leave exit status
        # 0, and the twelve of `manybad` reach the error limit, which exits 1.
        "fewbad": [(), ("-r",), ("-b", "-t", "3")],
        "manybad": [(), ("-r",)],
        # Division whose precision multiplier wraps, as in Release 2.07.
        # `dividezero` stops at the error limit, and `dividenested` evaluates a
        # comparison to false for every case.
        "divide5000": [(), ("-r",), ("-b", "-t", "3")],
        "divideneg": [(), ("-r",), ("-b", "-t", "3")],
        "dividemin": [(), ("-r",)],
        "dividezero": [(), ("-r",)],
        "divideint": [(), ("-r",)],
        "dividenested": [(), ("-r",)],
        # A quotient that Release 2.07 rounds in double precision.
        "dividefloat": [(), ("-r",)],
        # Leaves of 22,203 cases whose error estimate loses digits in float.
        "pruneleaves": [(), ("-g",), ("-r",)],
        # A subset test whose case count has 8 digits, and timestamps far
        # from the base year, both of which print differently in float.
        "widenames": [("-s",)],
        "farstamps": [(), ("-r",)],
    }
    cases: list[Case] = []
    for dataset, option_lists in options_by_dataset.items():
        cases.extend(Case(dataset, options) for options in option_lists)
    # Costs apply to tree and rule models, with boosting, and with winnowing.
    for dataset in ("mixed", "plain", "binary"):
        for options in [
            (),
            ("-r",),
            ("-b", "-t", "3"),
            *([("-w",)] if dataset != "mixed" else []),
            ("-s", "-r"),
            ("-p",),
            ("-X", "3", "-I", "5"),
            ("-e",),
        ]:
            cases.append(Case(dataset, options, costs=True))
    return cases


_TIMESTAMP = re.compile(r"^C5\.0 \[Release 2\.07 GPL Edition\].*$")
_ELAPSED = re.compile(r"^Time: .* secs$")
_MODEL_ID = re.compile(r'^id="See5/C5\.0 2\.07 GPL Edition [0-9-][0-9-]*"$')


def normalize_output(text: str) -> list[str]:
    lines = []
    for line in text.split("\n"):
        line = _TIMESTAMP.sub(
            "C5.0 [Release 2.07 GPL Edition] <timestamp>", line
        )
        line = _ELAPSED.sub("Time: <elapsed> secs", line)
        lines.append(line.rstrip())
    return lines


def normalize_model(text: str) -> list[str]:
    return [
        _MODEL_ID.sub('id="See5/C5.0 2.07 GPL Edition <date>"', line)
        for line in text.split("\n")
    ]


@dataclass(frozen=True)
class Result:
    case: Case
    problems: list[str]
    directory: Path


def _run(
    program: Path, case: Case, directory: Path, source: Path
) -> list[str]:
    """Train in `directory` and return the normalized output and model."""
    directory.mkdir(parents=True)
    extensions = ["names", "data", "test"] + (["costs"] if case.costs else [])
    for extension in extensions:
        name = f"{case.dataset}.{extension}"
        shutil.copy(source / name, directory / name)
    completed = subprocess.run(
        [str(program), "-f", case.dataset, *case.options],
        cwd=directory,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=TIMEOUT_SECONDS,
        check=False,
    )
    sections = [
        f"exit status {completed.returncode}",
        "stdout:",
        *normalize_output(completed.stdout.decode("ascii", "replace")),
        "stderr:",
        *normalize_output(completed.stderr.decode("ascii", "replace")),
    ]
    model_extension = case.model_extension
    if model_extension is not None:
        model = directory / f"{case.dataset}.{model_extension}"
        sections.append(f"model {model.name}:")
        if model.exists():
            text = model.read_text(encoding="ascii", errors="replace")
            sections.extend(normalize_model(text))
        else:
            sections.append("(missing)")
    return sections


def _compare(
    index: int,
    case: Case,
    reference: Path,
    candidate: Path,
    source: Path,
    work: Path,
) -> Result:
    base = work / f"case-{index:03d}"
    expected = _run(reference, case, base / "reference", source)
    actual = _run(candidate, case, base / "candidate", source)
    problems: list[str] = []
    # The first line of each report is the exit status.
    if expected[0] != actual[0]:
        problems.append(
            f"{expected[0]} in Release 2.07, but {actual[0]} in the library"
        )
    problems += list(
        difflib.unified_diff(
            expected,
            actual,
            "Release 2.07",
            "library",
            lineterm="",
            n=2,
        )
    )
    return Result(case, problems, base)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--only", default="", help="run cases containing TEXT")
    parser.add_argument("--keep", type=Path, help="keep files in DIRECTORY")
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args(argv[1:])

    cases = [case for case in _cases() if args.only in case.name]
    if not cases:
        print("no case matches", file=sys.stderr)
        return 2
    programs = [args.reference.resolve(), args.candidate.resolve()]
    for program in programs:
        if not program.is_file():
            print(f"missing program: {program}", file=sys.stderr)
            return 2

    with tempfile.TemporaryDirectory(prefix="c50-compare-") as scratch:
        work = args.keep.resolve() if args.keep else Path(scratch)
        work.mkdir(parents=True, exist_ok=True)
        source = work / "data"
        generate.write(source)
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            futures = [
                pool.submit(
                    _compare,
                    index,
                    case,
                    programs[0],
                    programs[1],
                    source,
                    work,
                )
                for index, case in enumerate(cases)
            ]
            results = [future.result() for future in futures]

        failures = [result for result in results if result.problems]
        for result in failures:
            print(f"DIFFERENT: {result.case.name}")
            for line in result.problems[:60]:
                print(f"    {line}")
            print(f"    files: {result.directory}")
        print(f"{len(results) - len(failures)} of {len(results)} cases match")
        if failures and not args.keep:
            print(
                "Rerun with --keep DIRECTORY to retain the files.",
                file=sys.stderr,
            )
        return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
