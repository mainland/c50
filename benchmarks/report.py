#!/usr/bin/env python3
"""Render benchmark results as the tables in docs/performance.md.

Usage:
    python benchmarks/report.py RESULTS --update docs/performance.md
    python benchmarks/report.py RESULTS --check docs/performance.md
    python benchmarks/report.py RESULTS

The page holds the generated text between two marker comments. ``--update``
replaces it, ``--check`` fails when it differs from the results, and without
an option the text is printed.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

BEGIN = "<!-- begin generated benchmark results -->"
END = "<!-- end generated benchmark results -->"
REPOSITORY = Path(__file__).resolve().parents[1]
MIB = 1024 * 1024


def median(samples: list[dict[str, Any]], key: str) -> float:
    return float(statistics.median(sample[key] for sample in samples))


def dataset_label(record: dict[str, Any]) -> str:
    return (
        f"{record['name']} ({record['rows']:,} rows, "
        f"{record['attributes']} attributes, {record['classes']} classes)"
    )


def classifier_label(kind: str, trials: int) -> str:
    return {"tree": "tree", "rules": "rules", "boost": f"{trials} boosted trees"}[kind]


def yes_no(value: bool) -> str:
    return "yes" if value else "**no**"


def render(results: dict[str, Any], path: str) -> str:
    """Return the generated Markdown for one results file."""
    if results["format_version"] != 1:
        raise SystemExit(f"unsupported results format {results['format_version']}")
    datasets = {d["name"]: d for d in results["datasets"]}
    machine = results["machine"]
    builds = results["builds"]
    commit = results["source"]["commit"]
    lines = [
        f"These results come from `{path}`. They were measured at commit "
        f"`{commit[:12]}` on {results['started'][:10]} with an "
        f"{machine['cpu']} ({machine['logical_cpus']} logical CPUs, "
        f"`{machine['cpu_governor']}` frequency governor) and "
        f"{machine['memory_bytes'] / 2**30:.0f} GiB of memory, running "
        f"`{machine['platform']}`. Each time is the median of "
        f"{results['repeats']} runs.",
        "",
        f"The library's `c5.0` was built with {builds['c50']['cxx_compiler']}, "
        "using `" + " ".join(
            part
            for part in builds["c50"]["learner_compile_command"].split()
            if part.startswith("-O") or part in ("-ffloat-store", "-DNDEBUG")
        ) + "`.",
    ]
    if "reference" in builds:
        lines[-1] += (
            f" C5.0 Release 2.07 was built with {builds['reference']['compiler']}, "
            f"using `{builds['reference']['command']}`, as its Makefile builds "
            "the production program."
        )
    lines += ["", "### C5.0 Release 2.07 and this library", ""]
    if "reference" in builds:
        lines += [
            "| Dataset | Classifier | C5.0 2.07, s | c50, s | Time ratio "
            "| C5.0 2.07 peak, MiB | c50 peak, MiB | Same classifier |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    else:
        lines += [
            "| Dataset | Classifier | c50, s | c50 peak, MiB |",
            "| --- | --- | ---: | ---: |",
        ]
    for row in results["comparisons"]:
        label = dataset_label(datasets[row["dataset"]])
        kind = classifier_label(row["classifier"], results["boost_trials"])
        c50 = row["samples"]["c50"]
        c50_time = median(c50, "wall_seconds")
        c50_peak = median(c50, "peak_rss_bytes") / MIB
        if "reference" in row["samples"]:
            reference = row["samples"]["reference"]
            reference_time = median(reference, "wall_seconds")
            lines.append(
                f"| {label} | {kind} | {reference_time:.2f} | {c50_time:.2f} "
                f"| {c50_time / reference_time:.2f} "
                f"| {median(reference, 'peak_rss_bytes') / MIB:,.0f} "
                f"| {c50_peak:,.0f} | {yes_no(row['same_classifier'])} |"
            )
        else:
            lines.append(f"| {label} | {kind} | {c50_time:.2f} | {c50_peak:,.0f} |")

    counts = sorted({int(c) for row in results["workers"] for c in row["samples"]})
    lines += [
        "",
        "### Split workers and tie order",
        "",
        "| Dataset | Tie order | "
        + " | ".join(f"{c} worker{'s' if c > 1 else ''}, s" for c in counts)
        + f" | Speedup at {counts[-1]} | Same classifier for all worker counts "
        "| Same classifier as reference order |",
        "| --- | --- | " + " | ".join("---:" for _ in counts) + " | ---: | --- | --- |",
    ]
    reference_digests = {
        row["dataset"]: row["samples"]["1"][0]["classifier_sha256"]
        for row in results["workers"]
        if row["ties"] == "reference"
    }
    for row in results["workers"]:
        times = [median(row["samples"][str(c)], "training_seconds") for c in counts]
        digest = row["samples"]["1"][0]["classifier_sha256"]
        same_as_reference = (
            "--"
            if row["ties"] == "reference"
            else yes_no(digest == reference_digests.get(row["dataset"])).replace(
                "**no**", "no"
            )
        )
        lines.append(
            f"| {dataset_label(datasets[row['dataset']])} | {row['ties']} | "
            + " | ".join(f"{t:.2f}" for t in times)
            + f" | {times[0] / times[-1]:.2f} | {yes_no(row['same_classifier'])} "
            f"| {same_as_reference} |"
        )
    return "\n".join(lines) + "\n"


def replace_block(page: str, block: str) -> str:
    start = page.find(BEGIN)
    end = page.find(END)
    if start < 0 or end < start:
        raise SystemExit("the page has no generated-results markers")
    return page[: start + len(BEGIN)] + "\n\n" + block + "\n" + page[end:]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--update", type=Path, metavar="PAGE")
    action.add_argument("--check", type=Path, metavar="PAGE")
    args = parser.parse_args(argv)
    results = json.loads(args.results.read_text())
    try:
        path = args.results.resolve().relative_to(REPOSITORY).as_posix()
    except ValueError:
        path = args.results.name
    block = render(results, path)
    if args.update:
        args.update.write_text(replace_block(args.update.read_text(), block))
    elif args.check:
        page = args.check.read_text()
        if replace_block(page, block) != page:
            print(
                f"{args.check} does not match {path}; run "
                f"benchmarks/report.py {path} --update {args.check}",
                file=sys.stderr,
            )
            return 1
    else:
        sys.stdout.write(block)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
