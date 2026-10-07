#!/usr/bin/env python3
"""Render benchmark results as the tables in docs/performance.md.

Usage:
    python benchmarks/report.py RESULTS --update docs/performance.md
    python benchmarks/report.py RESULTS --check docs/performance.md
    python benchmarks/report.py RESULTS

The page holds the generated text between two marker comments. ``--update``
replaces it, ``--check`` fails when it differs from the results, and without
an option the text is printed.
Before a release result exists, ``--check`` accepts only the unchanged
no-results placeholder.
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
PENDING = "No release result has been retained yet.\n"
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
    if results["format_version"] != 2:
        raise SystemExit(f"unsupported results format {results['format_version']}")
    datasets = {d["name"]: d for d in results["datasets"]}
    machine = results["machine"]
    builds = results["builds"]
    tag = results["source"].get("tag")
    release = f" at release `{tag}`" if tag else ""
    hardware = f"{machine['cpu'] or 'an unreported CPU'} ({machine['logical_cpus']} logical CPUs"
    if machine["cpu_governor"]:
        hardware += f", `{machine['cpu_governor']}` frequency governor"
    hardware += ")"
    if machine["memory_bytes"] is not None:
        hardware += f" and {machine['memory_bytes'] / 2**30:.0f} GiB of memory"
    lines = [
        f"These results come from `{path}`. They were measured on "
        f"{results['started'][:10]}{release} with {hardware}, running "
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
    # Results measured before the suite added the -O3 reference build lack it.
    plain = "reference_o3" in builds
    if plain:
        lines[-1] += (
            " The same source was also built with "
            f"`{builds['reference_o3']['command']}`, without `-ffloat-store`."
        )
    python_build = builds["c50"].get("python_build")
    if python_build is not None:
        flags = " ".join(
            part
            for part in python_build["learner_compile_command"].split()
            if part.startswith(("-O", "-flto", "-fvisibility"))
            or part in ("-ffloat-store", "-DNDEBUG", "-fno-fat-lto-objects")
        )
        lines += [
            "",
            "The Python extension's core was built separately with "
            f"{python_build['cxx_compiler']}, using `{flags}`.",
        ]
    elif "installed_python_package" in builds["c50"]:
        # A development version's local label, such as +g1a2b3c4, names a
        # commit, which a history rewrite would invalidate.
        version = builds["c50"]["installed_python_package"].split("+", 1)[0]
        lines += [
            "",
            "Python workloads used the installed `c50` package, version "
            f"`{version}`. Its compiler settings were not recorded.",
        ]
    lines += ["", "### C5.0 Release 2.07 and this library", ""]
    if "reference" in builds:
        lines += [
            "| Dataset | Classifier | C5.0 2.07, s "
            + ("| C5.0 2.07 -O3, s " if plain else "")
            + "| c50, s | Time ratio "
            + ("| Time ratio to -O3 " if plain else "")
            + "| C5.0 2.07 peak, MiB | c50 peak, MiB | Same classifier |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | "
            + ("---: | ---: | " if plain else "")
            + "--- |",
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
            times = [reference_time]
            ratios = [c50_time / reference_time]
            if plain:
                plain_time = median(row["samples"]["reference_o3"], "wall_seconds")
                times.append(plain_time)
                ratios.append(c50_time / plain_time)
            cells = [*times, c50_time, *ratios]
            lines.append(
                f"| {label} | {kind} | "
                + " | ".join(f"{cell:.2f}" for cell in cells)
                + f" | {median(reference, 'peak_rss_bytes') / MIB:,.0f} "
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

    lines += [
        "",
        "### Python estimator and low-level interface",
        "",
        "| Dataset | Classifier | c50.train, s | C50Classifier.fit, s | Fit ratio "
        "| predict_details, s | predict_proba, s | c50.train peak, MiB "
        "| C50Classifier peak, MiB | Same predictions |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in results["estimator"]:
        native = row["samples"]["native"]
        estimator = row["samples"]["estimator"]
        native_fit = median(native, "fit_seconds")
        estimator_fit = median(estimator, "fit_seconds")
        lines.append(
            f"| {dataset_label(datasets[row['dataset']])} "
            f"| {classifier_label(row['classifier'], results['boost_trials'])} "
            f"| {native_fit:.2f} | {estimator_fit:.2f} "
            f"| {estimator_fit / native_fit:.2f} "
            f"| {median(native, 'predict_seconds'):.2f} "
            f"| {median(estimator, 'predict_seconds'):.2f} "
            f"| {median(native, 'peak_rss_bytes') / MIB:,.0f} "
            f"| {median(estimator, 'peak_rss_bytes') / MIB:,.0f} "
            f"| {yes_no(row['same_predictions'])} |"
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
    if args.check and not args.results.exists():
        page = args.check.read_text()
        if replace_block(page, PENDING) == page:
            print(f"{args.check}: no release result has been retained yet")
            return 0
        print(f"{args.results} is missing but {args.check} reports results", file=sys.stderr)
        return 1
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
