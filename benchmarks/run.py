#!/usr/bin/env python3
"""Run a benchmark suite and write one JSON result.

The suite answers three questions on the same input files:

* How does this library's ``c5.0`` program compare with the imported C5.0
  Release 2.07 program in time and peak memory, and do both write the same
  classifier?
* How does training time change with the number of split workers and the
  tie order, and is the classifier the same for every worker count?
* How much time and memory does ``C50Classifier`` add to low-level training
  and prediction, and does it predict the same classes?

The driver builds both programs itself, from the current checkout and from
the canonical upstream archive, and records their provenance with every
sample. See ``benchmarks/README.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import datasets
from datasets import Dataset, SyntheticSpec

FORMAT_VERSION = 2
REPOSITORY = Path(__file__).resolve().parents[1]
ARCHIVE_SHA256 = "309db588eda420c06701bf8ae74c06a6c923e9a06e714a598ea761bcadfc5e2e"
REFERENCE_CC = ["gcc", "-ffloat-store"]
REFERENCE_CFLAGS = ["-O3"]
WORKER_COUNTS = (1, 2, 4, 8)
TIE_ORDERS = ("reference", "stable")

# The launcher that measured() runs commands through. main() builds it.
LAUNCHER: Path | None = None


@dataclass(frozen=True)
class Comparison:
    """A dataset and the classifier kinds to train with both programs."""

    dataset: str | SyntheticSpec
    classifiers: tuple[str, ...]


@dataclass(frozen=True)
class Suite:
    """A named set of comparisons and worker-scaling workloads."""

    comparisons: tuple[Comparison, ...]
    worker_datasets: tuple[str | SyntheticSpec, ...]
    worker_counts: tuple[int, ...]
    repeats: int
    boost_trials: int
    needs_network: bool = field(default=False)


SCALE = SyntheticSpec(rows=1_000_000, features=100, categorical_features=10)

SUITES = {
    # A fast run of every code path, for CI and development.
    "smoke": Suite(
        comparisons=(
            Comparison(SyntheticSpec(2000, 12, 2), ("tree", "rules", "boost")),
        ),
        worker_datasets=(SyntheticSpec(20_000, 12, 4),),
        worker_counts=(1, 2),
        repeats=1,
        boost_trials=3,
    ),
    "release": Suite(
        comparisons=(
            Comparison("adult", ("tree", "rules", "boost")),
            Comparison("covertype", ("tree", "rules", "boost")),
            Comparison(SCALE, ("tree",)),
        ),
        worker_datasets=("covertype", SCALE),
        worker_counts=WORKER_COUNTS,
        repeats=5,
        boost_trials=10,
        needs_network=True,
    ),
}


def run(command: Sequence[str | Path], **kwargs: Any) -> str:
    """Run a command and return its standard output."""
    completed = subprocess.run(
        [str(part) for part in command],
        check=True,
        capture_output=True,
        text=True,
        **kwargs,
    )
    return str(completed.stdout)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def classifier_digest(text: str) -> str:
    """Hash a serialized classifier without its dated id line."""
    first, separator, rest = text.partition("\n")
    if not separator or not first.startswith("id="):
        raise ValueError("serialized classifier has no id line")
    return hashlib.sha256(rest.encode()).hexdigest()


def source_provenance(allow_dirty: bool) -> dict[str, Any]:
    """Describe the checked-out source that the suite measures."""
    commit = run(["git", "rev-parse", "HEAD"], cwd=REPOSITORY).strip()
    status = run(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPOSITORY
    )
    if status and not allow_dirty:
        raise SystemExit(
            "the working tree has uncommitted changes; commit them or pass "
            "--allow-dirty"
        )
    describe = run(["git", "describe", "--tags", "--always", "--dirty"], cwd=REPOSITORY)
    tags = run(["git", "tag", "--points-at", "HEAD"], cwd=REPOSITORY).split()
    return {
        "commit": commit,
        "describe": describe.strip(),
        "dirty": bool(status),
        "tag": tags[0] if len(tags) == 1 and not status else None,
    }


def machine_provenance() -> dict[str, Any]:
    """Describe the machine, including settings that affect timing noise."""

    def read(path: str) -> str | None:
        try:
            return Path(path).read_text().strip()
        except OSError:
            return None

    cpu = None
    memory = None
    cpuinfo = read("/proc/cpuinfo") or ""
    match = re.search(r"^model name\s*:\s*(.+)$", cpuinfo, re.M)
    if match:
        cpu = match.group(1)
    meminfo = read("/proc/meminfo") or ""
    match = re.search(r"^MemTotal:\s*(\d+) kB", meminfo, re.M)
    if match:
        memory = int(match.group(1)) * 1024
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": cpu or platform.processor(),
        "logical_cpus": os.cpu_count(),
        "memory_bytes": memory,
        "cpu_governor": read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "boost": read("/sys/devices/system/cpu/cpufreq/boost"),
        "python": platform.python_version(),
    }


def compiler_version(compiler: str) -> str:
    return run([compiler, "--version"]).splitlines()[0]


def build_record(build: Path) -> dict[str, Any]:
    """Record the compiler settings of a CMake learning-core build.

    Args:
        build: The configured and compiled CMake build directory.

    Returns:
        JSON-compatible compiler settings and the learner compile command.
    """
    cache = (build / "CMakeCache.txt").read_text()

    def setting(name: str) -> str:
        match = re.search(rf"^{name}:[A-Z]+=(.*)$", cache, re.M)
        return match.group(1) if match else ""

    # Record the command that compiled one learner source, because CMake adds
    # per-target options, such as -ffloat-store, that the cache omits.
    commands = json.loads((build / "compile_commands.json").read_text())
    learner = next(c for c in commands if c["file"].endswith("src/contin.cpp"))
    command = learner.get("command") or " ".join(learner["arguments"])
    command = command.replace(str(build.resolve()), "<build>")
    command = command.replace(str(REPOSITORY), "<source>")
    return {
        "build_type": setting("CMAKE_BUILD_TYPE"),
        "cxx_compiler": compiler_version(setting("CMAKE_CXX_COMPILER")),
        "learner_compile_command": command,
        "private_core": setting("C50_PRIVATE_CORE") == "ON",
    }


def build_library(work: Path, jobs: int, python_module: bool) -> dict[str, Any]:
    """Build the ordinary CLI and the package's private Python core separately.

    Args:
        work: Directory for builds and benchmark scratch files.
        jobs: Number of parallel build jobs.
        python_module: Whether to build Python instead of using its installation.

    Returns:
        The CLI path, optional Python package path, and both builds' provenance.
    """
    build = work / "build"
    configure: list[str | Path] = [
        "cmake",
        "-S",
        REPOSITORY,
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_TESTING=OFF",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        "-DCMAKE_UNITY_BUILD=OFF",
        "-UCMAKE_CXX_VISIBILITY_PRESET",
        "-DCMAKE_VISIBILITY_INLINES_HIDDEN=OFF",
        # Reset cached IPO overrides so reused work directories follow the
        # ordinary CLI and Python package policies, respectively.
        "-UCMAKE_INTERPROCEDURAL_OPTIMIZATION*",
    ]
    run([
        *configure, "-B", build,
        "-DC50_BUILD_PYTHON=OFF",
        "-DC50_PRIVATE_CORE=OFF",
        "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF",
    ])
    run(["cmake", "--build", build, "--parallel", str(jobs), "--target", "c50_cli"])
    program = build / "c5.0"
    record = build_record(build)
    record["program_sha256"] = sha256(program)
    python_path = None
    if python_module:
        import pybind11

        python_build = work / "python-build"
        run([
            *configure, "-B", python_build,
            "-DC50_BUILD_PYTHON=ON",
            "-DC50_PRIVATE_CORE=ON",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
        ])
        run([
            "cmake", "--build", python_build, "--parallel", str(jobs),
            "--target", "c50_python",
        ])
        record["python_build"] = build_record(python_build)
        python_path = python_build / "python"
        # Match the extension by its import suffix. The package also holds
        # the _c50.pyi stub.
        record["python_module_sha256"] = sha256(
            next(
                path
                for suffix in importlib.machinery.EXTENSION_SUFFIXES
                if (path := python_path / "c50" / f"_c50{suffix}").exists()
            )
        )
    else:
        c50 = import_c50()

        record["installed_python_package"] = importlib.metadata.version("c50")
        record["python_module_sha256"] = sha256(Path(c50._c50.__file__))
    return {"program": program, "python_path": python_path, "record": record}


def build_reference(archive: Path, work: Path) -> dict[str, Any]:
    """Build the imported program as its Makefile builds the production c5.0.

    The Makefile concatenates the sources into one file and compiles it with
    ``gcc -ffloat-store -O3``. Doing the same here avoids its ``csh``
    dependency.
    """
    digest = sha256(archive)
    if digest != ARCHIVE_SHA256:
        raise SystemExit(f"{archive} has SHA-256 {digest}, not {ARCHIVE_SHA256}")
    source = work / "reference"
    shutil.rmtree(source, ignore_errors=True)
    source.mkdir(parents=True)
    with tarfile.open(archive) as bundle:
        bundle.extractall(source, filter="data")
    makefile = (source / "Makefile").read_text()
    match = re.search(r"^src\s*=\\\n((?:\t.*\n)+)", makefile, re.M)
    if not match:
        raise SystemExit("cannot find the source list in the upstream Makefile")
    files = [line.strip().rstrip("\\").strip() for line in match.group(1).splitlines()]
    parts = [(source / "defns.i").read_text(errors="surrogateescape")]
    parts += [(source / name).read_text(errors="surrogateescape") for name in files]
    combined = "".join(parts)
    combined = "\n".join(
        line for line in combined.split("\n") if not re.search("defns.i|extern.i", line)
    )
    (source / "c50gt.c").write_text(combined, errors="surrogateescape")
    program = source / "c5.0"
    run(
        [*REFERENCE_CC, *REFERENCE_CFLAGS, "-o", program, source / "c50gt.c", "-lm"],
        cwd=source,
    )
    return {
        "program": program,
        "record": {
            "archive_sha256": digest,
            "compiler": compiler_version(REFERENCE_CC[0]),
            "command": " ".join([*REFERENCE_CC, *REFERENCE_CFLAGS]),
            "program_sha256": sha256(program),
        },
    }


def build_launcher(work: Path) -> Path:
    """Compile ``measure.c``, which starts each measured command."""
    launcher = work / "measure"
    run(["cc", "-O2", "-o", launcher, Path(__file__).with_name("measure.c")])
    return launcher


def measured(command: Sequence[str | Path], cwd: Path, env: dict[str, str] | None = None) -> dict[str, Any]:
    """Run a command and return its wall time, CPU time, and peak RSS.

    The command runs under the launcher, so that its peak RSS does not
    include the memory of this process.
    """
    if LAUNCHER is None:
        raise RuntimeError("the measurement launcher has not been built")
    load = os.getloadavg()[0]
    usage_path = cwd / "usage.json"
    with (cwd / "stdout.txt").open("w") as stdout:
        start = time.perf_counter()
        completed = subprocess.run(
            [str(LAUNCHER), str(usage_path), *(str(part) for part in command)],
            cwd=cwd,
            stdout=stdout,
            stderr=subprocess.PIPE,
            env=env,
        )
        wall = time.perf_counter() - start
    if completed.returncode != 0:
        raise RuntimeError(
            f"{command[0]} failed with status {completed.returncode}: "
            f"{completed.stderr.decode()}"
        )
    usage = json.loads(usage_path.read_text())
    return {
        "wall_seconds": wall,
        "user_seconds": usage["user_seconds"],
        "system_seconds": usage["system_seconds"],
        "peak_rss_bytes": usage["maxrss"] * (1 if sys.platform == "darwin" else 1024),
        "load_average": load,
    }


def classifier_arguments(kind: str, boost_trials: int) -> list[str]:
    return {"tree": [], "rules": ["-r"], "boost": ["-t", str(boost_trials)]}[kind]


def run_program(program: Path, dataset: Dataset, arguments: list[str], work: Path) -> dict[str, Any]:
    """Train with one command-line program in a fresh directory."""
    directory = Path(tempfile.mkdtemp(dir=work))
    try:
        stem = dataset.stem.name
        for suffix in (".names", ".data"):
            (directory / f"{stem}{suffix}").symlink_to(dataset.stem.with_suffix(suffix))
        sample = measured([program, "-f", stem, *arguments], directory)
        suffix = ".rules" if "-r" in arguments else ".tree"
        sample["classifier_sha256"] = classifier_digest(
            (directory / f"{stem}{suffix}").read_text()
        )
        return sample
    finally:
        shutil.rmtree(directory)


def compare(
    suite: Suite,
    comparison: Comparison,
    dataset: Dataset,
    programs: dict[str, Path],
    work: Path,
) -> list[dict[str, Any]]:
    """Alternate the programs on one dataset for each classifier kind."""
    results = []
    for kind in comparison.classifiers:
        arguments = classifier_arguments(kind, suite.boost_trials)
        samples: dict[str, list[dict[str, Any]]] = {name: [] for name in programs}
        for repeat in range(suite.repeats):
            order = list(programs)
            if repeat % 2:
                order.reverse()
            for name in order:
                progress(f"{dataset.name} {kind} {name} {repeat + 1}/{suite.repeats}")
                samples[name].append(
                    run_program(programs[name], dataset, arguments, work)
                )
        digests = {s["classifier_sha256"] for runs in samples.values() for s in runs}
        results.append(
            {
                "dataset": dataset.name,
                "classifier": kind,
                "arguments": arguments,
                "same_classifier": len(digests) == 1,
                "samples": samples,
            }
        )
    return results


# Set in child processes to the directory of the Python module that the
# driver built.
MODULE_DIRECTORY = "C50_BENCHMARK_MODULE_DIRECTORY"


def import_c50() -> Any:
    """Import ``c50`` from the module directory that the driver built.

    An editable install adds an import finder that runs before the path
    search, so ``PYTHONPATH`` alone does not select the built module. Drop
    every finder that would import ``c50`` from elsewhere.
    """
    directory = os.environ.get(MODULE_DIRECTORY)
    if directory:
        root = Path(directory).resolve()
        sys.path.insert(0, str(root))

        def elsewhere(finder: Any) -> bool:
            if finder is importlib.machinery.PathFinder:
                return False
            try:
                spec = finder.find_spec("c50", None)
            except (AttributeError, ImportError, TypeError, ValueError):
                return False
            origin = None if spec is None else spec.origin
            return origin is not None and not Path(origin).resolve().is_relative_to(root)

        sys.meta_path[:] = [finder for finder in sys.meta_path if not elsewhere(finder)]
    import c50
    import c50._c50

    return c50


def module_digest(c50: Any) -> str:
    """Hash the native extension that a child process loaded."""
    return sha256(Path(c50._c50.__file__))


def check_module(sample: dict[str, Any], expected: str) -> None:
    """Fail unless a child process loaded the module being measured."""
    loaded = sample.pop("python_module_sha256")
    if loaded != expected:
        raise SystemExit(
            f"a child process loaded the c50 extension {loaded[:12]}, "
            f"not the measured extension {expected[:12]}"
        )


def train_child(names: Path, data: Path, ties: str, workers: int) -> None:
    """Train once in this process and print the timing as JSON."""
    c50 = import_c50()

    names_text = names.read_text()
    data_text = data.read_text()
    options = c50.Options()
    options.ties = {"reference": c50.TieOrder.REFERENCE, "stable": c50.TieOrder.STABLE}[ties]
    start = time.perf_counter()
    model = c50.train(names_text, data_text, c50.ModelKind.TREE, options, "", workers)
    seconds = time.perf_counter() - start
    print(
        json.dumps(
            {
                "training_seconds": seconds,
                "classifier_sha256": classifier_digest(model.serialized_data),
                "python_module_sha256": module_digest(c50),
            }
        )
    )


def scale_workers(
    suite: Suite, dataset: Dataset, library: dict[str, Any], work: Path
) -> list[dict[str, Any]]:
    """Train through the library with each tie order and worker count."""
    env = dict(os.environ)
    if library["python_path"] is not None:
        env[MODULE_DIRECTORY] = str(library["python_path"])
    results = []
    for ties in TIE_ORDERS:
        samples: dict[str, list[dict[str, Any]]] = {
            str(count): [] for count in suite.worker_counts
        }
        for repeat in range(suite.repeats):
            counts = list(suite.worker_counts)
            if repeat % 2:
                counts.reverse()
            for count in counts:
                progress(f"{dataset.name} ties={ties} workers={count} {repeat + 1}/{suite.repeats}")
                directory = Path(tempfile.mkdtemp(dir=work))
                try:
                    sample = measured(
                        [
                            sys.executable,
                            Path(__file__).resolve(),
                            "_train",
                            dataset.names_path,
                            dataset.data_path,
                            ties,
                            str(count),
                        ],
                        directory,
                        env,
                    )
                    sample.update(json.loads((directory / "stdout.txt").read_text()))
                finally:
                    shutil.rmtree(directory)
                check_module(sample, library["record"]["python_module_sha256"])
                samples[str(count)].append(sample)
        digests = {s["classifier_sha256"] for runs in samples.values() for s in runs}
        results.append(
            {
                "dataset": dataset.name,
                "ties": ties,
                "same_classifier": len(digests) == 1,
                "samples": samples,
            }
        )
    return results


def labels_digest(labels: Sequence[str]) -> str:
    """Hash predicted class labels in case order."""
    return hashlib.sha256("\n".join(labels).encode()).hexdigest()


def native_child(stem: Path, kind: str, trials: int) -> None:
    """Train and predict once through the low-level interface, and print the
    timings as JSON."""
    c50 = import_c50()

    names_text = stem.with_suffix(".names").read_text()
    data_text = stem.with_suffix(".data").read_text()
    options = c50.Options()
    options.trials = trials
    model_kind = c50.ModelKind.RULES if kind == "rules" else c50.ModelKind.TREE
    start = time.perf_counter()
    model = c50.train(names_text, data_text, model_kind, options)
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    details = model.predict_details(data_text)
    predict_seconds = time.perf_counter() - start
    print(
        json.dumps(
            {
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "predictions_sha256": labels_digest(details.labels),
                "python_module_sha256": module_digest(c50),
            }
        )
    )


def estimator_child(stem: Path, kind: str, trials: int) -> None:
    """Fit and predict once with ``C50Classifier`` on a DataFrame, and print
    the timings as JSON."""
    c50 = import_c50()
    from c50.sklearn import C50Classifier

    X, y = datasets.load_frame(stem)
    classifier = C50Classifier(
        model_kind="rules" if kind == "rules" else "tree",
        trials=trials,
        categorical_features="from_dtype",
        n_jobs=1,
    )
    start = time.perf_counter()
    classifier.fit(X, y)
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    classifier.predict_proba(X)
    predict_seconds = time.perf_counter() - start
    labels = [str(label) for label in classifier.predict(X)]
    print(
        json.dumps(
            {
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "predictions_sha256": labels_digest(labels),
                "python_module_sha256": module_digest(c50),
            }
        )
    )


def compare_interfaces(
    suite: Suite,
    comparison: Comparison,
    dataset: Dataset,
    library: dict[str, Any],
    work: Path,
) -> list[dict[str, Any]]:
    """Alternate the low-level interface and the estimator on one dataset.

    Each sample runs in a fresh process, so its peak memory includes Python,
    the imported modules, and the input that the interface reads: the data
    text for ``c50.train`` and a DataFrame for ``C50Classifier``.
    """
    env = dict(os.environ)
    if library["python_path"] is not None:
        env[MODULE_DIRECTORY] = str(library["python_path"])
    interfaces = ("native", "estimator")
    results = []
    for kind in comparison.classifiers:
        trials = suite.boost_trials if kind == "boost" else 1
        samples: dict[str, list[dict[str, Any]]] = {name: [] for name in interfaces}
        for repeat in range(suite.repeats):
            order = list(interfaces)
            if repeat % 2:
                order.reverse()
            for name in order:
                progress(f"{dataset.name} {kind} {name} {repeat + 1}/{suite.repeats}")
                directory = Path(tempfile.mkdtemp(dir=work))
                try:
                    sample = measured(
                        [
                            sys.executable,
                            Path(__file__).resolve(),
                            f"_{name}",
                            dataset.stem,
                            kind,
                            str(trials),
                        ],
                        directory,
                        env,
                    )
                    sample.update(json.loads((directory / "stdout.txt").read_text()))
                finally:
                    shutil.rmtree(directory)
                check_module(sample, library["record"]["python_module_sha256"])
                samples[name].append(sample)
        digests = {s["predictions_sha256"] for runs in samples.values() for s in runs}
        results.append(
            {
                "dataset": dataset.name,
                "classifier": kind,
                "same_predictions": len(digests) == 1,
                "samples": samples,
            }
        )
    return results


def progress(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", choices=tuple(SUITES), default="smoke")
    parser.add_argument(
        "--reference-archive",
        type=Path,
        help="C50.tgz, the canonical C5.0 Release 2.07 GPL Edition archive; "
        "without it, the comparison runs only this library's program",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path.home() / ".cache" / "c50-benchmarks",
        help="directory for datasets (default: %(default)s)",
    )
    parser.add_argument("--work", type=Path, help="build and scratch directory")
    parser.add_argument("--output", type=Path, help="result file (default: stdout)")
    parser.add_argument("--repeats", type=int, help="override the suite's repeats")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument(
        "--installed-python",
        action="store_true",
        help="run the Python workloads with the installed c50 package "
        "instead of building the Python module from this checkout",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="measure a working tree with uncommitted changes",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    global LAUNCHER
    if argv[:1] == ["_train"]:
        names, data, ties, count = argv[1:]
        train_child(Path(names), Path(data), ties, int(count))
        return 0
    if argv[:1] in (["_native"], ["_estimator"]):
        stem, kind, trials = argv[1:]
        child = native_child if argv[0] == "_native" else estimator_child
        child(Path(stem), kind, int(trials))
        return 0
    args = parse_args(argv)
    suite = SUITES[args.suite]
    if args.repeats is not None:
        if args.repeats < 1:
            raise SystemExit("--repeats must be positive")
        suite = Suite(**{**suite.__dict__, "repeats": args.repeats})
    if args.suite == "release" and args.reference_archive is None:
        raise SystemExit("the release suite requires --reference-archive")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)

    source = source_provenance(args.allow_dirty)
    work = (args.work or Path(tempfile.mkdtemp(prefix="c50-benchmarks-"))).resolve()
    work.mkdir(parents=True, exist_ok=True)
    progress(f"building in {work}")
    LAUNCHER = build_launcher(work)
    library = build_library(work, args.jobs, not args.installed_python)
    programs = {"c50": library["program"]}
    builds: dict[str, Any] = {"c50": library["record"]}
    if args.reference_archive is not None:
        reference = build_reference(args.reference_archive, work)
        programs = {"reference": reference["program"], **programs}
        builds["reference"] = reference["record"]

    prepared: dict[str, Dataset] = {}

    def dataset(name: str | SyntheticSpec) -> Dataset:
        key = name if isinstance(name, str) else name.name
        if key not in prepared:
            progress(f"preparing {key}")
            prepared[key] = datasets.prepare(name, args.cache)
        return prepared[key]

    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    comparisons: list[dict[str, Any]] = []
    for comparison in suite.comparisons:
        comparisons += compare(
            suite, comparison, dataset(comparison.dataset), programs, work
        )
    workers: list[dict[str, Any]] = []
    for name in suite.worker_datasets:
        workers += scale_workers(suite, dataset(name), library, work)
    estimator: list[dict[str, Any]] = []
    for comparison in suite.comparisons:
        estimator += compare_interfaces(
            suite, comparison, dataset(comparison.dataset), library, work
        )

    result = {
        "format_version": FORMAT_VERSION,
        "suite": args.suite,
        "repeats": suite.repeats,
        "boost_trials": suite.boost_trials,
        "started": started,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": source,
        "machine": machine_provenance(),
        "builds": builds,
        "datasets": [d.record() for d in prepared.values()],
        "comparisons": comparisons,
        "workers": workers,
        "estimator": estimator,
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        sys.stdout.write(text)
    mismatches = [
        f"{c['dataset']} {c['classifier']}" for c in comparisons if not c["same_classifier"]
    ] + [f"{w['dataset']} ties={w['ties']}" for w in workers if not w["same_classifier"]]
    for mismatch in mismatches:
        progress(f"classifiers differ: {mismatch}")
    for row in estimator:
        if not row["same_predictions"]:
            mismatches.append(f"{row['dataset']} {row['classifier']} estimator")
            progress(
                f"predictions differ: {row['dataset']} {row['classifier']} estimator"
            )
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
