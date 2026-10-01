# Building and testing

## Support and validation matrix

The native API requires C++17. Installed consumers are compiled as C++17 and
C++20. The Python package requires Python 3.10 or newer. The estimator's
minimum dependencies are joblib 1.3, NumPy 1.26, and scikit-learn 1.6.

| Target | Compiler or dependencies | Validation status |
| --- | --- | --- |
| Linux x86-64 | GCC 13, Clang 18 | Locally validated native, installed-consumer, and sanitizer checks |
| Linux x86-64, Python 3.12 | NumPy 2.x, scikit-learn 1.9 | Locally validated Python and type checks |
| Linux ARM64 | GCC on `ubuntu-24.04-arm` | Hosted native and installed-consumer checks passed at `ff5b953` |
| macOS ARM64 | AppleClang on `macos-14` | Hosted native and installed-consumer checks passed at `ff5b953` |
| Linux x86-64, Python 3.10 | NumPy 1.26.4, pandas 2.2.3, scikit-learn 1.6.1 | Locally validated installed package, runtime tests, and typing |
| Linux x86-64, Python 3.11 | NumPy 2.4.6, scikit-learn 1.9.1 | Locally validated installed package, runtime tests, and typing |
| Linux x86-64, Python 3.13 and 3.14 | NumPy 2.5.3, scikit-learn 1.9.1 | Locally validated installed wheels, runtime tests, and typing |
| Windows x86-64, Python 3.12 | MSVC 19.44 on `windows-2022` | Hosted native, installed-consumer, shell regression, and Python checks passed |

The [hosted validation run at `ff5b953`](https://github.com/mainland/c50/actions/runs/36495390708)
passed native checks with GCC and Clang on Linux x86-64, GCC on Linux ARM64,
and AppleClang on macOS ARM64. It also passed all four Python dependency
configurations and the AFL++ seed replay. The
[hosted sanitizer run at `ff5b953`](https://github.com/mainland/c50/actions/runs/36495390702)
passed the address, undefined-behavior, and leak configuration and the
ThreadSanitizer configuration. The macOS build previously failed because its
math library does not provide `finite()`. Those calls now use C++
`std::isfinite`. Free-threaded Python, GPU arrays, and other compiler or
platform combinations are unverified.
The Linux-only allocation-injection and ELF writable-global inventory checks
are omitted on macOS and Windows. Portable contract and installed-consumer tests
still run.

A [hosted Windows run](https://github.com/mainland/c50/actions/runs/36728151998) built the library, programs, and Python package
with MSVC at warning level 4 and passed all 18 CTest tests and the Python test
suite. The reference-compatibility digests and the CLI, report, and prediction
fixtures matched without Windows-specific exceptions. MSVC still reports about
250 warnings, mostly narrowing conversions in the imported sources.

Native CI configures a C++14 project default to verify that each target declares
its own C++17 requirement. The platform labels follow the
[GitHub-hosted runner reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).

## Windows

Build with Visual Studio 2022 or newer and CMake. The shell regression tests
run under the POSIX `sh` from [Git for Windows](https://gitforwindows.org),
which CMake finds in its default installation directory or on `PATH`:

```sh
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build --build-config Release --output-on-failure
```

`.gitattributes` checks out text files with LF line endings on every platform,
so the fixtures compare byte for byte. The programs write model files in
binary mode, so `.tree` and `.rules` files are identical to those written on
other platforms. Console output and `-o` output files use Windows CRLF line
endings, and the shell tests remove carriage returns before comparing them.
The root `Makefile` and the `xval` script are not supported on Windows.

## Legacy Make build

The compatibility build requires C and C++17 compilers and `make`:

```sh
make
make test
```

It produces `c5.0`, the training and evaluation program, and `report`, the
cross-validation report generator.

The installed `C50::cpp` CMake target links the compiled C++17 core. Manual
link commands must use a C++ driver and link the platform's thread and math
libraries as required. The install-consumer test builds standalone C++17 and
C++20 programs using only the installed public interface.

## GCC

Select GCC explicitly when configuring CMake:

```sh
cmake -S . -B build/gcc \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++
cmake --build build/gcc
ctest --test-dir build/gcc --output-on-failure
```

The `reference-compatibility` test checks deterministic classifier and
prediction digests captured with the imported RuleQuest sorting algorithms. See
{doc}`compatibility` for its scope and baseline-update policy.

A `VerbOpt` build also checks split-diagnostic order and the resulting
classifier. Configure the flag for C++ sources, then run the focused test:

```sh
cmake -S . -B build/verbose -DCMAKE_CXX_FLAGS=-DVerbOpt
cmake --build build/verbose
ctest --test-dir build/verbose -R cli-verbose-regression --output-on-failure
```

## Clang

Select Clang explicitly with both compiler settings:

```sh
cmake -S . -B build/clang \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++
cmake --build build/clang
ctest --test-dir build/clang --output-on-failure
```

## AddressSanitizer and UndefinedBehaviorSanitizer

`C50_ENABLE_SANITIZERS` enables AddressSanitizer and
UndefinedBehaviorSanitizer together:

```sh
cmake -S . -B build/sanitize \
    -DCMAKE_BUILD_TYPE=Debug \
    -DC50_ENABLE_SANITIZERS=ON
cmake --build build/sanitize
ASAN_OPTIONS=detect_leaks=0 \
    ctest --test-dir build/sanitize --output-on-failure
```

Leak detection is disabled in the example because ptrace restrictions prevent
LeakSanitizer from running reliably in some containers.

## LeakSanitizer

On a host where the AddressSanitizer runtime supports LeakSanitizer, reuse the
sanitizer build and enable leak detection explicitly:

```sh
ASAN_OPTIONS=detect_leaks=1 \
    ctest --test-dir build/sanitize --output-on-failure
```

LeakSanitizer runs through the AddressSanitizer configuration. It does not
require a separate CMake option.

## ThreadSanitizer

ThreadSanitizer is a separate configuration and cannot be combined with the
other sanitizer option:

```sh
cmake -S . -B build/tsan \
    -DCMAKE_BUILD_TYPE=Debug \
    -DC50_ENABLE_THREAD_SANITIZER=ON
cmake --build build/tsan
ctest --test-dir build/tsan --output-on-failure
```

Some Linux address-space layouts cause GCC ThreadSanitizer to terminate with an
`unexpected memory mapping` diagnostic. Disable address randomization for the
test process on those hosts:

```sh
setarch "$(uname -m)" -R \
    ctest --test-dir build/tsan --output-on-failure
```

## Python tests and distributions

### Installed package

Install the test extra into a Python 3.10-or-newer environment and run pytest:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install '.[test]'
.venv/bin/python -m pytest
```

The tutorial test runs the code in {doc}`tutorial` and checks the results that
its text reports. It downloads two datasets from OpenML on first use, so it is
skipped unless `C50_NETWORK_TESTS=1` is set:

```sh
C50_NETWORK_TESTS=1 .venv/bin/python -m pytest tests/python/test_tutorial.py
```

The package includes the PEP 561 `py.typed` marker, inline annotations for the
estimator, and a stub for the compiled extension. Run the strict source and
public-API type checks with:

```sh
.venv/bin/python -m mypy
.venv/bin/python -m mypy.stubtest \
    --allowlist tests/typing/stubtest-allowlist.txt c50._c50
```

### CMake and CTest

CMake leaves the Python binding disabled by default. From the repository root,
use the environment prepared above to enable the binding and its tests.
Install pybind11 in that environment so CMake can find its configuration files.
The selected Python interpreter must also have its development headers.

```sh
.venv/bin/python -m pip install 'pybind11>=2.13'
cmake -S . -B build/python \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DC50_BUILD_PYTHON=ON \
    -DBUILD_TESTING=ON \
    -DPython_EXECUTABLE="$PWD/.venv/bin/python" \
    -Dpybind11_DIR="$(.venv/bin/python -m pybind11 --cmakedir)"
cmake --build build/python --target c50_python --parallel 2
ctest --test-dir build/python \
    -R '^(python-.*|benchmark-smoke)$' --output-on-failure
```

The `c50_python` target builds the extension and its native library dependency.
CMake places the package under `build/python/python/c50`. CTest uses the selected
interpreter and sets `PYTHONPATH` for runtime checks to import this build.
The filter runs `python-api` (pytest), `benchmark-smoke`, `python-typecheck`
(mypy), and `python-stubtest`. The two typing checks are registered only in
non-sanitizer builds.

To run the native tests as well, build all targets before running CTest without
its filter:

```sh
cmake --build build/python --parallel 2
ctest --test-dir build/python --output-on-failure
```

### Distributions

The `c50` Python distribution has its own version sequence, independent of the
imported C5.0 Release 2.07 GPL Edition. `setuptools-scm` derives the package
version from Git tags beginning with `v` and a digit, such as `v1.0.0a1`.
The upstream `c5.0-2.07` tag does not participate in package version discovery.
Commits after a package tag receive development versions with a commit identifier.

The native library and its CMake package share this version, so C++ consumers
request it with `find_package(C50 1.0 CONFIG REQUIRED)`. Packages with the same
major version are compatible. CMake versions have no prerelease labels, so a
prerelease such as `1.0.0a1` has the CMake version of the release it precedes,
`1.0.0`. Before tagging a release, set the version in the `project()` command in
`CMakeLists.txt` to the tag's release number. A Python build from a release tag
fails at configuration when the two differ. The command-line programs and
serialized models keep the upstream identifier `2.07`.

Build from a clean checkout of the package release tag when preparing a release.
The checkout must include Git history and tags. Source distributions retain the
resolved version in `PKG-INFO`, so wheels built from them do not require Git.
An arbitrary source copy without Git or source-distribution metadata has no
version fallback and cannot produce a Python package.

Build source and binary distributions with `build` installed:

```sh
.venv/bin/python -m pip install build
.venv/bin/python -m build
```

The default `build` command creates a source distribution and then builds the
wheel from that distribution. Test both artifacts in separate clean environments
before publishing them. Read the installed package version with:

```sh
.venv/bin/python -c 'from importlib.metadata import version; print(version("c50"))'
```

Check that a source distribution holds exactly the tracked files, plus the
generated `PKG-INFO`, from a checkout of the revision that built it:

```sh
.venv/bin/python tests/check_sdist.py dist/c50-*.tar.gz
```

### Release wheels

`.github/workflows/wheels.yml` builds the source distribution, checks it, and
then uses [cibuildwheel](https://cibuildwheel.pypa.io) to build and test wheels
from it, which verifies that the source distribution is complete. It builds
CPython 3.10 through 3.14 wheels for manylinux x86-64 and ARM64, macOS arm64
and x86-64, and Windows AMD64. Each wheel is installed in a clean environment
and must pass the Python test suite. The macOS x86-64 wheels build in their
own job on an arm64 runner and are tested under Rosetta 2. The workflow runs on integration branches, topic
branches, pull requests, and `v`-prefixed tags, and its last job records the
SHA-256 digests of every tested distribution in `SHA256SUMS`. It does not
publish packages.

`[tool.cibuildwheel]` in `pyproject.toml` holds the configuration, so a local
Linux build with Docker uses the same settings:

```sh
uvx cibuildwheel --only cp312-manylinux_x86_64 dist/c50-*.tar.gz
```

musllinux, free-threaded CPython, and prerelease CPython versions are not
built.

## Sort and worker-equivalence tests

The default CTest run includes two tests of training internals.
`sort-properties` checks that the continuous-value sorts order and preserve
records. It requires the RuleQuest tie order from the reference sort and the
case order from the stable sort, exhaustively for short inputs and on large
random and floating-point-extreme inputs. `parallel-workload-equivalence`
trains a deterministic dense workload with one, two, four, and eight split
workers. It requires the same serialized classifier at 9,999, 10,000, and
20,000 rows, with subset splits, and with the stable tie order on tied values.

## API documentation

Build the warning-strict Doxygen reference directly with CMake:

```sh
cmake -S . -B build/docs \
    -DC50_BUILD_DOCUMENTATION=ON \
    -DBUILD_TESTING=OFF
cmake --build build/docs --target c50-docs
```

The HTML output is under `build/docs/docs/doxygen/html`, and the XML consumed by
Breathe is under `build/docs/docs/doxygen/xml`.

Build the complete Sphinx site with:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install '.[docs]'
.venv/bin/python -m sphinx -W --keep-going \
    -b html docs docs/_build/html
```

On Linux with GCC or Clang, `allocation-failure` injects failures into native
`malloc`, `calloc`, `realloc`, and `strdup` calls and C++ allocations. It checks
basic model loading, training, prediction, schema growth, and thresholded
class names, including reuse of the context after every failure. Run it under
AddressSanitizer with leak detection to check exceptional-path ownership:

```sh
ASAN_OPTIONS=detect_leaks=1 \
    ctest --test-dir build/sanitize -R allocation-failure --output-on-failure
```

## Continuous validation

`.github/workflows/validation.yml` runs the native suite with GCC and Clang,
including the optional native correctness workloads and installed C++17/C++20
consumers. The GCC job also exercises production and verbose Make builds. A
separate job installs the Python package, runs the estimator and typing checks,
and builds warning-strict Doxygen and Sphinx documentation. Its Python 3.12 job
also runs the tutorial test, with the OpenML data cached between runs.

The workflow runs on pull requests, pushes to `main`, `dev`, and topic branches,
and manual dispatch. It uses read-only repository permissions and pinned action
revisions. A checked-in workflow defines the intended checks. Hosted results
must be inspected before claiming a platform or dependency configuration passed.

`.github/workflows/sanitizers.yml` runs separate Clang address/undefined/leak
and GCC thread-sanitizer jobs on pull requests, integration-branch pushes,
manual dispatch, and a weekly schedule. Sanitizer failures fail the job.
The thread job disables address randomization for the test process to avoid
GCC TSan's Linux shadow-memory mapping conflict. The ordinary local test target
does not run these additional builds.

## Allocation-failure regression tests

On Linux with GCC or Clang, `allocation-failure` fails each allocation in turn
while training, loading, or predicting trees and rules. Its cases cover
three-classifier boosted ensembles, dense input with missing values, implicit
expressions, growing schemas, subset pruning, and branch compression. Each
failure must propagate as `std::bad_alloc`, and the same context must accept a
subsequent successful operation. Run this test with leak detection to check
exception-path ownership:

```sh
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
    ctest --test-dir build/sanitize -R allocation-failure --output-on-failure
```

The harness uses GNU linker allocation wrappers. It does not establish
allocation-failure coverage on other platforms or cover every learner path.

`parallel-failure` compiles the production split scheduler with private test
probes. The installed core contains no probes. The test fails every allocation
in each additional worker workspace, injects task-launch errors after earlier
tasks start, and throws from the calling worker, background workers, and all
workers together. It checks that all started tasks finish before workspace
release and operation return, then retrains with the same context and compares
classifiers and predictions. These cases also run in the sanitizer jobs.

`cpp-contracts` compares normalized classifiers and exact class predictions,
confidence values, and class scores at one, two, four, and eight split workers.
The cases include boosted trees and rules, dense missing values, ordered
attributes, not-applicable values, case weights, costs, sampling, subsets, and
serialized-model reloads. They exercise both parallel split evaluation and its
serial fallback conditions.

The same test runs four simultaneous independent fits, each requesting four
split workers, with trees, rules, and opposing class assignments. Repeated fits
must reproduce their single-worker classifiers and predictions, and concurrent
prediction through a shared immutable model must reproduce the reference.

## Parser fuzzing

```{include} ../tests/fuzz/README.md
:start-line: 2
```
