# Building and testing

## Support and validation matrix

The native API requires C++17. Installed consumers are compiled as C++17 and
C++20. The Python package requires Python 3.10 or newer. The estimator's
minimum dependencies are joblib 1.3, NumPy 1.26, and scikit-learn 1.6.

The workflows in `.github/workflows` validate each supported target, and the
repository's [Actions page](https://github.com/mainland/c50/actions) shows
their current results.

| Target | Compiler or dependencies | Workflow and job | Checks |
| --- | --- | --- | --- |
| Linux x86-64 | GCC and Clang on `ubuntu-24.04` | `validation.yml`, `native` | Native build, CTest, and installed consumers. With GCC, also the Make workflows and verbose diagnostics. |
| Linux x86-64 | Clang | `sanitizers.yml`, `native` | CTest under AddressSanitizer with leak detection and UndefinedBehaviorSanitizer |
| Linux x86-64 | GCC | `sanitizers.yml`, `native` | CTest under ThreadSanitizer |
| Linux x86-64 | AFL++ `afl-clang-lto` | `sanitizers.yml`, `afl-seed-replay` | Replay of the checked-in fuzz seeds |
| Linux ARM64 | GCC on `ubuntu-24.04-arm` | `validation.yml`, `native` | Native build, CTest, and installed consumers |
| Linux x86-64 and ARM64 | GCC on `ubuntu-24.04` and `ubuntu-24.04-arm` | `reference-comparison.yml`, `release-207` | Trees, rules, and console output compared with C5.0 Release 2.07 built from `main`. ARM64 also builds with `-ffp-contract=off`. |
| macOS ARM64 | AppleClang on `macos-15` | `validation.yml`, `native` | Native build, CTest, and installed consumers |
| Linux x86-64 and ARM64, macOS ARM64 | GCC on Linux, AppleClang on macOS | `validation.yml`, `private-core` | Release build and CTest with `C50_PRIVATE_CORE`, the Python packages' core configuration |
| Linux x86-64 | Clang on `ubuntu-24.04` | `validation.yml`, `private-core-clang` | Release build and CTest with `C50_PRIVATE_CORE` and no interprocedural optimization variable, so that the programs link against Clang's LLVM bitcode core |
| Windows x86-64 | MSVC on `windows-2022`, Python 3.12 | `validation.yml`, `windows` | Native build with the Python extension, CTest, installed consumers, and Python tests |
| Linux x86-64 | Python 3.12 and the pinned checkers | `validation.yml`, `lint` | The pre-commit hooks: ruff, ruff format, mypy, the license notice check, zizmor on the workflows, codespell, rumdl, and file hygiene checks |
| Python 3.10 | joblib 1.3.0, NumPy 1.26.4, pandas 2.2.3, scikit-learn 1.6.1 | `validation.yml`, `python-and-docs` | Python tests, mypy, stubtest, and the benchmark smoke test |
| Python 3.11 through 3.14 | Latest releases | `validation.yml`, `python-and-docs` | As for Python 3.10. Python 3.12 also runs the tutorial test and builds the documentation. |
| Wheels for CPython 3.10 through 3.14 | Linux x86-64 and ARM64, macOS arm64 and x86-64, Windows AMD64 | `wheels.yml`, `wheels` | Build from the source distribution and test each wheel |

Free-threaded Python, GPU arrays, and other compiler or platform combinations
are not tested. Seven tests run only on Linux. `in-memory-without-files`,
`allocation-failure`, and `parallel-failure` need GNU or Clang on Linux because
they link with `-Wl,--wrap`, and `writable-global-inventory` reads ELF symbols
with GNU `nm`. `core-namespace` lists the external symbols of `libc50_core.a`
with `nm` and fails on any outside the `c50` and standard-library namespaces,
except for a short allowlist. It does not run for a core compiled with
link-time optimization. `math-overloads` compiles each source twice, with and
without `-DVerbOpt`, at `-O0 -fno-inline`. It fails when `nm` lists a math
function overload for `float` or `long double`, such as `std::log(float)`,
which C++ selects where Release 2.07's C calls the `double` function. It also
fails when `nm` fails,
when no object lists a symbol, and when it does not report the `float`
overload that `tests/math_overload_control.cpp` calls. The calls that match
Release 2.07's precision cast the whole argument to `double`. An optimized
build cannot show these overloads, because it inlines `rintf` and `fabsf`. The
check does not list classification functions such as `isfinite`, which are
exact in every precision, and it runs in every Linux job that runs CTest.
`writable-global-inventory-lto` reads no symbols. It configures
nested builds with `-flto` flags, so CMake registers it only on Linux with
`nm`, a single-configuration generator, and GCC or Clang. The macOS and Windows
jobs omit them, so the no-file guarantee, the allocation-failure behavior, and
the parallel-failure guarantees are tested only on Linux. The
`locale-independence` test exits with status 77, which CTest reports as
skipped, on Windows, so the Windows branch of its locale scope has no C++ test.
A core compiled with link-time optimization, whether Clang's LLVM bitcode or
GCC's slim LTO objects, records no sections, so `nm` cannot tell constants from
writable data and CMake does not register `writable-global-inventory`. Native
builds of the same sources run it, and `writable-global-inventory-lto`
configures both kinds of build to check which one registers it. The portable
contract and installed-consumer tests still run. The Windows job treats level-4
warnings as errors, except C4244, C4267, and C4305, which report the implicit
narrowing conversions that the imported sources use to store values as float.
It also builds the Python extension with Visual Studio, a multi-configuration
generator, so its CTest run checks that the build tree places the extension
next to its package.

The `license-notices` pre-commit hook and the `python-license-notices` test run
`tests/check_license_notices.py`. It fails when a tracked source file that can
carry a comment lacks an `SPDX-License-Identifier` line. It also fails when a
file that descends from a file of Release 2.07, including the root `Makefile`,
lacks RuleQuest's copyright line or a `Modified` line, and when
`src/c50_model.cpp`, `src/c50_sort.h`, or `tests/sort_properties.cpp`, which
follow code of Release 2.07, lacks RuleQuest's credit. The check looks for the
copyright line only, not for the permission paragraph under it. It also fails
when `src/text.i` or `xval`, the imported files that this project did not
modify, differs from its blob on `main`. The script holds those blob
identifiers, so a clone without a `main` branch checks them too. When `main` or
`origin/main` exists, the script also checks that each source file of Release
2.07 has an entry in its tables and that the entry names a tracked file, so an
import that is renamed and stripped of its notices fails. The test skips
outside a Git checkout.

Native CI configures a C++14 project default to verify that each target
declares its own C++17 requirement. The platform labels follow the
[GitHub-hosted runner reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).

## Windows

Build with Visual Studio 2022 or newer and CMake. The shell regression tests
run under the POSIX `sh` from [Git for Windows](https://gitforwindows.org),
which CMake finds in its default installation directory or on `PATH`:

```sh
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build --build-config Release --output-on-failure
cmake --install build --config Release --prefix <prefix>
```

Without `--config Release`, Visual Studio builds and installs Debug.
Single-configuration generators build Release when no build type is set.
Set one explicitly, such as `-DCMAKE_BUILD_TYPE=Debug`, to override that
default. CMake 3.22 and newer also read a non-empty `CMAKE_BUILD_TYPE`
environment variable. Older versions and an empty variable leave the Release
default in place.

`.gitattributes` checks out text files with LF line endings on every platform,
so the fixtures compare byte for byte. The programs write model files in
binary mode, so `.tree` and `.rules` files are identical to those written on
other platforms. Console output and `-o` output files use Windows CRLF line
endings, and the shell tests remove carriage returns before comparing them.
The root `Makefile` and the `xval` script are not supported on Windows.

## Legacy Make build

The compatibility build requires a C++17 compiler and `make`:

```sh
make
make test
```

It produces `c5.0`, the training and evaluation program, and `report`, the
cross-validation report generator. Only `make reference-comparison` needs a C
compiler, `gcc`, which builds C5.0 Release 2.07.

The installed `C50::cpp` CMake target links the compiled C++17 core. Manual
link commands must use a C++ driver and link the platform's thread and math
libraries as required. The install-consumer test builds standalone C++17 and
C++20 programs using only the installed public interface.

## GCC

Select GCC explicitly when configuring CMake:

```sh
cmake -S . -B build/gcc \
    -DCMAKE_CXX_COMPILER=g++
cmake --build build/gcc
ctest --test-dir build/gcc --output-on-failure
```

The `reference-compatibility` test checks deterministic classifier and
prediction digests captured with the imported RuleQuest sorting algorithms. See
{doc}`compatibility` for its scope and baseline-update policy.

A `VerbOpt` build also checks split-diagnostic order and the resulting
classifier. The verbose code contains assertions that the build checks only
when `NDEBUG` is undefined, so configure it with `-DCMAKE_BUILD_TYPE=Debug`. A
build of another type compiles the assertions out and the test passes without
checking them. Configure the flag for C++ sources, then run the two CLI tests:

```sh
cmake -S . -B build/verbose -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_FLAGS=-DVerbOpt
cmake --build build/verbose
ctest --test-dir build/verbose -R '^cli(-verbose)?-regression$' \
    --output-on-failure
```

## Clang

Select Clang explicitly with both compiler settings:

```sh
cmake -S . -B build/clang \
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

### Python module under the sanitizers

The Python module is supported in a sanitizer build on Linux. Add
`-DC50_BUILD_PYTHON=ON` and the interpreter options of
{ref}`the CMake and CTest section <cmake-and-ctest>` to the configuration. The
interpreter does not link the AddressSanitizer runtime, so CMake preloads the
shared runtime into `python-api` and `benchmark-smoke` with `LD_PRELOAD`. Plain
`ctest` then works as in a build without Python. For GCC, CMake preloads
`libasan.so` and then `libstdc++.so`, because the interpreter does not link
libstdc++ and the runtime cannot intercept the first C++ exception without it.
For Clang, CMake links the module with `-shared-libasan` and preloads the
`libclang_rt.asan` shared library that the compiler reports. Configuration
fails if the compiler reports no such library or the platform is not Linux.

Leak detection is off for these two tests, because CPython does not free all
its memory at exit and LeakSanitizer would report the interpreter. CMake sets
their `ASAN_OPTIONS` to `detect_leaks=0:halt_on_error=1`, which replaces the
value in the calling environment. The other tests keep the caller's
`ASAN_OPTIONS`. The type checks `python-typecheck` and `python-stubtest` are
not registered in a sanitizer build.

The `address-undefined-leak` job of `.github/workflows/sanitizers.yml` runs
this configuration with Clang. The ThreadSanitizer job tests the native code
only.

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

The property test draws its examples with Hypothesis. Every pytest run, also
under CTest and in CI, uses the `reproducible` profile that
`tests/python/conftest.py` defines. The profile derandomizes the examples and
ignores the example database, so the same tree draws the same examples on every
host, and a failure can be reproduced and bisected. To draw new examples, name
the `explore` profile:

```sh
.venv/bin/python -m pytest --hypothesis-profile=explore \
    tests/python/test_dense_properties.py
```

A failing example from either profile prints a `@reproduce_failure` line with a
blob. Add that decorator to the failing test, above `@given`, to replay the
example, then remove it after the fix. The blob is valid only for the printed
Hypothesis version. The default profile need not draw an example that
`explore` found, so add a regression test for it.

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

### Lint and format

The Python sources, benchmark scripts, and tooling scripts follow the ruff
configuration in `pyproject.toml`. Code, comments, and docstrings are limited
to 79 columns. The pre-commit hooks run ruff and mypy from `.venv`, so install
the pinned checkers into it with pip 25.1 or newer, then run every hook over
the tree. The lint environment needs Python 3.11 or newer, because the pinned
`pre-commit` requires it, although the package supports Python 3.10:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install --group dev
.venv/bin/pre-commit run --all-files
```

The hooks also run codespell, which `pyproject.toml` configures, rumdl, which
`.rumdl.toml` configures, and the license notice check. Add a word that
codespell flags wrongly to `ignore-words-list` in `pyproject.toml`.

Run `.venv/bin/pre-commit install` once to run the hooks before each commit.
The hooks check and never rewrite files. Fix a finding with
`.venv/bin/python -m ruff check --fix` or `.venv/bin/python -m ruff format`.

(cmake-and-ctest)=

### CMake and CTest

CMake leaves the Python binding disabled by default. From the repository root,
use the environment prepared above to enable the binding and its tests.
Install pybind11 3.0 or later in that environment so CMake can find its
configuration files. The selected Python interpreter must also have its
development headers.

```sh
.venv/bin/python -m pip install 'pybind11>=3.0'
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
CMake places the package under `build/python/python/c50`. CTest uses the
selected interpreter and sets `PYTHONPATH` for runtime checks to import this
build. The filter runs `python-api` (pytest), `python-license-notices`,
`benchmark-smoke`, `python-typecheck` (mypy), and `python-stubtest`. The two
typing checks are registered only in non-sanitizer builds. `benchmark-smoke`
is registered only on Linux and macOS, because the benchmark suite requires
one of them. On Windows the filter runs the other four tests.

To run the native tests as well, build all targets before running CTest without
its filter:

```sh
cmake --build build/python --parallel 2
ctest --test-dir build/python --output-on-failure
```

### Distributions

The `c50` Python distribution has its own version sequence, independent of the
imported C5.0 Release 2.07 GPL Edition. `setuptools-scm` derives the package
version from Git tags beginning with `v` and a digit, such as `v1.0.0rc1`. The
upstream `c5.0-2.07` tag does not participate in package version discovery.
Commits after a package tag receive development versions with a commit
identifier.

The native library and its CMake package share this version, so C++ consumers
request it with `find_package(C50 1.0 CONFIG REQUIRED)`. Packages with the same
major version are compatible. CMake versions have no prerelease labels, so a
prerelease such as `1.0.0rc1` has the CMake version of the release it precedes,
`1.0.0`. Before tagging a release, set the version in the `project()` command
in `CMakeLists.txt` to the tag's release number. A Python build from a release
tag fails at configuration when the two differ. The command-line programs and
serialized models keep the upstream identifier `2.07`.

Build from a clean checkout of the package release tag when preparing a
release. The checkout must include Git history and tags. Source distributions
retain the resolved version in `PKG-INFO`, so wheels built from them do not
require Git. An arbitrary source copy without Git or source-distribution
metadata has no version fallback and cannot produce a Python package.

Python package builds set the CMake option `C50_PRIVATE_CORE`, which compiles
the learning core with link-time optimization and hidden symbol visibility.
The extension then exports none of the core's functions, so the compiler can
inline across the core's source files. If the compiler does not support
link-time optimization, CMake prints a warning and builds without it. A build
with this option does not install the C++ library or its CMake package,
because its objects contain intermediate code that only the same compiler can
link.

Build source and binary distributions with `build` installed:

```sh
.venv/bin/python -m pip install build
.venv/bin/python -m build
```

The default `build` command creates a source distribution and then builds the
wheel from that distribution. Test both artifacts in separate clean
environments before publishing them. Read the installed package version with:

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
and must pass the Python test suite. The macOS x86-64 wheels build in their own
job on an arm64 runner and are tested under Rosetta 2. The workflow runs on
integration branches, topic branches, pull requests, and `v`-prefixed tags, and
its last job records the SHA-256 digests of every tested distribution in
`SHA256SUMS`. It does not publish packages.

`[tool.cibuildwheel]` in `pyproject.toml` holds the configuration, so a local
Linux build with Docker uses the same settings:

```sh
uvx cibuildwheel --only cp312-manylinux_x86_64 dist/c50-*.tar.gz
```

musllinux, free-threaded CPython, and prerelease CPython versions are not
built.

The Windows wheels import `MSVCP140.dll`, which Python installers do not ship.
`[tool.cibuildwheel.windows]` runs `delvewheel repair`, which copies the DLL
into the wheel's `c50.libs` directory. The repair runs only on a Windows
build. Before a release, check that the release candidate's Windows wheels
list no unbundled `MSVCP140.dll` import and that one imports in an
environment without the Visual C++ runtime.

## Comparison with Release 2.07

The `reference-comparison` check trains generated data with the library's
`c5.0` and with C5.0 Release 2.07 and requires identical results. It builds
the imported program from the `main` branch with the Makefile's flags,
`gcc -ffloat-store -O3`, without `csh`, and adds `-ffp-contract=off`. That
flag keeps the reference to x86-64 arithmetic on targets with fused
multiply-add and changes nothing on x86-64. The check does not run in the
default CTest or `make test` runs, because it builds a second program and
trains dozens of cases. Run it from a build directory or with Make:

```sh
cmake --build build --target reference-comparison
make reference-comparison
```

Run the script directly to compare any `c5.0` program. The optional second
argument is a directory that holds the files of `main`. Without it, the script
extracts a ref with `git archive`: `$C50_REFERENCE_REF` if set, else `main`,
else `origin/main`. A clone that has none of them, such as a shallow clone of
another branch, fails with a message that asks for the variable or a directory.

```sh
tests/reference/run.sh ./c5.0
tests/reference/run.sh ./c5.0 /path/to/release-2.07
```

`tests/reference/compare.py` runs the cases and takes a `--only TEXT` filter
and `--keep DIRECTORY`, which retains the files of each case. The standard
library is the only Python dependency. `tests/reference/generate.py` writes the
datasets from fixed seeds, so every run and platform trains the same bytes.

The datasets have continuous, discrete, and ordered discrete attributes,
missing (`?`) and not-applicable (`N/A`) values, two to four classes, a
case-weight attribute, a costs file, and implicit attributes. The cases use
`-r`, `-b`, `-t`, `-w`, `-s`, `-g`, `-m`, `-c`, `-p`, `-S` with `-I`, `-X`,
`-u`, `-e`, and costs, alone and in combination, with `-f` naming each
dataset. The comparison normalizes what `tests/test_cli.sh` normalizes: the
start time, the elapsed time, the date of the model `id`, and trailing white
space. It normalizes nothing else, and any other difference fails the case. The
comparison also requires the same exit status. Two datasets with undeclared
discrete values check it: three bad values leave exit status 0, and twelve
reach the error limit and exit with status 1. Six datasets divide in an
implicit definition by a value that wraps Release 2.07's precision multiplier,
to a positive value, a negative value, 0, and the most negative `int`, for a
six-decimal numerator and an integer one, and in a nested comparison.

The comparison avoids the differences that
{doc}`compatibility` documents. Winnowing runs only on a dataset without a
case-weight attribute, because the library computes the average case weight
before winnowing and Release 2.07 does not. The cases do not train a
`timestamp` attribute. The comparison does not cover prediction and report
output.

`.github/workflows/reference-comparison.yml` runs the check on `ubuntu-24.04`
and on the ARM64 runner `ubuntu-24.04-arm`. The reference never fuses
multiplications and additions. The default ARM64 entry shows whether the
library's default build matches the x86-64 classifier, and a second ARM64
entry, which builds the library with `-ffp-contract=off`, shows whether that
flag restores it.

## Sort and worker-equivalence tests

The default CTest run includes two tests of training internals.
`sort-properties` checks that the continuous-value sorts order and preserve
records. It requires the RuleQuest tie order from the reference sort and the
case order from the stable sort, exhaustively for short inputs and on large
random and floating-point-extreme inputs. `parallel-workload-equivalence`
trains a deterministic dense workload with one, two, four, and eight split
workers. It requires the same serialized classifier at 9,999, 10,000, and
20,000 rows, with subset splits, and with the stable tie order on tied values.

## Class-count index check

The `index-overflow` test checks the flattened-index helper at the `INT_MAX`
boundary and runs by default. The end-to-end check trains a schema of 46,341
classes and a schema of 4,096 classes with 524,288 attribute values, through
`c5.0` and the C++ API. The class-table cases need about 8.6 GB of address
space. The value-table cases fill two tables of about 16 GiB each, the discrete
frequencies and the split workspace, so they need about 32 GiB resident. CTest
does not run the check. Run it on Linux or macOS with a host that has the
memory:

```sh
cmake --build build --target index-overflow-large
```

## Training fuzz hunt

CTest replays a seed corpus of the training target, which trains small
decoded problems through the text and dense interfaces and compares the
results. The seeded driver `c50_training_fuzz_driver` trains pseudorandom
inputs for as long as asked. It builds on every platform, and CTest runs it
only to check its argument handling. Run a long hunt on Linux or macOS with the
`training-fuzz-hunt` target, which needs a shell, and add the sanitizer options
to a build with `C50_ENABLE_SANITIZERS`:

```sh
C50_HUNT_SEED=1 C50_HUNT_COUNT=100000 \
    cmake --build build --target training-fuzz-hunt
```

The input format, the environment variables, and how to replay a finding are
in the Training target part of Parser fuzzing.

## Python property hunt

The Hypothesis test of the text and dense interfaces draws a few dozen
examples. `tests/slow/property_hunt.py` runs the same strategy and the same
check on as many examples as asked. It is not part of the default build or
tests. It writes each example to a log, flushed and synced to disk, before it
trains. A segmentation fault ends the process before Hypothesis can print its
example, so the last line of the log holds the input of a crash. Configure
with `C50_BUILD_PYTHON`, and add `C50_ENABLE_SANITIZERS` to run the extension
under the sanitizers, which the target preloads as `python-api` does:

```sh
C50_PROPERTY_HUNT_SEED=1 C50_PROPERTY_HUNT_EXAMPLES=20000 \
    cmake --build build --target property-hunt
```

`C50_PROPERTY_HUNT_SEED` defaults to 1, `C50_PROPERTY_HUNT_EXAMPLES` to 1,000,
and `C50_PROPERTY_HUNT_LOG` to `property-hunt.jsonl` in the build directory.
The same seed draws the same examples. After a crash or a failed check, train
the last logged example again, under a debugger if needed:

```sh
C50_PROPERTY_HUNT_REPLAY="$PWD/build/property-hunt.jsonl" \
    cmake --build build --target property-hunt
```

The target gives the replay the environment of a hunt, including the sanitizer
runtime that a sanitizer build needs before it can import the extension. To
replay outside CMake with a build that has no sanitizers, run the script
directly:

```sh
python tests/run_built_python.py build/python tests/slow/property_hunt.py \
    --replay build/property-hunt.jsonl
```

The script exits with status 2, before it opens a log, when Python runs
optimized (`-O`, `-OO`, or `PYTHONOPTIMIZE`), because that removes the
assertions of the checks and a replay would pass on a failing example.

A bounded run establishes only that no failure occurred in those examples.

## API documentation

Build the warning-strict Doxygen reference directly with CMake:

```sh
cmake -S . -B build/docs \
    -DC50_BUILD_DOCUMENTATION=ON \
    -DBUILD_TESTING=OFF
cmake --build build/docs --target c50-docs
```

The HTML output is under `build/docs/docs/doxygen/html`, and the XML consumed
by Breathe is under `build/docs/docs/doxygen/xml`.

Build the complete Sphinx site with:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install '.[docs]'
.venv/bin/python -m sphinx -W --keep-going \
    -b html docs docs/_build/html
```

On Linux with GCC or Clang, `allocation-failure` injects failures into native
`malloc`, `calloc`, `realloc`, and `strdup` calls and C++ allocations. It
checks basic model loading, training, prediction, schema growth, and
thresholded class names, including reuse of the context after every failure.
Run it under AddressSanitizer with leak detection to check exceptional-path
ownership:

```sh
ASAN_OPTIONS=detect_leaks=1 \
    ctest --test-dir build/sanitize -R allocation-failure --output-on-failure
```

## Continuous validation

`.github/workflows/validation.yml` runs the native suite with GCC and Clang,
including the optional native correctness workloads and installed C++17/C++20
consumers. The GCC job also exercises production and verbose Make builds. The
`private-core` job repeats the native suite in Release mode with
`C50_PRIVATE_CORE` and link-time optimization required, with GCC on Linux
x86-64 and ARM64 and AppleClang on macOS ARM64. The `private-core-clang` job
configures the same option with Clang on Linux x86-64 and leaves the
interprocedural optimization variables unset, to check that the programs link
against a core of LLVM bitcode, which the default linker loads through the
Clang plugin. A separate job installs the Python package, runs the estimator
and typing checks, and builds warning-strict Doxygen and Sphinx documentation.
Its Python 3.12 job also runs the tutorial test, with the OpenML data cached
between runs.

The `lint` job runs the pre-commit hooks over the whole tree with the checkers
that `pyproject.toml` pins: ruff, ruff format, mypy, the license notice check,
zizmor, which audits the workflows, codespell, rumdl, and file hygiene checks.

The workflow runs on pull requests, pushes to `main`, `dev`, and topic
branches, and manual dispatch. It uses read-only repository permissions and
pinned action revisions. A checked-in workflow defines the intended checks.
Hosted results must be inspected before claiming a platform or dependency
configuration passed.

`.github/workflows/reference-comparison.yml` runs the comparison with
Release 2.07 described above.

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

`dense-not-applicable` and `tests/python/test_dense_not_applicable.py` train
trees, rules, boosted ensembles, subset splits, winnowed attributes, sampling,
and costs from dense input with a not-applicable mask and from the equivalent
text input with `N/A`. They compare the serialized model without its time
stamp, the predictions, and the attribute usage. The values under the mask
hold the real value, NaN, or an out-of-range value in turn. The Python file
also checks the mask's shape, dtype, layout, and `copy=False` handling.

The same test runs four simultaneous independent fits, each requesting four
split workers, with trees, rules, and opposing class assignments. Repeated fits
must reproduce their single-worker classifiers and predictions, and concurrent
prediction through a shared immutable model must reproduce the reference.

## Parser fuzzing

```{include} ../tests/fuzz/README.md
:start-line: 2
```
