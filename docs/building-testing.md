# Building and testing

## Legacy Make build

The compatibility build requires a C compiler and `make`:

```sh
make
make test
```

It produces `c5.0`, the training and evaluation program, and `report`, the
cross-validation report generator.

## GCC

Select GCC explicitly when configuring CMake:

```sh
cmake -S . -B build/gcc \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++
cmake --build build/gcc
ctest --test-dir build/gcc --output-on-failure
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
LeakSanitizer from running reliably in some containers. Omit
`ASAN_OPTIONS=detect_leaks=0` on hosts where LeakSanitizer is supported.

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

Install the test extra into a Python 3.12-or-newer environment and run pytest:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install '.[test]'
.venv/bin/python -m pytest
```

The package includes the PEP 561 `py.typed` marker, inline annotations for the
estimator, and a stub for the compiled extension. Run the strict source and
public-API type checks with:

```sh
.venv/bin/python -m mypy
.venv/bin/python -m mypy.stubtest \
    --allowlist tests/typing/stubtest-allowlist.txt c50._c50
```

CTest runs both checks as `python-typecheck` and `python-stubtest` when the
Python binding is enabled in a non-sanitizer build.

Build source and binary distributions with `build` installed:

```sh
.venv/bin/python -m pip install build
.venv/bin/python -m build
```

Test the resulting wheel and source distribution in separate clean
environments before publishing them.

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
python3.12 -m venv .venv
.venv/bin/python -m pip install '.[docs]'
.venv/bin/python -m sphinx -W --keep-going \
    -b html docs docs/_build/html
```
