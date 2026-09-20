# Quickstart

## Build the native library and programs

CMake builds the C core once and links the command line and C++ interfaces to
it:

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Install the C and C++ interfaces into a chosen prefix:

```sh
cmake --install build --prefix <prefix>
```

CMake consumers link `C50::core` for the C interface or `C50::cpp` for the
header-only C++ facade:

```cmake
find_package(C50 2.07 CONFIG REQUIRED)
target_link_libraries(my_program PRIVATE C50::cpp)
```

## Install the Python package

Python 3.12 or later is required. The package build compiles the pybind11
extension and links it to the same C core through the C++ facade:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install .
```

The low-level Python interface accepts the contents of C5.0 names and data
files directly:

```python
import c50

names = """no, yes.

value: continuous.
"""
training_data = """0, no
1, no
2, yes
3, yes
"""

model = c50.train(names, training_data)
labels = model.predict("0, ?\n3, ?\n")
scores = model.predict_proba("0, ?\n3, ?\n")
```

Use `ModelKind.RULES` to construct a rules model. `Options` exposes the native
training controls.

## Use the scikit-learn estimator

Install the optional NumPy and scikit-learn dependencies with:

```sh
python -m pip install '.[sklearn]'
```

The estimator accepts dense array-like inputs and delegates to the same native
training and prediction implementation:

```python
import numpy as np

from c50.sklearn import C50Classifier

X = np.asarray([[0.0], [0.5], [2.5], [3.0]])
y = np.asarray(["low", "low", "high", "high"])

classifier = C50Classifier(minimum_cases=1).fit(X, y)
labels = classifier.predict([[0.25], [2.75]])
scores = classifier.predict_proba([[0.25], [2.75]])
```

See {doc}`estimator` for the array, categorical-value, missing-value, and cost
matrix contracts.
