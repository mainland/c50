# Quickstart

## Build the native library and programs

CMake builds the C++ core once and links the command-line and C++
interfaces to it:

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Install the C++ interface into a chosen prefix:

```sh
cmake --install build --prefix <prefix>
```

CMake consumers link `C50::cpp`:

```cmake
find_package(C50 2.07 CONFIG REQUIRED)
target_link_libraries(my_program PRIVATE C50::cpp)
```

Train and predict from memory through the compiled C++ interface:

```cpp
#include <c50/c50.hpp>

int main()
{
    c50::context workspace;
    c50::options options;
    options.minimum_cases = 1;
    auto model = c50::model::train(
        workspace, c50::model_kind::tree,
        "no, yes.\nvalue: continuous.\n",
        "0, no\n1, no\n2, yes\n3, yes\n", options);
    auto result = model.predict(workspace, "0, ?\n3, ?\n");
    return result.class_index(0) == 0 && result.class_index(1) == 1 ? 0 : 1;
}
```

## Install the Python package

Python 3.10 or later is required. The package build compiles the pybind11
extension and links it to the same core through the C++ API:

```sh
python3 -m venv .venv
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

With NumPy installed, the low-level dense interface avoids constructing data
and case strings. Categorical columns use zero-based indices into explicit
values in the names data, and NaN represents a missing value. Class indices
require an integer dtype and nonnegative values in the declared class range:

```python
import numpy as np

values = np.asarray([[0.0], [1.0], [2.0], [3.0]])
class_indices = np.asarray([0, 0, 1, 1], dtype=np.uintp)

model = c50.train_dense(names, values, class_indices)
details = model.predict_details_dense(np.asarray([[0.0], [3.0]]))
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

See {doc}`tutorial` for a walkthrough on public datasets, {doc}`estimator` for
the array, categorical-value, missing-value, and cost matrix contracts, and
{doc}`large-datasets` for the scale target and benchmark.
