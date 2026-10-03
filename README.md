# C5.0 modernization

This repository modernizes the single-threaded C5.0 Release 2.07 GPL Edition
while preserving its learning behavior and model compatibility. It provides a
compiled C++17 library, compatible command-line programs, and Python bindings.

The [project documentation](https://github.com/mainland/c50/blob/main/docs/index.md) contains build and testing guides,
ownership and concurrency contracts, and generated C++ and Python API
references. The repository includes configuration for publishing that site on
Read the Docs.

Geoffrey Mainland maintains this modernization project. The C5.0 sources were
subsequently migrated from C to C++; the imported implementation remains
attributed to RuleQuest Research Pty Ltd. in its source notices.

## Provenance

The initial source import came from RuleQuest Research's `C50.tgz`, downloaded
from the [official RuleQuest downloads
page](https://www.rulequest.com/download.html). RuleQuest identifies this
archive as the single-threaded **C5.0 Release 2.07 GPL Edition**.

The archive used for the import has this SHA-256 digest:

```text
309db588eda420c06701bf8ae74c06a6c923e9a06e714a598ea761bcadfc5e2e  C50.tgz
```

## Build

The compatibility programs require C and C++17 compilers and `make`:

```sh
make
```

This builds:

- `c5.0`, the classifier training and evaluation program.
- `report`, the cross-validation report generator.

CMake is also supported, and is required on Windows with Visual Studio:

```sh
cmake -S . -B build
cmake --build build
```

Install the native library, C++ header, and CMake package with:

```sh
cmake --install build --prefix <prefix>
```

CMake consumers link `C50::cpp`, which carries the compiled core and C++17
requirements:

```cmake
find_package(C50 1.0 CONFIG REQUIRED)
target_link_libraries(my_program PRIVATE C50::cpp)
```

The public header is `<c50/c50.hpp>`. The C++ API accepts
C5.0 names, training data, optional costs, and prediction cases from memory.
Trained models retain their serialized C5.0 representation for storage or
interchange with compatible tools. The library does not provide a C API or a
stable binary ABI. C++ consumers must rebuild with a compatible toolchain
when the library changes.

## Python

The Python bindings require Python 3.10 or later. They are implemented with
pybind11 and call the compiled C++ API. Install a published release from PyPI
with:

```sh
python -m pip install --pre c50
```

`--pre` is needed while only prerelease versions, such as `1.0.0a1`, are
published. Releases include wheels for Linux x86-64 and ARM64, macOS arm64 and
x86-64, and Windows x86-64. To build and install from a source checkout
instead, run this from the repository root:

```sh
python -m pip install .
```

The compatibility API accepts the same in-memory C5.0 text formats as the
C++ API:

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
scores = model.predict_scores("0, ?\n3, ?\n")

serialized = model.serialized_data
restored = c50.load(model.names_data, serialized, model.kind)
```

`ModelKind.RULES` selects a rules model, and `Options` exposes the native
training controls. Models are pickleable. Independent training and prediction
operations release the Python GIL and use separate native contexts, so they can
execute concurrently; callers should still serialize mutation of each Python
object.

Install the optional NumPy and scikit-learn dependencies to use the
array-oriented estimator:

```sh
python -m pip install 'c50[sklearn]'
```

```python
import numpy as np

from c50.sklearn import C50Classifier

X = np.asarray([[0.0], [0.5], [2.5], [3.0]])
y = np.asarray(["low", "low", "high", "high"])

classifier = C50Classifier(minimum_cases=1).fit(X, y)
labels = classifier.predict([[0.25], [2.75]])
scores = classifier.predict_proba([[0.25], [2.75]])
```

The estimator supports dense numeric and categorical inputs, missing values,
native training options, misclassification costs, pipelines, and grid search.
It uses the typed native data path rather than materializing a whole-dataset
text buffer and batches prediction to bound temporary native storage. Training
remains in-memory. Set `n_jobs` to enable bounded parallel evaluation of
eligible attribute splits with up to 8 workers. The default is one worker.
Other training phases remain serial. The detailed input, concurrency, and
[large-dataset](https://github.com/mainland/c50/blob/main/docs/large-datasets.md) contracts are in the project
documentation.

Run `./c5.0 -h` to see the available command-line options. C5.0 uses a file stem
to locate inputs such as `<stem>.names`, `<stem>.data`, and optional
`<stem>.test` and `<stem>.costs` files:

```sh
./c5.0 -f <stem>
```

## Tests

Run the CLI regression tests with:

```sh
make test
```

The tests compare normalized command output and serialized tree and rule models
against fixtures captured from the imported GPL implementation.

Run the same regression tests through CTest with:

```sh
ctest --test-dir build --output-on-failure
```

Configure an AddressSanitizer and UndefinedBehaviorSanitizer build with:

```sh
cmake -S . -B build/sanitize \
    -DCMAKE_BUILD_TYPE=Debug \
    -DC50_ENABLE_SANITIZERS=ON
cmake --build build/sanitize
ctest --test-dir build/sanitize --output-on-failure
```

Independent contexts can run concurrently, and immutable loaded models can be
shared across those operations. Access to an individual context must remain
serialized. To exercise this contract under ThreadSanitizer, configure with:

```sh
cmake -S . -B build/tsan \
    -DCMAKE_BUILD_TYPE=Debug \
    -DC50_ENABLE_THREAD_SANITIZER=ON
cmake --build build/tsan
ctest --test-dir build/tsan --output-on-failure
```

## License

The imported C5.0 source is distributed under the GNU General Public License,
version 3 or, at your option, any later version. See [gpl.txt](https://github.com/mainland/c50/blob/main/gpl.txt) and the
notices in the source files.

C5.0 and RuleQuest are associated with RuleQuest Research Pty Ltd. This
modernization project is based on the GPL source release and is not an official
RuleQuest distribution.
