# C5.0 modernization

This repository modernizes the single-threaded C5.0 Release 2.07 GPL Edition
while preserving its learning behavior and model compatibility. The intended
deliverables are a reusable C library, a header-only C++ facade linked to that
library, command-line compatibility, and Python bindings.

Geoffrey Mainland maintains this modernization project. The imported C5.0
implementation remains attributed to RuleQuest Research Pty Ltd. in its source
notices.

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

The legacy programs require a C compiler and `make`:

```sh
make
```

This builds:

- `c5.0`, the classifier training and evaluation program.
- `report`, the cross-validation report generator.

CMake is also supported:

```sh
cmake -S . -B build
cmake --build build
```

Install the C library, C header, header-only C++ facade, and CMake package with:

```sh
cmake --install build --prefix <prefix>
```

CMake consumers can use `C50::core` for the C API or `C50::cpp` for the C++11
facade. The C++ target links the compiled C core transitively:

```cmake
find_package(C50 2.07 CONFIG REQUIRED)
target_link_libraries(my_program PRIVATE C50::cpp)
```

The public headers are `<c50/c50.h>` and `<c50/c50.hpp>`. Both APIs accept
C5.0 names, training data, optional costs, and prediction cases from memory.
Trained models retain their serialized C5.0 representation for storage or
interchange with compatible tools.

## Python

The Python bindings require Python 3.12 or later. They are implemented with
pybind11 and call the C++ facade, which in turn links to the C core. Build and
install them from the repository root with:

```sh
python -m pip install .
```

Training data and prediction cases use the same in-memory C5.0 text formats as
the C and C++ APIs:

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

serialized = model.serialized_data
restored = c50.load(model.names_data, serialized, model.kind)
```

`ModelKind.RULES` selects a rules model, and `Options` exposes the native
training controls. Models are pickleable. Independent training and prediction
operations release the Python GIL and use separate native contexts, so they can
execute concurrently; callers should still serialize mutation of each Python
object.

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
version 3 or, at your option, any later version. See [gpl.txt](gpl.txt) and the
notices in the source files.

C5.0 and RuleQuest are associated with RuleQuest Research Pty Ltd. This
modernization project is based on the GPL source release and is not an official
RuleQuest distribution.
