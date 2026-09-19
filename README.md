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

## Current build

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

## License

The imported C5.0 source is distributed under the GNU General Public License,
version 3 or, at your option, any later version. See [gpl.txt](gpl.txt) and the
notices in the source files.

C5.0 and RuleQuest are associated with RuleQuest Research Pty Ltd. This
modernization project is based on the GPL source release and is not an official
RuleQuest distribution.
