# Parser fuzz targets

The six AFL++ parser targets use the public C++ interface and accept at most 4096
bytes per input. Each iteration reuses the context for a known valid model and
prediction after the tested operation. Parse, argument, unsupported-format, and
missing-cost errors are expected. Unexpected exceptions, sanitizer failures,
hangs, and failed recovery are findings.

| Target suffix | Input |
| --- | --- |
| `schema` | Complete names-file contents. |
| `implicit` | An expression inserted after `derived :=` in a fixed schema. |
| `tree` | Complete serialized tree or tree-ensemble contents. |
| `rules` | Complete serialized rules or rule-ensemble contents. |
| `cases` | Cases for continuous `x` and categorical `color`. |
| `costs` | Costs for classes `no` and `yes`. |
| `training` | A decoded classification problem, as the last section describes. |

The training target, which has its own section at the end, is not a parser
target. The fixed schema of the parser targets has classes `no` and `yes`,
continuous attribute `x`, and categorical attribute `color` with levels `red`,
`blue`, and `green`. The implicit and costs targets train on four fixed cases.
The remaining targets load or predict without training. The engine-neutral
target implementation also supports deterministic seed replay through CTest.
AFL++ uses the thin `LLVMFuzzerTestOneInput` compatibility entry point supplied
by `aflpp_driver`.

Install AFL++, then configure a separate build with its LLVM LTO compiler
wrappers:

```sh
cmake -S . -B build/afl \
    -DCMAKE_C_COMPILER=afl-clang-lto \
    -DCMAKE_CXX_COMPILER=afl-clang-lto++ \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DBUILD_TESTING=OFF -DC50_BUILD_FUZZERS=ON
cmake --build build/afl --target fuzz-seeds
cmake --build build/afl --target fuzz-smoke
```

`fuzz-seeds` replays every checked-in seed without mutation. A normal testing
build also registers this replay as the `fuzz-seeds` CTest test, which
includes the training seeds described below.
`fuzz-smoke` runs each target for five seconds with a five-second per-input
timeout and a 512 MiB memory limit. It retains queues and findings under
`build/afl/afl-output/` and resumes them on later invocations. Fuzz campaigns
remain outside the default CTest suite. The target fails while its output
directory contains a saved crash or hang. Preserve and triage each finding,
then remove that output directory before starting a campaign for the repair.

For a longer campaign, run a target directly. The parser dictionary supplements
the LTO compiler's automatically discovered tokens:

```sh
AFL_SKIP_CPUFREQ=1 afl-fuzz \
    -i tests/fuzz/corpus/implicit \
    -o build/afl/afl-output/implicit \
    -x tests/fuzz/parser.dict -m 512 -t 5000 \
    -- build/afl/c50_fuzz_implicit
```

Replace the input directory with `-` to resume an existing output directory.
Use multiple synchronized AFL++ instances for sustained campaigns.

For sanitizer validation, configure another AFL++ build with
`C50_ENABLE_SANITIZERS=ON`. The `fuzz-smoke` target disables AFL++'s virtual
address-space limit for that build because AddressSanitizer reserves a large
shadow mapping. Run sanitized campaigns inside a memory-limited container or
cgroup. Alternatively, generate a corpus with the fast build and replay it with
the sanitized target.

Preserve the command, compiler, source commit, AFL++ version, and original
crash input. Replay a finding by passing its file to the target. Minimize it
with:

```sh
afl-tmin -i CRASH -o MINIMIZED -- build/afl/c50_fuzz_implicit @@
```

Add the minimized input to the corpus and add a focused ordinary regression
with the repair. Keep the original artifact in the run record. A bounded
campaign establishes only that no finding occurred in those runs.

AFL++ is an external development dependency. The fuzz executables and AFL++
output are not installed. See the
[AFL++ documentation](https://github.com/AFLplusplus/AFLplusplus/blob/stable/docs/fuzzing_in_depth.md)
for corpus management, synchronized instances, resource limits, and finding
triage.

## Training target

The training target decodes up to 4096 bytes into a small classification
problem and trains it through the public C++ interface. It checks properties
and not only that training survives. For each decoded problem it trains once
from names and data text and once from dense arrays, and it requires:

- The same serialized classifier, apart from the first line, which holds the
  date, and the same costs.
- The same labels, confidences, and class scores for the training cases from
  both models. All must be finite.
- The same labels, confidences, and class scores from each model after it is
  saved and loaded.

Every byte string decodes to a valid problem, so any exception from training
is a finding, as is a violated property or a sanitizer report. The decoder
reads the following fields in order, one byte each unless noted. A byte past
the end of the input reads as zero. In the list, `b` is the byte.

| Field | Decoding |
| --- | --- |
| Model kind | `b % 2`: tree or rules. |
| Flags | Bit 0 subset splits, bit 1 winnowing, bit 2 global pruning, bit 3 probabilistic thresholds. |
| Trials | `1 + b % 3`. |
| Minimum cases | `b % 3` selects 1, 2, or 3. |
| Confidence factor | `b % 3` selects 0.05, 0.25, or 0.5. |
| Classes | `2 + b % 3`, named `class_0` onward. |
| Continuous attributes | `1 + b % 3`, named `feature_0` onward. |
| Categorical attributes | `b % 3`, named after the continuous ones. |
| Rows | `1 + b % 24`. |
| Cost entry | `b % (n + 1)` for `n` ordered class pairs. Zero means no costs, and `k` selects pair `k - 1`, with the predicted class varying slowest. |
| Cost | `2 + b % 8`. |
| Sample fraction | `b % 3` selects 0, 0.5, or 0.75. |
| Sample seed | Two bytes, big-endian, reduced modulo 4096. |
| Sampling | Used only with 12 or more rows and a nonzero fraction. |
| Levels | One byte per categorical attribute, `2 + b % 3`. |
| Cases | Per row, one byte per attribute, then one byte for the class, `b % classes`. |

A continuous byte `b` reduces to `m = b % 19`. Zero is missing, one is `N/A`,
and any other value is `(m - 10) / 2`. A categorical byte reduces modulo the
level count plus two. Zero is missing, one is `N/A`, and any other value `m`
is level `m - 2`. The dense arrays mark `N/A` with the not-applicable mask and
fill the masked value with NaN or `1e300` alternately, because the library
must ignore it.

CTest replays the seeds in `corpus/training` with the parser seeds, as
`fuzz-seeds`. The two `null-branch` seeds are the minimal inputs that crashed
pruning when a costs file made a subtree's error estimate exceed the number of
cases. Each trains a single continuous attribute and two classes with
confidence factor 0.05 and a cost of 5 or 3. The second also winnows. The
other seeds cover trees and rules, boosting, sampling, subsets, soft
thresholds, and not-applicable values.

The seeded driver `c50_training_fuzz_driver` builds on every platform, and
the `training-fuzz-driver-arguments` test checks its argument handling. The
driver trains for as long as asked, so only the `training-fuzz-hunt` target,
which needs a shell, runs a long hunt, and it exists only on Linux and macOS.
The driver accepts only nonnegative decimal integers and exits with status 2,
before it creates the input file, on anything else. It derives input `i` of
seed `s` from `std::mt19937_64`, whose sequence the C++ standard specifies, so
that two numbers replay any input on any platform. It writes the input to a
file before it trains, so that a crash leaves the input behind. Build and run
it through CMake, which reads its settings from the environment:

```sh
C50_HUNT_SEED=7 C50_HUNT_COUNT=100000 \
    cmake --build build --target training-fuzz-hunt
```

`C50_HUNT_SEED` defaults to 1, `C50_HUNT_COUNT` to 10,000, and `C50_HUNT_INPUT`
to `training-hunt-input` in the build directory. An exception prints the seed
and the index, and the input file holds the input. An abrupt stop, such as a
signal, an abort, or a sanitizer report, prints nothing. The input file then
holds the bytes of the input being trained, because the driver writes it
before each training. Replay those bytes directly. Make a directory `DIR`
with the empty subdirectories `schema`, `implicit`, `tree`, `rules`, `cases`,
and `costs`, and with the subdirectory `training` that holds the file as
`input`, then run `c50_fuzz_seed_replay` on it:

```bash
DIR=$(mktemp -d)
mkdir "$DIR"/{schema,implicit,tree,rules,cases,costs,training}
cp build/training-hunt-input "$DIR"/training/input
build/c50_fuzz_seed_replay "$DIR"
```

Alternatively, pass the file as the argument of a `c50_fuzz_training` harness
build, which runs each file named on its command line. To replay or keep a
finding by its seed and index, run the driver directly, then copy the file
into `corpus/training`:

```sh
build/c50_training_fuzz_driver SEED 1 input.bin INDEX
cp input.bin tests/fuzz/corpus/training/NAME
```

A bounded run establishes only that no finding occurred in those inputs.

The AFL++ harness `c50_fuzz_training` runs the same target, and
`fuzz-smoke-training` runs it for five seconds from `corpus/training` with no
dictionary, because the format is binary. It adds coverage guidance for longer
campaigns. Run it as described above, with `-i tests/fuzz/corpus/training` and
the target `build/afl/c50_fuzz_training`.
