# Parser fuzz targets

The six AFL++ targets use the public C++ interface and accept at most 4096 bytes
per input. Each iteration reuses the context for a known valid model and
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

The fixed schema has classes `no` and `yes`, continuous attribute `x`, and
categorical attribute `color` with levels `red`, `blue`, and `green`. The
implicit and costs targets train on four fixed cases. The remaining targets
load or predict without training. The engine-neutral target implementation also
supports deterministic seed replay through CTest. AFL++ uses the thin
`LLVMFuzzerTestOneInput` compatibility entry point supplied by `aflpp_driver`.

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
build also registers this replay as the `parser-fuzz-seeds` CTest test.
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

Preserve the command, compiler, source commit, AFL++ version, and original crash
input. Replay a finding by passing its file to the target. Minimize it with:

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
