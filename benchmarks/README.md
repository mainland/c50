# Large-dataset benchmark

`large_dataset.py` measures the Python estimator's dense training path and
batched prediction path. It generates a deterministic mixed continuous and
categorical workload and writes one JSON result to standard output. The result
includes wall and CPU time, peak-RSS checkpoints, and SHA-256 digests of stable
serialized model content and prediction scores.

Run a quick development workload from an installed development environment:

```sh
python benchmarks/large_dataset.py \
    --rows 100000 --features 50 --categorical-features 5
```

Run the current target envelope with one exact, pruned tree:

```sh
python benchmarks/large_dataset.py \
    --rows 1000000 --features 100 --categorical-features 10 \
    >large-dataset.json
```

Record the commit, compiler, build type, and command alongside retained JSON
results. Do not compare peak RSS values across operating systems without
accounting for their different resource-reporting semantics.

`run_matrix.py` runs each workload in a fresh process so peak RSS is local to
that workload. The default smoke profile is suitable for development:

```sh
python benchmarks/run_matrix.py --profile smoke >benchmark-matrix.json
```

The `standard` profile adds medium continuous and mixed workloads. The `large`
profile additionally runs the million-row target envelope and therefore
requires substantially more time and memory.

## Native benchmarks and profiling

The optional native benchmark targets require Catch2 3.x. Configure an
optimized build with debug information:

```sh
cmake -S . -B build/benchmarks \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DC50_BUILD_BENCHMARKS=ON
cmake --build build/benchmarks -j
```

The Catch2 target verifies `Cachesort` exhaustively for short inputs, on larger
representative distributions, and across finite floating-point extremes,
including denormal values and signed zero. The properties require sorted
output with the same records but do not require stable ordering of equal
values. CTest runs these correctness properties:

```sh
ctest --test-dir build/benchmarks -R native-sort-properties \
    --output-on-failure
```

`c50-native-workload` trains one deterministic tree through the public dense C++
API without importing Python, NumPy, or scikit-learn. Its generator differs
from the Python large-dataset generator, so compare its model hash only with
other runs of the native workload:

```sh
build/benchmarks/c50-native-workload \
    --rows 100000 --features 50 --categorical-features 5
```

Use `--workers N` to select 1 through 8 split-evaluation workers. Training
uses multiple workers only for non-verbose nodes with at least 10,000 rows
and multiple eligible attributes, without legacy split-value subsampling.
Training-row sampling remains eligible; other phases remain serial.
The optional CTest case `parallel-workload-equivalence` checks exact model
digests across worker counts near that threshold. Benchmark each worker count
on the same workload and compare both time and peak RSS; more workers may
increase memory use without improving throughput.

Use the native workload for CPU sampling:

```sh
perf record -o /tmp/c50-perf.data -F 199 -e cycles:u \
    -g --call-graph dwarf,8192 -- \
    build/benchmarks/c50-native-workload \
    --rows 100000 --features 50 --categorical-features 5 --quiet
perf report -i /tmp/c50-perf.data
```

Unprivileged profiling requires a host policy that permits user-space
performance counters. On Linux, `kernel.perf_event_paranoid=2` permits the
user-space-only `cycles:u` event used above without enabling kernel profiling.

Use Heaptrack for allocation profiles. `--quiet` avoids attributing the C
library's standard-output buffer to the workload at process exit:

```sh
heaptrack -o /tmp/c50-native-heaptrack \
    build/benchmarks/c50-native-workload \
    --rows 100000 --features 50 --categorical-features 5 --quiet
heaptrack_print -f /tmp/c50-native-heaptrack.zst \
    --print-leaks=1 --print-peaks=0 --print-allocators=0 \
    --print-temporary=0
```
