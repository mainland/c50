# Native benchmarks

Build the optional native benchmarks with Catch2 3 installed:

```sh
cmake -S . -B build/bench -DCMAKE_BUILD_TYPE=Release -DC50_BUILD_BENCHMARKS=ON
cmake --build build/bench
ctest --test-dir build/bench --output-on-failure
```

`c50-native-benchmarks` checks sort ordering and record preservation.
`c50-native-workload` generates a deterministic dense
training dataset and reports training time and a serialized-classifier digest.
Its `--help` output lists the workload options. Keep profiling runs separate
from the default local tests.
