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

## Model operation measurements

`large_dataset.py` also records repeated model-load, serialized-string-copy,
empty-prediction, and small/batched-prediction timings. Set
`--operation-repetitions N` to change the five repetitions. Predictions are
warmed before measurement and checked for exact score equality afterward.
`model_load` includes parsing, validation, and owned copies. Empty prediction
includes parser/setup costs but no case classification. Retrieving
`serialized_data` copies an existing string in the Python binding. It does not
measure native serialization, which runs during training. Use a sampled native
profile to attribute time within training. These operation boundaries overlap
and must not be summed as independent training phases.
