# Large datasets and interpretable trees

The scale objective is to construct interpretable C5.0 decision trees from
reasonably large in-memory datasets without changing the classifier produced
by the reference GPL implementation. The current measurement envelope is:

- 1,000,000 rows;
- 50--100 input attributes;
- no more than 10 classes;
- mixed continuous and categorical values;
- one pruned tree, with sampling disabled; and
- peak resident memory below 16 GiB.

This is a target to test, not a claim that every workload in the envelope
already meets it. Tree shape, categorical cardinality, missingness, and enabled
options can materially change time and memory use.

## Model size

A tree learned from a large dataset can have tens of thousands of
records, which is too many for direct inspection. Data
capacity and interpretability are separate acceptance criteria: practical
workloads need explicit model-size measurements and suitable pruning and
`minimum_cases` settings. The inspection API reports node counts, supported
leaves, depth, and feature use without application-side parsing. Python text
and JSON exports use the same owned metadata. See {doc}`model-inspection`.

## Implemented data path

The C++ API accepts a borrowed, row-major `c50::dense_dataset`. Its
training and prediction overloads call the compiled C++ core. Continuous values
are doubles, categorical values are zero-based indices into explicit schema
values, and NaN denotes a missing value.

The Python estimator encodes array input directly into this dense
representation. It no longer builds a second, whole-dataset C5.0 text buffer.
Numeric NumPy columns, including DataFrame columns with numeric NumPy dtypes,
use vectorized conversion. Categorical columns retain a fitted index mapping.
Prediction conversion and native prediction are split into batches of 65,536
rows by default. Set `prediction_batch_size` to another positive integer, or to
`None` to process one prediction batch.

The original text APIs remain available for file-format compatibility. A
differential native test trains equivalent text and dense inputs and requires
byte-identical serialized models and identical prediction details.

## Memory model and limits

Training is in-memory, not streaming or out-of-core. The estimator's encoded
float64 matrix and the core's case records coexist during training. The
original input may also remain live. The estimator passes its private encoded
matrix to the binding with `copy=False`, so the binding does not copy it again.
The core also allocates row- and attribute-dependent split, sort, pruning,
and boosting workspaces. Boosting adds substantial row-by-class state, so a single tree is
the appropriate baseline for both memory and interpretability.

The public dense interface rejects more than `INT_MAX` rows or features because
the release 2.07 algorithm uses signed `int` indices internally. Dense schemas
currently require ordinary explicit continuous or categorical input
attributes. Dynamic `discrete N`, implicit, ignored, label, class-attribute,
date, and time declarations remain available only through the text interface.
Sparse matrices and memory-mapped out-of-core training are not supported.

Per-batch case storage is bounded by `prediction_batch_size`, apart from the
caller's input and returned score array. One parsed predictor is retained for
the duration of each estimator prediction call. Training still requires the
complete encoded matrix and complete native case store.

The classifier is retained in the compatible C5.0 serialized tree format.
The command-line program prints the familiar tree report, while the library
interfaces expose `serialized_data` for storage and interchange. The C++
`model.inspect` and Python `Model.inspect()` APIs expose owned node metadata.
Python `export_text` and `export_json` render that metadata for inspection.

## Exact and approximate modes

The baseline uses `trials=1` and `sample_fraction=0`. This preserves exact
single-tree learning and is the mode used for compatibility work. Setting
`sample_fraction` above zero can reduce training work, but deliberately changes
the cases seen by the learner and may change the classifier. It must be
described as approximate sampling rather than a transparent optimization.

Set `ties` to `stable` to evaluate continuous splits with a faster sort that
keeps equal values in case order. Training remains deterministic on every
platform, but the classifier may differ from the reference learner. See
{doc}`compatibility`.

Set the estimator's `n_jobs`, or the native `split_workers`, to enable bounded
parallel evaluation of eligible attribute splits. At most 8 workers are used,
and one worker is the default. Nodes need at least 10,000 training rows and
multiple eligible attributes. Verbose diagnostics and legacy split-value
subsampling remain serial. Training-row sampling remains
eligible. The cross-worker regression test compares exact classifier digests
across worker counts, including subset splits. Other training phases remain
serial.

## Reproducible measurement

The repository includes `benchmarks/large_dataset.py`. It reports workload
parameters, platform information, elapsed and CPU time, peak-RSS checkpoints,
serialized model size, and deterministic model and prediction digests as JSON.

Start with a development-sized run:

```sh
python benchmarks/large_dataset.py \
    --rows 100000 --features 50 --categorical-features 5
```

Then measure the target envelope on a machine with sufficient memory:

```sh
python benchmarks/large_dataset.py \
    --rows 1000000 --features 100 --categorical-features 10 \
    >large-dataset.json
```

Retain the exact command, Git commit, compiler, optimization level, and JSON
result. Measure before changing split evaluation, sorting, or allocation
behavior. An optimization is acceptable only when the compatibility suite and
dense/text differential tests still pass.

Run the isolated smoke matrix while developing instrumentation or ownership
changes:

```sh
python benchmarks/run_matrix.py --profile smoke >benchmark-matrix.json
```

The `standard` profile adds two medium workloads. The `large` profile also runs
the million-row target envelope and is intended for deliberate baseline runs,
not routine tests.
