# Benchmarks

The benchmark suite measures the released behavior of the library. It answers
three questions:

1. How does the library's `c5.0` program compare with the imported C5.0
   Release 2.07 program in time and peak memory, and do both programs write
   the same classifier?
2. How does training time change with the number of split workers and with
   the tie order, and is the classifier the same for every worker count?
3. How much time and memory does the scikit-learn estimator `C50Classifier`
   add to the low-level interface, and does it predict the same classes?

`run.py` runs a suite and writes one JSON result. `report.py` renders a result
as the tables in `docs/performance.md`.

## Measurements

The comparison trains each dataset with both command-line programs, alternating
their order in successive repetitions. Both programs read the same `.names` and
`.data` files in a fresh directory. The recorded time is the wall time of the
whole process, which includes reading the data and evaluating the classifier on
the training cases. Peak memory is the process's maximum resident set size.
On Linux, the peak that `wait4` reports for a child includes the memory of the
process that started it, so `run.py` starts each measured process through
`measure.c`, a small launcher. The comparison requires the serialized
classifiers to be identical apart from the dated `id` line.

The worker measurements train a tree through the low-level Python function
`c50.train` with 1, 2, 4, and 8 split workers, in a fresh process for each run.
The recorded time covers only the `train` call, which parses the data text and
trains the classifier. Each worker count must produce the same classifier. The
stable tie order may produce a classifier that differs from the reference
order, and the result records whether it does.

The estimator measurements compare `C50Classifier` with the low-level
interface on the comparison datasets, alternating the two in successive
repetitions, each in a fresh process. The low-level run times `c50.train` on
the data text and then `Model.predict_details` on the same text. The estimator
run reads the files into a DataFrame with pandas categorical columns, which
is not timed, and then times `fit` with `categorical_features="from_dtype"`
and `predict_proba` on the training rows. `fit` includes the DataFrame
conversion, encoding, training, and the attribute-usage pass behind
`feature_importances_`. Peak memory covers the whole process, including the
input. Both interfaces must predict the same class for every training case.

The DataFrame gives each categorical column the values that the `.names`
file declares, in the same order, and `C50Classifier` keeps a categorical
dtype's order. Both interfaces therefore declare the same values in the same
order. This matters because C5.0 can build a different classifier when the
same values are declared in another order, as `docs/compatibility.md`
describes.

Every result records the source commit, the release tag when a clean checkout
of exactly one tag was measured, the compiler and the command that compiled a
learner source file, checksums of the measured programs and Python module, the
archive checksum of the reference program, the CPU and its frequency governor,
the load average before each run, and every sample. The rendered report names
the tag but not the commit, because a history rewrite would invalidate a
commit hash in the documentation.

## Datasets

| Name | Source | Rows | Attributes | Classes |
| --- | --- | ---: | ---: | ---: |
| `adult` | OpenML dataset 1590, version 2 | 48,842 | 14 | 2 |
| `covertype` | OpenML dataset 1596, version 4 | 581,012 | 54 | 7 |
| `synthetic-1000000x100-c10` | `datasets.py`, seed 1729 | 1,000,000 | 100 | 3 |

The synthetic workload has 90 continuous attributes, 10 categorical attributes
with eight values each, and 1% missing values. Its class depends on a weighted
sum of eight continuous attributes. It matches the scale target in
`docs/large-datasets.md`.

`datasets.py` writes each dataset once as C5.0 files and records their SHA-256
digests. Categorical values and classes become generated tokens such as `v3`
and `c1`, so no value can change the C5.0 input grammar. The OpenML datasets
are downloaded on first use.

## Running the suite

The suite requires Linux or macOS, CMake, a C++17 compiler, GCC for the
reference program, and Python with NumPy, pandas, scikit-learn, and pybind11.
The development environment described in `docs/building-testing.md` provides
them.

The `smoke` suite trains small synthetic datasets in a few seconds and needs no
network access. CI runs it through `tests/benchmark_smoke.py`:

```sh
python benchmarks/run.py --suite smoke --output smoke.json
```

The `release` suite requires the canonical upstream archive, `C50.tgz`, whose
SHA-256 digest must be
`309db588eda420c06701bf8ae74c06a6c923e9a06e714a598ea761bcadfc5e2e`:

```sh
python benchmarks/run.py --suite release \
    --reference-archive ../archive/C50.tgz \
    --output benchmarks/results/release.json
```

The driver builds the command-line program from the checkout in `Release`
mode with the ordinary core. It builds the Python extension separately with
`C50_PRIVATE_CORE=ON`, matching Python package builds: hidden native symbols
and LTO when the compiler supports it. Each build has its own recorded
compiler settings. `--installed-python` uses the installed Python package
instead and records its version and extension checksum.

The driver builds the reference program as the upstream Makefile builds its
production `c5.0`, with `gcc -ffloat-store -O3` on the concatenated sources.
It refuses to measure a working tree with uncommitted changes unless
`--allow-dirty` is given. Datasets are cached in `~/.cache/c50-benchmarks`
unless `--cache` names another directory.

The release suite repeats each measurement five times and takes several
hours. Run it on an otherwise idle machine. The result
records the load average and frequency governor, but it cannot correct for
contention.

Check the result and report through the installed Python package with
`python tests/benchmark_smoke.py`. With an LTO-capable compiler, check the
separate Python build as well:

```sh
python tests/benchmark_smoke.py --build-python
```

## Results

Retained results are under `benchmarks/results/`. Each release documents one
result in `docs/performance.md`. After running the release suite, regenerate the
page's tables:

```sh
python benchmarks/report.py benchmarks/results/release.json \
    --update docs/performance.md
```

Check that the page still matches the retained result with `--check`:

```sh
python benchmarks/report.py benchmarks/results/release.json \
    --check docs/performance.md
```

## Interpretation

Each table entry is the median of the repetitions on one machine. The results
describe that machine, compiler, and these datasets. They are not general time
or memory bounds. Peak resident set size has different semantics on Linux and
macOS, so do not compare memory across operating systems.

## Profiling

To profile training, run the built program on a cached dataset. For example,
sample the CPU with `perf` on Linux:

```sh
perf record -g -o /tmp/c50.perf -- \
    build/c5.0 -f ~/.cache/c50-benchmarks/files/covertype/covertype
perf report -i /tmp/c50.perf
```
