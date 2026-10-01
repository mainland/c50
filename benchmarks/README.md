# Benchmarks

The benchmark suite measures the released behavior of the library. It answers
two questions:

1. How does the library's `c5.0` program compare with the imported C5.0
   Release 2.07 program in time and peak memory, and do both programs write
   the same classifier?
2. How does training time change with the number of split workers and with
   the tie order, and is the classifier the same for every worker count?

`run.py` runs a suite and writes one JSON result.

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

Every result records the source commit, the release tag when a clean checkout
of exactly one tag was measured, the compiler and the command that compiled a
learner source file, checksums of the measured programs and Python module, the
archive checksum of the reference program, the CPU and its frequency governor,
the load average before each run, and every sample.

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

The driver builds the library from the checkout in `Release` mode, and builds
the reference program as the upstream Makefile builds its production
`c5.0`, with `gcc -ffloat-store -O3` on the concatenated sources. It refuses to
measure a working tree with uncommitted changes unless `--allow-dirty` is
given. Datasets are cached in `~/.cache/c50-benchmarks` unless `--cache`
names another directory.

The release suite repeats each measurement five times and takes several
hours. Run it on an otherwise idle machine. The result
records the load average and frequency governor, but it cannot correct for
contention.

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
