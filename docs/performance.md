# Performance

This page reports the release benchmark suite. The suite compares this
library's `c5.0` program with the imported C5.0 Release 2.07 program on the
same input files, measures training with different numbers of split workers
and both tie orders, and measures the overhead of the scikit-learn estimator
over the low-level Python interface. `benchmarks/report.py` generates the
tables from one retained result. The repository's `benchmarks/README.md`
describes the datasets and the measurement protocol, and explains how to
repeat the measurements.
The report checker accepts the no-results placeholder until a release result
is retained. Afterward, it requires the tables to match that result.

## Results

<!-- begin generated benchmark results -->

No release result has been retained yet.

<!-- end generated benchmark results -->

## Reading the tables

In the first table, each time is the wall time of the whole program, which
reads the data, trains the classifier, writes it, and evaluates it on the
training cases. The time ratio divides the library's time by the time of
C5.0 Release 2.07, so a ratio below one means that the library is faster. Peak
memory is the maximum resident set size of the process. "Same classifier"
means that every run of both programs wrote the same serialized classifier,
apart from the dated `id` line.

In the second table, each time covers one call of the low-level Python function
`c50.train`, which parses the data text and trains a tree. The speedup divides
the one-worker time by the time with the most workers. Only nodes with at least
10,000 training cases and more than one eligible attribute use more than one
worker, so the speedup depends on the dataset and on the shape of the tree.
"Same classifier for all worker counts" means that every run with the tie
order wrote the same classifier. The stable tie order may produce a classifier
that differs from the reference order, as {doc}`compatibility` describes, and
the last column reports whether it did.

The third table compares the scikit-learn estimator `C50Classifier` with the
low-level interface on the same data, with one split worker. The `c50.train`
time covers parsing the data text and training. The `C50Classifier.fit` time
covers converting and encoding a DataFrame with categorical columns, training,
and computing the attribute usage behind `feature_importances_` with a second
pass over the training rows. The fit ratio divides the estimator's time by the
`c50.train` time, so a ratio above one is the estimator's overhead. The
prediction times cover `Model.predict_details` on the data text and
`predict_proba` on the DataFrame, both for the training rows. Peak memory
covers the whole Python process, including the interpreter, the imported
modules, and the input that each interface reads. "Same predictions" means that
both interfaces predicted the same class for every training case in every run.

The results describe one machine, one compiler, and these datasets. They are
not general time or memory bounds.
