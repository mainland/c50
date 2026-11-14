# Ownership and concurrency

## Native C++ interface

`c50::context` owns mutable parser, training, and prediction workspace. A
context permits one active operation at a time. Callers must serialize all
access to a context. A failed operation performs cleanup and leaves the
context available for another operation.

Set the maximum split-evaluation worker count with `context.split_workers(n)`
before training. The default is one. Counts from 1 through 8 are accepted.
Eligible attribute evaluations use separate worker scratch state within that
operation. This setting does not permit concurrent calls on the same context.

Every operation is independent of the host locale. Training, loading,
prediction, inspection, and serialization read and write numbers in the C
locale, so model text always uses `.` as the decimal point, even when the
caller's process or thread locale uses `,`. An operation selects the C locale
for the calling thread, and for each split worker thread it starts, and
restores the caller's thread locale when it returns or throws. It does not
change the process locale. If the C locale cannot be created, the operation
throws `std::bad_alloc` when the host is out of memory. If the
platform cannot create the C locale for another reason, the operation throws
`c50::exception` with `error_code::internal_error`. On Windows the operation
enables the per-thread locale and calls `setlocale` instead. The Python
binding inherits this behavior.

Input and operation failures throw `c50::exception`, whose `code()` returns an
`error_code` and whose `what()` supplies a diagnostic. Allocation failures throw
`std::bad_alloc`. Invalid prediction indices throw `std::out_of_range`. The
operation boundary runs cleanup and rethrows the original exception. If cleanup
also fails, the original operation failure takes precedence. Library code does
not terminate the host process.

Training with more than one split worker can also throw `std::system_error`,
which is not a `c50::exception`. It reports that the system could not start a
worker thread, for example when the user's process limit is exhausted, and its
`code()` carries the operating-system error. Only `model::train` starts split
workers, and only when `split_workers` is above one and a node holds at least
10,000 cases. Loading, prediction, inspection, and serialization start no
threads. The failed call joins every worker that it started and leaves the
context reusable.
`tests/parallel_failure.cpp` injects the failure and asserts this behavior.
Catch `std::system_error` separately, or retry with `split_workers(1)`.

`c50::model::train` and `c50::model::load` copy the retained schema, serialized
classifier, and applicable costs. Models are immutable and independent of their
originating contexts. Multiple threads may use a model concurrently with
separate contexts. Each call currently parses the retained classifier into its
context's workspace.

`model.prepare_predictor()` returns a move-only `predictor` that owns a parsed
classifier, serialized recovery data, and an independent workspace. Reuse it
for small prediction batches to avoid parsing the classifier on each call.
It may outlive the model. Callers must serialize operations on one predictor.
Independent predictors may run concurrently. Each result owns its storage.
Native case records are released after each call. Parser scratch may remain
allocated until the predictor is destroyed. After a native failure, the next
prediction rebuilds the parsed state from the retained model bytes.
This trades retained memory for lower repeated-call latency.

`model.inspect` returns an owned, copyable classifier snapshot with class and
feature names, nodes, branches, rules, and structural statistics. It uses the
same exclusive context rule as prediction. The snapshot survives both the
model and context, and changes to a snapshot do not affect the classifier. See
{doc}`model-inspection` for serialization-precision and missing-value semantics.

`model.predict` returns an independently owned `c50::predictions` batch.
Results retain class names, labels, confidence values, and scores after the
originating model or context is destroyed. Indexed access checks bounds.

Contexts, models, predictors, and prediction batches are move-only. Moved-from
objects may only be destroyed or assigned another object. References returned
by model and result accessors remain valid until the owning data is destroyed
or replaced.
Destruction and move assignment require exclusive access to the object.

Text views and dense arrays are borrowed for the duration of a call. The caller
must keep their storage alive and unchanged until the call returns. Dense
row strides are measured in `double` elements, with zero meaning the feature
count. NaN denotes missing data. Use text input for not-applicable values and
schemas unsupported by the dense interface.

## Stack use

Tree construction, pruning, rule generation, classification, and model parsing
recurse once per tree level on the calling thread. They run on the caller's
stack, because the library does not start a thread for them. The `c5.0` program
raises its stack limit to 20 MiB (or links with `/STACK:20971520` on Windows
with MSVC) before it runs. A library caller gets no such provision and must
supply the stack. Split workers run attribute evaluation, which does not recurse
with tree depth.

The recursion depth is the tree depth. The learner stops splitting a node when
fewer than two branches would hold `minimum_cases` cases, so the depth is at most
the number of training cases divided by `minimum_cases`. The learner reaches
that bound only when each split peels off about `minimum_cases` cases, and a
split must gain more than its cost, so realistic data stays far below it.
A synthetic set reaches it: 1,100 binary attributes and 1,100 classes, one
class per attribute, with two cases per class in which only that class's
attribute is true. The 2,200 cases train a tree of 1,100 levels, the deepest
measured, in about 23 seconds and 82 MB on x86-64 Linux with the `c5.0`
program. Each split peels off one class, so the depth grows with the number of
classes. Assigning classes by anything else, such as alternating them across
attributes, trains a single leaf.

Two limits cap what the library accepts. The parser rejects a serialized tree
of 1,024 or more levels with `error_code::parse_error`. A ruleset index limited
to 512 levels falls back to iterative matching. Training does not enforce the
tree limit. `model::train` returns a deeper tree, which its own `predict`,
`inspect`, and `prepare_predictor` accept, but a `load` of its serialized text
throws the same `parse_error`, and so does unpickling it in Python. Release
2.07 prints such a tree. See
{doc}`compatibility`.

Measured stack use on x86-64 Linux, in bytes per tree level, adds to a base of
about 10 KiB for training and 40 KiB for parsing:

| Build | Parsing a tree (`load`, `predict`, `inspect`) | Training a tree |
| --- | --- | --- |
| GCC 14 `-O3` (Release) | about 140 | about 215 |
| GCC 14 `-O0` (Debug) | about 140 | about 255 |
| Clang AddressSanitizer and UBSan | about 3,300 | about 3,800 |

Parsing is bounded by the 1,024-level limit, so any accepted tree parses in
under 160 KiB in a Release build and under 3.4 MiB in a sanitizer build.
Parsing a ruleset needs at most 71 KiB in a Release build, because rule indexes
stop at 512 levels. Training has no bound of its own, but its stack use is
about 215 bytes per level in a Release build, so a tree that the library
can still return (under 1,024 levels) trains in about 230 KiB.

Callers should compare these figures with their thread stacks. The platform
defaults, which this repository did not measure, are 8 MiB for a Linux main
thread and a glibc secondary thread, 1 MiB for a Windows thread, 512 KiB for a
macOS secondary thread, and 128 KiB for a musl thread. A Release build fits
all of them except musl. A sanitizer build needs more than 1 MiB at the
1,024-level limit and must run on a thread with a larger stack, for example
through `threading.stack_size` in Python or `pthread_attr_setstacksize`.
Python threads use the platform default stack unless `threading.stack_size`
sets another.

## Python interface

Each Python `Model` training, loading, or prediction call constructs a local
native context. Native work releases the Python GIL. Independent operations can
therefore execute concurrently, and the binding does not depend on hidden
process-global classifier state.

Training with `split_workers` above one can raise `RuntimeError` that is not a
`C50Error`, when the system cannot start a worker thread. The binding does not
translate the native `std::system_error`, so pybind11 reports it as a plain
`RuntimeError` whose message is the operating-system text, such as "Resource
temporarily unavailable". Catch `RuntimeError` to cover both it and `C50Error`,
which derives from `RuntimeError`, or retry with `split_workers=1`.

The binding does not copy text arguments. A `str` cannot change, and the call
keeps it alive, so native code reads its UTF-8 representation in place after
the binding releases the GIL.

`train_dense` copies `values` before releasing the GIL. With `copy=False`, it
reads a float64, C-contiguous array directly, and the caller must ensure that no
thread modifies the array until training returns. Other dtypes and layouts are
converted into a temporary array. The scikit-learn estimator uses `copy=False`
for its private encoded matrix.

Python `Model` objects expose immutable classifier data. `Options` remains a
mutable value object. `train` and `train_dense` copy it when the call begins,
so changing an `Options` instance while training runs, from another thread,
does not affect that call.

`Model.prepare_predictor()` creates a Python `Predictor` with the same owned
native state described above. Its `predict_details` and `predict_details_dense`
methods return ordinary `Predictions`. Calls on one Python predictor acquire an
instance lock after releasing the GIL. Independent predictors can run in
parallel. Dense inputs are copied before releasing the GIL. Persist the source
`Model`, which supports pickling, and recreate predictors after loading it.

The scikit-learn estimator creates one predictor per `predict` or `predict_proba`
call and reuses it across that call's batches. It retains no mutable prediction
workspace between calls, so independent calls on a fitted estimator remain safe.
