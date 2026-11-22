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
throws `std::bad_alloc` when the host is out of memory. If the platform cannot
create the C locale for another reason, the operation throws `c50::exception`
with `error_code::internal_error`. Each context and each predictor owns its
locale, so constructing a context, calling `model::prepare_predictor`, and
making the first prediction with a predictor after a failed one can throw these
exceptions too. On Windows the operation enables the per-thread locale and
calls `setlocale` instead. The Python binding inherits this behavior.

Input and operation failures throw `c50::exception`, whose `code()` returns an
`error_code` and whose `what()` supplies a diagnostic. Allocation failures
throw `std::bad_alloc`. Invalid prediction indices throw `std::out_of_range`.
The operation boundary runs cleanup and rethrows the original exception. If
cleanup also fails, the original operation failure takes precedence. Library
code does not terminate the host process.

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
same exclusive context rule as prediction. The snapshot survives both the model
and context, and changes to a snapshot do not affect the classifier. See
{doc}`model-inspection` for serialization-precision and missing-value
semantics.

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
count. NaN denotes missing data. The optional `dense_dataset::not_applicable`
mask marks not-applicable values. It holds `row_count * feature_count` bytes
with no padding, whatever the row stride, and the caller keeps it alive and
unchanged for the call, like `values`. A nonzero byte makes the value not
applicable and hides the matching element of `values`; the library never
writes the mask. Concurrent operations may share one mask if none writes it.
Each dense operation checks the schema before the dataset. A schema that the
dense interface does not support raises `error_code::unsupported` whatever the
rows, width, stride, or buffers. A dataset that fails a later check raises
`error_code::invalid_argument`. A buffer is required only when the operation
can address one of its elements, so `values` may be null when `row_count` or
`feature_count` is zero.
Use text input for schemas unsupported by the dense interface.

## Stack use

Tree construction, pruning, rule generation, classification, and model parsing
recurse once per tree level on the calling thread. They run on the caller's
stack, because the library does not start a thread for them. The `c5.0` program
raises its stack limit to 20 MiB (or links with `/STACK:20971520` on Windows
with MSVC) before it runs. A library caller gets no such provision and must
supply the stack. Split workers run attribute evaluation, which does not
recurse with tree depth.

The recursion depth is the tree depth. The learner stops splitting a node when
fewer than two branches would hold `minimum_cases` cases, so the depth is at
most the number of training cases divided by `minimum_cases`. The learner
reaches that bound only when each split peels off about `minimum_cases` cases,
and a split must gain more than its cost, so realistic data stays far below it.
A synthetic set reaches it: 1,100 binary attributes and 1,100 classes, one
class per attribute, with two cases per class in which only that class's
attribute is true. The 2,200 cases train a tree of 1,100 levels, the deepest
measured, in about 23 seconds and 82 MB on x86-64 Linux with the `c5.0`
program. Each split peels off one class, so the depth grows with the number of
classes. Assigning classes by anything else, such as alternating them across
attributes, trains a single leaf.

Two limits cap what the library accepts. `model::load` rejects a serialized
tree of more than `max_tree_depth` levels with `error_code::parse_error`. The
limit counts levels with the root as level 1 and defaults to
`default_max_tree_depth`, 1,024. A later 1.x release may raise the default but
never lowers it. Pass `std::nullopt` (`None` in Python) to remove it for text
that you trust. A ruleset index limited to 512 levels falls back to iterative
matching, so no trained ruleset is unusable because of depth.

Training enforces no tree limit, and Release 2.07 enforces none when it prints
or reads a tree. The library therefore parses without a limit the text that it
wrote itself. A model that `model::train` returns, and a model that unpickling
restores, apply no limit when `predict`, `inspect`, and `prepare_predictor`
parse the text again. A model that `model::load` returns applies the
`max_tree_depth` that the caller gave, the same value on every later parse, so
a model loaded with `std::nullopt` stays usable. Copies of a Python `Model`
(`copy.copy`, `copy.deepcopy`) and its pickles restore through the unpickling
path and apply no limit, whatever limit the original used. A model with no
limit still needs stack in proportion to its depth to parse and to predict, as
in Release 2.07.

Measured stack use on x86-64 Linux, in bytes per tree level, with the base in
parentheses. Parsing includes freeing the parsed tree, because both recurse
once per level. Classification is `Classify` on a parsed predictor, where a
case with an unknown value follows every branch and a case with known values
follows one:

| Build | Parsing (`load`, `predict`, `inspect`, `prepare_predictor`) | Classifying, known values | Classifying, unknown value | Training |
| --- | --- | --- | --- | --- |
| GCC `-O3` (Release) | 144 (10 KiB) | 64 | 64 | about 215 |
| GCC `-O0` (Debug) | 144 (10 KiB) | 112 | 176 | about 255 |
| Clang AddressSanitizer and UBSan | 3,360 (11 KiB) | 1,664 | 2,016 | about 3,800 |

A call to `predict` or `inspect` parses and then classifies, but the two
phases do not nest, so the larger of the two figures applies: parsing. The
parsing and classification figures come from a probe that loads a synthetic
tree of 4,000 and of 12,000 levels (1,000 and 2,000 in the sanitizer build),
runs the operation on a thread with a set stack size, and bisects the smallest
size that does not crash. The slope between the two depths is the per-level
figure, and the intercept is the base. The compilers were GCC 13.3 and
Clang 18.1. The training figures come from an earlier measurement with GCC 14
and a deeper trained tree, and are rounded. Treat all figures as estimates for
planning, because they depend on the compiler and its version.

At the default limit, any accepted tree parses in under 160 KiB in a Release
build and under 3.4 MiB in a sanitizer build. Without the limit, a Release
build needs about 1.4 MiB for a tree of 10,000 levels, and a sanitizer build
needs about 32 MiB. Training has no bound of its own, but a tree of 1,024
levels trains in about 230 KiB in a Release build.

Callers should compare these figures with their thread stacks. The platform
defaults, which this repository did not measure, are 8 MiB for a Linux main
thread and a glibc secondary thread, 1 MiB for a Windows thread, 512 KiB for a
macOS secondary thread, and 128 KiB for a musl thread. A Release build parses
a tree of 1,024 levels on all of them except musl. A sanitizer build needs more
than 1 MiB at the default limit and must run on a thread with a larger stack,
for example through `threading.stack_size` in Python or
`pthread_attr_setstacksize`. Python threads use the platform default stack
unless `threading.stack_size` sets another. A service that parses models from
untrusted sources on small stacks should keep the limit or lower it.

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

`train_dense` copies `values` and `not_applicable` before releasing the GIL.
With `copy=False`, it reads a float64, C-contiguous, and aligned `values`
array and a C-contiguous bool `not_applicable` array directly, and the caller
must ensure that no thread modifies either array until training returns. Other
dtypes and layouts of `values`, an unaligned `values` array, and other layouts
of `not_applicable` are converted into a temporary array. A `not_applicable`
array must have a bool dtype and the shape of `values`. The scikit-learn
estimator uses `copy=False` for its private encoded matrix.

Python `Model` objects expose immutable classifier data. `Options` remains a
mutable value object. `train` and `train_dense` copy it when the call begins,
so changing an `Options` instance while training runs, from another thread,
does not affect that call.

`Model.prepare_predictor()` creates a Python `Predictor` with the same owned
native state described above. Its `predict_details` and `predict_details_dense`
methods return ordinary `Predictions`. Calls on one Python predictor acquire an
instance lock after releasing the GIL. Independent predictors can run in
parallel. Dense inputs, including `not_applicable`, are copied before
releasing the GIL. Persist the source
`Model`, which supports pickling, and recreate predictors after loading it.

The scikit-learn estimator creates one predictor per `predict` or
`predict_proba` call and reuses it across that call's batches. It retains no
mutable prediction workspace between calls, so independent calls on a fitted
estimator remain safe.
