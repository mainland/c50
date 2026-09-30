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

Input and operation failures throw `c50::exception`, whose `code()` returns an
`error_code` and whose `what()` supplies a diagnostic. Allocation failures throw
`std::bad_alloc`. Invalid prediction indices throw `std::out_of_range`. The
operation boundary runs cleanup and rethrows the original exception. If cleanup
also fails, the original operation failure takes precedence. Library code does
not terminate the host process.

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

## Python interface

Each Python `Model` training, loading, or prediction call constructs a local
native context. Native work releases the Python GIL. Independent operations can
therefore execute concurrently, and the binding does not depend on hidden
process-global classifier state.

`train_dense` copies `values` before releasing the GIL. With `copy=False`, it
reads a float64, C-contiguous array directly, and the caller must ensure that no
thread modifies the array until training returns. Other dtypes and layouts are
converted into a temporary array. The scikit-learn estimator uses `copy=False`
for its private encoded matrix.

Python `Model` objects expose immutable classifier data. `Options` remains a
mutable value object, so callers must not modify one instance concurrently with
a training call that uses it.

`Model.prepare_predictor()` creates a Python `Predictor` with the same owned
native state described above. Its `predict_details` and `predict_details_dense`
methods return ordinary `Predictions`. Calls on one Python predictor acquire an
instance lock after releasing the GIL. Independent predictors can run in
parallel. Dense inputs are copied before releasing the GIL. Persist the source
`Model`, which supports pickling, and recreate predictors after loading it.

The scikit-learn estimator creates one predictor per `predict` or `predict_proba`
call and reuses it across that call's batches. It retains no mutable prediction
workspace between calls, so independent calls on a fitted estimator remain safe.
