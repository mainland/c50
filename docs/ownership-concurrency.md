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

`c50::model::train` and `c50::model::load` copy the retained schema, serialized
classifier, and applicable costs. Models are immutable and independent of their
originating contexts. Multiple threads may use a model concurrently with
separate contexts. Each call currently parses the retained classifier into its
context's workspace.

`model.predict` returns an independently owned `c50::predictions` batch.
Results retain class names, labels, confidence values, and scores after the
originating model or context is destroyed. Indexed access checks bounds.

Contexts, models, and prediction batches are move-only. Moved-from objects may
only be destroyed or assigned another object. References returned by model and
result accessors remain valid until the owning data is destroyed or replaced.
Destruction and move assignment require exclusive access to the object.

Text views and dense arrays are borrowed for the duration of a call. The caller
must keep their storage alive and unchanged until the call returns. Dense
row strides are measured in `double` elements, with zero meaning the feature
count. NaN denotes missing data. Use text input for not-applicable values and
schemas unsupported by the dense interface.

Each dense operation checks the schema before the dataset. A schema that the
dense interface does not support raises `error_code::unsupported` whatever the
rows, width, stride, or buffers. A dataset that fails a later check raises
`error_code::invalid_argument`. A buffer is required only when the operation
can address one of its elements, so `values` may be null when `row_count` or
`feature_count` is zero.

## Python interface

Each Python training, loading, or prediction call constructs a local native
context. Native work releases the Python GIL. Independent operations can
therefore execute concurrently, and the binding does not depend on hidden
process-global classifier state.

Python `Model` objects expose immutable classifier data. `Options` remains a
mutable value object. `train` and `train_dense` copy it when the call begins,
so changing an `Options` instance while training runs, from another thread,
does not affect that call.
