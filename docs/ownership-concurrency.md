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

`model.inspect` returns an owned, copyable tree snapshot with class and feature
names, nodes, branches, and structural statistics. It uses the same exclusive
context rule as prediction. The snapshot survives both the model and context,
and changes to a snapshot do not affect the classifier. See
{doc}`model-inspection` for serialization-precision and missing-value semantics.

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

## Python interface

Each Python training, loading, or prediction call constructs a local native
context. Native work releases the Python GIL. Independent operations can
therefore execute concurrently, and the binding does not depend on hidden
process-global classifier state.

Python `Model` objects expose immutable classifier data. `Options` remains a
mutable value object, so callers must not modify one instance concurrently with
a training call that uses it.
