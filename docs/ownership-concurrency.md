# Ownership and concurrency

## C interface

`c50_context` owns mutable parser, allocator, diagnostic, training, and
prediction state. A context permits one active operation at a time. A failed
operation records a status and diagnostic in its context, performs operation
cleanup, and leaves the context available for another operation.

`c50_model_train` and `c50_model_load` return independently owned models. A
model retains copies of its names data, serialized classifier, and applicable
costs data. Destroying the context used to create a model does not invalidate
the model.

Models are immutable after construction. Multiple threads may use the same
model concurrently when each operation has a different context. The owner must
keep the model alive until all operations finish.

`c50_model_predict` returns independently owned results. Prediction results
retain their class names and values after the originating context or model is
destroyed.

## C++ facade

`c50::context`, `c50::model`, and `c50::predictions` provide RAII ownership of
their corresponding C handles. These types are move-only. A native failure
becomes `c50::exception`, which retains the corresponding `c50_status`.

The facade does not add synchronization. Calls using the same `c50::context`
must remain serialized. An immutable `c50::model` may be shared across threads
when each call receives a separate context.

## Python interface

Each Python training, loading, or prediction call constructs a local native
context. Native work releases the Python GIL. Independent operations can
therefore execute concurrently, and the binding does not depend on hidden
process-global classifier state.

Python `Model` objects expose immutable classifier data. `Options` remains a
mutable value object, so callers must not modify one instance concurrently with
a training call that uses it.
