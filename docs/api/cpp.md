# C++ API reference

The C++17 interface is compiled into the core library. Include
`<c50/c50.hpp>` and link `C50::cpp`. Options are ordinary value types.
Contexts, models, predictors, and prediction batches are move-only owners. See
{doc}`../ownership-concurrency` for lifetime and exception contracts.

```{doxygenenum} c50::error_code
:project: C50
```

```{doxygenenum} c50::model_kind
:project: C50
```

```{doxygenclass} c50::exception
:project: C50
:members:
```

```{doxygenclass} c50::context
:project: C50
:members:
```

```{doxygenstruct} c50::options
:project: C50
:members:
```

```{doxygenclass} c50::model
:project: C50
:members:
```

```{doxygenclass} c50::predictions
:project: C50
:members:
```

```{doxygenstruct} c50::dense_dataset
:project: C50
:members:
```

```{doxygenclass} c50::predictor
:project: C50
:members:
```

## Owned inspection records

Inspection records own their metadata and are copyable. See
{doc}`../model-inspection` for ordering, support counts, and prediction limits.

```{doxygenenum} c50::node_kind
:project: C50
```

```{doxygenenum} c50::condition_kind
:project: C50
```

```{doxygenstruct} c50::split_condition
:project: C50
:members:
```

```{doxygenstruct} c50::continuous_threshold
:project: C50
:members:
```

```{doxygenstruct} c50::tree_branch
:project: C50
:members:
```

```{doxygenstruct} c50::tree_node
:project: C50
:members:
```

```{doxygenstruct} c50::tree_inspection
:project: C50
:members:
```

```{doxygenstruct} c50::rule_inspection
:project: C50
:members:
```

```{doxygenstruct} c50::ruleset_inspection
:project: C50
:members:
```

```{doxygenstruct} c50::model_inspection
:project: C50
:members:
```
