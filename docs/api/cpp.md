# C++ API reference

The C++17 interface is compiled into the core library. Include
`<c50/c50.hpp>` and link `C50::cpp`. Options are ordinary value types.
Contexts, models, and prediction batches are move-only owners. See
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
