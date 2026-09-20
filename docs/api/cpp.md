# C++ API reference

The C++11 interface is header-only but links transitively to the compiled C
core through the `C50::cpp` CMake target. It supplies ownership and exception
handling without implementing classifier behavior.

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

```{doxygenclass} c50::options
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
