# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests that every array-like input gives the result of a NumPy array.

`fit` reads `X`, `y`, and `sample_weight`, and the prediction methods read `X`.
Each container below holds the same numbers as a NumPy array, and each role
must produce the classifier or predictions that the NumPy arrays produce. The
stubs expose one conversion protocol each, as the libraries that the guide
names do, so that the tests run without those libraries. The tests for the
libraries themselves use `pytest.importorskip`.
"""

from __future__ import annotations

import array
import datetime
import mmap
import os
import sys
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest
import sklearn
from c50.sklearn import C50Classifier, _host_array
from model_text import without_date
from scipy import sparse

_rng = np.random.default_rng(5)
X = _rng.normal(size=(60, 3))
Y = (X[:, 0] + 0.5 * X[:, 1] + _rng.normal(scale=0.5, size=60) > 0).astype(
    np.int64
)
W = np.linspace(0.5, 2.0, 60)


class _Dtype:
    """A dtype that NumPy cannot interpret, as PyTorch's is."""

    def __repr__(self) -> str:
        return "fake.dtype"


class _Stub:
    """Base of the stubs, which hold a NumPy array and expose one protocol."""

    def __init__(self, values: np.ndarray) -> None:
        self._values = values


class _TensorLike(_Stub):
    """Has `__array__` and a `dtype` that is not a NumPy dtype."""

    dtype = _Dtype()

    def __array__(
        self, dtype: Any = None, copy: bool | None = None
    ) -> np.ndarray:
        return np.asarray(self._values, dtype=dtype)


class _PolarsLike(_TensorLike):
    """A `_TensorLike` whose `to_numpy` accepts no `dtype`."""

    def to_numpy(self) -> np.ndarray:
        return self._values


class _InterfaceOnly(_Stub):
    """Has `__array_interface__` and nothing else."""

    @property
    def __array_interface__(self) -> dict[str, Any]:
        return dict(self._values.__array_interface__)


class _NamespaceLike(_TensorLike):
    """Has `__array_namespace__`, as an Array API array does."""

    def __array_namespace__(self, api_version: str | None = None) -> Any:
        return np


class _Refusing(_TensorLike):
    """Refuses its conversion, as a tensor in device memory does."""

    def __array__(
        self, dtype: Any = None, copy: bool | None = None
    ) -> np.ndarray:
        raise RuntimeError("cannot convert a device array to NumPy")


class _MissingNone(_TensorLike):
    """Converts to an object array that holds `None`."""

    def __array__(
        self, dtype: Any = None, copy: bool | None = None
    ) -> np.ndarray:
        values = np.asarray(self._values, dtype=object).copy()
        values[3] = None
        return values


class _MissingNaN(_TensorLike):
    """Converts to an array that holds NaN."""

    def __array__(
        self, dtype: Any = None, copy: bool | None = None
    ) -> np.ndarray:
        values = np.asarray(self._values, dtype=dtype).copy()
        values[3] = np.nan
        return values


class _ColumnFrame(_TensorLike):
    """Has `columns`, as a data frame of another library does."""

    columns = ("a", "b", "c")


def _xarray(values: np.ndarray) -> Any:
    return pytest.importorskip("xarray").DataArray(values)


def _polars_frame(values: np.ndarray) -> Any:
    return pytest.importorskip("polars").DataFrame(values)


def _pyarrow_table(values: np.ndarray) -> Any:
    pa = pytest.importorskip("pyarrow")
    return pa.table({str(i): values[:, i] for i in range(values.shape[1])})


def _array_api_strict(values: np.ndarray) -> Any:
    xp = pytest.importorskip("array_api_strict")
    return xp.asarray(values)


def _masked(values: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.MaskedArray(values, mask=np.zeros(values.shape, dtype=bool))


# Each container maps a NumPy array to the same numbers. The set holds the
# numbers of dimensions that the container can have.
_Make = Callable[[np.ndarray], Any]
CONTAINERS: dict[str, tuple[_Make, frozenset[int]]] = {
    "ndarray": (lambda a: a, frozenset({1, 2})),
    "masked": (_masked, frozenset({1, 2})),
    "list": (lambda a: a.tolist(), frozenset({1, 2})),
    "tuple": (
        lambda a: (
            tuple(map(tuple, a.tolist())) if a.ndim == 2 else tuple(a.tolist())
        ),
        frozenset({1, 2}),
    ),
    "pandas-frame": (lambda a: pd.DataFrame(a), frozenset({2})),
    "pandas-series": (lambda a: pd.Series(a), frozenset({1})),
    "pandas-index": (lambda a: pd.Index(a), frozenset({1})),
    "pandas-extension": (
        lambda a: pd.array(
            a, dtype="Int64" if a.dtype.kind == "i" else "Float64"
        ),
        frozenset({1}),
    ),
    "pandas-extension-series": (
        lambda a: pd.Series(
            pd.array(a, dtype="Int64" if a.dtype.kind == "i" else "Float64")
        ),
        frozenset({1}),
    ),
    "array-api-strict": (_array_api_strict, frozenset({1, 2})),
    "tensor-like": (_TensorLike, frozenset({1, 2})),
    "polars-like": (_PolarsLike, frozenset({1, 2})),
    "interface-only": (_InterfaceOnly, frozenset({1, 2})),
    "namespace": (_NamespaceLike, frozenset({1, 2})),
    "memoryview": (
        lambda a: memoryview(np.ascontiguousarray(a)),
        frozenset({1, 2}),
    ),
    "column-frame": (_ColumnFrame, frozenset({2})),
    "xarray": (_xarray, frozenset({1, 2})),
    "polars-frame": (_polars_frame, frozenset({2})),
    "pyarrow-table": (_pyarrow_table, frozenset({2})),
    "array.array": (
        lambda a: array.array("q" if a.dtype.kind == "i" else "d", a),
        frozenset({1}),
    ),
}

ROLES = (
    "fit_X",
    "y",
    "sample_weight",
    "predict_X",
    "predict_proba_X",
    "predict_log_proba_X",
    "score_X",
)
_DIMENSIONS = {"y": 1, "sample_weight": 1}
CASES = [
    pytest.param(role, name, id=f"{role}-{name}")
    for role in ROLES
    for name, (_, dims) in CONTAINERS.items()
    if _DIMENSIONS.get(role, 2) in dims
]


def _fit(X: Any = X, y: Any = Y, sample_weight: Any = W) -> C50Classifier:
    return C50Classifier(minimum_cases=1).fit(
        X, y, sample_weight=sample_weight
    )


_reference = _fit()


def _data_for(classifier: C50Classifier) -> Any:
    """Return `X` in the form that the classifier was fitted on."""
    if hasattr(classifier, "feature_names_in_"):
        return pd.DataFrame(X, columns=classifier.feature_names_in_)
    return X


def _same_classifier(a: C50Classifier, b: C50Classifier) -> None:
    assert without_date(a.model_.serialized_data) == without_date(
        b.model_.serialized_data
    )
    np.testing.assert_array_equal(a.classes_, b.classes_)
    np.testing.assert_array_equal(
        a.predict_proba(_data_for(a)), b.predict_proba(_data_for(b))
    )
    np.testing.assert_array_equal(
        a.predict(_data_for(a)), b.predict(_data_for(b))
    )


@pytest.mark.filterwarnings("ignore:X has feature names")
@pytest.mark.parametrize(("role", "name"), CASES)
def test_container_gives_the_numpy_result(role: str, name: str) -> None:
    make = CONTAINERS[name][0]
    if role == "fit_X":
        _same_classifier(_fit(X=make(X)), _reference)
    elif role == "y":
        _same_classifier(_fit(y=make(Y)), _reference)
    elif role == "sample_weight":
        _same_classifier(_fit(sample_weight=make(W)), _reference)
    elif role == "predict_X":
        np.testing.assert_array_equal(
            _reference.predict(make(X)), _reference.predict(X)
        )
    elif role == "predict_proba_X":
        probabilities = _reference.predict_proba(make(X))
        assert isinstance(probabilities, np.ndarray)
        np.testing.assert_array_equal(
            probabilities, _reference.predict_proba(X)
        )
    elif role == "predict_log_proba_X":
        logarithms = _reference.predict_log_proba(make(X))
        assert isinstance(logarithms, np.ndarray)
        np.testing.assert_array_equal(
            logarithms, _reference.predict_log_proba(X)
        )
    else:
        assert _reference.score(make(X), Y) == _reference.score(X, Y)


def _call(role: str, value: Any) -> object:
    if role == "fit_X":
        return _fit(X=value)
    if role == "y":
        return _fit(y=value)
    if role == "sample_weight":
        return _fit(sample_weight=value)
    if role == "predict_X":
        return _reference.predict(value)
    if role == "predict_proba_X":
        return _reference.predict_proba(value)
    if role == "predict_log_proba_X":
        return _reference.predict_log_proba(value)
    return _reference.score(value, Y)


_NAMES = {
    "fit_X": "X",
    "y": "y",
    "sample_weight": "sample_weight",
    "predict_X": "X",
    "predict_proba_X": "X",
    "predict_log_proba_X": "X",
    "score_X": "X",
}


@pytest.mark.parametrize("role", ROLES)
def test_refused_conversion_raises_value_error(role: str) -> None:
    values = X if _DIMENSIONS.get(role, 2) == 2 else Y.astype(float)
    with pytest.raises(ValueError, match=_NAMES[role]) as caught:
        _call(role, _Refusing(values))
    assert "cannot be converted to a NumPy array" in str(caught.value)
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_other_array_inputs_keep_a_clear_error_when_unreadable() -> None:
    class OnlyNamespace:
        def __array_namespace__(self, api_version: str | None = None) -> Any:
            return np

    messages = {"fit_X": "Expected 2D array", "y": "y should be a 1d array"}
    for role, message in messages.items():
        with pytest.raises(ValueError, match=message):
            _call(role, OnlyNamespace())


def test_pandas_missing_weight_still_raises_type_error() -> None:
    weights = pd.array(np.arange(60) % 3 + 1, dtype="Int64")
    weights[3] = pd.NA
    with pytest.raises(TypeError, match="numbers or strings"):
        _fit(sample_weight=weights)
    with pytest.raises(TypeError, match="numbers or strings"):
        _fit(sample_weight=pd.Series(weights))


def test_missing_weight_in_another_library_raises_its_converted_error() -> (
    None
):
    with pytest.raises(ValueError, match="finite"):
        _fit(sample_weight=_MissingNaN(W))
    with pytest.raises(TypeError, match="numbers or strings"):
        _fit(sample_weight=_MissingNone(W))


def test_other_array_weights_hold_their_own_elements() -> None:
    # A tensor-like weight is read as objects, so a bad element is still judged
    # by its type and not by a rounded copy.
    weights = np.asarray(W, dtype=object)
    weights[0] = 1j
    with pytest.raises(TypeError, match="numbers or strings"):
        _fit(sample_weight=_TensorLike(weights))


def test_frames_and_unreadable_objects_pass_the_boundary_unchanged() -> None:
    class InterchangeFrame(_TensorLike):
        def __dataframe__(self) -> None:
            return None

    class Plain:
        dtype = np.dtype(float)

    kept = [
        InterchangeFrame(X),
        _ColumnFrame(X),
        pd.DataFrame(X),
        pd.Series(W),
        pd.Index(W),
        pd.array(W, dtype="Float64"),
        sparse.csr_matrix(X),
        Plain(),
        {"a": 1},
        range(3),
    ]
    for value in kept:
        assert _host_array(value, "X") is value
    assert isinstance(_host_array(_TensorLike(X), "X"), np.ndarray)


@pytest.mark.parametrize("dispatch", [False, True])
def test_dispatch_setting_does_not_change_conversion(dispatch: bool) -> None:
    if dispatch and os.environ.get("SCIPY_ARRAY_API") != "1":
        pytest.skip("array_api_dispatch requires SCIPY_ARRAY_API=1")
    with sklearn.config_context(array_api_dispatch=dispatch):
        _same_classifier(
            _fit(
                X=_TensorLike(X),
                y=_TensorLike(Y),
                sample_weight=_TensorLike(W),
            ),
            _reference,
        )


def test_torch_cpu_tensors_work_as_x_y_and_weights() -> None:
    torch = pytest.importorskip("torch")
    candidate = _fit(
        X=torch.tensor(X), y=torch.tensor(Y), sample_weight=torch.tensor(W)
    )
    _same_classifier(candidate, _reference)
    probabilities = candidate.predict_proba(torch.tensor(X))
    assert isinstance(probabilities, np.ndarray)
    np.testing.assert_array_equal(probabilities, _reference.predict_proba(X))
    assert candidate.score(torch.tensor(X), torch.tensor(Y)) == (
        _reference.score(X, Y)
    )
    flags = _fit(X=torch.tensor(X > 0), y=torch.tensor(Y > 0))
    assert flags.predict(torch.tensor(X > 0)).shape == (60,)


def test_torch_tensors_that_cannot_convert_raise_value_error() -> None:
    torch = pytest.importorskip("torch")
    cases = {
        "meta": torch.empty((60, 3), device="meta"),
        "gradient": torch.tensor(X, requires_grad=True),
        "bfloat16": torch.tensor(X, dtype=torch.bfloat16),
    }
    for tensor in cases.values():
        with pytest.raises(ValueError, match="X cannot be converted"):
            _fit(X=tensor)
        with pytest.raises(ValueError, match="X cannot be converted"):
            _reference.predict(tensor)
    with pytest.raises(ValueError, match="sample_weight cannot be converted"):
        _fit(sample_weight=torch.tensor(W, requires_grad=True))
    with pytest.raises(ValueError, match="finite"):
        _fit(sample_weight=torch.tensor(np.where(W > 1, np.inf, W)))


def test_polars_series_weights_train() -> None:
    pl = pytest.importorskip("polars")
    for make in (pl.Series, lambda w: pl.DataFrame({"w": w})["w"]):
        _same_classifier(_fit(sample_weight=make(W)), _reference)
    ints = np.arange(60) % 3 + 1
    expected = _fit(sample_weight=ints)
    _same_classifier(_fit(sample_weight=pl.Series(ints)), expected)
    _same_classifier(
        _fit(sample_weight=pl.Series(ints % 2 == 0)),
        _fit(sample_weight=ints % 2 == 0),
    )


def test_polars_and_pyarrow_missing_weights_raise_value_error() -> None:
    pl = pytest.importorskip("polars")
    pa = pytest.importorskip("pyarrow")
    nulls = [None if i == 3 else float(w) for i, w in enumerate(W)]
    for weights in (
        pl.Series(nulls),
        pl.Series([np.nan, *W[1:]]),
        pa.array(nulls),
        pa.chunked_array([nulls]),
    ):
        with pytest.raises(ValueError, match="finite"):
            _fit(sample_weight=weights)


def _reads_feature_names(library: str) -> bool:
    """Return whether the installed scikit-learn reads the library's names.

    scikit-learn reads the names of pyarrow tables and polars 1.x frames
    through `__dataframe__`, which polars 2 lacks. It reads polars 2 names
    itself from version 1.9.0 on.
    """
    if library == "pyarrow":
        return True
    polars = pytest.importorskip("polars")
    major, minor = (int(part) for part in sklearn.__version__.split(".")[:2])
    return int(polars.__version__.split(".")[0]) < 2 or (major, minor) >= (
        1,
        9,
    )


@pytest.mark.parametrize("library", ["polars", "pyarrow"])
def test_data_frames_keep_their_feature_names(library: str) -> None:
    names = ["alpha", "beta", "gamma"]
    columns = {name: X[:, i] for i, name in enumerate(names)}
    reference = _fit(X=pd.DataFrame(columns))
    module = pytest.importorskip(library)
    frame = (module.DataFrame if library == "polars" else module.table)(
        columns
    )
    classifier = _fit(X=frame)
    _same_classifier(classifier, reference)
    if _reads_feature_names(library):
        np.testing.assert_array_equal(classifier.feature_names_in_, names)
        # The names make the column order part of the contract.
        with pytest.raises(ValueError, match="feature names"):
            classifier.predict(frame.select(names[::-1]))
        named = C50Classifier(
            minimum_cases=1, categorical_features=["beta"]
        ).fit(frame, Y)
        assert list(named.categorical_features_) == [1]
    else:
        assert not hasattr(classifier, "feature_names_in_")
        with pytest.raises(
            ValueError, match="require input with feature names"
        ):
            C50Classifier(minimum_cases=1, categorical_features=["beta"]).fit(
                frame, Y
            )
    np.testing.assert_array_equal(
        classifier.predict(frame), reference.predict(_data_for(reference))
    )


def test_polars_and_pyarrow_columns_work_as_y_and_weights() -> None:
    pl = pytest.importorskip("polars")
    pa = pytest.importorskip("pyarrow")
    for y, weights in (
        (pl.Series(Y), pl.Series(W)),
        (pa.array(Y), pa.array(W)),
        (pa.chunked_array([Y]), pa.chunked_array([W])),
    ):
        _same_classifier(_fit(y=y, sample_weight=weights), _reference)


_STAMPS = np.datetime64("2020-01-01", "ns") + np.arange(60).astype(
    "timedelta64[ns]"
)


@pytest.mark.parametrize("kind", ["datetime64[ns]", "timedelta64[ns]"])
def test_interface_only_nanosecond_weights_raise_type_error(kind: str) -> None:
    # The stub has no `dtype` attribute, so the conversion alone decides what
    # the weight check sees.
    weights = _STAMPS.astype(kind)
    with pytest.raises(TypeError, match="numbers or strings"):
        _fit(sample_weight=_InterfaceOnly(weights))
    with pytest.raises(TypeError, match="numbers or strings"):
        _fit(sample_weight=weights)


def test_polars_nanosecond_weights_raise_type_error() -> None:
    pl = pytest.importorskip("polars")
    stamps = pl.Series(_STAMPS)
    weights = [
        stamps,
        stamps.dt.replace_time_zone("UTC"),
        pl.Series(_STAMPS - _STAMPS[0]),
        pl.Series(_STAMPS - _STAMPS[0], dtype=pl.Duration("ns")),
    ]
    for value in weights:
        with pytest.raises(TypeError, match="numbers or strings"):
            _fit(sample_weight=value)


def test_pyarrow_nanosecond_weights_raise_type_error() -> None:
    pa = pytest.importorskip("pyarrow")
    stamps = pa.array(_STAMPS)
    durations = pa.array(_STAMPS - _STAMPS[0])
    weights = [
        stamps,
        pa.chunked_array([stamps]),
        pa.array(_STAMPS, pa.timestamp("ns", tz="UTC")),
        durations,
        pa.chunked_array([durations, durations]),
    ]
    for value in weights:
        with pytest.raises(TypeError, match="numbers or strings"):
            _fit(sample_weight=value)


class _LibraryError(Exception):
    """An exception type of a library, as polars raises its own."""


class _RaisesOwnError(_TensorLike):
    """Refuses its conversion with an exception of its own kind."""

    def __array__(
        self, dtype: Any = None, copy: bool | None = None
    ) -> np.ndarray:
        raise _LibraryError("the library cannot convert this array")


@pytest.mark.parametrize("role", ROLES)
def test_any_exception_of_a_conversion_raises_value_error(role: str) -> None:
    values = X if _DIMENSIONS.get(role, 2) == 2 else Y.astype(float)
    with pytest.raises(ValueError, match=_NAMES[role]) as caught:
        _call(role, _RaisesOwnError(values))
    assert "cannot be converted to a NumPy array" in str(caught.value)
    assert isinstance(caught.value.__cause__, _LibraryError)


@pytest.mark.parametrize("role", ROLES)
def test_exhausted_memory_propagates_from_a_conversion(role: str) -> None:
    class OutOfMemory(_TensorLike):
        def __array__(
            self, dtype: Any = None, copy: bool | None = None
        ) -> np.ndarray:
            raise MemoryError("the array does not fit")

    values = X if _DIMENSIONS.get(role, 2) == 2 else Y.astype(float)
    with pytest.raises(MemoryError, match="does not fit"):
        _call(role, OutOfMemory(values))


class _FailsOnce:
    """A table whose first conversion raises and whose later ones succeed."""

    columns = (0, 1, 2)
    shape = X.shape

    def __init__(self, error: BaseException) -> None:
        self.error = error
        self.calls = 0

    def __array__(
        self, dtype: Any = None, copy: bool | None = None
    ) -> np.ndarray:
        self.calls += 1
        if self.calls == 1:
            raise self.error
        return np.asarray(X, dtype=dtype)


@pytest.mark.parametrize("role", ["fit_X", "predict_X"])
def test_a_table_conversion_that_fails_once_raises_value_error(
    role: str,
) -> None:
    table = _FailsOnce(RuntimeError("the library failed"))
    with pytest.raises(ValueError, match="X cannot be converted") as caught:
        _call(role, table)
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert table.calls == 1


@pytest.mark.parametrize("role", ["fit_X", "predict_X"])
def test_exhausted_memory_propagates_from_a_table_conversion(
    role: str,
) -> None:
    table = _FailsOnce(MemoryError("the table does not fit"))
    with pytest.raises(MemoryError, match="does not fit"):
        _call(role, table)
    assert table.calls == 1


@pytest.mark.parametrize("role", ["fit_X", "predict_X"])
@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit])
def test_a_base_exception_propagates_from_a_table_conversion(
    role: str, error: type[BaseException]
) -> None:
    # The time probe converts a table before scikit-learn does. An interrupt
    # during that conversion is not an `Exception`, so the estimator neither
    # translates nor ignores it.
    table = _FailsOnce(error())
    with pytest.raises(error):
        _call(role, table)
    assert table.calls == 1


def test_an_exception_that_is_not_an_exception_propagates() -> None:
    class Panic(BaseException):
        pass

    class Panicking(_TensorLike):
        def __array__(
            self, dtype: Any = None, copy: bool | None = None
        ) -> np.ndarray:
            raise Panic

    with pytest.raises(Panic):
        _fit(X=Panicking(X))


def _raising_probes() -> dict[str, Any]:
    class ColumnsRaise(_TensorLike):
        @property
        def columns(self) -> list[str]:
            raise RuntimeError("columns not available")

    closed = mmap.mmap(-1, 64)
    closed.close()
    released = memoryview(np.ones(60))
    released.release()
    cases: dict[str, Any] = {
        "columns": ColumnsRaise(X),
        "closed-mmap": closed,
        "released-memoryview": released,
    }
    if sys.version_info >= (3, 12):

        class Locked:
            def __buffer__(self, flags: int) -> memoryview:
                raise BufferError("buffer is locked")

        cases["buffer"] = Locked()
    return cases


@pytest.mark.parametrize("role", ["fit_X", "y", "sample_weight", "predict_X"])
def test_failing_probes_of_an_input_raise_value_error(role: str) -> None:
    for value in _raising_probes().values():
        with pytest.raises(ValueError, match=_NAMES[role]) as caught:
            _call(role, value)
        assert "cannot be converted to a NumPy array" in str(caught.value)
        assert caught.value.__cause__ is not None


def test_polars_lazy_frames_fail_without_a_column_warning() -> None:
    # scikit-learn warns about the data types of a lazy frame in some versions.
    # Only the question of whether the frame has columns is the estimator's.
    pl = pytest.importorskip("polars")
    lazy = pl.LazyFrame({"a": X[:, 0]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="2D array"):
            _fit(X=lazy)
    assert not [w for w in caught if "column names" in str(w.message)]


def test_buffer_objects_keep_their_results() -> None:
    weights = np.arange(60) % 3 + 1
    expected = _fit(sample_weight=weights.astype(np.uint8))
    for buffer in (
        bytearray(weights.astype(np.uint8).tobytes()),
        memoryview(weights.astype(np.uint8)),
        array.array("B", weights.astype(np.uint8).tobytes()),
    ):
        _same_classifier(_fit(sample_weight=buffer), expected)


@pytest.mark.parametrize("name", ["Int128", "UInt128"])
def test_polars_128_bit_integers_are_refused_by_name(name: str) -> None:
    pl = pytest.importorskip("polars")
    dtype = getattr(pl, name, None)
    if dtype is None:
        pytest.skip(f"this polars has no {name}")
    column = pl.Series(np.arange(60) % 3, dtype=dtype)
    try:
        np.asarray(column)
    except Exception:
        panic = None
    except BaseException as exc:
        # polars 2.0 panics, and a panic is not an Exception.
        panic = type(exc)
    else:
        pytest.skip(f"this polars converts {name}")
    if panic is None:
        with pytest.raises(ValueError, match="y cannot be converted"):
            _fit(y=column)
        with pytest.raises(ValueError, match="sample_weight cannot be conv"):
            _fit(sample_weight=column)
    else:
        with pytest.raises(panic):
            _fit(y=column)
        with pytest.raises(panic):
            _fit(sample_weight=column)


COSTS = np.array([[0.0, 2.5], [1.5, 0.0]])
_two_classes = (X[:, 0] > 0).astype(np.int64)


def _fit_costs(cost_matrix: Any) -> C50Classifier:
    return C50Classifier(minimum_cases=1, cost_matrix=cost_matrix).fit(
        X, _two_classes
    )


def _same_costs(candidate: C50Classifier, reference: C50Classifier) -> None:
    _same_classifier(candidate, reference)
    np.testing.assert_array_equal(
        candidate.cost_matrix_, reference.cost_matrix_
    )


def test_a_masked_x_is_read_by_its_data_values() -> None:
    # The mask does not make a cell missing, as the guide states.
    masked = _masked(X)
    masked[:20, 0] = np.ma.masked
    reference = _fit()
    candidate = _fit(X=masked)
    _same_classifier(candidate, reference)
    np.testing.assert_array_equal(
        candidate.predict_proba(masked), reference.predict_proba(X)
    )


def test_cost_matrix_containers_give_the_numpy_result() -> None:
    reference = _fit_costs(COSTS)
    containers: list[Any] = [
        COSTS.tolist(),
        _masked(COSTS),
        _TensorLike(COSTS),
        _PolarsLike(COSTS),
        _InterfaceOnly(COSTS),
        _NamespaceLike(COSTS),
        memoryview(np.ascontiguousarray(COSTS)),
        pd.DataFrame(COSTS),
        COSTS.astype(np.float32),
    ]
    for cost_matrix in containers:
        candidate = _fit_costs(cost_matrix)
        _same_classifier(candidate, reference)
        np.testing.assert_array_equal(
            candidate.cost_matrix_, reference.cost_matrix_
        )
    _same_classifier(_fit_costs(_array_api_strict(COSTS)), reference)
    torch = pytest.importorskip("torch")
    _same_classifier(_fit_costs(torch.tensor(COSTS)), reference)


def test_cost_matrix_that_cannot_convert_raises_value_error() -> None:
    cases: dict[str, Any] = {"refusing": _Refusing(COSTS)}
    try:
        xp = pytest.importorskip("array_api_strict")
        cases["device"] = xp.asarray(COSTS, device=xp.Device("device1"))
    except pytest.skip.Exception:
        pass
    try:
        torch = pytest.importorskip("torch")
        cases["gradient"] = torch.tensor(COSTS, requires_grad=True)
        cases["bfloat16"] = torch.tensor(COSTS, dtype=torch.bfloat16)
        cases["meta"] = torch.empty((2, 2), device="meta")
    except pytest.skip.Exception:
        pass
    for case in cases.values():
        with pytest.raises(ValueError, match="cost_matrix cannot be conv"):
            _fit_costs(case)


@pytest.mark.parametrize(
    "cost_matrix",
    [
        np.array([[0, 2 + 5j], [1, 0]]),
        [[0, 2 + 5j], [1, 0]],
        np.array([[0, 2], [1, 0]]).astype("timedelta64[D]"),
        np.array([[0, 2], [1, 0]]).astype("timedelta64[ns]"),
        np.array([[0, 2], [1, 0]]).astype("datetime64[D]"),
    ],
    ids=["complex-array", "complex-list", "days", "nanoseconds", "dates"],
)
def test_cost_matrix_of_another_kind_raises_value_error(
    cost_matrix: Any,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="real numbers"):
            _fit_costs(cost_matrix)


def _cost_matrix_with(element: object) -> np.ndarray:
    matrix = np.array([[0, 0], [1, 0]], dtype=object)
    matrix[0, 1] = element
    return matrix


class _FloatDate(datetime.date):
    def __float__(self) -> float:
        return 2.0


class _FloatDateTime(datetime.datetime):
    def __float__(self) -> float:
        return 2.0


class _FloatTimedelta(datetime.timedelta):
    def __float__(self) -> float:
        return 2.0


class _FloatComplex(complex):
    def __float__(self) -> float:
        return 2.0


_SCAN_MESSAGE = "not complex, datetime, or timedelta"
_CONVERSION_MESSAGE = "convert to float"


def _not_real_costs() -> dict[str, tuple[Any, str]]:
    """Cost matrices with a refused element, and the message of the refusal.

    The scan of `encode_costs` reaches only the elements that convert to
    `float`, and refuses those that have a refused type or dtype. The builtin
    date, duration, and complex values fail the conversion first.
    """
    two = np.timedelta64(2, "ns")
    scan, conversion = _SCAN_MESSAGE, _CONVERSION_MESSAGE
    return {
        "timedelta64-object-array": (
            np.array([[0, two], [two, 0]], object),
            scan,
        ),
        "int-and-timedelta64-list": (
            [[0, two], [np.timedelta64(1, "ns"), 0]],
            scan,
        ),
        "datetime64-object-array": (
            _cost_matrix_with(np.datetime64(2, "ns")),
            scan,
        ),
        "complex128": (_cost_matrix_with(np.complex128(2)), scan),
        "complex64": (_cost_matrix_with(np.complex64(2)), scan),
        "clongdouble": (_cost_matrix_with(np.clongdouble(2)), scan),
        "date-subclass": (_cost_matrix_with(_FloatDate(2000, 1, 2)), scan),
        "datetime-subclass": (
            _cost_matrix_with(_FloatDateTime(2000, 1, 2)),
            scan,
        ),
        "timedelta-subclass": (
            _cost_matrix_with(_FloatTimedelta(days=2)),
            scan,
        ),
        "complex-subclass": (_cost_matrix_with(_FloatComplex(2)), scan),
        "0d-timedelta64-array": (_cost_matrix_with(np.array(two)), scan),
        "0d-datetime64-array": (
            _cost_matrix_with(np.array(np.datetime64(2, "ns"))),
            scan,
        ),
        "0d-complex128-array": (
            _cost_matrix_with(np.array(2 + 0j)),
            scan,
        ),
        "datetime": (
            _cost_matrix_with(datetime.datetime(2000, 1, 2)),
            conversion,
        ),
        "date": (_cost_matrix_with(datetime.date(2000, 1, 2)), conversion),
        "timedelta": (
            _cost_matrix_with(datetime.timedelta(days=2)),
            conversion,
        ),
        "pandas-timestamp": (
            _cost_matrix_with(pd.Timestamp("2000-01-02")),
            conversion,
        ),
        "pandas-timedelta": (
            _cost_matrix_with(pd.Timedelta(days=2)),
            conversion,
        ),
        "complex": (_cost_matrix_with(2 + 0j), conversion),
    }


@pytest.mark.parametrize("name", list(_not_real_costs()))
def test_cost_matrix_with_time_or_complex_objects_raises_value_error(
    name: str,
) -> None:
    # NumPy converts a time scalar in an object array to a count of units, and
    # a complex scalar to its real part. Neither is a cost. The scan refuses
    # them by their type or dtype kind. Each type that only `_NOT_REAL` covers,
    # and each dtype kind that `_is_not_real` checks, has a case here. The
    # builtin values fail to convert, and the message says which refusal
    # raised.
    matrix, message = _not_real_costs()[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match=message):
            _fit_costs(matrix)


def test_cost_matrix_with_a_0d_xarray_element_raises_value_error() -> None:
    xarray = pytest.importorskip("xarray")
    two = np.timedelta64(2, "ns")
    for element in (
        xarray.DataArray(two),
        xarray.DataArray(np.datetime64(2, "ns")),
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match=_SCAN_MESSAGE):
                _fit_costs(_cost_matrix_with(element))


class _RaisingDtype:
    """Converts to float, and raises `error` when `dtype` is read."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    def __float__(self) -> float:
        return 2.0

    @property
    def dtype(self) -> np.dtype:
        raise self._error


class _GetattrKeyError:
    """Converts to float, and raises `KeyError` for every other attribute."""

    def __float__(self) -> float:
        return 2.0

    def __getattr__(self, name: str) -> object:
        raise KeyError(name)


class _RecursiveDtype:
    """Converts to float, and reads its own `dtype` without end."""

    def __float__(self) -> float:
        return 2.0

    @property
    def dtype(self) -> np.dtype:
        return self.dtype


class _WarningDtype:
    """Converts to float, and warns when `dtype` is read."""

    def __float__(self) -> float:
        return 2.0

    @property
    def dtype(self) -> np.dtype:
        warnings.warn("dtype is deprecated", DeprecationWarning, stacklevel=2)
        return np.dtype(np.float64)


def _dtype_read_failures() -> dict[str, tuple[Any, type[BaseException]]]:
    return {
        "runtime-error": (
            _RaisingDtype(RuntimeError("no dtype here")),
            RuntimeError,
        ),
        "type-error": (_RaisingDtype(TypeError("no dtype here")), TypeError),
        "not-implemented": (
            _RaisingDtype(NotImplementedError("no dtype here")),
            NotImplementedError,
        ),
        "getattr-key-error": (_GetattrKeyError(), KeyError),
        "recursion": (_RecursiveDtype(), RecursionError),
        "warning-as-error": (_WarningDtype(), DeprecationWarning),
    }


@pytest.mark.parametrize("name", list(_dtype_read_failures()))
def test_cost_element_whose_dtype_read_raises_value_error(name: str) -> None:
    # Reading `dtype` follows the policy of the conversion to float: any
    # exception but `AttributeError` and `MemoryError` becomes the documented
    # `ValueError`, chained to the original.
    element, cause = _dtype_read_failures()[name]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match=_CONVERSION_MESSAGE) as caught:
            _fit_costs(_cost_matrix_with(element))
    assert isinstance(caught.value.__cause__, cause)
    assert str(caught.value.__cause__) in str(caught.value)


def test_cost_element_without_a_readable_dtype_trains() -> None:
    # `AttributeError` means that the element has no dtype.
    element = _RaisingDtype(AttributeError("no dtype here"))
    _same_costs(
        _fit_costs(_cost_matrix_with(element)),
        _fit_costs(_cost_matrix_with(2.0)),
    )


def test_memory_error_reading_a_cost_element_dtype_propagates() -> None:
    with pytest.raises(MemoryError):
        _fit_costs(_cost_matrix_with(_RaisingDtype(MemoryError())))


def test_cost_element_whose_dtype_read_warns_trains_without_a_filter() -> None:
    with pytest.warns(DeprecationWarning, match="dtype is deprecated"):
        candidate = _fit_costs(_cost_matrix_with(_WarningDtype()))
    _same_costs(candidate, _fit_costs(_cost_matrix_with(2.0)))


@pytest.mark.parametrize(
    ("element", "value"),
    [
        (np.array(2), 2.0),
        (np.array(2.5), 2.5),
        (np.float32(2), 2.0),
        (np.True_, 1.0),
    ],
    ids=["0d-int", "0d-float", "float32", "bool"],
)
def test_cost_matrix_with_a_real_numpy_element_still_trains(
    element: object, value: float
) -> None:
    reference = _fit_costs(np.array([[0.0, value], [1.0, 0.0]]))
    _same_costs(_fit_costs(_cost_matrix_with(element)), reference)


class _RealWithDtypeAttribute:
    """Converts to float, and has a `dtype` attribute of any kind."""

    def __init__(self, dtype: object) -> None:
        self.dtype = dtype

    def __float__(self) -> float:
        return 2.0


class _RealWithComplexArray:
    """Converts to float, and to a complex array through `__array__`."""

    def __float__(self) -> float:
        return 2.0

    def __array__(
        self, dtype: Any = None, copy: bool | None = None
    ) -> np.ndarray:
        return np.asarray(2 + 0j, dtype=dtype)


@pytest.mark.parametrize(
    "element",
    [
        _RealWithDtypeAttribute("complex128"),
        _RealWithDtypeAttribute(np.complex128),
        _RealWithDtypeAttribute(None),
        _RealWithComplexArray(),
    ],
    ids=["dtype-string", "dtype-type", "dtype-none", "complex-array"],
)
def test_cost_element_converting_to_float_trains_with_that_value(
    element: object,
) -> None:
    # The scan refuses an element by its type, or by a NumPy dtype that it
    # carries as `dtype`. It converts no element to an array, and it reads no
    # other `dtype` as a NumPy dtype, so an object that converts to float is a
    # real cost.
    _same_costs(
        _fit_costs(_cost_matrix_with(element)),
        _fit_costs(_cost_matrix_with(2.0)),
    )


def _cost_objects() -> dict[str, Any]:
    return {
        "complex-objects": np.array([[0, 2 + 5j], [1, 0]], dtype=object),
        "timedelta-objects": [[0, datetime.timedelta(days=2)], [1, 0]],
        "pandas-na": pd.DataFrame([[0, pd.NA], [1, 0]], dtype="Int64"),
        "dict": {0: 1.0, 1: 2.0},
        "generator": (row for row in [[0, 2], [1, 0]]),
        "text": [["a", "b"], ["c", "d"]],
    }


@pytest.mark.parametrize("name", list(_cost_objects()))
def test_cost_matrix_that_does_not_convert_to_float_raises_value_error(
    name: str,
) -> None:
    with pytest.raises(ValueError, match="real numbers") as excinfo:
        _fit_costs(_cost_objects()[name])
    assert excinfo.value.__cause__ is not None


class _FloatRaises:
    def __init__(self, error: BaseException) -> None:
        self.error = error

    def __float__(self) -> float:
        raise self.error


def test_cost_matrix_that_exhausts_memory_propagates_memory_error() -> None:
    # The conversion of one element may exhaust memory. That is not a property
    # of the matrix, so it is not a ValueError.
    matrix = _cost_matrix_with(_FloatRaises(MemoryError("exhausted")))
    with pytest.raises(MemoryError, match="exhausted"):
        _fit_costs(matrix)
    matrix = _cost_matrix_with(_FloatRaises(ValueError("refused")))
    with pytest.raises(ValueError, match="real numbers") as excinfo:
        _fit_costs(matrix)
    assert isinstance(excinfo.value.__cause__, ValueError)


@pytest.mark.parametrize("name", ["polars", "pyarrow"])
def test_score_reads_y_and_weights_of_other_libraries(name: str) -> None:
    make = {
        "polars": lambda a: pytest.importorskip("polars").Series(a),
        "pyarrow": lambda a: pytest.importorskip("pyarrow").array(a),
    }[name]
    expected = _reference.score(X, Y, sample_weight=W)
    assert _reference.score(X, make(Y)) == _reference.score(X, Y)
    assert _reference.score(X, make(Y), sample_weight=make(W)) == expected


@pytest.mark.parametrize(
    "name",
    [
        name
        for name, (_, dims) in CONTAINERS.items()
        # scikit-learn's own parameter check refuses objects that are only
        # buffers or only expose `__array_interface__`, as the `y` of `score`.
        if 1 in dims and name not in {"interface-only", "memoryview"}
    ],
)
def test_score_reads_y_and_weights_of_every_container(name: str) -> None:
    make = CONTAINERS[name][0]
    expected = _reference.score(X, Y, sample_weight=W)
    assert _reference.score(X, make(Y)) == _reference.score(X, Y)
    assert _reference.score(X, make(Y), sample_weight=make(W)) == expected


def _mixed_rows() -> tuple[np.ndarray, list[Any], list[str], list[bool]]:
    rng = np.random.default_rng(3)
    numbers = [
        None if i % 7 == 0 else float(v)
        for i, v in enumerate(rng.normal(size=60))
    ]
    colors = rng.choice(["red", "green", "blue"], 60).tolist()
    flags = (rng.random(60) > 0.5).tolist()
    rows = np.empty((60, 3), dtype=object)
    for i in range(60):
        rows[i] = [numbers[i], colors[i], flags[i]]
    return rows, numbers, colors, flags


def _mixed_frames() -> dict[str, Any]:
    _, numbers, colors, flags = _mixed_rows()
    frames: dict[str, Any] = {
        "pandas-nullable": pd.DataFrame(
            {
                "a": pd.array(numbers, dtype="Float64"),
                "b": pd.array(colors, dtype="string"),
                "c": pd.array(flags, dtype="boolean"),
            }
        )
    }
    try:
        pl = pytest.importorskip("polars")
        frames["polars"] = pl.DataFrame(
            {"a": numbers, "b": colors, "c": flags}
        )
        frames["polars-categorical"] = pl.DataFrame(
            {
                "a": numbers,
                "b": pl.Series(colors, dtype=pl.Categorical),
                "c": flags,
            }
        )
        frames["polars-enum"] = pl.DataFrame(
            {
                "a": numbers,
                "b": pl.Series(colors, dtype=pl.Enum(sorted(set(colors)))),
                "c": flags,
            }
        )
    except pytest.skip.Exception:
        pass
    try:
        pa = pytest.importorskip("pyarrow")
        frames["pyarrow"] = pa.table({"a": numbers, "b": colors, "c": flags})
        frames["pyarrow-dictionary"] = pa.table(
            {
                "a": numbers,
                "b": pa.array(colors).dictionary_encode(),
                "c": flags,
            }
        )
        frames["pandas-arrow"] = pd.DataFrame(
            {
                "a": pd.array(numbers, dtype="double[pyarrow]"),
                "b": pd.array(colors, dtype="string[pyarrow]"),
                "c": pd.array(flags, dtype="bool[pyarrow]"),
            }
        )
    except pytest.skip.Exception:
        pass
    return frames


def test_mixed_type_frames_give_the_result_of_an_object_array() -> None:
    # Columns of strings, Booleans, categories, and nulls in each library make
    # the classifier that the same values in an object array make.
    rows, _, _, _ = _mixed_rows()
    labels = np.arange(60) % 2
    expected = C50Classifier(minimum_cases=1).fit(rows, labels)
    assert list(expected.categorical_features_) == [1, 2]
    for name, frame in _mixed_frames().items():
        candidate = C50Classifier(minimum_cases=1).fit(frame, labels)
        assert without_date(candidate.model_.serialized_data) == without_date(
            expected.model_.serialized_data
        ), name
        assert [list(c) for c in candidate.categories_] == [
            list(c) for c in expected.categories_
        ], name
        np.testing.assert_array_equal(
            candidate.predict(frame), expected.predict(rows), err_msg=name
        )


@pytest.mark.parametrize(
    "dtype", ["UInt8", "UInt64", "Int32", "int64[pyarrow]", "uint8[pyarrow]"]
)
def test_pandas_nullable_and_arrow_integer_columns_work(dtype: str) -> None:
    if "pyarrow" in dtype:
        pytest.importorskip("pyarrow")
    integers = np.arange(60) % 4
    columns = {"a": integers, "b": (integers * 7) % 5}
    frame = pd.DataFrame(
        {k: pd.array(v, dtype=dtype) for k, v in columns.items()}
    )
    expected = C50Classifier(minimum_cases=1).fit(pd.DataFrame(columns), Y)
    candidate = C50Classifier(minimum_cases=1).fit(frame, Y)
    assert without_date(candidate.model_.serialized_data) == without_date(
        expected.model_.serialized_data
    )
    np.testing.assert_array_equal(
        candidate.predict(frame), expected.predict(pd.DataFrame(columns))
    )


def test_nested_sequences_of_arrays_work_as_x() -> None:
    rows = list(X)
    for nested in (rows, tuple(rows), [row.tolist() for row in X]):
        _same_classifier(_fit(X=nested), _reference)
        np.testing.assert_array_equal(
            _reference.predict(nested), _reference.predict(X)
        )
    torch = pytest.importorskip("torch")
    tensors = [torch.tensor(row) for row in X]
    _same_classifier(_fit(X=tensors), _reference)
    np.testing.assert_array_equal(
        _reference.predict(tensors), _reference.predict(X)
    )


def test_masked_x_is_read_without_its_mask_and_checked_for_infinity() -> None:
    masked = _masked(X)
    masked.mask[2, 1] = True
    _same_classifier(_fit(X=masked), _reference)
    infinite = _masked(X)
    infinite[2, 1] = np.inf
    infinite.mask[2, 1] = True
    for value in (infinite, _masked(np.where(X > 5, np.inf, X))):
        value[2, 1] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            _fit(X=value)
