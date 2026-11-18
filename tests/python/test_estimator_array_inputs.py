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
    return _reference.score(value, Y)


_NAMES = {
    "fit_X": "X",
    "y": "y",
    "sample_weight": "sample_weight",
    "predict_X": "X",
    "predict_proba_X": "X",
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

    for role in ("fit_X", "y"):
        with pytest.raises(ValueError, match=r"array|shape|1d|2D"):
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

    class ColumnFrame(_TensorLike):
        columns = ("a", "b", "c")

    class Plain:
        dtype = np.dtype(float)

    kept = [
        InterchangeFrame(X),
        ColumnFrame(X),
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
