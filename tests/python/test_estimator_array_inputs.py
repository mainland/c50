# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests of estimator input conversion and cost validation."""

from __future__ import annotations

import datetime
import warnings
from typing import Any

import numpy as np
import pytest
from c50.sklearn import C50Classifier
from model_text import without_date

_rng = np.random.default_rng(5)
X = _rng.normal(size=(60, 3))
Y = (X[:, 0] + 0.5 * X[:, 1] + _rng.normal(scale=0.5, size=60) > 0).astype(
    np.int64
)


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


def test_foreign_dtype_does_not_prevent_fit_or_prediction() -> None:
    values = _PolarsLike(X)
    classifier = C50Classifier(minimum_cases=1).fit(values, Y)
    reference = C50Classifier(minimum_cases=1).fit(X, Y)

    assert without_date(classifier.model_.serialized_data) == without_date(
        reference.model_.serialized_data
    )
    np.testing.assert_array_equal(
        classifier.predict(values), reference.predict(X)
    )
    np.testing.assert_array_equal(
        classifier.predict_proba(values), reference.predict_proba(X)
    )


def _call(role: str, values: Any) -> Any:
    classifier = C50Classifier(minimum_cases=1)
    if role == "fit_X":
        return classifier.fit(values, Y)
    return classifier.fit(X, Y).predict(values)


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


_two_classes = (X[:, 0] > 0).astype(np.int64)


def _fit_costs(cost_matrix: Any) -> C50Classifier:
    return C50Classifier(minimum_cases=1, cost_matrix=cost_matrix).fit(
        X, _two_classes
    )


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
    `float`, and refuses those that have a refused type. The builtin date,
    duration, and complex values fail the conversion first.
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
        "datetime": (
            _cost_matrix_with(datetime.datetime(2000, 1, 2)),
            conversion,
        ),
        "date": (_cost_matrix_with(datetime.date(2000, 1, 2)), conversion),
        "timedelta": (
            _cost_matrix_with(datetime.timedelta(days=2)),
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
    # them by their type. Each type in `_NOT_REAL` has a case here. The builtin
    # values fail to convert, and the message says which refusal raised.
    matrix, message = _not_real_costs()[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match=message):
            _fit_costs(matrix)


@pytest.mark.parametrize("name", ["Timestamp", "Timedelta"])
def test_cost_matrix_with_pandas_time_element_raises_value_error(
    name: str,
) -> None:
    pandas = pytest.importorskip("pandas")
    element = {
        "Timestamp": pandas.Timestamp("2000-01-02"),
        "Timedelta": pandas.Timedelta(days=2),
    }[name]
    with pytest.raises(ValueError, match=_CONVERSION_MESSAGE):
        _fit_costs(_cost_matrix_with(element))


def _cost_objects() -> dict[str, Any]:
    return {
        "complex-objects": np.array([[0, 2 + 5j], [1, 0]], dtype=object),
        "timedelta-objects": [[0, datetime.timedelta(days=2)], [1, 0]],
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


def test_cost_matrix_with_a_nullable_pandas_value_raises_value_error() -> None:
    pd = pytest.importorskip("pandas")
    matrix = pd.DataFrame([[0, pd.NA], [1, 0]], dtype="Int64")
    with pytest.raises(ValueError, match=_CONVERSION_MESSAGE) as excinfo:
        _fit_costs(matrix)
    assert excinfo.value.__cause__ is not None
