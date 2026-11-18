# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests that dates and durations are categorical in every unit.

NumPy converts dates and durations in some units to `int` when it converts an
array to `object`, while the other units become `datetime.date`,
`datetime.datetime`, and `datetime.timedelta`. The estimator keeps the values
of the first group as `numpy.datetime64` and `numpy.timedelta64` scalars, so
that each unit is categorical, and a model fitted on one unit rejects the same
instants in another unit as unseen. Each case below fits or predicts with the
same values in one unit, one container, and one role.
"""

from __future__ import annotations

import datetime
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from c50._data import is_integer_time_dtype
from c50.sklearn import C50Classifier
from model_text import without_date

OFFSETS = np.arange(24) % 4
LABELS = np.asarray(["no", "yes", "no", "no"])[OFFSETS]
UNITS = ("D", "s", "us", "ns")
ALL_UNITS = (
    *("Y", "M", "W", "D", "h", "m", "s", "ms", "us"),
    *("ns", "ps", "fs", "as"),
)
KINDS = ("date", "duration")
# The units of each kind that NumPy converts to `int` under `astype(object)`.
# `test_the_scalar_units_are_the_ones_that_numpy_converts_to_int` checks them
# against NumPy, and the other units are the ones that were categorical before.
SCALAR_UNITS = {
    "date": ("ns", "ps", "fs", "as"),
    "duration": ("Y", "M", "ns", "ps", "fs", "as"),
}


def _values(kind: str, unit: str) -> np.ndarray:
    steps = OFFSETS.astype(f"timedelta64[{unit}]")
    if kind == "duration":
        return steps
    return np.datetime64("1970-01-01", unit) + steps


def _object_array(values: np.ndarray) -> np.ndarray:
    result = np.empty((len(values), 1), dtype=object)
    result[:, 0] = list(values)
    return result


def _polars(values: np.ndarray) -> Any:
    pl = pytest.importorskip("polars")
    return pl.DataFrame({"t": pl.Series(values)})


def _pyarrow(values: np.ndarray) -> Any:
    pa = pytest.importorskip("pyarrow")
    return pa.table({"t": pa.array(values)})


def _polars_zoned(values: np.ndarray) -> Any:
    pl = pytest.importorskip("polars")
    return pl.DataFrame({"t": pl.Series(values).dt.replace_time_zone("UTC")})


def _pyarrow_zoned(values: np.ndarray) -> Any:
    pa = pytest.importorskip("pyarrow")
    return pa.table({"t": pa.array(values, pa.timestamp("ns", tz="UTC"))})


# Each container maps a one-dimensional array to a feature matrix, with the
# units that the container can hold and the kinds that it can hold.
_Make = Callable[[np.ndarray], Any]
CONTAINERS: dict[str, tuple[_Make, tuple[str, ...], tuple[str, ...]]] = {
    "ndarray": (lambda v: v.reshape(-1, 1), ALL_UNITS, KINDS),
    "object-array": (_object_array, ALL_UNITS, KINDS),
    "nested-list": (lambda v: [[x] for x in v], ALL_UNITS, KINDS),
    "polars": (_polars, ("us", "ns"), KINDS),
    "pyarrow": (_pyarrow, ("s", "us", "ns"), KINDS),
    "polars-zoned": (_polars_zoned, ("ns",), ("date",)),
    "pyarrow-zoned": (_pyarrow_zoned, ("ns",), ("date",)),
}
CASES = [
    pytest.param(kind, unit, name, id=f"{kind}-{unit}-{name}")
    for name, (_, units, kinds) in CONTAINERS.items()
    for kind in kinds
    for unit in units
]


def _fit(X: Any, **parameters: Any) -> C50Classifier:
    return C50Classifier(minimum_cases=1, **parameters).fit(X, LABELS)


def _reference(kind: str) -> C50Classifier:
    """Fit on microseconds, a unit that was categorical before."""
    return _fit(_values(kind, "us").reshape(-1, 1))


def _is_scalar_category(category: object, kind: str, unit: str) -> bool:
    if unit in SCALAR_UNITS[kind]:
        return isinstance(
            category, np.datetime64 if kind == "date" else np.timedelta64
        )
    return isinstance(
        category,
        (datetime.date, datetime.datetime)
        if kind == "date"
        else datetime.timedelta,
    )


@pytest.mark.parametrize(("kind", "unit", "name"), CASES)
def test_fit_is_categorical_and_matches_microseconds(
    kind: str, unit: str, name: str
) -> None:
    make = CONTAINERS[name][0]
    classifier = _fit(make(_values(kind, unit)))
    reference = _reference(kind)
    assert list(classifier.categorical_features_) == [0]
    assert without_date(classifier.model_.serialized_data) == without_date(
        reference.model_.serialized_data
    )
    assert len(classifier.categories_[0]) == 4
    assert all(
        _is_scalar_category(category, kind, unit)
        for category in classifier.categories_[0]
    )


@pytest.mark.parametrize(("kind", "unit", "name"), CASES)
def test_predict_matches_microseconds(kind: str, unit: str, name: str) -> None:
    make = CONTAINERS[name][0]
    values = make(_values(kind, unit))
    classifier = _fit(values)
    assert list(classifier.categorical_features_) == [0]
    reference = _reference(kind)
    expected = reference.predict(_values(kind, "us").reshape(-1, 1))
    np.testing.assert_array_equal(classifier.predict(values), expected)
    np.testing.assert_array_equal(
        classifier.predict_proba(values),
        reference.predict_proba(_values(kind, "us").reshape(-1, 1)),
    )


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("unit", ALL_UNITS)
def test_named_categorical_column_holds_the_expected_form(
    kind: str, unit: str
) -> None:
    classifier = _fit(
        _values(kind, unit).reshape(-1, 1), categorical_features=[0]
    )
    assert all(
        _is_scalar_category(category, kind, unit)
        for category in classifier.categories_[0]
    )


@pytest.mark.parametrize("kind", ["M", "m"])
@pytest.mark.parametrize("unit", ALL_UNITS)
def test_the_scalar_units_are_the_ones_that_numpy_converts_to_int(
    kind: str, unit: str
) -> None:
    dtype = np.dtype(f"{kind}8[{unit}]")
    converted = np.zeros(1, dtype=dtype).astype(object)[0]
    name = "date" if kind == "M" else "duration"
    assert isinstance(converted, int) == (unit in SCALAR_UNITS[name])
    assert is_integer_time_dtype(dtype) == (unit in SCALAR_UNITS[name])


@pytest.mark.parametrize("kind", KINDS)
def test_not_a_time_is_missing_in_every_unit(kind: str) -> None:
    def column(unit: str) -> np.ndarray:
        values = _values(kind, unit).copy()
        values[::5] = np.array(["NaT"], dtype=values.dtype)[0]
        return values

    reference = _fit(column("us").reshape(-1, 1))
    for unit in ALL_UNITS:
        classifier = _fit(column(unit).reshape(-1, 1))
        assert without_date(classifier.model_.serialized_data) == without_date(
            reference.model_.serialized_data
        )
        assert len(classifier.categories_[0]) == 4
        np.testing.assert_array_equal(
            classifier.predict(column(unit).reshape(-1, 1)),
            reference.predict(column("us").reshape(-1, 1)),
        )


@pytest.mark.parametrize(
    ("kind", "unit"),
    [(kind, unit) for kind in KINDS for unit in SCALAR_UNITS[kind]],
)
def test_scalar_units_and_other_units_never_match(
    kind: str, unit: str
) -> None:
    # The categories match by type. A NumPy scalar never equals a `datetime`
    # object, so the first value, which has the same instant in every unit, is
    # unseen in a model of the other form. Units within the NumPy group are
    # separate categories, which the tests below pin.
    scalars = _values(kind, unit).reshape(-1, 1)
    for other_unit in ("us", "D"):
        others = _values(kind, other_unit).reshape(-1, 1)
        for fitted, other in ((scalars, others), (others, scalars)):
            classifier = _fit(fitted)
            with pytest.raises(ValueError, match="unseen category"):
                classifier.predict(other)
            lenient = _fit(fitted, unknown_categories="missing")
            assert lenient.predict(other).shape == (len(other),)


# Pairs of units of the scalar group and the factor from the first unit to the
# second, so that the pair holds equal instants or durations.
UNIT_PAIRS = [
    pytest.param("date", "ns", "ps", 1000, id="date-ns-ps"),
    pytest.param("duration", "ns", "ps", 1000, id="duration-ns-ps"),
    pytest.param("duration", "Y", "M", 12, id="duration-Y-M"),
]


def _scaled(kind: str, unit: str, factor: int) -> np.ndarray:
    """Return the values of ``_values(kind, ...)``, equal in a finer unit."""
    steps = (OFFSETS * factor).astype(f"timedelta64[{unit}]")
    if kind == "duration":
        return steps
    return np.datetime64("1970-01-01", unit) + steps


@pytest.mark.parametrize(("kind", "coarse", "fine", "factor"), UNIT_PAIRS)
def test_equal_values_in_two_scalar_units_are_not_matched(
    kind: str, coarse: str, fine: str, factor: int
) -> None:
    # NumPy 2.2 and later hash equal values of two units alike, and earlier
    # releases do not, so the estimator keys a scalar by its unit.
    first = _values(kind, coarse).reshape(-1, 1)
    second = _scaled(kind, fine, factor).reshape(-1, 1)
    assert np.all(first == second)
    for fitted, other in ((first, second), (second, first)):
        classifier = _fit(fitted)
        with pytest.raises(ValueError, match="unseen category"):
            classifier.predict(other)
        lenient = _fit(fitted, unknown_categories="missing")
        assert lenient.predict(other).shape == (len(other),)


def _mixed_rows(
    kind: str, coarse: str, fine: str, factor: int, order: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return one column that holds every value in two units, and its labels.

    Each unit holds the same four instants or durations.
    """
    rows = [
        (value, LABELS[i])
        for unit, scale in ((coarse, 1), (fine, factor))
        for i, value in enumerate(_scaled(kind, unit, scale))
    ]
    if order == "reversed":
        rows.reverse()
    elif order == "interleaved":
        rows = rows[::2] + rows[1::2]
    return (
        _object_array([value for value, _ in rows]),
        np.asarray([label for _, label in rows]),
    )


@pytest.mark.parametrize(("kind", "coarse", "fine", "factor"), UNIT_PAIRS)
def test_mixed_unit_columns_do_not_depend_on_row_order(
    kind: str, coarse: str, fine: str, factor: int
) -> None:
    # Equal values of two units are two categories, so the four values of each
    # unit give eight. They are listed by unit, and then by count.
    expected = [
        *((coarse, count) for count in range(4)),
        *((fine, count * factor) for count in range(4)),
    ]
    reference = None
    for order in ("given", "reversed", "interleaved"):
        X, y = _mixed_rows(kind, coarse, fine, factor, order)
        classifier = C50Classifier(minimum_cases=1).fit(X, y)
        assert [
            (np.datetime_data(value.dtype)[0], int(value.view("i8")))
            for value in classifier.categories_[0]
        ] == expected
        text = without_date(classifier.model_.serialized_data)
        reference = text if reference is None else reference
        assert text == reference


@pytest.mark.parametrize("calendar", ["Y", "M"])
def test_calendar_and_linear_durations_train_in_one_column(
    calendar: str,
) -> None:
    # NumPy refuses to compare a calendar duration with a linear one, so the
    # estimator must not ask it to, whatever the NumPy release.
    units = (calendar, "ns")
    values = [
        np.timedelta64(int(count), units[(i // 4) % 2])
        for i, count in enumerate(OFFSETS)
    ]
    X = _object_array(values)
    classifier = _fit(X)
    assert len(classifier.categories_[0]) == 8
    assert classifier.predict(X).shape == (len(X),)
    lenient = _fit(X, unknown_categories="missing")
    assert lenient.predict(X).shape == (len(X),)
