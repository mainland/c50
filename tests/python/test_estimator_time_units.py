# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests that dates and durations are categorical in every unit.

NumPy converts nanosecond dates and durations to `int` when it converts an
array to `object`, while the coarser units become `datetime.date`,
`datetime.datetime`, and `datetime.timedelta`. The estimator keeps nanosecond
values as `numpy.datetime64` and `numpy.timedelta64` scalars, so that each unit
is categorical, and a model fitted on one unit rejects the same instants in
another unit as unseen. Each case below fits or predicts with the same values
in one unit, one container, and one role.
"""

from __future__ import annotations

import datetime
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from c50.sklearn import C50Classifier
from model_text import without_date

OFFSETS = np.arange(24) % 4
LABELS = np.asarray(["no", "yes", "no", "no"])[OFFSETS]
UNITS = ("D", "s", "us", "ns")
KINDS = ("date", "duration")


def _values(kind: str, unit: str) -> np.ndarray:
    steps = OFFSETS.astype(f"timedelta64[{unit}]")
    if kind == "duration":
        return steps
    return np.datetime64("2020-01-01", unit) + steps


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
    "ndarray": (lambda v: v.reshape(-1, 1), UNITS, KINDS),
    "object-array": (_object_array, UNITS, KINDS),
    "nested-list": (lambda v: [[x] for x in v], UNITS, KINDS),
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
    if unit == "ns":
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
    reference = _reference(kind)
    expected = reference.predict(_values(kind, "us").reshape(-1, 1))
    np.testing.assert_array_equal(classifier.predict(values), expected)
    np.testing.assert_array_equal(
        classifier.predict_proba(values),
        reference.predict_proba(_values(kind, "us").reshape(-1, 1)),
    )


@pytest.mark.parametrize("kind", KINDS)
def test_named_categorical_column_holds_numpy_scalars(kind: str) -> None:
    classifier = _fit(
        _values(kind, "ns").reshape(-1, 1), categorical_features=[0]
    )
    assert all(
        _is_scalar_category(category, kind, "ns")
        for category in classifier.categories_[0]
    )


@pytest.mark.parametrize("kind", KINDS)
def test_not_a_time_is_missing_in_every_unit(kind: str) -> None:
    def column(unit: str) -> np.ndarray:
        values = _values(kind, unit).copy()
        values[::5] = np.array(["NaT"], dtype=values.dtype)[0]
        return values

    reference = _fit(column("us").reshape(-1, 1))
    for unit in UNITS:
        classifier = _fit(column(unit).reshape(-1, 1))
        assert without_date(classifier.model_.serialized_data) == without_date(
            reference.model_.serialized_data
        )
        assert len(classifier.categories_[0]) == 4
        np.testing.assert_array_equal(
            classifier.predict(column(unit).reshape(-1, 1)),
            reference.predict(column("us").reshape(-1, 1)),
        )


@pytest.mark.parametrize("kind", KINDS)
def test_instants_in_another_unit_are_unseen(kind: str) -> None:
    nanoseconds = _values(kind, "ns").reshape(-1, 1)
    microseconds = _values(kind, "us").reshape(-1, 1)
    for fitted, other in (
        (nanoseconds, microseconds),
        (microseconds, nanoseconds),
    ):
        classifier = _fit(fitted)
        with pytest.raises(ValueError, match="unseen category"):
            classifier.predict(other)
        lenient = _fit(fitted, unknown_categories="missing")
        assert lenient.predict(other).shape == (len(other),)
