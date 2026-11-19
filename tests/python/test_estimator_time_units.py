# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests that dates and durations are categorical, unit by unit.

NumPy converts dates and durations in some units to `int` when it converts an
array to `object`, while the other units become `datetime.date`,
`datetime.datetime`, and `datetime.timedelta`. The estimator keeps the values
of the first group as `numpy.datetime64` and `numpy.timedelta64` scalars, so
that each unit is categorical, and a model fitted on one unit rejects the same
instants in another unit as unseen. Each case below fits or predicts with the
same values in one unit, one container, and one role. The last tests cover the
values that NumPy converts to `int` whatever their unit, which are continuous.
"""

from __future__ import annotations

import datetime
from collections.abc import Callable
from functools import partial
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


@pytest.mark.parametrize("kind", KINDS)
def test_equal_counts_of_different_steps_are_ordered_categories(
    kind: str,
) -> None:
    # `2ns` and `ns` are the same unit with different steps. Values of equal
    # count in them are different instants, so they are different categories,
    # ordered by step and then by count, whatever the order of the rows.
    make = np.datetime64 if kind == "date" else np.timedelta64
    pairs = [(count, step) for count in (1, 2) for step in (2, 1)] * 4
    values = [make(count, f"{step}ns") for count, step in pairs]
    labels = np.asarray(
        ["yes" if step * count == 2 else "no" for count, step in pairs]
    )
    forward = C50Classifier(minimum_cases=1).fit(_object_array(values), labels)
    backward = C50Classifier(minimum_cases=1).fit(
        _object_array(values[::-1]), labels[::-1]
    )
    expected = [(1, 1), (1, 2), (2, 1), (2, 2)]
    for classifier in (forward, backward):
        categories = [
            (np.datetime_data(c.dtype)[1], int(c.view("i8")))
            for c in classifier.categories_[0]
        ]
        assert categories == expected
    assert without_date(forward.model_.serialized_data) == without_date(
        backward.model_.serialized_data
    )


# Values that NumPy converts to `int` for reasons other than their unit, with
# the units that convert other values to objects. The predicate looks at the
# unit only, so these stay integers, and the feature trains as continuous.
VALUE_DEPENDENT = {
    "date-after-9999": np.datetime64("10000-01-01", "D")
    + OFFSETS.astype("timedelta64[D]"),
    "date-before-year-1": np.datetime64("-0001-01-01", "D")
    + OFFSETS.astype("timedelta64[D]"),
    "seconds-after-9999": np.datetime64("300000-01-01", "s")
    + OFFSETS.astype("timedelta64[s]"),
    "duration-beyond-max": ((OFFSETS + 1) * 10**10).astype("timedelta64[D]"),
    "generic-duration": OFFSETS.astype("timedelta64"),
}


NUMPY_WRAPS = np.lib.NumpyVersion(np.__version__) < "2.2.0"


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(
            name,
            marks=pytest.mark.skipif(
                name == "duration-beyond-max" and NUMPY_WRAPS,
                reason="NumPy before 2.2 wraps it, which the next test pins",
            ),
        )
        for name in VALUE_DEPENDENT
    ],
)
def test_values_that_numpy_converts_for_other_reasons_stay_integers(
    name: str,
) -> None:
    values = VALUE_DEPENDENT[name]
    assert all(isinstance(x, int) for x in values.astype(object))
    assert not is_integer_time_dtype(values.dtype)
    classifier = _fit(values.reshape(-1, 1))
    assert len(classifier.categorical_features_) == 0


def test_numpy_before_2_2_wraps_some_durations_beyond_the_maximum() -> None:
    # NumPy releases before 2.2 convert some durations in `W`, `D`, `h`, `m`,
    # `s`, and `ms` longer than about nine million years, positive or negative,
    # to a wrong `timedelta` instead of an `int`. The wrong value can equal
    # another value of the column. The estimator does not repair it, which the
    # guide states.
    values = [5, 2**32 + 5, 2**63 - 1, -(2**32 + 5)]
    durations = np.array(values, dtype="timedelta64[D]")
    converted = list(durations.astype(object))
    column = durations[np.arange(len(OFFSETS)) % 4].reshape(-1, 1)
    categories = _fit(column).categories_[0]
    days = datetime.timedelta(days=5)
    if NUMPY_WRAPS:
        wrong = [days, days, datetime.timedelta(days=-1)]
        wrong.append(datetime.timedelta(days=-5))
        assert converted == wrong
        assert len(categories) == 3
    else:
        assert converted[0] == days
        assert converted[1:] == values[1:]
        assert all(type(x) is int for x in converted[1:])
        assert len(categories) == 4


def test_integers_among_dates_are_categories_of_their_own() -> None:
    values = _values("date", "D").astype("datetime64[D]")
    values[0] = np.datetime64("10000-01-01", "D")
    classifier = _fit(values.reshape(-1, 1))
    assert list(classifier.categorical_features_) == [0]
    kinds = {type(category) for category in classifier.categories_[0]}
    assert kinds == {int, datetime.date}


def _empty_polars() -> Any:
    pl = pytest.importorskip("polars")
    return pl.DataFrame({"t": pl.Series([], dtype=pl.Datetime("ns"))})


def _empty_pyarrow() -> Any:
    pa = pytest.importorskip("pyarrow")
    return pa.table({"t": pa.array([], pa.timestamp("ns"))})


@pytest.mark.parametrize(
    ("make", "empty"), [(_polars, _empty_polars), (_pyarrow, _empty_pyarrow)]
)
@pytest.mark.parametrize("role", ["fit", "predict", "predict_proba"])
def test_a_table_of_nanoseconds_without_rows_is_rejected(
    make: _Make, empty: Callable[[], Any], role: str
) -> None:
    # The sample-count check of scikit-learn applies in every container.
    table = empty()
    classifier = _fit(make(_values("date", "ns")))
    call = (
        partial(C50Classifier(minimum_cases=1).fit, table, LABELS[:0])
        if role == "fit"
        else partial(getattr(classifier, role), table)
    )
    with pytest.raises(ValueError, match=r"0 sample\(s\)"):
        call()


class _Frame:
    """Has `columns`, as a data frame does, and converts to an array."""

    columns = ("t",)

    def __init__(self, values: np.ndarray) -> None:
        self._values = values

    def __array__(
        self, dtype: Any = None, copy: bool | None = None
    ) -> np.ndarray:
        return np.asarray(self._values, dtype=dtype)


@pytest.mark.parametrize("shape", [(), (24,)], ids=["scalar", "vector"])
def test_a_table_of_nanoseconds_must_be_two_dimensional(
    shape: tuple[int, ...],
) -> None:
    # Only a two-dimensional table of such values takes the path of the tables.
    # Any other shape goes to scikit-learn, which reports it as it does for
    # every other container.
    values = _values("date", "ns")
    table = _Frame(values.reshape(shape) if shape else values[0])
    classifier = _fit(values.reshape(-1, 1))
    with pytest.raises(ValueError, match="Expected 2D array"):
        C50Classifier(minimum_cases=1).fit(table, LABELS)
    with pytest.raises(ValueError, match="Expected 2D array"):
        classifier.predict(table)


def test_polars_table_through_to_pandas_keeps_the_date_categorical() -> None:
    # The guide tells users to pass a pandas `DataFrame` to control the type of
    # each column, and names `to_pandas()` as the way to get one from another
    # data frame. The table mixes a date with a float column.
    pl = pytest.importorskip("polars")
    pytest.importorskip("pyarrow")
    pd = pytest.importorskip("pandas")
    dates = _values("date", "us")
    floats = np.linspace(0.5, 2.5, len(dates))
    table = pl.DataFrame({"t": pl.Series(dates), "x": floats})
    expected = pd.DataFrame({"t": dates, "x": floats})
    classifier = _fit(table.to_pandas())
    reference = _fit(expected)
    assert list(classifier.categorical_features_) == [0]
    assert len(classifier.categories_[0]) == 4
    assert classifier.categories_[0].tolist() == (
        reference.categories_[0].tolist()
    )
    assert without_date(classifier.model_.serialized_data) == without_date(
        reference.model_.serialized_data
    )


def test_polars_boolean_through_to_pandas_stays_categorical() -> None:
    # The rule for a data frame other than pandas covers every column type,
    # and its remedy is the same `to_pandas()`. The table mixes a Boolean
    # column with a float column.
    pl = pytest.importorskip("polars")
    pytest.importorskip("pyarrow")
    pd = pytest.importorskip("pandas")
    flags = np.arange(len(LABELS)) % 3 == 0
    floats = np.linspace(0.5, 2.5, len(LABELS))
    table = pl.DataFrame({"b": flags, "x": floats})
    classifier = _fit(table.to_pandas())
    reference = _fit(pd.DataFrame({"b": flags, "x": floats}))
    assert list(classifier.categorical_features_) == [0]
    assert classifier.categories_[0].tolist() == [False, True]
    assert without_date(classifier.model_.serialized_data) == without_date(
        reference.model_.serialized_data
    )


def _mixed_pyarrow_table(kind: str) -> Any:
    pa = pytest.importorskip("pyarrow")
    days = [datetime.date(2020, 1, 1 + int(i)) for i in OFFSETS]
    numbers = np.arange(len(OFFSETS)) % 7
    if kind == "date32-int64":
        return pa.table(
            {"t": pa.array(days, pa.date32()), "n": pa.array(numbers)}
        )
    stamps = [
        datetime.datetime.combine(day, datetime.time(12)) for day in days
    ]
    return pa.table(
        {
            "t": pa.array(stamps, pa.timestamp("us")),
            "x": pa.array(numbers / 7.0),
        }
    )


@pytest.mark.parametrize("kind", ["date32-int64", "timestamp-float64"])
def test_a_pyarrow_table_of_a_date_and_a_number_trains_and_predicts(
    kind: str,
) -> None:
    # NumPy cannot convert such a table without a dtype, while scikit-learn
    # converts it through `object`, so the estimator leaves it to scikit-learn.
    # The table gives the classifier that its `to_pandas()` frame gives.
    table = _mixed_pyarrow_table(kind)
    frame = table.to_pandas()
    classifier = _fit(table)
    reference = _fit(frame)
    assert list(classifier.categorical_features_) == [0]
    assert without_date(classifier.model_.serialized_data) == without_date(
        reference.model_.serialized_data
    )
    assert classifier.predict(table).tolist() == LABELS.tolist()
    np.testing.assert_array_equal(
        classifier.predict_proba(table), reference.predict_proba(frame)
    )
