"""Tests that estimator categories match numbers by value, not by type."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from fractions import Fraction
from itertools import permutations
from typing import Any, Literal

import numpy as np
import pandas as pd
import pytest
from c50._data import fit_schema
from c50.sklearn import C50Classifier

Policy = Literal["error", "missing"]
POLICIES: list[Policy] = ["error", "missing"]
CODES = [0, 1, 2] * 8
LABELS = np.asarray(["no", "yes", "no"] * 8)


def make_matrix(dtype: type) -> np.ndarray:
    return np.asarray(CODES, dtype=dtype).reshape(-1, 1)


def make_frame(dtype: type) -> pd.DataFrame:
    return pd.DataFrame({"code": np.asarray(CODES, dtype=dtype)})


def make_categorical_frame(dtype: type) -> pd.DataFrame:
    return pd.DataFrame(
        {"code": pd.Categorical(np.asarray(CODES, dtype=dtype))}
    )


def make_classifier(policy: Policy, **options: Any) -> C50Classifier:
    return C50Classifier(minimum_cases=1, unknown_categories=policy, **options)


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize(
    ("fit_type", "predict_type"), [(int, float), (float, int)]
)
def test_numpy_categories_match_across_int_and_float(
    policy: Policy, fit_type: type, predict_type: type
) -> None:
    model = make_classifier(policy, categorical_features=[0])
    model.fit(make_matrix(fit_type), LABELS)

    query = np.asarray([[1], [0], [2], [1]], dtype=predict_type)
    assert model.predict(query).tolist() == ["yes", "no", "no", "yes"]
    # The same classifier results whichever type fitted it.
    assert model.categories_[0].tolist() == [0, 1, 2]


@pytest.mark.parametrize("policy", POLICIES)
def test_float_prediction_matrix_with_missing_value_matches_int_fit(
    policy: Policy,
) -> None:
    # A NaN promotes an integer column to float64, as pandas does on reading a
    # file with a blank cell.
    model = make_classifier(policy, categorical_features=[0])
    model.fit(make_matrix(int), LABELS)

    query = np.asarray([[1.0], [np.nan], [0.0]])
    reference = model.predict_proba(np.asarray([[1], [np.nan], [0]]))
    np.testing.assert_array_equal(model.predict_proba(query), reference)
    assert model.predict(query)[0] == "yes"


@pytest.mark.parametrize("policy", POLICIES)
def test_model_fitted_with_missing_values_accepts_integer_input(
    policy: Policy,
) -> None:
    X = make_matrix(float)
    X[3, 0] = np.nan
    model = make_classifier(policy, categorical_features=[0]).fit(X, LABELS)

    assert model.predict(np.asarray([[1], [2]])).tolist() == ["yes", "no"]


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize(
    ("fit_type", "predict_type"), [(int, float), (float, int)]
)
def test_pandas_categories_match_across_int_and_float(
    policy: Policy, fit_type: type, predict_type: type
) -> None:
    model = make_classifier(policy, categorical_features=["code"])
    model.fit(make_frame(fit_type), LABELS)

    query = pd.DataFrame({"code": np.asarray([1, 0, 2], dtype=predict_type)})
    assert model.predict(query).tolist() == ["yes", "no", "no"]
    with_missing = pd.DataFrame({"code": [1.0, np.nan, 0.0]})
    assert model.predict(with_missing)[0] == "yes"


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize(
    ("fit_type", "predict_type"), [(int, float), (float, int)]
)
def test_from_dtype_categories_match_across_int_and_float(
    policy: Policy, fit_type: type, predict_type: type
) -> None:
    model = make_classifier(policy, categorical_features="from_dtype")
    model.fit(make_categorical_frame(fit_type), LABELS)

    query = make_categorical_frame(predict_type).iloc[:3]
    assert model.predict(query).tolist() == ["no", "yes", "no"]
    # An object column carries no dtype, but must match as well.
    as_object = pd.DataFrame({"code": pd.Series([1, 0, 2], dtype=object)})
    assert model.predict(as_object).tolist() == ["yes", "no", "no"]
    as_float = pd.DataFrame({"code": pd.Series([1.0, 0.0, 2.0], dtype=object)})
    assert model.predict(as_float).tolist() == ["yes", "no", "no"]


def test_object_column_merges_equal_int_and_float() -> None:
    values = np.asarray([1, 1.0, 2, 0.0, 0, 2.0], dtype=object)
    schema = fit_schema(values.reshape(-1, 1), categorical_features=[0])

    assert schema.categories == ((0, 1, 2),)
    assert [type(v) for v in schema.categories[0]] == [int, int, int]
    encoded = schema.dense_data(values.reshape(-1, 1), "error")
    assert encoded[:, 0].tolist() == [1, 1, 2, 0, 0, 2]


def test_declaration_order_does_not_depend_on_the_type_seen_first() -> None:
    forward = np.asarray([1, 1.0, 2.0, 2, 0.5], dtype=object).reshape(-1, 1)
    backward = forward[::-1].copy()

    one = fit_schema(forward, categorical_features=[0])
    other = fit_schema(backward, categorical_features=[0])

    assert one.categories == other.categories == ((0.5, 1, 2),)
    assert [type(v) for v in one.categories[0]] == [float, int, int]
    assert one.names_data(2) == other.names_data(2)


def test_mixed_object_column_orders_numbers_first() -> None:
    values = np.asarray(["b", 2.0, 1, "a", 1.0], dtype=object).reshape(-1, 1)

    schema = fit_schema(values, categorical_features=[0])

    assert schema.categories == ((1, 2.0, "a", "b"),)


def test_boolean_and_integer_categories_stay_distinct() -> None:
    values = np.asarray([True, 1, False, 0, 1.0], dtype=object).reshape(-1, 1)

    schema = fit_schema(values, categorical_features=[0])

    assert schema.categories == ((False, True, 0, 1),)
    encoded = schema.dense_data(values, "error")[:, 0]
    assert encoded[0] != encoded[1]
    assert encoded[1] != encoded[2]
    assert encoded[1] == 3
    assert encoded[4] == 3


def test_float32_categories_match_only_values_equal_as_doubles() -> None:
    fitted = np.asarray([0.5, 0.25, 0.1, 0.5] * 4, dtype=np.float64)
    labels = np.asarray(["a", "b", "a", "a"] * 4)
    model = make_classifier("error", categorical_features=[0])
    model.fit(fitted.reshape(-1, 1), labels)

    # 0.5 and 0.25 are exact in float32, so they match.
    exact = np.asarray([[0.5], [0.25]], dtype=np.float32)
    assert model.predict(exact).tolist() == ["a", "b"]
    # float32(0.1) is 0.10000000149..., which is not the double 0.1.
    inexact = np.asarray([[0.1]], dtype=np.float32)
    with pytest.raises(ValueError, match="unseen category"):
        model.predict(inexact)
    model.set_params(unknown_categories="missing")
    expected = model.predict_proba(np.asarray([[np.nan]]))
    np.testing.assert_array_equal(model.predict_proba(inexact), expected)


def test_negative_zero_does_not_depend_on_row_order() -> None:
    # -0.0 equals 0.0 and shares its category, so the stored value must not
    # depend on which of the two occurs first in the rows.
    column = np.asarray(
        [-0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 2.0]
    )
    labels = np.asarray(["a", "b"] * 6)
    order = np.arange(column.size)[::-1]

    forward = C50Classifier(categorical_features=[0]).fit(
        column.reshape(-1, 1), labels
    )
    backward = C50Classifier(categorical_features=[0]).fit(
        column[order].reshape(-1, 1), labels[order]
    )

    for fitted in (forward, backward):
        assert [str(v) for v in fitted.categories_[0]] == ["0.0", "1.0", "2.0"]
    assert forward.model_.names_data == backward.model_.names_data
    assert forward.model_.serialized_data == backward.model_.serialized_data


@pytest.mark.parametrize(
    ("forms", "expected"),
    [
        ([Decimal("0"), Decimal("-0"), Decimal("0.00")], Decimal("0")),
        ([Decimal("1"), Decimal("1.0"), Decimal("1.00")], Decimal("1")),
        ([Decimal("100"), Decimal("1E+2")], Decimal("100")),
        ([0j, complex(-0.0, 0.0), complex(0.0, -0.0)], 0j),
        ([complex(1, 0), complex(1, -0.0)], complex(1, 0)),
    ],
    ids=[
        "decimal-zero",
        "decimal-one",
        "decimal-exponent",
        "complex-zero",
        "complex-one",
    ],
)
def test_equal_decimal_and_complex_forms_do_not_depend_on_row_order(
    forms: list[object], expected: object
) -> None:
    for rows in permutations([*forms, 7, 8]):
        column = np.asarray(rows, dtype=object).reshape(-1, 1)
        schema = fit_schema(column, categorical_features=[0])
        stored = [v for v in schema.categories[0] if v == expected]

        assert len(stored) == 1
        assert repr(stored[0]) == repr(expected)


def object_column(*values: object) -> np.ndarray:
    return np.asarray(values, dtype=object).reshape(-1, 1)


@pytest.mark.parametrize("nan", [Decimal("NaN"), Decimal("-NaN")])
def test_decimal_nan_is_a_missing_value(nan: Decimal) -> None:
    column = object_column(Decimal(1), nan, Decimal(2), nan)
    schema = fit_schema(column, categorical_features=[0])

    assert schema.categories == ((Decimal(1), Decimal(2)),)
    encoded = schema.dense_data(column, "error")
    assert np.isnan(encoded[:, 0]).tolist() == [False, True, False, True]


def test_decimal_signaling_nan_is_a_missing_value() -> None:
    # A signaling NaN is unhashable, so it must be recognized as missing before
    # a category key is built.
    column = object_column(Decimal(1), Decimal("sNaN"), Decimal(2))
    schema = fit_schema(column, categorical_features=[0])

    assert schema.categories == ((Decimal(1), Decimal(2)),)
    assert np.isnan(schema.dense_data(column, "error")[:, 0]).tolist() == [
        False,
        True,
        False,
    ]


def test_decimal_nan_fits_and_predicts() -> None:
    column = object_column(*[Decimal(code) for code in CODES])
    column[3, 0] = Decimal("NaN")
    model = make_classifier("error", categorical_features=[0])
    model.fit(column, LABELS)

    query = object_column(Decimal(1), Decimal("NaN"), Decimal(0))
    assert model.predict(query)[0] == "yes"


class Unordered:
    """A hashable value whose comparison raises ArithmeticError."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __lt__(self, other: object) -> bool:
        raise ArithmeticError(self.name)

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Unordered) and other.name == self.name


def test_arithmetic_error_in_ordering_keeps_first_occurrence_order() -> None:
    values = [Unordered("b"), Unordered("a"), Unordered("c")]
    schema = fit_schema(object_column(*values), categorical_features=[0])

    assert [v.name for v in schema.categories[0]] == ["b", "a", "c"]


class ValueUnordered(Unordered):
    """A hashable value whose comparison raises ValueError."""

    def __lt__(self, other: object) -> bool:
        raise ValueError(self.name)


def test_value_error_in_ordering_keeps_first_occurrence_order() -> None:
    values = [ValueUnordered("b"), ValueUnordered("a"), ValueUnordered("c")]
    schema = fit_schema(object_column(*values), categorical_features=[0])

    assert [v.name for v in schema.categories[0]] == ["b", "a", "c"]


def test_mixed_frequency_periods_keep_first_occurrence_order() -> None:
    # pandas 2.2 raises IncompatibleFrequency, a ValueError, when comparing
    # Period values of different frequencies. pandas 3 raises a TypeError.
    values = [
        pd.Period("2026-02", "M"),
        pd.Period("2026-01-05", "D"),
        pd.Period("2026-01", "M"),
    ]
    schema = fit_schema(object_column(*values), categorical_features=[0])

    assert list(schema.categories[0]) == values


def test_int_and_decimal_are_separate_categories() -> None:
    column = object_column(1, Decimal(1), 2, Decimal(2))
    schema = fit_schema(column, categorical_features=[0])

    assert len(schema.categories[0]) == 4
    assert {type(v) for v in schema.categories[0]} == {int, Decimal}
    # Fraction is a Real, so it matches the integer by value.
    merged = fit_schema(
        object_column(1, Fraction(1), 2), categorical_features=[0]
    )
    assert merged.categories == ((1, 2),)


def test_model_fitted_on_integers_rejects_equal_decimal() -> None:
    model = make_classifier("error", categorical_features=[0])
    model.fit(make_matrix(int), LABELS)

    with pytest.raises(ValueError, match="unseen category Decimal"):
        model.predict(object_column(Decimal(1)))
    missing = make_classifier("missing", categorical_features=[0]).fit(
        make_matrix(int), LABELS
    )
    assert missing.predict(object_column(Decimal(1))).shape == (1,)


def test_equal_aware_datetimes_keep_their_first_form() -> None:
    utc = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
    plus_two = utc.astimezone(timezone(timedelta(hours=2)))
    assert utc == plus_two
    other = datetime(2026, 1, 2, tzinfo=timezone.utc)

    first = fit_schema(
        object_column(utc, plus_two, other), categorical_features=[0]
    )
    second = fit_schema(
        object_column(plus_two, utc, other), categorical_features=[0]
    )

    assert len(first.categories[0]) == len(second.categories[0]) == 2
    assert first.categories[0][0].utcoffset() == timedelta(0)
    assert second.categories[0][0].utcoffset() == timedelta(hours=2)
