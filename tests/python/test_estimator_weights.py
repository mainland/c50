# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for estimator sample weights and C5.0 case weights."""

from __future__ import annotations

import datetime as dt
import decimal
import enum
import fractions
import numbers
import pickle
import warnings

import c50
import numpy as np
import pandas as pd
import pytest
from c50.sklearn import C50Classifier
from model_text import without_date


def weighted_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return overlapping classes whose weighted classifier differs."""
    rng = np.random.default_rng(29)
    count = 120
    X = rng.normal(size=(count, 3))
    y = np.where(
        X[:, 0] + 0.5 * X[:, 1] + rng.normal(size=count) > 0, "yes", "no"
    )
    weights = np.where(X[:, 2] > 0, 4.0, 0.5) * rng.uniform(
        0.5, 1.5, size=count
    )
    return X, y, weights


def test_unit_and_scaled_weights_preserve_the_classifier() -> None:
    X, y, weights = weighted_problem()

    unweighted = C50Classifier().fit(X, y)
    unit = C50Classifier().fit(X, y, sample_weight=np.ones(len(y)))
    scalar = C50Classifier().fit(X, y, sample_weight=3.0)

    for fitted in (unit, scalar):
        assert without_date(fitted.model_.serialized_data) == without_date(
            unweighted.model_.serialized_data
        )
        np.testing.assert_array_equal(
            fitted.predict_proba(X), unweighted.predict_proba(X)
        )

    weighted = C50Classifier().fit(X, y, sample_weight=weights)
    for scale in (1e-30, 7.5, 1e30):
        scaled = C50Classifier().fit(X, y, sample_weight=weights * scale)
        assert without_date(scaled.model_.serialized_data) == without_date(
            weighted.model_.serialized_data
        )


def test_weights_change_the_classifier() -> None:
    X, y, weights = weighted_problem()

    unweighted = C50Classifier().fit(X, y)
    weighted = C50Classifier().fit(X, y, sample_weight=weights)

    assert without_date(weighted.model_.serialized_data) != without_date(
        unweighted.model_.serialized_data
    )
    assert "case weight: continuous." in weighted.model_.names_data
    assert "case weight" not in unweighted.model_.names_data


@pytest.mark.parametrize(
    ("sample_scale", "class_scale"),
    [(1e308, None), (1e-308, None), (1e200, 1e200), (1e-200, 1e-200)],
)
def test_extreme_uniform_weights_preserve_the_classifier(
    sample_scale: float,
    class_scale: float | None,
) -> None:
    X, y, _ = weighted_problem()
    expected = C50Classifier().fit(X, y)
    weights = np.full(len(y), sample_scale)
    class_weight = (
        None
        if class_scale is None
        else {
            "yes": class_scale,
            "no": class_scale,
        }
    )
    with np.errstate(all="raise"):
        fitted = C50Classifier(class_weight=class_weight).fit(
            X, y, sample_weight=weights
        )
    assert without_date(fitted.model_.serialized_data) == without_date(
        expected.model_.serialized_data
    )
    np.testing.assert_array_equal(
        fitted.predict_proba(X), expected.predict_proba(X)
    )
    np.testing.assert_array_equal(weights, np.full(len(y), sample_scale))


def test_extreme_products_do_not_silently_remove_positive_samples() -> None:
    X = np.arange(8.0).reshape(-1, 1)
    y = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    with pytest.raises(ValueError, match="single-precision"):
        C50Classifier(class_weight={0: 1e-200, 1: 1.0}).fit(
            X, y, sample_weight=np.full(8, 1e-200)
        )


def test_subnormal_products_preserve_relative_weights() -> None:
    X, y, weights = weighted_problem()
    expected = C50Classifier().fit(X, y, sample_weight=weights)
    with np.errstate(all="raise"):
        fitted = C50Classifier(class_weight={"yes": 1e-160, "no": 1e-160}).fit(
            X, y, sample_weight=weights * 1e-160
        )
    assert without_date(fitted.model_.serialized_data) == without_date(
        expected.model_.serialized_data
    )
    np.testing.assert_array_equal(
        fitted.predict_proba(X), expected.predict_proba(X)
    )


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"model_kind": "rules"},
        {"trials": 3},
        {"winnow": True},
        {"subset_splits": True, "global_pruning": False},
    ],
)
def test_weights_match_the_native_case_weight_attribute(
    options: dict[str, object],
) -> None:
    X, y, weights = weighted_problem()
    groups = np.asarray(["red", "green", "blue"], dtype=object)[
        (np.abs(X[:, 2]) * 3).astype(int) % 3
    ]
    X = np.column_stack((X.astype(object), groups))
    classifier = C50Classifier(**options).fit(X, y, sample_weight=weights)

    relative = weights / weights.mean()
    classes = {label: index for index, label in enumerate(classifier.classes_)}
    categories = {
        value: index for index, value in enumerate(classifier.categories_[3])
    }
    rows = [
        f"{float(row[0])!r}, {float(row[1])!r}, {float(row[2])!r}, "
        f"value_{categories[row[3]]}, {float(weight)!r}, "
        f"class_{classes[label]}"
        for row, weight, label in zip(X, relative, y, strict=True)
    ]
    native = c50.Model.train(
        classifier.model_.names_data,
        "\n".join(rows) + "\n",
        classifier.model_.kind,
        classifier._native_options(),
    )

    assert without_date(classifier.model_.serialized_data) == without_date(
        native.serialized_data
    )


def test_zero_weights_remove_samples() -> None:
    X, y, weights = weighted_problem()
    X = np.column_stack((X.astype(object), np.where(X[:, 2] > 0, "a", "b")))
    extra = np.asarray(
        [[9.0, 9.0, 9.0, "c"], [-9.0, -9.0, -9.0, "a"]], dtype=object
    )

    kept = C50Classifier().fit(X, y, sample_weight=weights)
    padded = C50Classifier().fit(
        np.vstack((X, extra)),
        np.concatenate((y, ["maybe", "yes"])),
        sample_weight=np.concatenate((weights, [0.0, 0.0])),
    )

    assert without_date(padded.model_.serialized_data) == without_date(
        kept.model_.serialized_data
    )
    assert padded.classes_.tolist() == ["no", "yes"]
    assert padded.categories_[3].tolist() == kept.categories_[3].tolist()
    np.testing.assert_array_equal(
        padded.predict_proba(X), kept.predict_proba(X)
    )


def test_dataframe_weights_and_weighted_model_round_trip() -> None:
    X, y, weights = weighted_problem()
    frame = pd.DataFrame(
        {
            "signal": X[:, 0],
            "noise": X[:, 1],
            "side": np.where(X[:, 2] > 0, "right", "left"),
        }
    )
    weights[::5] = 0

    classifier = C50Classifier(prediction_batch_size=7).fit(
        frame, y, sample_weight=weights
    )
    expected = C50Classifier().fit(
        frame.to_numpy(dtype=object), y, sample_weight=weights
    )

    assert without_date(classifier.model_.serialized_data) == without_date(
        expected.model_.serialized_data
    )
    probabilities = classifier.predict_proba(frame)
    np.testing.assert_array_equal(
        probabilities, expected.predict_proba(frame.to_numpy(dtype=object))
    )
    restored = pickle.loads(pickle.dumps(classifier))
    np.testing.assert_array_equal(restored.predict_proba(frame), probabilities)


@pytest.mark.parametrize(
    ("sample_weight", "message"),
    [
        (np.ones(7), "shape"),
        (np.ones((8, 1)), "shape"),
        (np.asarray([1, 1, 1, 1, 1, 1, 1, -1.0]), "nonnegative"),
        (np.asarray([1, 1, 1, 1, 1, 1, 1, np.nan]), "finite"),
        (np.asarray([1, 1, 1, 1, 1, 1, 1, np.inf]), "finite"),
        (np.zeros(8), "zero for every sample"),
        (np.asarray([1, 1, 1, 1, 1, 1, 1, 1e-60]), "single-precision"),
    ],
)
def test_invalid_weights_are_rejected(
    sample_weight: np.ndarray, message: str
) -> None:
    X = np.arange(8.0).reshape(-1, 1)
    y = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])

    with pytest.raises(ValueError, match=message):
        C50Classifier().fit(X, y, sample_weight=sample_weight)


def test_class_weights_are_sample_weights_by_class() -> None:
    X, y, weights = weighted_problem()
    by_class = np.where(y == "yes", 3.0, 0.5)

    expected = C50Classifier().fit(X, y, sample_weight=by_class)
    mapped = C50Classifier(class_weight={"yes": 3.0, "no": 0.5}).fit(X, y)
    scaled = C50Classifier(class_weight={"yes": 12.0, "no": 2.0}).fit(X, y)
    partial = C50Classifier(class_weight={"yes": 6.0}).fit(X, y)
    combined = C50Classifier(class_weight={"yes": 3.0, "no": 0.5}).fit(
        X, y, sample_weight=weights
    )

    assert without_date(mapped.model_.serialized_data) == without_date(
        expected.model_.serialized_data
    )
    assert without_date(scaled.model_.serialized_data) == without_date(
        expected.model_.serialized_data
    )
    assert without_date(partial.model_.serialized_data) == without_date(
        expected.model_.serialized_data
    )
    assert without_date(combined.model_.serialized_data) == without_date(
        C50Classifier()
        .fit(X, y, sample_weight=weights * by_class)
        .model_.serialized_data
    )
    np.testing.assert_array_equal(
        mapped.predict_proba(X), expected.predict_proba(X)
    )


def test_balanced_class_weights_favor_the_minority_class() -> None:
    rng = np.random.default_rng(31)
    X = rng.normal(size=(300, 2))
    y = np.where(
        X[:, 0] + rng.normal(scale=0.8, size=300) > 1.2, "rare", "common"
    )
    counts = {
        label: np.count_nonzero(y == label) for label in ("common", "rare")
    }
    balanced_weights = np.asarray(
        [len(y) / (2 * counts[label]) for label in y]
    )

    plain = C50Classifier().fit(X, y)
    balanced = C50Classifier(class_weight="balanced").fit(X, y)
    expected = C50Classifier().fit(X, y, sample_weight=balanced_weights)

    assert without_date(balanced.model_.serialized_data) == without_date(
        expected.model_.serialized_data
    )
    assert np.count_nonzero(balanced.predict(X) == "rare") > np.count_nonzero(
        plain.predict(X) == "rare"
    )


@pytest.mark.parametrize(
    ("class_weight", "error", "message"),
    [
        ({"yes": -1.0}, ValueError, "nonnegative"),
        ({"yes": np.inf}, ValueError, "finite"),
        ({"maybe": 2.0}, ValueError, "not in class_weight"),
        ("uniform", ValueError, "class_weight"),
        ({"yes": 0.0}, ValueError, "two classes"),
    ],
)
def test_invalid_class_weights_are_rejected(
    class_weight: object, error: type[Exception], message: str
) -> None:
    X, y, _ = weighted_problem()

    with pytest.raises(error, match=message):
        C50Classifier(class_weight=class_weight).fit(X, y)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("weight", "error"),
    [
        (c50.NOT_APPLICABLE, TypeError),
        (object(), TypeError),
        ("heavy", ValueError),
    ],
)
def test_non_numeric_sample_weight_raises_the_documented_exception(
    weight: object, error: type[Exception]
) -> None:
    # `docs/estimator.md` and the `fit` docstring document this rule. A
    # non-numeric object is a TypeError and a string that does not parse as a
    # number is a ValueError.
    X, y, weights = weighted_problem()
    values = np.asarray(weights, dtype=object)
    values[3] = weight
    with pytest.raises(error):
        C50Classifier().fit(X, y, sample_weight=values)


def test_non_numeric_continuous_value_raises_in_every_prediction_method() -> (
    None
):
    # The `Raises` sections of the prediction methods name this TypeError.
    X, y, _ = weighted_problem()
    fitted = C50Classifier().fit(X, y)
    cases = np.asarray(X[:2], dtype=object)
    cases[0, 0] = "x"
    for method in (
        fitted.predict,
        fitted.predict_proba,
        fitted.predict_log_proba,
    ):
        with pytest.raises(TypeError, match="continuous"):
            method(cases)


def _object_weights(element: object) -> np.ndarray:
    values = np.asarray(weighted_problem()[2], dtype=object)
    values[3] = element
    return values


REJECTED_WEIGHTS = {
    "datetime64 array": lambda: np.full(120, "2020-01-01", dtype="M8[D]"),
    "NaT array": lambda: np.full(120, "NaT", dtype="M8[D]"),
    "timedelta64 array": lambda: np.ones(120).astype("m8[s]"),
    "complex array": lambda: np.ones(120, dtype=complex),
    "complex64 array": lambda: np.ones(120, dtype=np.complex64),
    "structured array": lambda: np.ones(120, dtype=[("w", "f8")]),
    "None": lambda: _object_weights(None),
    "pandas.NA": lambda: _object_weights(pd.NA),
    "pandas.NaT": lambda: _object_weights(pd.NaT),
    "date": lambda: _object_weights(dt.date(2020, 1, 1)),
    "datetime": lambda: _object_weights(dt.datetime(2020, 1, 1)),
    "time": lambda: _object_weights(dt.time(1)),
    "timedelta": lambda: _object_weights(dt.timedelta(1)),
    "numpy datetime64": lambda: _object_weights(np.datetime64("2020-01-01")),
    "numpy timedelta64": lambda: _object_weights(np.timedelta64(1, "s")),
    "complex": lambda: _object_weights(1 + 0j),
    "list in an array": lambda: _object_weights([1.0]),
    "list with None": lambda: [None, *weighted_problem()[2][1:]],
    "scalar None": lambda: np.asarray(None),
    "scalar date": lambda: dt.date(2020, 1, 1),
    "scalar complex": lambda: 2 + 0j,
    "scalar numpy timedelta64": lambda: np.timedelta64(5, "s"),
    "datetime Series": lambda: pd.Series(pd.to_datetime(["2020-01-01"] * 120)),
    "timedelta Series": lambda: pd.Series(pd.to_timedelta(np.ones(120), "s")),
    "Int64 Series with NA": lambda: pd.Series(
        [1, None, *[1] * 118], dtype="Int64"
    ),
    "Float64 Series with NA": lambda: pd.Series(
        [1, None, *[1] * 118], dtype="Float64"
    ),
    "boolean Series with NA": lambda: pd.Series(
        [True, None, *[True] * 118], dtype="boolean"
    ),
}


@pytest.mark.parametrize("name", sorted(REJECTED_WEIGHTS))
def test_weights_that_are_not_numbers_or_strings_raise_type_error(
    name: str,
) -> None:
    # The rule is applied before any conversion to float64, which would turn
    # dates into numbers, None into NaN, and complex numbers into their real
    # parts.
    X, y, _ = weighted_problem()
    classifier = C50Classifier()

    with pytest.raises(TypeError, match="numbers or strings"):
        classifier.fit(X, y, sample_weight=REJECTED_WEIGHTS[name]())
    assert not hasattr(classifier, "model_")


def _accepted_weights() -> dict[str, tuple[object, np.ndarray]]:
    """Map a label to a weight input and the float64 weights it denotes."""
    weights = np.where(np.arange(120) % 3 == 0, 1.0, 2.0)
    ints = np.where(np.arange(120) % 3 == 0, 1, 2)
    return {
        "float64 list": (weights.tolist(), weights),
        "int tuple": (tuple(ints.tolist()), weights),
        "bool list": ([True] * 120, np.ones(120)),
        "float32 array": (weights.astype(np.float32), weights),
        "float16 array": (weights.astype(np.float16), weights),
        "int8 array": (ints.astype(np.int8), weights),
        "uint64 array": (ints.astype(np.uint64), weights),
        "bool array": (np.ones(120, dtype=bool), np.ones(120)),
        "string list": ([str(v) for v in ints], weights),
        "bytes array": (ints.astype("S1"), weights),
        "object array": (weights.astype(object), weights),
        "object array of numpy scalars": (
            np.asarray([np.float32(v) for v in weights], dtype=object),
            weights,
        ),
        "object array of Fractions": (
            np.asarray(
                [fractions.Fraction(int(v)) for v in ints], dtype=object
            ),
            weights,
        ),
        "object array of numpy bool": (
            np.asarray([np.bool_(True)] * 120, dtype=object),
            np.ones(120),
        ),
        "object array of Decimals": (
            np.asarray([decimal.Decimal(int(v)) for v in ints], dtype=object),
            weights,
        ),
        "Decimal list": ([decimal.Decimal(int(v)) for v in ints], weights),
        "scalar Decimal": (decimal.Decimal("2.5"), np.ones(120)),
        "scalar int": (3, np.ones(120)),
        "scalar float": (2.5, np.ones(120)),
        "scalar numpy float32": (np.float32(2), np.ones(120)),
        "scalar string": ("2", np.ones(120)),
        "int64 Series": (pd.Series(ints), weights),
        "float64 Series": (pd.Series(weights), weights),
        "Int64 Series": (pd.Series(ints, dtype="Int64"), weights),
        "Float64 Series": (pd.Series(weights, dtype="Float64"), weights),
        "boolean Series": (
            pd.Series([True] * 120, dtype="boolean"),
            np.ones(120),
        ),
        "string Series": (pd.Series([str(v) for v in ints]), weights),
    }


@pytest.mark.parametrize("name", sorted(_accepted_weights()))
def test_numeric_and_string_weights_train_the_float64_classifier(
    name: str,
) -> None:
    X, y, _ = weighted_problem()
    weights, expected = _accepted_weights()[name]

    fitted = C50Classifier().fit(X, y, sample_weight=weights)
    reference = C50Classifier().fit(X, y, sample_weight=expected)

    # The id line holds the local date of each fit.
    assert without_date(fitted.model_.serialized_data) == without_date(
        reference.model_.serialized_data
    )


@pytest.mark.parametrize(
    "weights",
    [
        pd.Series([np.nan, *[1.0] * 119]),
        pd.Series([np.inf, *[1.0] * 119]),
        np.asarray([np.nan, *[1.0] * 119], dtype=object),
        ["inf", *["1"] * 119],
        np.asarray(
            [decimal.Decimal("NaN"), *[decimal.Decimal(1)] * 119], dtype=object
        ),
        np.asarray(
            [decimal.Decimal("Infinity"), *[decimal.Decimal(1)] * 119],
            dtype=object,
        ),
        decimal.Decimal("Infinity"),
    ],
)
def test_nonfinite_numeric_weights_remain_value_errors(
    weights: object,
) -> None:
    X, y, _ = weighted_problem()

    with pytest.raises(ValueError, match="finite"):
        C50Classifier().fit(X, y, sample_weight=weights)


def _mixed_weights() -> dict[str, tuple[list[object], np.ndarray]]:
    """Map a label to a mixed list and the object array of its elements."""
    cases: dict[str, list[object]] = {
        "numpy complex128 and strings": [np.complex128(1 + 2j), "1", "1", "1"],
        "numpy complex64 and bytes": [np.complex64(1 + 2j), b"1", b"1", b"1"],
        "python complex and strings": [1 + 2j, "1", "1", "1"],
        "numpy datetime64 and strings": [np.datetime64("1970-01-02")]
        + ["1"] * 3,
        "date and strings": [dt.date(1970, 1, 2), "1", "1", "1"],
        "None and strings": [None, "1", "1", "1"],
        "complex and floats": [1 + 2j, 1.0, 1.0, 1.0],
    }
    result = {}
    for name, elements in cases.items():
        array = np.empty(len(elements), dtype=object)
        for index, element in enumerate(elements):
            array[index] = element
        result[name] = (elements, array)
    return result


@pytest.mark.parametrize("name", sorted(_mixed_weights()))
def test_mixed_lists_and_object_arrays_raise_the_same_type_error(
    name: str,
) -> None:
    # NumPy infers a string array from a list of strings and a complex number
    # or a date. The estimator must judge and convert the original elements.
    elements, array = _mixed_weights()[name]
    X = [[0], [1], [2], [3]]
    y = [0, 0, 1, 1]
    messages = []
    for weights in (elements, array):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(TypeError, match="numbers or strings") as info:
                C50Classifier().fit(X, y, sample_weight=weights)
        messages.append(str(info.value))
    assert messages[0] == messages[1]


def test_lists_of_zero_dimensional_arrays_train_the_float64_classifier() -> (
    None
):
    X = [[0], [1], [2], [3]]
    y = [0, 0, 1, 1]
    expected = C50Classifier().fit(X, y, sample_weight=[1.0, 2.0, 1.0, 2.0])

    for weights in (
        [np.array(1.0), np.array(2.0), np.array(1.0), np.array(2.0)],
        [np.array("1"), np.array("2"), np.array("1"), np.array("2")],
        [np.array(1), np.array(2.0), "1", decimal.Decimal(2)],
    ):
        fitted = C50Classifier().fit(X, y, sample_weight=weights)
        assert without_date(fitted.model_.serialized_data) == without_date(
            expected.model_.serialized_data
        )

    for weights in (
        [np.array(1j), "1", "1", "1"],
        [np.array(np.datetime64("1970-01-02")), 1.0, 1.0, 1.0],
    ):
        with pytest.raises(TypeError, match="numbers or strings"):
            C50Classifier().fit(X, y, sample_weight=weights)


@pytest.mark.parametrize(
    "weights",
    [
        [[1.0, 2.0], [3.0, 4.0]],
        [[1.0]] * 4,
        [(1.0,)] * 4,
        [np.array([1.0])] * 4,
        np.ones((4, 1)),
    ],
)
def test_weights_with_two_dimensions_fail_the_shape_check(
    weights: object,
) -> None:
    with pytest.raises(ValueError, match=r"one-dimensional|shape"):
        C50Classifier().fit(
            [[0], [1], [2], [3]], [0, 0, 1, 1], sample_weight=weights
        )


def _sequence_elements() -> dict[str, object]:
    return {
        "list": [2.0],
        "tuple": (2.0,),
        "range": range(1),
        "Series": pd.Series([2.0]),
        "one-element array": np.array([2.0]),
        "empty array": np.array([]),
    }


@pytest.mark.parametrize("name", sorted(_sequence_elements()))
@pytest.mark.parametrize("container", ["list", "object array"])
def test_weights_that_are_sequences_raise_type_error(
    name: str, container: str
) -> None:
    # A sequence or an array that is not 0-d is not a weight, whatever holds
    # it, so it raises like any other object that is not a number.
    elements: list[object] = [_sequence_elements()[name], 1.0, 1.0, 1.0]
    weights: object = elements
    if container == "object array":
        weights = np.empty(4, dtype=object)
        for index, element in enumerate(elements):
            weights[index] = element  # type: ignore[index]

    with pytest.raises(TypeError, match="numbers or strings"):
        C50Classifier().fit(
            [[0], [1], [2], [3]], [0, 0, 1, 1], sample_weight=weights
        )


@pytest.mark.parametrize(
    ("weights", "error", "message"),
    [
        ([(1.0,)] * 4, ValueError, "has shape"),
        ([[1.0, 2.0]] * 4, ValueError, "has shape"),
        ([[None]] * 4, TypeError, "numbers or strings"),
        ([[1j]] * 4, TypeError, "numbers or strings"),
        ([[1.0], [1.0, 2.0], [1.0], [1.0]], TypeError, "numbers or strings"),
    ],
    ids=[
        "equal-length tuples of floats",
        "equal-length lists of floats",
        "equal-length lists of None",
        "equal-length lists of complex",
        "ragged lists of floats",
    ],
)
def test_the_element_check_comes_before_the_shape_check(
    weights: list[object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        C50Classifier().fit(
            [[0], [1], [2], [3]], [0, 0, 1, 1], sample_weight=weights
        )


def test_a_masked_weight_array_is_read_without_its_mask() -> None:
    # scikit-learn reads a masked array through np.asarray, which drops the
    # mask. A caller who wants a sample removed sets its weight to zero.
    X, y, weights = weighted_problem()
    mask = np.arange(len(weights)) < 60
    masked = np.ma.array(weights, mask=mask)
    removed = np.where(mask, 0.0, weights)

    fitted = C50Classifier().fit(X, y, sample_weight=masked)
    unmasked = C50Classifier().fit(X, y, sample_weight=weights)
    zeroed = C50Classifier().fit(X, y, sample_weight=removed)

    assert without_date(fitted.model_.serialized_data) == without_date(
        unmasked.model_.serialized_data
    )
    assert without_date(fitted.model_.serialized_data) != without_date(
        zeroed.model_.serialized_data
    )


def test_zero_dimensional_arrays_in_object_arrays_follow_their_dtype() -> None:
    X = [[0], [1], [2], [3]]
    y = [0, 0, 1, 1]
    expected = C50Classifier().fit(X, y, sample_weight=[2.0, 1.0, 1.0, 1.0])

    held = np.empty(4, dtype=object)
    held[:] = [np.array(2.0), 1.0, 1.0, 1.0]
    fitted = C50Classifier().fit(X, y, sample_weight=held)
    assert without_date(fitted.model_.serialized_data) == without_date(
        expected.model_.serialized_data
    )

    held[0] = np.array(1j)
    with pytest.raises(TypeError, match="numbers or strings"):
        C50Classifier().fit(X, y, sample_weight=held)


STRING_DTYPE = getattr(getattr(np, "dtypes", None), "StringDType", None)
needs_string_dtype = pytest.mark.skipif(
    STRING_DTYPE is None, reason="NumPy has no variable-width string dtype"
)


@needs_string_dtype
def test_variable_width_numeric_strings_train_the_float64_classifier() -> None:
    assert STRING_DTYPE is not None
    X = [[0], [1], [2], [3]]
    y = [0, 0, 1, 1]
    weights = np.array(["1", "2", "1", "2"], dtype=STRING_DTYPE())

    fitted = C50Classifier().fit(X, y, sample_weight=weights)
    reference = C50Classifier().fit(X, y, sample_weight=[1.0, 2.0, 1.0, 2.0])

    assert without_date(fitted.model_.serialized_data) == without_date(
        reference.model_.serialized_data
    )


@needs_string_dtype
def test_variable_width_strings_that_do_not_parse_raise_value_error() -> None:
    assert STRING_DTYPE is not None
    weights = np.array(["1", "x", "1", "2"], dtype=STRING_DTYPE())

    with pytest.raises(ValueError, match="could not convert"):
        C50Classifier().fit(
            [[0], [1], [2], [3]], [0, 0, 1, 1], sample_weight=weights
        )


@needs_string_dtype
@pytest.mark.parametrize(
    ("sentinel", "error"),
    [
        (None, TypeError),
        (pd.NA, TypeError),
        (float("nan"), ValueError),
        ("2", None),
    ],
    ids=["None", "pandas.NA", "NaN", "string"],
)
def test_variable_width_string_missing_sentinels_follow_the_element_rule(
    sentinel: object, error: type[Exception] | None
) -> None:
    assert STRING_DTYPE is not None
    X = [[0], [1], [2], [3]]
    y = [0, 0, 1, 1]
    dtype = STRING_DTYPE(na_object=sentinel)
    weights = np.array(["1", sentinel, "1", "2"], dtype=dtype)

    if error is None:
        fitted = C50Classifier().fit(X, y, sample_weight=weights)
        reference = C50Classifier().fit(
            X, y, sample_weight=[1.0, 2.0, 1.0, 2.0]
        )
        assert without_date(fitted.model_.serialized_data) == without_date(
            reference.model_.serialized_data
        )
    else:
        with pytest.raises(error):
            C50Classifier().fit(X, y, sample_weight=weights)


def _three_class_problem() -> tuple[list[list[int]], list[int]]:
    return [[0]] * 8 + [[1]] * 8 + [[2]] * 8, [0] * 8 + [1] * 8 + [2] * 8


@pytest.mark.parametrize(
    "scale",
    [
        decimal.Decimal(1),
        decimal.Decimal("1e-300"),
        decimal.Decimal("1e300"),
        decimal.Decimal("5e-324"),
        fractions.Fraction(1, 10**300),
        10**300,
    ],
    ids=["1", "1e-300", "1e300", "5e-324", "Fraction 1e-300", "int 1e300"],
)
def test_scaled_weights_within_float64_keep_every_class(scale: object) -> None:
    X, y = _three_class_problem()
    weights = [scale] * 8 + [5 * scale] * 16  # type: ignore[operator]
    reference = C50Classifier().fit(X, y, sample_weight=[1.0] * 8 + [5.0] * 16)

    fitted = C50Classifier().fit(X, y, sample_weight=weights)

    assert fitted.classes_.tolist() == [0, 1, 2]
    np.testing.assert_array_equal(
        fitted.predict([[0], [1], [2]]), reference.predict([[0], [1], [2]])
    )


# NumPy's longdouble is as narrow as float64 on Windows and on Apple ARM64, and
# a double-double on PowerPC has float64's exponent range, so the itemsize does
# not tell. Compare the ranges, overflow and underflow separately.
_LONGDOUBLE_OVERFLOWS_LATER = (
    np.finfo(np.longdouble).maxexp > np.finfo(np.float64).maxexp
)
_LONGDOUBLE_UNDERFLOWS_LATER = (
    np.finfo(np.longdouble).smallest_subnormal
    < np.finfo(np.float64).smallest_subnormal
)


def _longdouble_above_float64() -> np.longdouble:
    """Return a longdouble that float64 cannot hold, or infinity."""
    if _LONGDOUBLE_OVERFLOWS_LATER:
        return np.longdouble("1e4000")
    return np.longdouble(np.inf)


def _longdouble_below_float64() -> np.longdouble:
    """Return a longdouble that float64 rounds to zero, or zero."""
    if _LONGDOUBLE_UNDERFLOWS_LATER:
        return np.longdouble("1e-4000")
    return np.longdouble(0)


def _out_of_range_message(weights: object) -> str:
    """Return the message that a weight outside float64 raises.

    A longdouble as narrow as float64 holds infinity and zero themselves, so
    the first is an infinite weight and the second an all-zero array.
    """
    if isinstance(weights, np.floating) and np.isinf(weights):
        return "must contain only finite values"
    if (
        isinstance(weights, np.ndarray)
        and weights.dtype.kind == "f"
        and not weights.any()
    ):
        return "cannot be zero for every sample"
    return "rounded to float64"


def _out_of_range_weights() -> dict[str, object]:
    return {
        "Decimal 1e-324 scale": [decimal.Decimal("1e-324")] * 8
        + [decimal.Decimal("5e-324")] * 16,
        "Decimal 1e400": decimal.Decimal("1e400"),
        "int 10**400": 10**400,
        "int 10**400 list": [10**400] * 24,
        "Fraction 10**400": fractions.Fraction(10**400),
        "Fraction 1 / 10**400": fractions.Fraction(1, 10**400),
        "string 1e400": "1e400",
        "string 1e-400": ["1e-400"] * 24,
        "bytes array 1e-400": np.array([b"1e-400"] * 24),
        "longdouble 1e4000": _longdouble_above_float64(),
        "longdouble 1e-4000 array": np.full(24, _longdouble_below_float64()),
        "object array with one underflow": np.asarray(
            [1.0] * 23 + [decimal.Decimal("1e-400")], dtype=object
        ),
    }


@pytest.mark.parametrize("name", sorted(_out_of_range_weights()))
def test_weights_outside_float64_raise_value_error(name: str) -> None:
    # A weight that rounds to infinity or to zero would change the ratios
    # between weights, so scale invariance cannot hold for it.
    X, y = _three_class_problem()
    weights = _out_of_range_weights()[name]
    classifier = C50Classifier()

    with pytest.raises(ValueError, match=_out_of_range_message(weights)):
        classifier.fit(X, y, sample_weight=weights)
    assert not hasattr(classifier, "model_")


def test_subnormal_float64_weights_keep_their_rescaling() -> None:
    X, y = _three_class_problem()
    weights = np.array([5e-324] * 8 + [2.5e-323] * 16)
    reference = C50Classifier().fit(X, y, sample_weight=[1.0] * 8 + [5.0] * 16)

    fitted = C50Classifier().fit(X, y, sample_weight=weights)

    assert without_date(fitted.model_.serialized_data) == without_date(
        reference.model_.serialized_data
    )


def _zero_weight_inputs() -> dict[str, object]:
    """Map a label to weights that hold zeros outside the float64 path."""
    values = [0, 1, 2, 1, 2, 1, 2, 0]
    return {
        "float list": [float(v) for v in values],
        "int list": values,
        "string list": [str(v) for v in values],
        "unicode array": np.array([str(v) for v in values]),
        "bytes array": np.array([str(v).encode() for v in values]),
        "Decimal list": [decimal.Decimal(v) for v in values],
        "Decimal zero among floats": [decimal.Decimal(0), *values[1:]],
        "Fraction list": [fractions.Fraction(v) for v in values],
        "Fraction zero among floats": [fractions.Fraction(0), *values[1:]],
        "longdouble array": np.array(values, dtype=np.longdouble),
        "longdouble list": [np.longdouble(v) for v in values],
        "Float64 Series": pd.Series(values, dtype="Float64"),
        "Int64 Series": pd.Series(values, dtype="Int64"),
        "object Series": pd.Series([float(v) for v in values], dtype=object),
    }


@pytest.mark.parametrize("name", sorted(_zero_weight_inputs()))
def test_zero_weights_outside_the_float64_path_remove_samples(
    name: str,
) -> None:
    # Each zero is exact, so it removes its sample instead of raising the range
    # error that a nonzero weight rounding to zero raises.
    X = [[0], [1], [2], [3]] * 2
    y = [0, 0, 1, 1] * 2
    expected = [0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0]

    fitted = C50Classifier().fit(
        X, y, sample_weight=_zero_weight_inputs()[name]
    )
    reference = C50Classifier().fit(X, y, sample_weight=expected)

    assert without_date(fitted.model_.serialized_data) == without_date(
        reference.model_.serialized_data
    )
    np.testing.assert_array_equal(fitted.predict(X), reference.predict(X))


class _ForeignReal:
    """A registered real number that is not in the closed list of types.

    Its zero does not equal 0, and NumPy cannot pass it to ``isfinite``, like
    the real numbers of arbitrary-precision libraries.
    """

    def __init__(self, value: float) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        return False

    __hash__ = object.__hash__

    def __float__(self) -> float:
        return self.value


numbers.Real.register(_ForeignReal)


def _foreign_real_weights() -> dict[str, object]:
    def array(values: list[object]) -> np.ndarray:
        result = np.empty(len(values), dtype=object)
        for index, value in enumerate(values):
            result[index] = value
        return result

    result: dict[str, object] = {}
    for label, value in (
        ("zero", 0.0),
        ("in range", 2.0),
        ("beyond float64", float("inf")),
    ):
        foreign = _ForeignReal(value)
        elements = [foreign, *[1.0] * 7]
        result[f"{label} in a list"] = elements
        result[f"{label} in an object array"] = array(elements)
        result[f"{label} as a scalar"] = foreign
    return result


@pytest.mark.parametrize("name", sorted(_foreign_real_weights()))
def test_real_numbers_outside_the_accepted_types_raise_type_error(
    name: str,
) -> None:
    classifier = C50Classifier()

    with pytest.raises(TypeError, match="numbers or strings"):
        classifier.fit(
            [[0], [1], [2], [3]] * 2,
            [0, 0, 1, 1] * 2,
            sample_weight=_foreign_real_weights()[name],
        )
    assert not hasattr(classifier, "model_")


class _Count(int):
    """An int whose float conversion disagrees with its value."""

    def __float__(self) -> float:
        return 0.0


class _Reading(float):
    """A float whose float conversion disagrees with its value."""

    def __float__(self) -> float:
        return 0.0


class _Amount(decimal.Decimal):
    """A Decimal that claims to equal zero."""

    def __eq__(self, other: object) -> bool:
        return True

    __hash__ = decimal.Decimal.__hash__


class _Text(str):
    """A str whose float conversion disagrees with its value."""

    def __float__(self) -> float:
        return 0.0


class _Level(enum.IntEnum):
    NONE = 0
    LOW = 1
    HIGH = 2


class _Double(np.float64):
    """A user subclass of a NumPy scalar type."""

    def __float__(self) -> float:
        return 0.0


class _Single(np.float32):
    """A user subclass of a NumPy scalar type that is not a Python float."""

    def __float__(self) -> float:
        return 0.0


_SUBCLASS_ELEMENTS: dict[str, tuple[object, object]] = {
    "int subclass": (_Count(5), 1),
    "float subclass": (_Reading(5.0), 1.0),
    "Decimal subclass": (_Amount(5), decimal.Decimal(1)),
    "str subclass": (_Text("5"), "1"),
    "IntEnum": (_Level.HIGH, 1),
    "numpy float64 subclass": (_Double(5.0), np.float64(1.0)),
    "numpy float32 subclass": (_Single(5.0), np.float32(1.0)),
}


@pytest.mark.parametrize("name", sorted(_SUBCLASS_ELEMENTS))
@pytest.mark.parametrize("container", ["list", "object array", "scalar"])
def test_subclasses_of_accepted_weight_types_raise_type_error(
    name: str, container: str
) -> None:
    # NumPy converts an object through its own __float__, which a subclass can
    # make disagree with the value that the range checks read.
    element, one = _SUBCLASS_ELEMENTS[name]
    X = [[0], [1], [2], [3]] * 2
    y = [0, 0, 1, 1] * 2
    weights: object = [element, *[one] * 7]
    if container == "object array":
        weights = np.asarray(weights, dtype=object)
    elif container == "scalar":
        weights = element
    classifier = C50Classifier()

    with pytest.raises(TypeError, match="numbers or strings"):
        classifier.fit(X, y, sample_weight=weights)
    assert not hasattr(classifier, "model_")


# Every scalar type that NumPy names with a dtype code. A sized alias such as
# `np.int64` names only one of the types that share a width, so a platform that
# has `np.longlong` beside `np.int_`, or `np.intc` beside `np.int32`, reaches a
# type that no alias spells.
_NUMPY_SCALAR_TYPES = sorted(
    {np.dtype(code).type for code in "?" + np.typecodes["AllInteger"] + "efdg"}
    | {np.str_, np.bytes_},
    key=lambda make: (make.__name__, make.__qualname__),
)


def _numpy_scalar(make: type, value: int) -> object:
    if make is np.bytes_:
        return np.bytes_(str(value).encode())
    return make(value)  # type: ignore[call-arg]


@pytest.mark.parametrize("container", ["list", "scalar"])
@pytest.mark.parametrize(
    "make", _NUMPY_SCALAR_TYPES, ids=lambda make: make.__name__
)
def test_numpy_scalar_types_stay_accepted(make: type, container: str) -> None:
    X = [[0], [1], [2], [3]] * 2
    y = [0, 0, 1, 1] * 2
    ones = [1, 2, 1, 2, 1, 2, 1, 1]
    if make is np.bool_:
        ones = [1] * 8
    if container == "scalar":
        reference = C50Classifier().fit(X, y, sample_weight=float(ones[0]))
        weights = _numpy_scalar(make, ones[0])
    else:
        reference = C50Classifier().fit(
            X, y, sample_weight=[float(v) for v in ones]
        )
        weights = [_numpy_scalar(make, v) for v in ones]

    fitted = C50Classifier().fit(X, y, sample_weight=weights)

    assert without_date(fitted.model_.serialized_data) == without_date(
        reference.model_.serialized_data
    )


def _weight_outcome(weights: object) -> tuple[str, str]:
    """Fit on eight samples and return the model text or the error."""
    X = [[0], [1], [2], [3]] * 2
    y = [0, 0, 1, 1] * 2
    try:
        fitted = C50Classifier().fit(X, y, sample_weight=weights)
    except (TypeError, ValueError) as exc:
        return type(exc).__name__, str(exc)
    return "trained", without_date(fitted.model_.serialized_data)


@pytest.mark.parametrize("text", ["0\x00", "inf\x00", "1e-400\x00", "0.5\x00"])
@pytest.mark.parametrize("make", [np.str_, np.bytes_])
def test_numpy_string_scalars_read_trailing_nuls_as_numpy_does(
    make: type, text: str
) -> None:
    value = make(text if make is np.str_ else text.encode())
    array = np.array([value] + [make("1" if make is np.str_ else b"1")] * 7)
    assert array.dtype.kind in "US"

    assert _weight_outcome([value, *[1.0] * 7]) == _weight_outcome(array)


def test_python_string_with_a_nul_stays_rejected() -> None:
    for text in ("0.5\x00", "0\x00"):
        with pytest.raises(ValueError, match="could not convert"):
            C50Classifier().fit(
                [[0], [1], [2], [3]],
                [0, 0, 1, 1],
                sample_weight=[text, 1.0, 1.0, 1.0],
            )


_LONGDOUBLE = np.longdouble

# Each type maps to the values that represent 0, 1, and 2, and to the values
# that round to infinity and to zero, where the type can leave the range.
_WEIGHT_TYPES: dict[str, tuple[object, object, object, object, object]] = {
    "bool": (False, True, True, None, None),
    "int": (0, 1, 2, 10**400, None),
    "float": (0.0, 1.0, 2.0, None, None),
    "numpy bool": (
        np.bool_(False),
        np.bool_(True),
        np.bool_(True),
        None,
        None,
    ),
    "numpy int64": (np.int64(0), np.int64(1), np.int64(2), None, None),
    "numpy float16": (np.float16(0), np.float16(1), np.float16(2), None, None),
    "numpy float32": (np.float32(0), np.float32(1), np.float32(2), None, None),
    "numpy float64": (np.float64(0), np.float64(1), np.float64(2), None, None),
    "numpy longdouble": (
        _LONGDOUBLE(0),
        _LONGDOUBLE(1),
        _LONGDOUBLE(2),
        _longdouble_above_float64(),
        _longdouble_below_float64() if _LONGDOUBLE_UNDERFLOWS_LATER else None,
    ),
    "Decimal": (
        decimal.Decimal(0),
        decimal.Decimal(1),
        decimal.Decimal(2),
        decimal.Decimal("1e400"),
        decimal.Decimal("1e-400"),
    ),
    "Fraction": (
        fractions.Fraction(0),
        fractions.Fraction(1),
        fractions.Fraction(2),
        fractions.Fraction(10**400),
        fractions.Fraction(1, 10**400),
    ),
    "str": ("0", "1", "2", "1e400", "1e-400"),
    "bytes": (b"0", b"1", b"2", b"1e400", b"1e-400"),
}


@pytest.mark.parametrize("name", sorted(_WEIGHT_TYPES))
def test_each_accepted_weight_type_follows_the_range_rule(name: str) -> None:
    zero, one, two, over, under = _WEIGHT_TYPES[name]
    X = [[0], [1], [2], [3]] * 2
    y = [0, 0, 1, 1] * 2
    weights = [zero, one, two, one, two, one, two, zero]
    reference = C50Classifier().fit(
        X,
        y,
        sample_weight=[float(w) for w in weights],  # type: ignore[arg-type]
    )

    # At zero and in range, the weights train the float64 classifier.
    fitted = C50Classifier().fit(X, y, sample_weight=weights)
    assert without_date(fitted.model_.serialized_data) == without_date(
        reference.model_.serialized_data
    )

    # On either side of the range, a weight that float64 cannot hold raises.
    for outside in (over, under):
        if outside is None:
            continue
        for held in ([one] * 7 + [outside], outside):
            with pytest.raises(
                ValueError, match=_out_of_range_message(outside)
            ):
                C50Classifier().fit(X, y, sample_weight=held)


ZERO_STRINGS = [
    "0e1000000000000000000",
    "0E+1000000000000000000",
    "0e-1999999999999999998",
    "0e-3000000000000000000",
    "0e100000000000000000000000",
    "+0e1000000000000000000",
    "-0e1000000000000000000",
    "-0.00E-3000000000000000000",
    "0_0e1000000000000000000",
    " \t0e1000000000000000000\n",
    "\N{ARABIC-INDIC DIGIT ZERO}e1000000000000000000",
]


def _zero_string_weights() -> dict[str, object]:
    result: dict[str, object] = {}
    for text in ZERO_STRINGS:
        result[f"list {text!r}"] = [text, *["1"] * 7]
        result[f"unicode array {text!r}"] = np.array([text, *["1"] * 7])
        if text.isascii():
            result[f"bytes array {text!r}"] = np.array(
                [text.encode(), *[b"1"] * 7]
            )
    return result


@pytest.mark.parametrize("name", sorted(_zero_string_weights()))
def test_zero_strings_with_huge_exponents_remove_samples(name: str) -> None:
    # A zero mantissa makes the weight zero whatever the exponent, although
    # Decimal cannot parse an exponent of this size.
    X = [[0], [1], [2], [3]] * 2
    y = [0, 0, 1, 1] * 2

    fitted = C50Classifier().fit(
        X, y, sample_weight=_zero_string_weights()[name]
    )
    reference = C50Classifier().fit(X, y, sample_weight=[0.0] + [1.0] * 7)

    assert without_date(fitted.model_.serialized_data) == without_date(
        reference.model_.serialized_data
    )


@pytest.mark.parametrize(
    "text",
    [
        "1e-3000000000000000000",
        "1e3000000000000000000",
        "0.1e-2000000000000000000",
    ],
)
def test_nonzero_strings_with_huge_exponents_still_raise(text: str) -> None:
    for weights in ([text, *["1"] * 7], np.array([text, *["1"] * 7])):
        with pytest.raises(ValueError, match="rounded to float64"):
            C50Classifier().fit(
                [[0], [1], [2], [3]] * 2,
                [0, 0, 1, 1] * 2,
                sample_weight=weights,
            )


@pytest.mark.parametrize("trapped", [True, False])
def test_string_weights_ignore_the_callers_decimal_context(
    trapped: bool,
) -> None:
    # Decimal reports an unparsable string through the thread's context. With
    # the trap off it returns NaN, which hid the zero mantissa, and with the
    # trap on it left the caller's flag set.
    X = [[0], [1], [2], [3]] * 2
    y = [0, 0, 1, 1] * 2
    reference = C50Classifier().fit(X, y, sample_weight=[0.0] + [1.0] * 7)
    zero = ["0e1000000000000000000"] + ["1"] * 7
    nonzero = ["1e1000000000000000000"] + ["1"] * 7

    with decimal.localcontext() as context:
        context.traps[decimal.InvalidOperation] = trapped
        context.clear_flags()
        fitted = C50Classifier().fit(X, y, sample_weight=zero)
        assert not any(context.flags.values())
        with pytest.raises(ValueError, match="rounded to float64"):
            C50Classifier().fit(X, y, sample_weight=nonzero)
        assert not any(context.flags.values())
        assert context.traps[decimal.InvalidOperation] is trapped

    assert without_date(fitted.model_.serialized_data) == without_date(
        reference.model_.serialized_data
    )


def test_each_distinct_zero_is_checked_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A column of strings holds a few spellings of zero, and a Python-level
    # check of each of a million zeros made conversion several times slower.
    # Equal objects of different types are still checked separately.
    calls: list[object] = []
    exact_check = c50.sklearn._weight_is_zero

    def counting(element: object) -> bool:
        calls.append(element)
        return exact_check(element)

    monkeypatch.setattr("c50.sklearn._weight_is_zero", counting)
    zeros: list[object] = [
        decimal.Decimal(0),
        fractions.Fraction(0),
        0,
        False,
        0.0,
        "0",
        "0.00",
        b"0",
    ]
    weights = [zero for zero in zeros for _ in range(250)] + [1.0] * 250
    count = len(weights)
    X = [[i % 4] for i in range(count)]
    y = [i % 2 for i in range(count)]
    arrays = [
        np.array(["0"] * 250 + ["1"] * 250),
        np.array([b"0"] * 250 + [b"1"] * 250),
    ]

    C50Classifier().fit(X, y, sample_weight=weights)
    assert sorted(map(repr, calls)) == sorted(
        map(
            repr,
            [
                decimal.Decimal(0),
                fractions.Fraction(0),
                "0",
                "0.00",
                b"0",
            ],
        )
    )

    for array in arrays:
        calls.clear()
        C50Classifier().fit(
            [[i % 4] for i in range(500)],
            [i % 2 for i in range(500)],
            sample_weight=array,
        )
        assert len(calls) == 1


def test_exact_weight_checks_skip_types_that_float64_holds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A zero of a type that float64 holds exactly is not a lost weight, and a
    # Python-level call for each of a million zeros made conversion ten times
    # slower.
    def fail(element: object) -> bool:
        raise AssertionError(f"exact check of {element!r}")

    monkeypatch.setattr("c50.sklearn._weight_is_zero", fail)
    monkeypatch.setattr("c50.sklearn._weight_is_finite", fail)
    X = [[0], [1], [2], [3]]
    y = [0, 0, 1, 1]

    for weights in (
        [0.0, 1.0, 2.0, 1.0],
        [0, 1, 2, 1],
        [False, True, True, True],
        [np.float32(0), 1.0, np.float16(2), np.bool_(True)],
        [np.int64(0), np.uint8(1), 2, 1.0],
        pd.Series([0.0, 1.0, 2.0, 1.0], dtype="Float64"),
    ):
        C50Classifier().fit(X, y, sample_weight=weights)
