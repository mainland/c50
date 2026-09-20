"""Tests for array-to-C5.0 data encoding."""

from __future__ import annotations

import numpy as np
import pytest

from c50._data import encode_costs, fit_schema


def test_schema_encodes_continuous_and_categorical_features() -> None:
    X = np.asarray(
        [
            [0.5, "red"],
            [1.25, "blue"],
            [2.0, "red"],
        ],
        dtype=object,
    )

    schema = fit_schema(X, categorical_features=(1,))

    assert schema.categorical_indices == (1,)
    assert schema.categories == ((), ("red", "blue"))
    assert schema.names_data(2) == (
        "class_0, class_1.\n\n"
        "feature_0: continuous.\n"
        "feature_1: discrete 2.\n"
    )
    assert schema.training_data(X, np.asarray([0, 1, 0])) == (
        "0.5, value_0, class_0\n"
        "1.25, value_1, class_1\n"
        "2, value_0, class_0\n"
    )


def test_schema_infers_categories_and_encodes_missing_values() -> None:
    X = np.asarray(
        [
            [1.0, "alpha"],
            [np.nan, None],
        ],
        dtype=object,
    )

    schema = fit_schema(X, categorical_features=None)

    assert schema.categorical_indices == (1,)
    assert schema.prediction_data(X, "error") == (
        "1, value_0, ?\n"
        "?, ?, ?\n"
    )


def test_schema_accepts_named_categorical_features() -> None:
    X = np.asarray([[1, 10], [2, 20]], dtype=float)

    schema = fit_schema(
        X,
        categorical_features=("code",),
        feature_names=("measurement", "code"),
    )

    assert schema.categorical_indices == (1,)
    assert schema.categories == ((), (10.0, 20.0))


def test_schema_rejects_invalid_categorical_selection() -> None:
    X = np.asarray([[1, 2], [3, 4]])

    with pytest.raises(ValueError, match="feature names"):
        fit_schema(X, categorical_features=("kind",))
    with pytest.raises(ValueError, match="outside"):
        fit_schema(X, categorical_features=(2,))
    with pytest.raises(ValueError, match="repeated"):
        fit_schema(X, categorical_features=(1, 1))


def test_schema_controls_unseen_category_behavior() -> None:
    schema = fit_schema(
        np.asarray([["red"], ["blue"]], dtype=object),
        categorical_features=(0,),
    )
    unseen = np.asarray([["green"]], dtype=object)

    with pytest.raises(ValueError, match="unseen category 'green'"):
        schema.prediction_data(unseen, "error")
    assert schema.prediction_data(unseen, "missing") == "?, ?\n"


def test_schema_rejects_nonfinite_continuous_values() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        fit_schema(
            np.asarray([[1.0], [np.inf]]),
            categorical_features=(),
        )


def test_cost_matrix_encoding_uses_predicted_by_actual_order() -> None:
    assert encode_costs(2, [[0, 5], [2, 0]]) == (
        "class_0, class_1: 5\n"
        "class_1, class_0: 2\n"
    )


def test_cost_matrix_validation() -> None:
    with pytest.raises(ValueError, match="shape"):
        encode_costs(2, [[0, 1]])
    with pytest.raises(ValueError, match="nonnegative"):
        encode_costs(2, [[0, -1], [1, 0]])
    with pytest.raises(ValueError, match="diagonal"):
        encode_costs(2, [[1, 1], [1, 0]])
