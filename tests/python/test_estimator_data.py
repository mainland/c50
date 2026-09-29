"""Tests for array-to-C5.0 data encoding."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from c50._data import dataframe_columns, encode_costs, fit_schema


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
        "feature_1: value_0, value_1.\n"
    )
    assert schema.dense_data(X, "error") == pytest.approx(
        np.asarray([[0.5, 0], [1.25, 1], [2, 0]], dtype=float)
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
    encoded = schema.dense_data(X, "error")
    assert encoded[0] == pytest.approx([1.0, 0.0])
    assert np.all(np.isnan(encoded[1]))


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
        schema.dense_data(unseen, "error")
    assert np.isnan(schema.dense_data(unseen, "missing")[0, 0])


def test_schema_rejects_nonfinite_continuous_values() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        fit_schema(
            np.asarray([[1.0], [np.inf]]),
            categorical_features=(),
        )


def test_strings_are_categories_not_missing_values() -> None:
    X = np.asarray([["nan"], [b"nan"], [""], [None], [np.nan]], dtype=object)

    schema = fit_schema(X, categorical_features=None)

    assert schema.categories == (("nan", b"nan", ""),)
    encoded = schema.dense_data(X, "error")[:, 0]
    np.testing.assert_array_equal(encoded, [0, 1, 2, np.nan, np.nan])


def test_dataframe_columns_keep_numeric_dtypes() -> None:
    frame = pd.DataFrame(
        {
            "real": [0.5, 1.5, 2.5],
            "count": np.asarray([1, 2, 3], dtype=np.uint8),
            "flag": [True, False, True],
            "word": ["a", "b", "a"],
            "nullable": pd.array([1, None, 3], dtype="Int64"),
        }
    )

    columns = dataframe_columns(frame)

    assert columns.shape == (3, 5)
    assert [array.dtype for array in columns.arrays] == [
        np.dtype(np.float64),
        np.dtype(np.uint8),
        np.dtype(bool),
        np.dtype(object),
        np.dtype(object),
    ]
    assert columns[1:].shape == (2, 5)
    assert columns[1:].arrays[1].tolist() == [2, 3]

    schema = fit_schema(columns, categorical_features=None)
    assert schema.categorical_indices == (2, 3)
    np.testing.assert_array_equal(
        schema.dense_data(columns[1:], "error"),
        np.asarray([[1.5, 2, 1, 1, np.nan], [2.5, 3, 0, 0, 3]]),
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
