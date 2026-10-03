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
    assert schema.categories == ((), ("blue", "red"))
    assert schema.names_data(2) == (
        "class_0, class_1.\n\n"
        "feature_0: continuous.\n"
        "feature_1: value_0, value_1.\n"
    )
    assert schema.dense_data(X, "error") == pytest.approx(
        np.asarray([[0.5, 1], [1.25, 0], [2, 1]], dtype=float)
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

    assert schema.categories == ((b"nan", "", "nan"),)
    encoded = schema.dense_data(X, "error")[:, 0]
    np.testing.assert_array_equal(encoded, [2, 0, 1, np.nan, np.nan])


def test_dataframe_columns_keep_numeric_dtypes() -> None:
    frame = pd.DataFrame(
        {
            "real": [0.5, 1.5, 2.5],
            "count": np.asarray([1, 2, 3], dtype=np.uint8),
            "flag": [True, False, True],
            "word": pd.Series(["a", "b", "a"], dtype=object),
            "when": pd.to_datetime([0, 1, 0], unit="D"),
        }
    )

    columns = dataframe_columns(frame)

    assert columns.shape == (3, 5)
    assert [getattr(array, "dtype", None) for array in columns.arrays] == [
        np.dtype(np.float64),
        np.dtype(np.uint8),
        np.dtype(bool),
        np.dtype(object),
        np.dtype(object),
    ]
    assert columns[1:].shape == (2, 5)
    assert columns[1:].arrays[1].tolist() == [2, 3]

    schema = fit_schema(columns, categorical_features=None)
    assert schema.categorical_indices == (2, 3, 4)
    np.testing.assert_array_equal(
        schema.dense_data(columns[1:], "error"),
        np.asarray([[1.5, 2, 0, 1, 1], [2.5, 3, 1, 0, 0]]),
    )


def test_coded_columns_use_dtype_or_sorted_order() -> None:
    frame = pd.DataFrame(
        {
            "group": pd.Categorical(
                ["b", None, "c", "b", "a"], categories=["a", "b", "c", "d"]
            ),
            "word": pd.array(["y", "z", None, "y", "x"], dtype="string"),
            "count": pd.array([1, None, 3, 4, 5], dtype="Int64"),
        }
    )

    columns = dataframe_columns(frame)
    schema = fit_schema(columns[1:], categorical_features=None)

    assert [type(array).__name__ for array in columns.arrays] == [
        "CodedColumn",
        "CodedColumn",
        "NullableNumbers",
    ]
    # A categorical dtype declares its order, and strings are sorted, so
    # neither depends on the order of the rows.
    assert schema.categories == (("a", "b", "c"), ("x", "y", "z"), ())
    np.testing.assert_array_equal(
        schema.dense_data(columns, "error"),
        np.asarray(
            [
                [1, 1, 1],
                [np.nan, 2, np.nan],
                [2, np.nan, 3],
                [1, 1, 4],
                [0, 0, 5],
            ]
        ),
    )
    declared = fit_schema(columns, categorical_features=(0, 1, 2))
    assert declared.categories[2] == (1, 3, 4, 5)
    assert all(type(value) is int for value in declared.categories[2])


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


def test_declared_unused_categories_predict_as_missing() -> None:
    from c50.sklearn import C50Classifier

    colors = ["red", "red", "blue", "blue", "red", "blue"] * 3
    size = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] * 3
    y = ["a", "a", "b", "b", "a", "b"] * 3
    used = pd.CategoricalDtype(["blue", "red"])
    declared = pd.CategoricalDtype(["blue", "green", "red"])
    train_used = pd.DataFrame(
        {"color": pd.Series(colors, dtype=used), "size": size}
    )
    train_declared = train_used.astype({"color": declared})

    options = {"categorical_features": "from_dtype", "minimum_cases": 1}
    reference = C50Classifier(**options).fit(train_used, y)
    fitted = C50Classifier(**options).fit(train_declared, y)
    # Declared categories do not change the classifier or fitted categories.
    assert fitted.model_.names_data == reference.model_.names_data
    assert fitted.model_.serialized_data == reference.model_.serialized_data
    assert [list(c) for c in fitted.categories_] == [
        list(c) for c in reference.categories_
    ]

    green = pd.DataFrame(
        {"color": pd.Series(["green", "green"], dtype=declared), "size": [1.0, 6.0]}
    )
    missing = pd.DataFrame(
        {"color": pd.Series([None, None], dtype=declared), "size": [1.0, 6.0]}
    )
    np.testing.assert_array_equal(
        fitted.predict_proba(green), fitted.predict_proba(missing)
    )
    # Object arrays carry no dtype, but the category was declared in fitting.
    np.testing.assert_array_equal(
        fitted.predict_proba(green.astype({"color": object})),
        fitted.predict_proba(missing),
    )

    undeclared = pd.DataFrame(
        {"color": pd.Series(["purple"], dtype="category"), "size": [1.0]}
    )
    with pytest.raises(ValueError, match="unseen category 'purple'"):
        fitted.predict(undeclared)
    with pytest.raises(ValueError, match="unseen category 'green'"):
        reference.predict(green)



def test_ordered_categoricals_are_ordered_attributes() -> None:
    from c50.sklearn import C50Classifier

    levels = ["tiny", "small", "medium", "large", "huge"]
    rng = np.random.default_rng(1)
    values = np.asarray(levels)[rng.integers(0, len(levels), 300)]
    y = np.where(np.isin(values, ["large", "huge"]), "big", "little")

    def fit(ordered: bool) -> C50Classifier:
        dtype = pd.CategoricalDtype(levels, ordered=ordered)
        X = pd.DataFrame({"size": pd.Series(values, dtype=dtype)})
        return C50Classifier(categorical_features="from_dtype").fit(X, y)

    ordered = fit(True)
    unordered = fit(False)
    assert "feature_0: [ordered] value_0," in ordered.model_.names_data
    assert "[ordered]" not in unordered.model_.names_data
    # Both follow the dtype's category order rather than first occurrence.
    assert list(ordered.categories_[0]) == levels
    assert list(unordered.categories_[0]) == levels

    # An ordered test has three branches: N/A, at or below a cut, and above.
    root = ordered.model_.inspect().trees[0].nodes[0]
    assert len(root.branches) == 3
    assert "'size' in {'tiny', 'small', 'medium'}" in ordered.export_text()
    query = pd.DataFrame(
        {"size": pd.Series(["small", "huge"], dtype=pd.CategoricalDtype(levels, ordered=True))}
    )
    assert ordered.predict(query).tolist() == ["little", "big"]


def test_category_order_does_not_depend_on_row_order() -> None:
    frame = pd.DataFrame(
        {
            "word": pd.Series(["pear", "fig", None, "apple", "fig"], dtype=object),
            "text": pd.array(["y", "z", None, "y", "x"], dtype="string"),
            "group": pd.Categorical(
                ["b", None, "c", "b", "a"], categories=["c", "a", "b", "d"]
            ),
            "mixed": pd.Series(["b", 2, "a", 1, True], dtype=object),
        }
    )
    reversed_frame = frame.iloc[::-1].reset_index(drop=True)

    schema = fit_schema(dataframe_columns(frame), categorical_features=None)
    reversed_schema = fit_schema(
        dataframe_columns(reversed_frame), categorical_features=None
    )

    # Strings sort, a categorical dtype keeps its declared order without unused
    # categories, and mixed values group by type name: bool, int, then str.
    assert schema.categories == (
        ("apple", "fig", "pear"),
        ("x", "y", "z"),
        ("c", "a", "b"),
        (True, 1, 2, "a", "b"),
    )
    assert reversed_schema.categories == schema.categories
    assert reversed_schema.names_data(2) == schema.names_data(2)


def test_permuted_training_rows_train_the_same_classifier() -> None:
    from c50.sklearn import C50Classifier

    rng = np.random.default_rng(7)
    rows = 600
    colors = np.asarray(["red", "green", "blue", "cyan", "gray", "pink"])
    shapes = ["circle", "square", "star", "hex"]
    color = colors[rng.integers(0, len(colors), rows)]
    shape = np.asarray(shapes)[rng.integers(0, len(shapes), rows)]
    size = rng.normal(size=rows)
    y = np.where(
        np.isin(color, ["red", "pink"]) ^ (shape == "star") ^ (size > 0.5),
        "yes",
        "no",
    )
    frame = pd.DataFrame(
        {
            "color": pd.Series(color, dtype=object),
            "shape": pd.Categorical(shape, categories=shapes),
            "size": size,
        }
    )

    def body(X: pd.DataFrame, labels: np.ndarray) -> str:
        model = C50Classifier(minimum_cases=1).fit(X, labels).model_
        return model.serialized_data.split("\n", 1)[1]

    reference = body(frame, y)
    for seed in range(3):
        order = np.random.default_rng(seed).permutation(rows)
        permuted = frame.iloc[order].reset_index(drop=True)
        assert body(permuted, y[order]) == reference
