# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The estimator marks not-applicable values with ``c50.NOT_APPLICABLE``."""

from __future__ import annotations

import copy
import json
import pickle
from decimal import Decimal
from typing import Any

import c50
import c50.sklearn
import numpy as np
import pandas as pd
import pytest
from c50._data import fit_schema
from c50.sklearn import C50Classifier
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

NA = c50.NOT_APPLICABLE
CLASSES = ("hi", "lo", "mid")
G_LEVELS = ("a", "b", "c")
H_LEVELS = ("p", "q", "r", "s", "t")
COLUMNS = ("x", "y", "g", "h")


def table(
    rows: int, seed: int, with_na: bool
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.intp]]:
    """Return values, a not-applicable mask, and class indices.

    NaN marks missing values. The class depends on N/A as well as on the
    values, so that the learner has a reason to split on N/A.
    """
    rng = np.random.default_rng(seed)
    values = np.empty((rows, 4))
    values[:, 0] = rng.integers(0, 80, rows) / 4
    values[:, 1] = rng.integers(0, 80, rows) / 4
    values[:, 2] = rng.integers(0, 3, rows)
    values[:, 3] = rng.integers(0, 5, rows)
    mask = np.zeros((rows, 4), dtype=np.bool_)
    if with_na:
        for column, period in enumerate((4, 6, 5, 4)):
            mask[:, column] = rng.integers(0, period, rows) == 0
        mask[3::17, :] = True
    values[rng.integers(0, 11, rows) == 0, 0] = np.nan
    values[rng.integers(0, 9, rows) == 0, 3] = np.nan
    labels = np.where(values[:, 0] > 12, 1, 0)
    labels = np.where(
        (values[:, 3] == 3) | (values[:, 3] == 4),
        1 + (values[:, 0] <= 10),
        labels,
    )
    labels = np.where(mask[:, 2], 2, labels)
    labels = np.where(mask[:, 0], 0, labels)
    labels = np.where(mask[:, 3], (labels + 1) % 3, labels)
    return values, mask, labels.astype(np.intp)


def cells(
    values: NDArray[np.float64],
    mask: NDArray[np.bool_],
    g_levels: tuple[str, ...] = G_LEVELS,
    h_levels: tuple[str, ...] = H_LEVELS,
) -> NDArray[Any]:
    """Return an object array with the sentinel under the mask.

    Missing values are NaN in the continuous and first categorical column and
    ``None`` in the second categorical column.
    """
    result = np.empty(values.shape, dtype=object)
    for row, (line, flags) in enumerate(zip(values, mask, strict=True)):
        result[row, 0] = float(line[0])
        result[row, 1] = float(line[1])
        result[row, 2] = (
            np.nan if np.isnan(line[2]) else g_levels[int(line[2])]
        )
        result[row, 3] = None if np.isnan(line[3]) else h_levels[int(line[3])]
        for column in np.flatnonzero(flags):
            result[row, column] = NA
    return result


def names_of(
    g_levels: tuple[str, ...], h_levels: tuple[str, ...], ordered: bool
) -> str:
    """Return the names data that the estimator builds for this table."""
    flag = "[ordered] " if ordered else ""
    classes = ", ".join(f"class_{k}" for k in range(len(CLASSES)))
    return (
        f"{classes}.\n\n"
        "feature_0: continuous.\n"
        "feature_1: continuous.\n"
        f"feature_2: {flag}"
        + ", ".join(f"value_{k}" for k in range(len(g_levels)))
        + ".\n"
        f"feature_3: {flag}"
        + ", ".join(f"value_{k}" for k in range(len(h_levels)))
        + ".\n"
    )


def text_of(
    values: NDArray[np.float64],
    mask: NDArray[np.bool_],
    classes: NDArray[np.intp] | None,
) -> str:
    """Return the data text of a table with ``N/A`` under the mask.

    Category ``k`` of a feature is the token ``value_k`` and class ``k`` is
    ``class_k``, as the estimator names them.
    """
    lines = []
    for row, flags in zip(values, mask, strict=True):
        fields = []
        for column, (value, na) in enumerate(zip(row, flags, strict=True)):
            if na:
                fields.append("N/A")
            elif np.isnan(value):
                fields.append("?")
            elif column < 2:
                fields.append(repr(float(value)))
            else:
                fields.append(f"value_{int(value)}")
        lines.append(",".join(fields))
    if classes is None:
        return "".join(f"{line},?\n" for line in lines)
    return "".join(
        f"{line},class_{k}\n" for line, k in zip(lines, classes, strict=True)
    )


def labels_of(classes: NDArray[np.intp]) -> NDArray[Any]:
    return np.asarray([CLASSES[k] for k in classes], dtype=object)


def category_frame(
    objects: NDArray[Any],
    g_levels: tuple[str, ...],
    h_levels: tuple[str, ...],
    ordered: bool,
) -> pd.DataFrame:
    """Return a frame whose categorical columns list the sentinel."""
    frame = pd.DataFrame(objects, columns=list(COLUMNS))
    frame["g"] = pd.Categorical(
        frame["g"], categories=[*g_levels, NA], ordered=ordered
    )
    frame["h"] = pd.Categorical(
        frame["h"], categories=[*h_levels, NA], ordered=ordered
    )
    return frame


def body(model_text: str) -> list[str]:
    """Return a model without the line that has the time stamp."""
    return [
        line for line in model_text.splitlines() if not line.startswith("id=")
    ]


def native_probabilities(
    predictions: c50.Predictions,
) -> NDArray[np.float64]:
    """Normalize native scores as ``predict_proba`` does."""
    scores = np.asarray(predictions.scores, dtype=np.float64)
    totals = scores.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.asarray(scores / totals)
    # A row without a score assigns probability one to the predicted class.
    for row in np.flatnonzero(totals[:, 0] == 0):
        result[row] = 0.0
        result[row, int(predictions.class_indices[row])] = 1.0
    return result


SETTINGS: dict[str, dict[str, Any]] = {
    "tree": {},
    "rules": {"model_kind": "rules"},
    "boosted tree": {"trials": 4},
    "boosted rules": {"model_kind": "rules", "trials": 3},
    "subsets and winnowing": {"subset_splits": True, "winnow": True},
}


def native_options(settings: dict[str, Any]) -> dict[str, Any]:
    options = c50.Options()
    options.trials = settings.get("trials", 1)
    options.subset_splits = settings.get("subset_splits", False)
    options.winnow = settings.get("winnow", False)
    return {
        "kind": (
            c50.ModelKind.RULES
            if settings.get("model_kind") == "rules"
            else c50.ModelKind.TREE
        ),
        "options": options,
    }


def estimator(settings: dict[str, Any]) -> C50Classifier:
    # The reference tie order is the default of the text interface.
    return C50Classifier(ties="reference", **settings)


INPUTS = ["object array", "object frame", "category frame", "ordered frame"]


def build_input(
    kind: str, objects: NDArray[Any]
) -> NDArray[Any] | pd.DataFrame:
    if kind == "object array":
        return objects
    if kind == "object frame":
        return pd.DataFrame(objects, columns=list(COLUMNS))
    ordered = kind == "ordered frame"
    return category_frame(objects, LEVELS[kind][0], LEVELS[kind][1], ordered)


# Category order of each input kind. The pandas category kinds declare their
# levels in an order that differs from the sorted one.
LEVELS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "object array": (G_LEVELS, H_LEVELS),
    "object frame": (G_LEVELS, H_LEVELS),
    "category frame": (("c", "a", "b"), ("t", "p", "s", "q", "r")),
    "ordered frame": (("c", "a", "b"), ("t", "p", "s", "q", "r")),
}


def cases_for(kind: str, values: NDArray[np.float64]) -> NDArray[Any]:
    """Return object cells for ``values``, whose codes follow LEVELS."""
    g_levels, h_levels = LEVELS[kind]
    return cells(
        values, np.zeros(values.shape, dtype=bool), g_levels, h_levels
    )


@pytest.mark.parametrize("kind", INPUTS)
@pytest.mark.parametrize("name", sorted(SETTINGS))
def test_estimator_matches_text_input_with_not_applicable(
    name: str, kind: str
) -> None:
    settings = SETTINGS[name]
    g_levels, h_levels = LEVELS[kind]
    ordered = kind == "ordered frame"
    values, mask, classes = table(160, 12345, True)
    case_values, case_mask, _ = table(70, 777, True)

    names = names_of(g_levels, h_levels, ordered)
    expected_model = c50.train(
        names, text_of(values, mask, classes), **native_options(settings)
    )
    assert any(
        "N/A" in line for line in expected_model.serialized_data.splitlines()
    )

    fitted = estimator(settings).fit(
        build_input(kind, cells(values, mask, g_levels, h_levels)),
        labels_of(classes),
    )
    assert body(fitted.model_.serialized_data) == body(
        expected_model.serialized_data
    )

    cases = build_input(
        kind, cells(case_values, case_mask, g_levels, h_levels)
    )
    expected = expected_model.predict_details(
        text_of(case_values, case_mask, None)
    )
    assert fitted.predict(cases).tolist() == [
        CLASSES[int(label.removeprefix("class_"))] for label in expected.labels
    ]
    np.testing.assert_array_equal(
        fitted.predict_proba(cases), native_probabilities(expected)
    )
    with np.errstate(divide="ignore"):
        np.testing.assert_array_equal(
            fitted.predict_log_proba(cases),
            np.log(fitted.predict_proba(cases)),
        )


@pytest.mark.parametrize("kind", ["object array", "category frame"])
@pytest.mark.parametrize("name", ["tree", "rules", "boosted tree"])
def test_not_applicable_only_at_prediction(name: str, kind: str) -> None:
    settings = SETTINGS[name]
    g_levels, h_levels = LEVELS[kind]
    values, mask, classes = table(160, 12345, False)
    assert not mask.any()
    expected_model = c50.train(
        names_of(g_levels, h_levels, False),
        text_of(values, mask, classes),
        **native_options(settings),
    )
    train = cells(values, mask, g_levels, h_levels)
    fitted = estimator(settings).fit(
        build_input(kind, train), labels_of(classes)
    )
    assert body(fitted.model_.serialized_data) == body(
        expected_model.serialized_data
    )

    case_values, case_mask, _ = table(70, 777, True)
    cases = build_input(
        kind, cells(case_values, case_mask, g_levels, h_levels)
    )
    expected = expected_model.predict_details(
        text_of(case_values, case_mask, None)
    )
    np.testing.assert_array_equal(
        fitted.predict_proba(cases), native_probabilities(expected)
    )
    if settings.get("model_kind") == "rules":
        # A ruleset trained without N/A has no condition that tests for it.
        return
    # N/A is not missing, and the two classify differently somewhere.
    as_missing = cells(case_values, case_mask, g_levels, h_levels)
    as_missing[case_mask] = None
    assert not np.array_equal(
        fitted.predict_proba(build_input(kind, as_missing)),
        fitted.predict_proba(cases),
    )


def test_training_attribute_usage_matches_text() -> None:
    values, mask, classes = table(160, 12345, True)
    model = c50.train(
        names_of(G_LEVELS, H_LEVELS, False), text_of(values, mask, classes)
    )
    fitted = estimator({}).fit(cells(values, mask), labels_of(classes))
    usage = np.asarray(
        model.attribute_usage(text_of(values, mask, classes))[:4]
    )
    np.testing.assert_allclose(fitted.attribute_usage_, usage / len(classes))


def test_batches_and_zero_weights_keep_the_mask() -> None:
    values, mask, classes = table(160, 12345, True)
    weights = np.ones(len(classes))
    weights[::5] = 0
    keep = weights > 0
    X = cells(values, mask)
    y = labels_of(classes)
    reference = estimator({}).fit(X[keep], y[keep])
    weighted = estimator({"prediction_batch_size": 7}).fit(
        X, y, sample_weight=weights
    )
    assert body(weighted.model_.serialized_data) == body(
        reference.model_.serialized_data
    )
    np.testing.assert_array_equal(
        weighted.predict_proba(X), reference.predict_proba(X)
    )
    np.testing.assert_array_equal(
        weighted.attribute_usage_, reference.attribute_usage_
    )


def test_pandas_na_is_missing_not_not_applicable() -> None:
    values, mask, classes = table(120, 4, False)
    base = cells(values, mask)
    base[::6, 0] = np.nan
    fitted = estimator({}).fit(base, labels_of(classes))
    markers = (None, np.nan, pd.NA, pd.NaT, Decimal("NaN"), Decimal("sNaN"))
    reference = None
    for marker in markers:
        probe = cells(values, mask)[:30]
        probe[:, 0] = marker
        probe[:, 3] = marker
        result = fitted.predict_proba(probe)
        if reference is None:
            reference = result
        np.testing.assert_array_equal(result, reference)
    probe = cells(values, mask)[:30]
    probe[:, 0] = NA
    probe[:, 3] = NA
    assert not np.array_equal(fitted.predict_proba(probe), reference)


def test_sentinel_does_not_make_a_numeric_column_categorical() -> None:
    column = np.asarray([[1.5], [NA], [2.5], [None], [np.nan]], dtype=object)
    schema = fit_schema(column, categorical_features=None)
    assert schema.categorical_indices == ()
    assert schema.categories == ((),)
    # Without the sentinel, an object that is not a number makes the column
    # categorical, as before.
    other = np.asarray([[1.5], ["x"], [2.5]], dtype=object)
    assert fit_schema(
        other, categorical_features=None
    ).categorical_indices == (0,)


def test_all_missing_and_sentinel_columns_follow_the_missing_rule() -> None:
    missing = np.asarray([[None], [np.nan], [None]], dtype=object)
    only_na = np.asarray([[NA], [NA], [None]], dtype=object)
    assert fit_schema(missing, None).categorical_indices == ()
    assert fit_schema(only_na, None).categorical_indices == ()
    y = np.asarray(["a", "b", "a"])
    for X in (missing, only_na):
        # The feature is continuous and carries no information.
        fitted = C50Classifier(minimum_cases=1).fit(
            np.column_stack([X, [0.0, 1.0, 0.0]]), y
        )
        assert fitted.categorical_features_.tolist() == []


def test_categories_exclude_the_sentinel() -> None:
    values, mask, classes = table(160, 12345, True)
    fitted = estimator({}).fit(cells(values, mask), labels_of(classes))
    assert fitted.categorical_features_.tolist() == [2, 3]
    assert [c.tolist() for c in fitted.categories_] == [
        [],
        [],
        list(G_LEVELS),
        list(H_LEVELS),
    ]
    for categories in fitted.categories_:
        assert all(value is not NA for value in categories)


@pytest.mark.parametrize("ordered", [False, True])
def test_category_dtype_listing_the_sentinel(ordered: bool) -> None:
    g_levels, h_levels = ("c", "a", "b"), ("t", "p", "s", "q", "r")
    values, mask, classes = table(160, 12345, True)
    frame = category_frame(
        cells(values, mask, g_levels, h_levels), g_levels, h_levels, ordered
    )
    assert NA in frame["g"].cat.categories
    fitted = estimator({}).fit(frame, labels_of(classes))
    # The remaining categories keep their declared order.
    assert fitted.categories_[2].tolist() == list(g_levels)
    assert fitted.categories_[3].tolist() == list(h_levels)
    assert fitted.categorical_features_.tolist() == [2, 3]
    assert fitted.predict(frame).shape == (160,)


def test_category_dtype_of_numbers_and_the_sentinel_is_continuous() -> None:
    values, mask, classes = table(160, 12345, True)
    objects = cells(values, mask)
    frame = pd.DataFrame(objects, columns=list(COLUMNS))
    frame["x"] = pd.Categorical(
        frame["x"],
        categories=[*sorted(set(values[:, 0][~np.isnan(values[:, 0])])), NA],
    )
    fitted = estimator({}).fit(frame, labels_of(classes))
    assert fitted.categorical_features_.tolist() == [2, 3]
    reference = estimator({}).fit(
        pd.DataFrame(objects, columns=list(COLUMNS)), labels_of(classes)
    )
    assert body(fitted.model_.serialized_data) == body(
        reference.model_.serialized_data
    )


def test_continuous_features_accept_the_sentinel() -> None:
    values, mask, classes = table(160, 12345, True)
    # Features 2 and 3 are continuous here, so the codes are numbers.
    numbers_only = np.empty(values.shape, dtype=object)
    numbers_only[:] = values
    numbers_only[mask] = NA
    forced = estimator({"categorical_features": []}).fit(
        numbers_only, labels_of(classes)
    )
    assert forced.categorical_features_.tolist() == []
    names = (
        names_of(G_LEVELS, H_LEVELS, False)
        .replace(
            "feature_2: value_0, value_1, value_2.", "feature_2: continuous."
        )
        .replace(
            "feature_3: value_0, value_1, value_2, value_3, value_4.",
            "feature_3: continuous.",
        )
    )
    expected = c50.train(
        names,
        "".join(
            ",".join(
                "N/A" if na else "?" if np.isnan(v) else repr(float(v))
                for v, na in zip(row, flags, strict=True)
            )
            + f",class_{k}\n"
            for row, flags, k in zip(values, mask, classes, strict=True)
        ),
    )
    assert body(forced.model_.serialized_data) == body(
        expected.serialized_data
    )
    # Another object is still not a number.
    with pytest.raises(TypeError, match="continuous"):
        estimator({"categorical_features": []}).fit(
            np.asarray([[NA], ["x"], [1.0], [2.0]], dtype=object),
            ["a", "b", "a", "b"],
        )


def test_class_labels_cannot_be_not_applicable() -> None:
    X = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    y = np.asarray(["a", NA, "b", "a"], dtype=object)
    with pytest.raises(ValueError, match="NOT_APPLICABLE"):
        C50Classifier(minimum_cases=1).fit(X, y)
    with pytest.raises(ValueError, match="NOT_APPLICABLE"):
        C50Classifier(minimum_cases=1).fit(X, pd.Series(y))


def test_unknown_categories_and_declared_categories_are_unchanged() -> None:
    X = np.asarray(
        [["a", 0.0], ["b", 1.0], [NA, 2.0], ["a", 3.0], ["b", 4.0]] * 4,
        dtype=object,
    )
    y = np.asarray(["u", "v", "u", "u", "v"] * 4)
    fitted = C50Classifier(minimum_cases=1).fit(X, y)
    unseen = np.asarray([["zzz", 1.0]], dtype=object)
    with pytest.raises(ValueError, match="unseen category 'zzz'"):
        fitted.predict(unseen)
    lenient = clone(fitted).set_params(unknown_categories="missing").fit(X, y)
    assert lenient.predict(unseen).shape == (1,)

    declared = pd.DataFrame(
        {
            "g": pd.Categorical(
                ["a", "b", NA, "a", "b"] * 4,
                categories=["a", "b", "never", NA],
            ),
            "v": [0.0, 1.0, 2.0, 3.0, 4.0] * 4,
        }
    )
    again = C50Classifier(minimum_cases=1).fit(declared, y)
    assert again.categories_[0].tolist() == ["a", "b"]
    probe = declared.iloc[:3].copy()
    probe["g"] = pd.Categorical(
        ["never"] * 3, categories=declared["g"].dtype.categories
    )
    assert again.predict(probe).shape == (3,)


def test_exports_show_the_not_applicable_branch() -> None:
    values, mask, classes = table(160, 12345, True)
    fitted = estimator({}).fit(
        pd.DataFrame(cells(values, mask), columns=list(COLUMNS)),
        labels_of(classes),
    )
    text = fitted.export_text()
    assert "'x' is N/A ->" in text
    assert "'h' in {N/A} ->" in text
    # The sentinel is a marker, not a level of any feature.
    assert "NOT_APPLICABLE" not in text
    kinds = {
        condition["kind"]
        for node in json.loads(fitted.export_json())["trees"][0]["nodes"]
        for branch in node.get("branches", ())
        for condition in [branch["condition"]]
    }
    assert "not_applicable" in kinds
    assert "NOT_APPLICABLE" not in fitted.export_json()


def test_clone_pipeline_and_cross_validation_carry_the_sentinel() -> None:
    values, mask, classes = table(240, 12345, True)
    X = cells(values, mask)
    y = labels_of(classes)
    base = estimator({})
    assert clone(base).get_params() == base.get_params()

    pipeline = Pipeline(
        [
            ("identity", FunctionTransformer(validate=False)),
            ("tree", clone(base)),
        ]
    )
    pipeline.fit(X, y)
    assert pipeline.score(X, y) == base.fit(X, y).score(X, y)

    for data in (X, pd.DataFrame(X, columns=list(COLUMNS))):
        scores = cross_val_score(base, data, y, cv=3)
        assert scores.shape == (3,)
        assert np.all(scores > 0.3)

    search = GridSearchCV(base, {"minimum_cases": [2, 5]}, cv=3).fit(X, y)
    assert search.best_estimator_.predict(X).shape == y.shape


def test_pickled_estimator_predicts_with_the_sentinel() -> None:
    values, mask, classes = table(160, 12345, True)
    X = cells(values, mask)
    fitted = estimator({}).fit(X, labels_of(classes))
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        restored = pickle.loads(pickle.dumps(fitted, protocol))
        np.testing.assert_array_equal(
            restored.predict_proba(X), fitted.predict_proba(X)
        )
    # The sentinel in the data survives pickling as the same object.
    assert pickle.loads(pickle.dumps(X))[mask].tolist() == [NA] * int(
        mask.sum()
    )
    assert all(v is NA for v in pickle.loads(pickle.dumps(X))[mask])
    assert all(v is NA for v in copy.deepcopy(X)[mask])


def test_existing_inputs_do_not_build_a_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import c50._data as data

    seen: list[object] = []
    original = data.Schema.dense_data

    def spy(self: Any, *args: Any, **kwargs: Any) -> Any:
        encoded = original(self, *args, **kwargs)
        seen.append(encoded.not_applicable)
        return encoded

    monkeypatch.setattr(data.Schema, "dense_data", spy)
    values, _, classes = table(120, 4, False)
    y = labels_of(classes)
    inputs = [
        values,
        cells(values, np.zeros(values.shape, dtype=bool)),
        pd.DataFrame(values, columns=list(COLUMNS)),
        category_frame(
            cells(values, np.zeros(values.shape, dtype=bool)),
            G_LEVELS,
            H_LEVELS,
            False,
        ),
    ]
    for X in inputs:
        fitted = estimator({"prediction_batch_size": 50}).fit(X, y)
        fitted.predict(X)
        fitted.predict_proba(X)
    assert len(seen) > 8
    assert all(mask is None for mask in seen)

    # The native calls receive no mask either.
    calls: list[object] = []
    train_dense = c50.Model.train_dense

    def train_spy(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs.get("not_applicable"))
        return train_dense(*args, **kwargs)

    monkeypatch.setattr(c50.Model, "train_dense", staticmethod(train_spy))
    estimator({}).fit(values, y)
    assert calls == [None]


def test_string_n_a_is_still_an_ordinary_level() -> None:
    # A marker object other than the sentinel stays an ordinary category.
    X = np.asarray([["N/A"], ["a"], ["N/A"], ["a"]] * 3, dtype=object)
    y = np.asarray(["u", "v", "u", "v"] * 3)
    fitted = C50Classifier(minimum_cases=1).fit(X, y)
    assert fitted.categories_[0].tolist() == ["N/A", "a"]
