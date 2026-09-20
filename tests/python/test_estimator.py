"""Behavioral tests for the scikit-learn-compatible estimator."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

import c50
from c50.sklearn import C50Classifier


X_BINARY = np.asarray([[0.0], [0.5], [1.0], [1.5], [2.5], [3.0], [3.5], [4.0]])
Y_BINARY = np.asarray(["low", "low", "low", "low", "high", "high", "high", "high"])


def test_constructor_is_cloneable_and_does_not_fit() -> None:
    classifier = C50Classifier(trials=3, categorical_features=(1,))

    copied = clone(classifier)

    assert copied.get_params() == classifier.get_params()
    assert copied.trials == 3
    assert copied.categorical_features == (1,)
    assert not hasattr(classifier, "model_")


def test_numeric_fit_predict_and_predict_proba() -> None:
    classifier = C50Classifier().fit(X_BINARY, Y_BINARY)

    assert classifier.n_features_in_ == 1
    assert classifier.categorical_features_.tolist() == []
    assert classifier.categories_[0].tolist() == []
    assert classifier.classes_.tolist() == ["high", "low"]
    assert classifier.predict([[0.25], [3.75]]).tolist() == ["low", "high"]

    probabilities = classifier.predict_proba([[0.25], [3.75]])
    assert probabilities.shape == (2, 2)
    assert probabilities.sum(axis=1) == pytest.approx([1.0, 1.0])
    assert classifier.classes_[probabilities.argmax(axis=1)].tolist() == [
        "low",
        "high",
    ]


def test_multiclass_integer_labels_preserve_dtype() -> None:
    X = np.asarray([[0], [0.1], [1], [1.1], [2], [2.1]], dtype=float)
    y = np.asarray([10, 10, 20, 20, 30, 30])

    classifier = C50Classifier(minimum_cases=1).fit(X, y)
    predictions = classifier.predict([[0], [1], [2]])

    assert classifier.classes_.tolist() == [10, 20, 30]
    assert predictions.dtype == y.dtype
    assert predictions.tolist() == [10, 20, 30]


def test_categorical_features_and_missing_values() -> None:
    X = np.asarray(
        [
            [0.0, "red"],
            [0.5, "blue"],
            [1.0, None],
            [1.5, "blue"],
            [2.5, "red"],
            [3.0, "blue"],
            [3.5, None],
            [4.0, "blue"],
        ],
        dtype=object,
    )
    classifier = C50Classifier(categorical_features=(1,)).fit(X, Y_BINARY)

    assert classifier.categorical_features_.tolist() == [1]
    assert classifier.categories_[1].tolist() == ["red", "blue"]
    assert classifier.predict([[0.25, "red"], [3.75, None]]).tolist() == [
        "low",
        "high",
    ]


def test_unseen_category_policy() -> None:
    X = np.asarray(
        [[0, "red"], [1, "blue"], [2, "red"], [3, "blue"]],
        dtype=object,
    )
    y = np.asarray(["low", "low", "high", "high"])

    strict = C50Classifier(categorical_features=(1,), minimum_cases=1).fit(X, y)
    with pytest.raises(ValueError, match="unseen category 'green'"):
        strict.predict([[1.5, "green"]])

    permissive = C50Classifier(
        categorical_features=(1,),
        unknown_categories="missing",
        minimum_cases=1,
    ).fit(X, y)
    assert permissive.predict([[1.5, "green"]]).shape == (1,)


def test_rules_options_costs_and_refit() -> None:
    classifier = C50Classifier(
        model_kind="rules",
        trials=2,
        minimum_cases=1,
        cost_matrix=[[0, 3], [2, 0]],
    ).fit(X_BINARY, Y_BINARY)

    assert classifier.model_.kind == c50.ModelKind.RULES
    assert classifier.cost_matrix_.tolist() == [[0, 3], [2, 0]]
    assert classifier.model_.costs_data

    classifier.fit(X_BINARY, np.where(Y_BINARY == "low", "left", "right"))
    assert classifier.classes_.tolist() == ["left", "right"]


def test_pickle_round_trip() -> None:
    classifier = C50Classifier().fit(X_BINARY, Y_BINARY)

    restored = pickle.loads(pickle.dumps(classifier))

    assert restored.predict([[0.25], [3.75]]).tolist() == ["low", "high"]
    assert np.array_equal(restored.classes_, classifier.classes_)


def test_validation_errors_are_reported_at_the_boundary() -> None:
    classifier = C50Classifier()

    with pytest.raises(NotFittedError):
        classifier.predict([[1]])
    with pytest.raises(ValueError, match="at least two classes"):
        classifier.fit([[0], [1]], ["only", "only"])
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        classifier.fit([[0], [1]], ["low"])
    with pytest.raises(ValueError, match="model_kind"):
        C50Classifier(model_kind="forest").fit(X_BINARY, Y_BINARY)
