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
    classifier = C50Classifier(
        trials=3, split_workers=2, categorical_features=(1,)
    )

    copied = clone(classifier)

    assert copied.get_params() == classifier.get_params()
    assert copied.trials == 3
    assert copied.categorical_features == (1,)
    assert copied.split_workers == 2
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


def test_worker_count_preserves_estimator_model() -> None:
    serial = C50Classifier(split_workers=1).fit(X_BINARY, Y_BINARY)
    parallel = C50Classifier(split_workers=2).fit(X_BINARY, Y_BINARY)

    assert parallel.model_.serialized_data == serial.model_.serialized_data
    assert parallel.predict(X_BINARY).tolist() == serial.predict(X_BINARY).tolist()


def test_stable_ties_fit_and_validation() -> None:
    classifier = C50Classifier(ties="stable")

    assert clone(classifier).ties == "stable"
    classifier.fit(X_BINARY, Y_BINARY)
    assert classifier.predict([[0.25], [3.75]]).tolist() == ["low", "high"]
    with pytest.raises(ValueError, match="ties"):
        C50Classifier(ties="fast").fit(X_BINARY, Y_BINARY)  # type: ignore[arg-type]


@pytest.mark.parametrize("workers", [0, 9, 1.5, True])
def test_invalid_worker_count(workers: object) -> None:
    classifier = C50Classifier(split_workers=workers)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="split_workers"):
        classifier.fit(X_BINARY, Y_BINARY)


def test_prediction_batches_preserve_results() -> None:
    unbatched = C50Classifier(prediction_batch_size=None).fit(X_BINARY, Y_BINARY)
    batched = C50Classifier(prediction_batch_size=1).fit(X_BINARY, Y_BINARY)
    cases = [[0.25], [1.25], [2.75], [3.75]]

    assert batched.predict(cases).tolist() == unbatched.predict(cases).tolist()
    assert batched.predict_proba(cases) == pytest.approx(
        unbatched.predict_proba(cases)
    )


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("prediction_batch_size", -1),
        ("prediction_batch_size", 0),
        ("prediction_batch_size", 1.5),
        ("unknown_categories", "ignore"),
    ],
)
def test_prediction_parameters_are_validated_after_fit(
    parameter: str, value: object
) -> None:
    classifier = C50Classifier().fit(X_BINARY, Y_BINARY)
    classifier.set_params(**{parameter: value})

    with pytest.raises(ValueError, match=parameter):
        classifier.predict(X_BINARY)
    with pytest.raises(ValueError, match=parameter):
        classifier.predict_proba(X_BINARY)


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


def test_labels_and_categories_cannot_change_the_c5_input_grammar() -> None:
    X = np.asarray(
        [["red, blue"], ["period. | slash\\"], ["red, blue"], ["other: value"]],
        dtype=object,
    )
    y = np.asarray(["class, one", "class. two", "class, one", "class. two"])

    classifier = C50Classifier(
        categorical_features=(0,),
        minimum_cases=1,
    ).fit(X, y)

    assert classifier.predict(X).tolist() == y.tolist()
    assert "red, blue" not in classifier.model_.names_data
    assert "class, one" not in classifier.model_.names_data


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
    with pytest.raises(ValueError, match="prediction_batch_size"):
        C50Classifier(prediction_batch_size=0).fit(X_BINARY, Y_BINARY)


@pytest.mark.parametrize(
    "costs", [[[0, 0], [0, 0]], [[0, 0], [1, 0]], [[0, 1e100], [1, 0]]]
)
def test_unusable_native_costs_raise(costs: list[list[float]]) -> None:
    X = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    with pytest.raises(ValueError):
        C50Classifier(cost_matrix=costs).fit(X, [0, 0, 1, 1])


def test_ignore_costs_discards_training_and_prediction_costs() -> None:
    X = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    ordinary = C50Classifier().fit(X, [0, 0, 1, 1])
    ignored = C50Classifier(ignore_costs=True, cost_matrix=[[0, 0], [0, 0]]).fit(
        X, [0, 0, 1, 1]
    )
    assert ignored.model_.serialized_data == ordinary.model_.serialized_data
    assert ignored.model_.costs_data == ""
    np.testing.assert_array_equal(ignored.predict_proba(X), ordinary.predict_proba(X))


@pytest.mark.parametrize(
    ("parameters", "labels"),
    [
        ({"split_workers": 0}, Y_BINARY),
        ({}, Y_BINARY[:-1]),
        ({}, np.repeat("only", len(Y_BINARY))),
        ({"categorical_features": (2,)}, Y_BINARY),
        ({"cost_matrix": [[0, 0], [0, 0]]}, Y_BINARY),
    ],
)
def test_failed_refit_clears_model_and_metadata(
    parameters: dict[str, object], labels: np.ndarray
) -> None:
    classifier = C50Classifier().fit(X_BINARY, Y_BINARY)
    classifier.set_params(**parameters)
    two_features = np.column_stack((X_BINARY, X_BINARY))

    with pytest.raises(ValueError):
        classifier.fit(two_features, labels)

    assert not classifier.__sklearn_is_fitted__()
    assert not any(name.endswith("_") for name in vars(classifier))
    assert not hasattr(classifier, "_schema")
    assert not hasattr(classifier, "_label_encoder")
    with pytest.raises(NotFittedError):
        classifier.predict(X_BINARY)
    with pytest.raises(NotFittedError):
        classifier.predict_proba(two_features)

    classifier.set_params(split_workers=1, categorical_features=None, cost_matrix=None)
    classifier.fit(two_features, Y_BINARY)
    assert classifier.n_features_in_ == 2
    np.testing.assert_array_equal(classifier.predict(two_features), Y_BINARY)
