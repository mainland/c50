"""Behavioral tests for the scikit-learn-compatible estimator."""

from __future__ import annotations

import json
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
        trials=3, n_jobs=2, categorical_features=(1,)
    )

    copied = clone(classifier)

    assert copied.get_params() == classifier.get_params()
    assert copied.trials == 3
    assert copied.categorical_features == (1,)
    assert copied.n_jobs == 2
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


def test_predict_log_proba_is_the_log_of_predict_proba() -> None:
    classifier = C50Classifier(minimum_cases=1).fit(X_BINARY, Y_BINARY)
    probabilities = classifier.predict_proba(X_BINARY)

    with np.errstate(divide="ignore"):
        expected = np.log(probabilities)
    np.testing.assert_array_equal(classifier.predict_log_proba(X_BINARY), expected)
    with pytest.raises(NotFittedError):
        C50Classifier().predict_log_proba(X_BINARY)


def test_worker_count_preserves_estimator_model() -> None:
    serial = C50Classifier().fit(X_BINARY, Y_BINARY)

    for n_jobs in (1, 2, 64, -1, -2):
        parallel = C50Classifier(n_jobs=n_jobs).fit(X_BINARY, Y_BINARY)
        assert parallel.model_.serialized_data == serial.model_.serialized_data
        assert (
            parallel.predict(X_BINARY).tolist() == serial.predict(X_BINARY).tolist()
        )


@pytest.mark.parametrize(
    ("n_jobs", "expected"),
    [(None, 1), (1, 1), (3, 3), (8, 8), (9, 8), (64, 8)],
)
def test_n_jobs_selects_native_workers(n_jobs: int | None, expected: int) -> None:
    assert C50Classifier(n_jobs=n_jobs)._native_workers() == expected


def test_n_jobs_follows_joblib_configuration() -> None:
    joblib = pytest.importorskip("joblib")

    with joblib.parallel_config(n_jobs=3):
        assert C50Classifier()._native_workers() == 3
        assert C50Classifier(n_jobs=2)._native_workers() == 2
    assert 1 <= C50Classifier(n_jobs=-1)._native_workers() <= 8


def test_stable_ties_fit_and_validation() -> None:
    classifier = C50Classifier(ties="stable")

    assert clone(classifier).ties == "stable"
    classifier.fit(X_BINARY, Y_BINARY)
    assert classifier.predict([[0.25], [3.75]]).tolist() == ["low", "high"]
    with pytest.raises(ValueError, match="ties"):
        C50Classifier(ties="fast").fit(X_BINARY, Y_BINARY)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("parameters", "name"),
    [
        ({"trials": 1.5}, "trials"),
        ({"trials": 0}, "trials"),
        ({"trials": True}, "trials"),
        ({"subset_splits": "yes"}, "subset_splits"),
        ({"global_pruning": None}, "global_pruning"),
        ({"winnow": 1}, "winnow"),
        ({"minimum_cases": "2"}, "minimum_cases"),
        ({"confidence_factor": None}, "confidence_factor"),
        ({"sample_fraction": False}, "sample_fraction"),
    ],
)
def test_invalid_parameter_types_name_the_parameter(
    parameters: dict[str, object], name: str
) -> None:
    with pytest.raises(ValueError, match=name):
        C50Classifier(**parameters).fit(X_BINARY, Y_BINARY)  # type: ignore[arg-type]


def test_numpy_scalar_parameters_match_python_scalars() -> None:
    X = np.linspace(0, 4, 40).reshape(-1, 1)
    y = np.where(X[:, 0] > 2, "high", "low")
    y[::7] = "high"
    python = C50Classifier(
        trials=3, winnow=True, minimum_cases=1.0, sample_fraction=0.5, random_state=2
    ).fit(X, y)
    numpy = C50Classifier(
        trials=np.int64(3),
        winnow=np.bool_(True),
        minimum_cases=np.float32(1.0),
        sample_fraction=np.float64(0.5),
        random_state=np.int64(2),
    ).fit(X, y)

    assert numpy.model_.serialized_data == python.model_.serialized_data


@pytest.mark.parametrize("n_jobs", [0, 1.5, True, "2"])
def test_invalid_n_jobs(n_jobs: object) -> None:
    classifier = C50Classifier(n_jobs=n_jobs)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_jobs"):
        classifier.fit(X_BINARY, Y_BINARY)


def test_random_state_controls_only_sampling() -> None:
    X = np.linspace(0, 4, 40).reshape(-1, 1)
    y = np.where(X[:, 0] > 2, "high", "low")
    y[::7] = "high"

    def model(**parameters: object) -> str:
        return str(
            C50Classifier(minimum_cases=1, **parameters)
            .fit(X, y)
            .model_.serialized_data
        )

    state = np.random.RandomState(5)
    before = state.get_state()[1].copy()
    assert model(random_state=state) == model()
    np.testing.assert_array_equal(state.get_state()[1], before)

    sampled = {"sample_fraction": 0.5}
    assert model(random_state=3, **sampled) == model(random_state=3, **sampled)
    first = np.random.RandomState(8)
    second = np.random.RandomState(8)
    assert model(random_state=first, **sampled) == model(
        random_state=second, **sampled
    )
    assert len({model(random_state=seed, **sampled) for seed in range(8)}) > 1

    with pytest.raises(ValueError):
        C50Classifier(random_state="seed").fit(X, y)  # type: ignore[arg-type]


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


def test_attribute_usage_and_feature_importances() -> None:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(400, 4))
    y = np.where(X[:, 0] + 0.4 * X[:, 1] > 0, "yes", "no")

    for options in ({}, {"model_kind": "rules"}, {"trials": 4}):
        classifier = C50Classifier(**options).fit(X, y)
        counts = classifier.model_.attribute_usage_dense(X)
        np.testing.assert_array_equal(
            classifier.attribute_usage_, np.asarray(counts) / len(y)
        )
        assert classifier.attribute_usage_[0] == 1
        assert classifier.feature_importances_.sum() == pytest.approx(1)
        np.testing.assert_array_equal(
            classifier.feature_importances_.argsort(),
            classifier.attribute_usage_.argsort(),
        )
        batched = C50Classifier(prediction_batch_size=7, **options).fit(X, y)
        np.testing.assert_array_equal(
            batched.attribute_usage_, classifier.attribute_usage_
        )


def test_attribute_usage_excludes_weights_and_zero_weight_samples() -> None:
    rng = np.random.default_rng(4)
    X = rng.normal(size=(200, 3))
    y = np.where(X[:, 0] > 0, 1, 0)
    weights = rng.uniform(0.5, 2, size=len(y))
    weights[:50] = 0

    weighted = C50Classifier().fit(X, y, sample_weight=weights)
    kept = C50Classifier().fit(X[50:], y[50:], sample_weight=weights[50:])

    assert weighted.attribute_usage_.shape == (3,)
    np.testing.assert_array_equal(weighted.attribute_usage_, kept.attribute_usage_)


def test_single_leaf_has_zero_importances() -> None:
    X = np.arange(8.0).reshape(-1, 1)
    y = np.asarray([0, 1] * 4)

    classifier = C50Classifier().fit(X, y)

    assert classifier.model_.inspect().trees[0].leaf_count == 1
    np.testing.assert_array_equal(classifier.attribute_usage_, [0])
    np.testing.assert_array_equal(classifier.feature_importances_, [0])


def test_exports_use_original_names_and_labels() -> None:
    rng = np.random.default_rng(8)
    count = 300
    color = rng.choice(["red", "green", "blue"], size=count)
    size = rng.normal(size=count)
    X = np.column_stack((size.astype(object), color))
    y = np.where((color == "red") | (size > 1), 1, 0)

    for options in ({"subset_splits": True}, {"model_kind": "rules"}):
        classifier = C50Classifier(categorical_features=[1], **options).fit(X, y)
        text = classifier.export_text()
        exported = json.loads(classifier.export_json())

        assert "'feature_1'" in text and "'red'" in text
        assert "value_" not in text and "class_" not in text
        assert exported["feature_names"] == ["feature_0", "feature_1"]
        assert exported["class_names"] == ["0", "1"]
        assert "value_" not in classifier.export_json()

    weighted = C50Classifier().fit(X, y, sample_weight=rng.uniform(0.5, 2, count))
    assert json.loads(weighted.export_json())["feature_names"] == [
        "feature_0",
        "feature_1",
        "case weight",
    ]
    assert "children omitted" in weighted.export_text(max_depth=0)
    with pytest.raises(NotFittedError):
        C50Classifier().export_text()


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
        ({"n_jobs": 0}, Y_BINARY),
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

    classifier.set_params(n_jobs=None, categorical_features=None, cost_matrix=None)
    classifier.fit(two_features, Y_BINARY)
    assert classifier.n_features_in_ == 2
    np.testing.assert_array_equal(classifier.predict(two_features), Y_BINARY)
