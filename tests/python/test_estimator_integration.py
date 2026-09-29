"""Integration checks for the scikit-learn estimator protocol."""

from __future__ import annotations

import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.base import clone
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import get_tags
from sklearn.utils.estimator_checks import check_estimator

from c50.sklearn import C50Classifier


X = np.asarray(
    [
        [0.0, 0.0],
        [0.2, 0.1],
        [0.4, 0.3],
        [0.6, 0.4],
        [2.0, 2.1],
        [2.2, 2.0],
        [2.4, 2.3],
        [2.6, 2.5],
    ]
)
Y = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])


def test_common_estimator_checks() -> None:
    check_estimator(C50Classifier(minimum_cases=1))


def test_default_estimator_checks_record_weight_scaling() -> None:
    check_estimator(
        C50Classifier(),
        expected_failed_checks={
            "check_sample_weight_equivalence_on_dense_data": (
                "C5.0 rescales case weights to their mean, so a weight is not "
                "equivalent to repeating a sample."
            )
        },
    )


def test_pipeline_and_grid_search() -> None:
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("classify", C50Classifier(minimum_cases=1)),
        ]
    )
    search = GridSearchCV(
        pipeline,
        {
            "classify__model_kind": ["tree", "rules"],
            "classify__trials": [1, 2],
        },
        cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=7),
    )

    search.fit(X, Y)

    assert search.best_estimator_.predict(X).shape == Y.shape
    assert 0 <= search.best_score_ <= 1


def test_feature_selection_uses_feature_importances() -> None:
    rng = np.random.default_rng(6)
    features = rng.normal(size=(300, 5))
    labels = np.where(features[:, 2] - 0.5 * features[:, 4] > 0, 1, 0)

    selector = SelectFromModel(C50Classifier(), threshold="mean").fit(
        features, labels
    )

    assert selector.get_support().tolist() == [False, False, True, False, True]


def test_grid_search_accepts_multiple_evaluation_metrics() -> None:
    values = np.arange(24, dtype=float).reshape(-1, 1)
    labels = np.asarray([0] * 17 + [1, 0, 1, 0, 1, 1, 1])
    folds = list(
        StratifiedKFold(3, shuffle=True, random_state=11).split(values, labels)
    )
    search = GridSearchCV(
        C50Classifier(),
        {"minimum_cases": [1, 4]},
        scoring={"balanced_accuracy": "balanced_accuracy", "f1_macro": "f1_macro"},
        refit="balanced_accuracy",
        cv=folds,
    ).fit(values, labels)

    for index, parameters in enumerate(search.cv_results_["params"]):
        balanced_scores = []
        f1_scores = []
        for training, validation in folds:
            classifier = C50Classifier(**parameters).fit(
                values[training], labels[training]
            )
            predictions = classifier.predict(values[validation])
            balanced_scores.append(
                balanced_accuracy_score(labels[validation], predictions)
            )
            f1_scores.append(f1_score(labels[validation], predictions, average="macro"))
        assert search.cv_results_["mean_test_balanced_accuracy"][index] == pytest.approx(
            np.mean(balanced_scores)
        )
        assert search.cv_results_["mean_test_f1_macro"][index] == pytest.approx(
            np.mean(f1_scores)
        )
    assert search.best_score_ == max(search.cv_results_["mean_test_balanced_accuracy"])
    assert (
        search.best_estimator_.get_params()["minimum_cases"]
        == search.best_params_["minimum_cases"]
    )


def test_clone_and_pickle_preserve_predictions() -> None:
    classifier = C50Classifier(minimum_cases=1, trials=2).fit(X, Y)
    copied = clone(classifier)
    restored = pickle.loads(pickle.dumps(classifier))

    assert not hasattr(copied, "model_")
    assert np.array_equal(restored.predict(X), classifier.predict(X))
    assert np.array_equal(restored.predict_proba(X), classifier.predict_proba(X))


def test_independent_estimators_can_fit_concurrently() -> None:
    def fit_and_predict(labels: np.ndarray) -> list[int]:
        classifier = C50Classifier(minimum_cases=1).fit(X, labels)
        return classifier.predict(X).tolist()

    reversed_labels = 1 - Y
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(fit_and_predict, Y),
            executor.submit(fit_and_predict, reversed_labels),
        ]

    assert futures[0].result() == Y.tolist()
    assert futures[1].result() == reversed_labels.tolist()


def test_one_fitted_estimator_can_predict_concurrently() -> None:
    classifier = C50Classifier(minimum_cases=1).fit(X, Y)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(classifier.predict, X) for _ in range(8)]

    for future in futures:
        assert future.result().tolist() == Y.tolist()


def test_sampling_is_reproducible_for_a_fixed_seed() -> None:
    parameters = {
        "minimum_cases": 1,
        "sample_fraction": 0.75,
        "random_state": 13,
    }

    first = C50Classifier(**parameters).fit(X, Y)
    second = C50Classifier(**parameters).fit(X, Y)

    assert first.model_.serialized_data == second.model_.serialized_data
    assert np.array_equal(first.predict(X), second.predict(X))


def test_sparse_and_wrong_dimension_inputs_are_rejected() -> None:
    classifier = C50Classifier(minimum_cases=1)

    with pytest.raises(TypeError, match="Sparse data was passed"):
        classifier.fit(sparse.csr_matrix(X), Y)
    with pytest.raises(ValueError, match="2D array"):
        classifier.fit(X[:, 0], Y)


def test_pandas_categories_names_and_missing_values() -> None:
    frame = pd.DataFrame(
        {
            "signal": X[:, 0],
            "group": pd.Categorical(["a", "a", None, "a", "b", "b", "b", "b"]),
        }
    )
    classifier = C50Classifier(
        minimum_cases=1, categorical_features=["group"]
    ).fit(frame, pd.Series(Y))

    assert classifier.feature_names_in_.tolist() == ["signal", "group"]
    assert classifier.categorical_features_.tolist() == [1]
    assert classifier.categories_[1].tolist() == ["a", "b"]
    np.testing.assert_array_equal(classifier.predict(frame), Y)
    restored = pickle.loads(pickle.dumps(classifier))
    np.testing.assert_array_equal(
        restored.predict_proba(frame), classifier.predict_proba(frame)
    )
    with pytest.raises(ValueError, match="order"):
        classifier.predict(frame[["group", "signal"]])


def test_nullable_dataframe_values_are_missing() -> None:
    groups = ["a", "b"] * (len(Y) // 2) + ["a"] * (len(Y) % 2)
    frame = pd.DataFrame(
        {
            "signal": pd.array(X[:, 0], dtype="Float64"),
            "group": pd.array(groups, dtype="string"),
        }
    )
    frame.loc[1, "signal"] = pd.NA
    frame.loc[2, "group"] = pd.NA
    classifier = C50Classifier(minimum_cases=1).fit(frame, Y)

    assert classifier.feature_names_in_.tolist() == ["signal", "group"]
    assert classifier.categories_[1].tolist() == ["a", "b"]
    np.testing.assert_array_equal(
        classifier.predict(frame.drop(index=[1, 2])), np.delete(Y, [1, 2])
    )
    assert np.isfinite(classifier.predict_proba(frame)).all()


def test_dataframe_columns_match_object_array_conversion() -> None:
    rng = np.random.default_rng(11)
    count = 200
    frame = pd.DataFrame(
        {
            "real": rng.normal(size=count),
            "narrow": rng.normal(size=count).astype(np.float32),
            "count": rng.integers(-3, 4, size=count),
            "flag": rng.integers(0, 2, size=count).astype(bool),
            "word": rng.choice(["x", "y", "z"], size=count),
            "group": pd.Categorical(rng.choice(["p", "q", None], size=count)),
            "gap": np.where(
                rng.random(count) < 0.2, np.nan, rng.normal(size=count)
            ),
        }
    )
    labels = np.where(
        frame["real"] + 0.3 * frame["count"] + (frame["word"] == "x") > 0.5,
        "yes",
        "no",
    )
    values = frame.to_numpy(dtype=object)

    from_frame = C50Classifier(minimum_cases=1).fit(frame, labels)
    from_values = C50Classifier(minimum_cases=1).fit(values, labels)

    assert from_frame.model_.serialized_data == from_values.model_.serialized_data
    assert from_frame.categorical_features_.tolist() == [3, 4, 5]
    for frame_categories, value_categories in zip(
        from_frame.categories_, from_values.categories_, strict=True
    ):
        assert [(type(value), value) for value in frame_categories] == [
            (type(value), value) for value in value_categories
        ]
    expected = from_values.predict_proba(values)
    np.testing.assert_array_equal(from_frame.predict_proba(frame), expected)
    from_frame.set_params(prediction_batch_size=7)
    np.testing.assert_array_equal(from_frame.predict_proba(frame), expected)


def coded_frame() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(12)
    count = 240
    words = rng.choice(["red", "green", "blue"], size=count)
    frame = pd.DataFrame(
        {
            "signal": rng.normal(size=count),
            "group": pd.Categorical(np.where(rng.random(count) < 0.1, None, words)),
            "word": pd.array(
                np.where(rng.random(count) < 0.1, None, words[::-1]), dtype="string"
            ),
            "count": pd.array(
                np.where(rng.random(count) < 0.1, None, rng.integers(0, 9, size=count)),
                dtype="Int64",
            ),
            "level": pd.array(rng.normal(size=count), dtype="Float64"),
        }
    )
    labels = np.where(
        frame["signal"] + (words == "red") - (words[::-1] == "blue") > 0.2,
        "yes",
        "no",
    )
    return frame, labels


def test_coded_and_nullable_columns_match_object_conversion() -> None:
    frame, labels = coded_frame()
    values = frame.to_numpy(dtype=object)
    weights = np.where(np.arange(len(labels)) % 5 == 0, 0.0, 1.0)

    for options in ({}, {"categorical_features": ["group", "word", "count"]}):
        from_frame = C50Classifier(minimum_cases=1, **options).fit(
            frame, labels, sample_weight=weights
        )
        indices = [frame.columns.get_loc(name) for name in options.get(
            "categorical_features", []
        )]
        from_values = C50Classifier(
            minimum_cases=1,
            categorical_features=indices if options else None,
        ).fit(values, labels, sample_weight=weights)

        assert from_frame.model_.serialized_data == from_values.model_.serialized_data
        for frame_categories, value_categories in zip(
            from_frame.categories_, from_values.categories_, strict=True
        ):
            assert [(type(value), value) for value in frame_categories] == [
                (type(value), value) for value in value_categories
            ]
        np.testing.assert_array_equal(
            from_frame.predict_proba(frame), from_values.predict_proba(values)
        )


def test_categorical_prediction_uses_fitted_categories() -> None:
    frame, labels = coded_frame()
    classifier = C50Classifier(minimum_cases=1).fit(frame, labels)
    expected = classifier.predict_proba(frame)

    recoded = frame.copy()
    recoded["group"] = recoded["group"].cat.set_categories(
        ["violet", "blue", "red", "green"]
    )
    recoded["word"] = recoded["word"].astype(object)
    np.testing.assert_array_equal(classifier.predict_proba(recoded), expected)

    unseen = frame.copy()
    unseen["group"] = unseen["group"].cat.add_categories(["violet"])
    np.testing.assert_array_equal(classifier.predict_proba(unseen), expected)
    unseen.loc[[4, 9], "group"] = "violet"
    with pytest.raises(ValueError, match="unseen category 'violet' in feature 1"):
        classifier.predict(unseen)
    classifier.set_params(unknown_categories="missing")
    missing = frame.copy()
    missing.loc[[4, 9], "group"] = None
    np.testing.assert_array_equal(
        classifier.predict_proba(unseen), classifier.predict_proba(missing)
    )


def test_dataframe_input_errors_match_array_input() -> None:
    frame = pd.DataFrame({"signal": X[:, 0], "noise": X[:, 1]})
    classifier = C50Classifier(minimum_cases=1)

    with pytest.raises(ValueError, match="1d array"):
        classifier.fit(frame, np.column_stack([Y, Y]))
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        classifier.fit(frame, Y[:-1])
    with pytest.raises(ValueError, match="contains NaN"):
        classifier.fit(frame, np.where(Y == 0, np.nan, 1.0))
    with pytest.raises(ValueError, match="0 sample"):
        classifier.fit(frame.iloc[:0], Y[:0])

    classifier.fit(frame, Y)
    with pytest.raises(ValueError, match="0 sample"):
        classifier.predict(frame.iloc[:0])
    with pytest.raises(ValueError, match="seen at fit time, yet now missing"):
        classifier.predict(frame[["signal"]])


def test_estimator_declares_numpy_only_array_support() -> None:
    assert not get_tags(C50Classifier()).array_api_support
    classifier = C50Classifier(minimum_cases=1).fit(X, Y)
    assert isinstance(classifier.predict(X), np.ndarray)
    assert isinstance(classifier.predict_proba(X), np.ndarray)
