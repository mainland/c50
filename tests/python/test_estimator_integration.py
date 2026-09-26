"""Integration checks for the scikit-learn estimator protocol."""

from __future__ import annotations

import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from scipy import sparse
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
        "random_seed": 13,
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
