"""Owned reusable prediction state and Python concurrency contracts."""

from __future__ import annotations

import gc
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from c50 import Model, ModelKind, Predictions
from c50.sklearn import C50Classifier


def assert_equal(actual: Predictions, expected: Predictions) -> None:
    assert actual.class_names == expected.class_names
    assert actual.class_indices == expected.class_indices
    assert actual.confidences == expected.confidences
    assert actual.scores == expected.scores


@pytest.mark.parametrize("kind", [ModelKind.TREE, ModelKind.RULES])
def test_predictor_lifetime_recovery_and_concurrency(kind: ModelKind) -> None:
    model = Model.train(
        "low, high.\nx: continuous.\n",
        "0, low\n1, low\n3, high\n4, high\n", kind,
    )
    values = np.asarray([[0.0], [4.0], [np.nan]])
    expected = model.predict_details_dense(values)
    predictor = model.prepare_predictor()
    del model
    gc.collect()
    for _ in range(3):
        assert_equal(predictor.predict_details_dense(values), expected)
        assert_equal(predictor.predict_details("0, ?\n4, ?\n?, ?"), expected)
        with pytest.raises(ValueError):
            predictor.predict_details("invalid, ?\n4, ?\n")
        with pytest.raises(ValueError, match="two-dimensional"):
            predictor.predict_details_dense(np.asarray([0.0]))
        assert_equal(predictor.predict_details_dense(values), expected)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: predictor.predict_details_dense(values), range(16)))
    for result in results:
        assert_equal(result, expected)
    del predictor
    gc.collect()
    assert_equal(results[0], expected)


@pytest.mark.parametrize("kind", ["tree", "rules"])
def test_estimator_batches_preserve_exact_missing_scores(kind: str) -> None:
    values = np.asarray([[0.0], [1.0], [3.0], [4.0], [np.nan]])
    labels = np.asarray([0, 0, 1, 1, 1])
    classifier = C50Classifier(model_kind=kind, minimum_cases=1).fit(values, labels)
    expected = classifier.predict_proba(values)
    classifier.set_params(prediction_batch_size=1)
    np.testing.assert_array_equal(classifier.predict_proba(values), expected)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: classifier.predict_proba(values), range(8)))
    for result in results:
        np.testing.assert_array_equal(result, expected)
