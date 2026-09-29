"""Tests for estimator sample weights and C5.0 case weights."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

import c50
from c50.sklearn import C50Classifier


def weighted_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return overlapping classes whose weighted classifier differs."""
    rng = np.random.default_rng(29)
    count = 120
    X = rng.normal(size=(count, 3))
    y = np.where(X[:, 0] + 0.5 * X[:, 1] + rng.normal(size=count) > 0, "yes", "no")
    weights = np.where(X[:, 2] > 0, 4.0, 0.5) * rng.uniform(0.5, 1.5, size=count)
    return X, y, weights


def test_unit_and_scaled_weights_preserve_the_classifier() -> None:
    X, y, weights = weighted_problem()

    unweighted = C50Classifier().fit(X, y)
    unit = C50Classifier().fit(X, y, sample_weight=np.ones(len(y)))
    scalar = C50Classifier().fit(X, y, sample_weight=3.0)

    for fitted in (unit, scalar):
        assert fitted.model_.serialized_data == unweighted.model_.serialized_data
        np.testing.assert_array_equal(
            fitted.predict_proba(X), unweighted.predict_proba(X)
        )

    weighted = C50Classifier().fit(X, y, sample_weight=weights)
    for scale in (1e-30, 7.5, 1e30):
        scaled = C50Classifier().fit(X, y, sample_weight=weights * scale)
        assert scaled.model_.serialized_data == weighted.model_.serialized_data


def test_weights_change_the_classifier() -> None:
    X, y, weights = weighted_problem()

    unweighted = C50Classifier().fit(X, y)
    weighted = C50Classifier().fit(X, y, sample_weight=weights)

    assert weighted.model_.serialized_data != unweighted.model_.serialized_data
    assert "case weight: continuous." in weighted.model_.names_data
    assert "case weight" not in unweighted.model_.names_data


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"model_kind": "rules"},
        {"trials": 3},
        {"winnow": True},
        {"subset_splits": True, "global_pruning": False},
    ],
)
def test_weights_match_the_native_case_weight_attribute(
    options: dict[str, object],
) -> None:
    X, y, weights = weighted_problem()
    groups = np.asarray(["red", "green", "blue"], dtype=object)[
        (np.abs(X[:, 2]) * 3).astype(int) % 3
    ]
    X = np.column_stack((X.astype(object), groups))
    classifier = C50Classifier(**options).fit(X, y, sample_weight=weights)

    relative = weights / weights.mean()
    classes = {label: index for index, label in enumerate(classifier.classes_)}
    categories = {
        value: index for index, value in enumerate(classifier.categories_[3])
    }
    rows = [
        f"{float(row[0])!r}, {float(row[1])!r}, {float(row[2])!r}, "
        f"value_{categories[row[3]]}, {float(weight)!r}, class_{classes[label]}"
        for row, weight, label in zip(X, relative, y, strict=True)
    ]
    native = c50.Model.train(
        classifier.model_.names_data,
        "\n".join(rows) + "\n",
        classifier.model_.kind,
        classifier._native_options(),
    )

    assert classifier.model_.serialized_data == native.serialized_data


def test_zero_weights_remove_samples() -> None:
    X, y, weights = weighted_problem()
    X = np.column_stack((X.astype(object), np.where(X[:, 2] > 0, "a", "b")))
    extra = np.asarray([[9.0, 9.0, 9.0, "c"], [-9.0, -9.0, -9.0, "a"]], dtype=object)

    kept = C50Classifier().fit(X, y, sample_weight=weights)
    padded = C50Classifier().fit(
        np.vstack((X, extra)),
        np.concatenate((y, ["maybe", "yes"])),
        sample_weight=np.concatenate((weights, [0.0, 0.0])),
    )

    assert padded.model_.serialized_data == kept.model_.serialized_data
    assert padded.classes_.tolist() == ["no", "yes"]
    assert padded.categories_[3].tolist() == kept.categories_[3].tolist()
    np.testing.assert_array_equal(padded.predict_proba(X), kept.predict_proba(X))


def test_dataframe_weights_and_weighted_model_round_trip() -> None:
    X, y, weights = weighted_problem()
    frame = pd.DataFrame(
        {
            "signal": X[:, 0],
            "noise": X[:, 1],
            "side": np.where(X[:, 2] > 0, "right", "left"),
        }
    )
    weights[::5] = 0

    classifier = C50Classifier(prediction_batch_size=7).fit(
        frame, y, sample_weight=weights
    )
    expected = C50Classifier().fit(
        frame.to_numpy(dtype=object), y, sample_weight=weights
    )

    assert classifier.model_.serialized_data == expected.model_.serialized_data
    probabilities = classifier.predict_proba(frame)
    np.testing.assert_array_equal(
        probabilities, expected.predict_proba(frame.to_numpy(dtype=object))
    )
    restored = pickle.loads(pickle.dumps(classifier))
    np.testing.assert_array_equal(restored.predict_proba(frame), probabilities)


@pytest.mark.parametrize(
    ("sample_weight", "message"),
    [
        (np.ones(7), "shape"),
        (np.ones((8, 1)), "shape"),
        (np.asarray([1, 1, 1, 1, 1, 1, 1, -1.0]), "nonnegative"),
        (np.asarray([1, 1, 1, 1, 1, 1, 1, np.nan]), "finite"),
        (np.asarray([1, 1, 1, 1, 1, 1, 1, np.inf]), "finite"),
        (np.zeros(8), "zero for every sample"),
        (np.asarray([1, 1, 1, 1, 1, 1, 1, 1e-60]), "single-precision"),
    ],
)
def test_invalid_weights_are_rejected(
    sample_weight: np.ndarray, message: str
) -> None:
    X = np.arange(8.0).reshape(-1, 1)
    y = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])

    with pytest.raises(ValueError, match=message):
        C50Classifier().fit(X, y, sample_weight=sample_weight)


def test_class_weights_are_sample_weights_by_class() -> None:
    X, y, weights = weighted_problem()
    by_class = np.where(y == "yes", 3.0, 0.5)

    expected = C50Classifier().fit(X, y, sample_weight=by_class)
    mapped = C50Classifier(class_weight={"yes": 3.0, "no": 0.5}).fit(X, y)
    scaled = C50Classifier(class_weight={"yes": 12.0, "no": 2.0}).fit(X, y)
    partial = C50Classifier(class_weight={"yes": 6.0}).fit(X, y)
    combined = C50Classifier(class_weight={"yes": 3.0, "no": 0.5}).fit(
        X, y, sample_weight=weights
    )

    assert mapped.model_.serialized_data == expected.model_.serialized_data
    assert scaled.model_.serialized_data == expected.model_.serialized_data
    assert partial.model_.serialized_data == expected.model_.serialized_data
    assert (
        combined.model_.serialized_data
        == C50Classifier().fit(X, y, sample_weight=weights * by_class)
        .model_.serialized_data
    )
    np.testing.assert_array_equal(mapped.predict_proba(X), expected.predict_proba(X))


def test_balanced_class_weights_favor_the_minority_class() -> None:
    rng = np.random.default_rng(31)
    X = rng.normal(size=(300, 2))
    y = np.where(X[:, 0] + rng.normal(scale=0.8, size=300) > 1.2, "rare", "common")
    counts = {label: np.count_nonzero(y == label) for label in ("common", "rare")}
    balanced_weights = np.asarray([len(y) / (2 * counts[label]) for label in y])

    plain = C50Classifier().fit(X, y)
    balanced = C50Classifier(class_weight="balanced").fit(X, y)
    expected = C50Classifier().fit(X, y, sample_weight=balanced_weights)

    assert balanced.model_.serialized_data == expected.model_.serialized_data
    assert np.count_nonzero(balanced.predict(X) == "rare") > np.count_nonzero(
        plain.predict(X) == "rare"
    )


@pytest.mark.parametrize(
    ("class_weight", "error", "message"),
    [
        ({"yes": -1.0}, ValueError, "nonnegative"),
        ({"yes": np.inf}, ValueError, "finite"),
        ({"maybe": 2.0}, ValueError, "not in class_weight"),
        ("uniform", ValueError, "class_weight"),
        ({"yes": 0.0}, ValueError, "two classes"),
    ],
)
def test_invalid_class_weights_are_rejected(
    class_weight: object, error: type[Exception], message: str
) -> None:
    X, y, _ = weighted_problem()

    with pytest.raises(error, match=message):
        C50Classifier(class_weight=class_weight).fit(X, y)  # type: ignore[arg-type]
