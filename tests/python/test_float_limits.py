# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Training on values near the limits of float.

The threshold between two neighbouring values is half of their float sum, which
overflows for two large values of the same sign.
"""

from __future__ import annotations

import c50
import numpy as np
import pytest
from c50.sklearn import C50Classifier

NAMES = "a, b.\n\nx: continuous.\n"
PAIRS = [(-3e38, -2e38), (2e38, 3e38)]
KINDS = [c50.ModelKind.TREE, c50.ModelKind.RULES]


def cases(pair: tuple[float, float]) -> str:
    return "".join(f"{value!r}, ?\n" for value in pair)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("pair", PAIRS)
def test_text_training_separates_a_pair_whose_sum_overflows(
    pair: tuple[float, float], kind: c50.ModelKind
) -> None:
    low, high = pair
    data = f"{low!r}, a\n" * 5 + f"{high!r}, b\n" * 5
    model = c50.train(NAMES, data, kind)
    assert model.predict(cases(pair)) == ["a", "b"]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("pair", PAIRS)
def test_dense_training_separates_a_pair_whose_sum_overflows(
    pair: tuple[float, float], kind: c50.ModelKind
) -> None:
    values = np.array([[pair[0]]] * 5 + [[pair[1]]] * 5)
    classes = np.array([0] * 5 + [1] * 5)
    model = c50.train_dense(NAMES, values, classes, kind)
    assert model.predict(cases(pair)) == ["a", "b"]


@pytest.mark.parametrize("model_kind", ["tree", "rules"])
@pytest.mark.parametrize("pair", PAIRS)
def test_estimator_separates_a_pair_whose_sum_overflows(
    pair: tuple[float, float], model_kind: str
) -> None:
    X = np.array([[pair[0]]] * 5 + [[pair[1]]] * 5)
    y = np.array(["a"] * 5 + ["b"] * 5)
    estimator = C50Classifier(model_kind=model_kind)  # type: ignore[arg-type]
    estimator.fit(X, y)
    predicted = estimator.predict(np.array([[pair[0]], [pair[1]]]))
    assert list(predicted) == ["a", "b"]


SOFT_PAIRS = [*PAIRS, (-3.4e38, -3e36), (3e36, 3.4e38)]


@pytest.mark.parametrize("pair", SOFT_PAIRS)
def test_soft_thresholds_stay_finite_for_large_values(
    pair: tuple[float, float],
) -> None:
    low, high = pair
    data = f"{low!r}, a\n" * 5 + f"{high!r}, b\n" * 5
    options = c50.Options()
    options.probabilistic_thresholds = True
    model = c50.train(NAMES, data, options=options)
    assert "inf" not in model.serialized_data
    assert model.predict(cases(pair)) == ["a", "b"]


@pytest.mark.parametrize("pair", SOFT_PAIRS)
def test_estimator_with_soft_thresholds_handles_large_values(
    pair: tuple[float, float],
) -> None:
    X = np.array([[pair[0]]] * 5 + [[pair[1]]] * 5)
    y = np.array(["a"] * 5 + ["b"] * 5)
    estimator = C50Classifier(probabilistic_thresholds=True).fit(X, y)
    assert list(estimator.predict(X)) == list(y)


# A float with the bits of the unknown marker, and a double that rounds to the
# smallest float, which has the bits of the not-applicable marker. The learner
# stores them as known values, and softening the cut must not read past the
# node. Release 2.07 trains an inconsistent tree on these values without soft
# thresholds, so the tests check that training completes and a model predicts,
# not which class.
MARKER_VALUES = [1.5777218104420236e-30, 1e-45]


@pytest.mark.parametrize("value", MARKER_VALUES)
@pytest.mark.parametrize("kind", KINDS)
def test_dense_soft_thresholds_train_on_marker_values(
    value: float, kind: c50.ModelKind
) -> None:
    names = "no, yes.\n\nx: continuous.\n"
    values = np.array([[5.0], [5.0], [value], [value]])
    classes = np.array([0, 0, 1, 1])
    options = c50.Options()
    options.probabilistic_thresholds = True
    model = c50.train_dense(names, values, classes, kind, options=options)
    assert model.predict("5, ?\n")[0] in {"no", "yes"}


@pytest.mark.parametrize("value", MARKER_VALUES)
def test_estimator_with_soft_thresholds_handles_marker_values(
    value: float,
) -> None:
    X = np.array([[5.0], [5.0], [value], [value]])
    y = np.array(["no", "no", "yes", "yes"])
    estimator = C50Classifier(probabilistic_thresholds=True).fit(X, y)
    assert estimator.predict(np.array([[5.0]]))[0] in {"no", "yes"}
