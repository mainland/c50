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
