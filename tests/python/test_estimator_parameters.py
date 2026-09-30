"""Range and conversion errors of the estimator's numeric parameters."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from c50.sklearn import C50Classifier

X = np.asarray([[0.0], [0.5], [1.0], [1.5], [2.5], [3.0], [3.5], [4.0]])
Y = np.asarray(["low", "low", "low", "low", "high", "high", "high", "high"])


@pytest.mark.parametrize(
    "trials",
    [0, -1, 1001, 2**32, 10**400],
    ids=["0", "-1", "1001", "2pow32", "1e400"],
)
def test_trials_out_of_range_is_a_value_error(trials: int) -> None:
    with pytest.raises(ValueError, match="trials"):
        C50Classifier(trials=trials).fit(X, Y)


@pytest.mark.parametrize("trials", [1, 2, 1000])
def test_trials_in_range_are_accepted(trials: int) -> None:
    classifier = C50Classifier(trials=trials)

    classifier._validate_configuration()


@pytest.mark.parametrize("name", ["minimum_cases", "confidence_factor"])
@pytest.mark.parametrize(
    "value", [10**400, -(10**400)], ids=["huge", "huge-negative"]
)
def test_unconvertible_reals_name_the_parameter(name: str, value: int) -> None:
    parameters: dict[str, Any] = {name: value}

    with pytest.raises(ValueError, match=name):
        C50Classifier(**parameters).fit(X, Y)


def test_unconvertible_sample_fraction_names_the_parameter() -> None:
    with pytest.raises(ValueError, match="sample_fraction"):
        C50Classifier(sample_fraction=10**400).fit(X, Y)
