# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

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


class IndexOnly:
    """An object that converts to an integer through __index__ alone."""

    def __index__(self) -> int:
        return 2


# The estimator requires numbers.Integral. The low-level API accepts any
# __index__ object, so these values differ from c50.Options, and bool is
# rejected with ValueError here, not TypeError.
NOT_INTEGRAL = [True, np.True_, IndexOnly(), np.asarray(2)]


@pytest.mark.parametrize("name", ["trials", "n_jobs", "prediction_batch_size"])
@pytest.mark.parametrize(
    "value", NOT_INTEGRAL, ids=["True", "np.True_", "index-only", "0d-array"]
)
def test_integer_parameters_reject_non_integral_values_at_fit(
    name: str, value: object
) -> None:
    parameters: dict[str, Any] = {name: value}
    classifier = C50Classifier(**parameters)

    with pytest.raises(ValueError, match=name):
        classifier.fit(X, Y)


@pytest.mark.parametrize("name", ["trials", "n_jobs", "prediction_batch_size"])
def test_integer_parameters_accept_numpy_integers(name: str) -> None:
    parameters: dict[str, Any] = {name: np.int64(2)}

    C50Classifier(**parameters).fit(X, Y)
