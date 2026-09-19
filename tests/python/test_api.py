"""Public API tests for the Python bindings."""

from __future__ import annotations

import pickle
from collections.abc import Callable

import pytest

import c50


NAMES = """low, high.

signal: continuous.
group: alpha, beta.
"""

TRAINING = """0, alpha, low
0.5, beta, low
1, alpha, low
1.5, beta, low
2.5, alpha, high
3, beta, high
3.5, alpha, high
4, beta, high
"""

CASES = """0.5, alpha, ?
3.5, beta, ?
"""


def test_exception_hierarchy() -> None:
    assert issubclass(c50.C50Error, RuntimeError)


def test_options_defaults_and_mutation() -> None:
    options = c50.Options()

    assert options.trials == 1
    assert options.subset_splits is False
    assert options.winnow is False
    assert options.global_pruning is True
    assert options.probabilistic_thresholds is False
    assert options.ignore_costs is False
    assert options.minimum_cases == 2
    assert options.confidence_factor == 0.25
    assert options.sample_fraction == 0
    assert options.random_seed == 0

    options.trials = 3
    options.subset_splits = True
    options.random_seed = 7
    assert options.trials == 3
    assert options.subset_splits is True
    assert options.random_seed == 7


@pytest.mark.parametrize("factory", [c50.train, c50.Model.train])
def test_training_prediction_and_serialization(
    factory: Callable[..., c50.Model],
) -> None:
    model = factory(NAMES, TRAINING)

    assert model.kind == c50.ModelKind.TREE
    assert model.classes_ == ["low", "high"]
    assert model.predict(CASES) == ["low", "high"]
    assert model.predict_proba(CASES) == [[1.0, 0.0], [0.0, 1.0]]
    assert model.names_data == NAMES
    assert 'entries="1"' in model.serialized_data
    assert model.costs_data == ""

    details = model.predict_details(CASES)
    assert len(details) == 2
    assert details.class_names == ["low", "high"]
    assert details.class_indices == [0, 1]
    assert details.labels == ["low", "high"]
    assert details.confidences == [1.0, 1.0]
    assert details.scores == [[1.0, 0.0], [0.0, 1.0]]


def test_tree_and_rules_round_trips() -> None:
    tree = c50.train(NAMES, TRAINING)
    loaded_tree = c50.load(
        tree.names_data,
        tree.serialized_data,
        tree.kind,
        tree.costs_data,
    )
    assert loaded_tree.predict(CASES) == ["low", "high"]

    options = c50.Options()
    options.winnow = True
    rules = c50.train(
        NAMES,
        TRAINING,
        c50.ModelKind.RULES,
        options,
    )
    assert rules.kind == c50.ModelKind.RULES
    assert 'rules="' in rules.serialized_data
    loaded_rules = c50.Model.load(
        rules.names_data,
        rules.serialized_data,
        rules.kind,
        rules.costs_data,
    )
    assert loaded_rules.predict(CASES) == ["low", "high"]


def test_pickle_round_trip() -> None:
    model = c50.train(NAMES, TRAINING)
    restored = pickle.loads(pickle.dumps(model))

    assert restored.kind == model.kind
    assert restored.serialized_data == model.serialized_data
    assert restored.predict(CASES) == ["low", "high"]


def test_native_failures_become_python_exceptions() -> None:
    options = c50.Options()
    options.trials = 0
    with pytest.raises(ValueError, match="trials"):
        c50.train(NAMES, TRAINING, options=options)

    model = c50.train(NAMES, TRAINING)
    with pytest.raises(ValueError):
        model.predict("malformed")


def test_empty_prediction_batch() -> None:
    model = c50.train(NAMES, TRAINING)

    assert model.predict("") == []
    assert model.predict_proba("") == []
    assert len(model.predict_details("")) == 0
