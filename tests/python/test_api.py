"""Public API tests for the Python bindings."""

from __future__ import annotations

import pickle
from collections.abc import Callable

import numpy as np
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

DENSE_TRAINING = np.asarray(
    [
        [0, 0],
        [0.5, 1],
        [1, 0],
        [1.5, 1],
        [2.5, 0],
        [3, 1],
        [3.5, 0],
        [4, 1],
    ],
    dtype=np.float64,
)
DENSE_CLASSES = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uintp)
DENSE_CASES = np.asarray([[0.5, 0], [3.5, 1]], dtype=np.float64)


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
    assert options.ties == c50.TieOrder.REFERENCE

    options.trials = 3
    options.subset_splits = True
    options.random_seed = 7
    options.ties = c50.TieOrder.STABLE
    assert options.trials == 3
    assert options.subset_splits is True
    assert options.random_seed == 7
    assert options.ties == c50.TieOrder.STABLE


def test_stable_ties_are_deterministic() -> None:
    # More rows than the stable sort's insertion-sort limit, with many ties.
    rng = np.random.default_rng(11)
    values = rng.integers(0, 6, size=(400, 3)).astype(np.float64)
    values[rng.random(values.shape) < 0.05] = np.nan
    classes = (np.nan_to_num(values).sum(axis=1) > 7).astype(np.uintp)
    names = "no, yes.\n\nx0: continuous.\nx1: continuous.\nx2: continuous.\n"
    options = c50.Options()
    options.ties = c50.TieOrder.STABLE
    options.trials = 3

    first = c50.train_dense(names, values, classes, options=options)
    repeated = c50.train_dense(names, values, classes, options=options)
    parallel = c50.train_dense(
        names, values, classes, options=options, split_workers=4
    )

    assert repeated.serialized_data == first.serialized_data
    assert parallel.serialized_data == first.serialized_data
    assert len(first.predict_details_dense(values)) == len(values)


def test_stable_ties_agree_across_text_and_dense_training() -> None:
    options = c50.Options()
    options.ties = c50.TieOrder.STABLE
    text_model = c50.train(NAMES, TRAINING, options=options)
    dense_model = c50.train_dense(
        NAMES, DENSE_TRAINING, DENSE_CLASSES, options=options
    )

    assert text_model.serialized_data == dense_model.serialized_data
    assert dense_model.predict(CASES) == ["low", "high"]


@pytest.mark.parametrize("factory", [c50.train, c50.Model.train])
def test_training_prediction_and_serialization(
    factory: Callable[..., c50.Model],
) -> None:
    model = factory(NAMES, TRAINING)

    assert model.kind == c50.ModelKind.TREE
    assert model.class_names == ["low", "high"]
    assert model.predict(CASES) == ["low", "high"]
    assert model.predict_scores(CASES) == [[1.0, 0.0], [0.0, 1.0]]
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


@pytest.mark.parametrize("factory", [c50.train_dense, c50.Model.train_dense])
def test_dense_training_matches_text_training(
    factory: Callable[..., c50.Model],
) -> None:
    text_model = c50.train(NAMES, TRAINING)
    dense_model = factory(NAMES, DENSE_TRAINING, DENSE_CLASSES)

    assert dense_model.serialized_data == text_model.serialized_data
    details = dense_model.predict_details_dense(DENSE_CASES)
    assert details.class_indices == [0, 1]
    assert details.scores == [[1.0, 0.0], [0.0, 1.0]]


def test_dense_training_without_copy_matches_copy() -> None:
    expected = c50.Model.train_dense(NAMES, DENSE_TRAINING, DENSE_CLASSES)

    for values in (
        DENSE_TRAINING,
        np.asfortranarray(DENSE_TRAINING),
        DENSE_TRAINING.astype(np.float32),
        np.repeat(DENSE_TRAINING, 2, axis=1)[:, ::2],
    ):
        model = c50.Model.train_dense(
            NAMES, values, DENSE_CLASSES, copy=False
        )
        assert model.serialized_data == expected.serialized_data
    assert (
        c50.train_dense(NAMES, DENSE_TRAINING, DENSE_CLASSES, copy=False)
        .serialized_data
        == expected.serialized_data
    )


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_training_worker_count_preserves_model(workers: int) -> None:
    text_model = c50.train(NAMES, TRAINING, split_workers=workers)
    dense_model = c50.train_dense(
        NAMES, DENSE_TRAINING, DENSE_CLASSES, split_workers=workers
    )

    assert text_model.serialized_data == dense_model.serialized_data
    assert dense_model.predict(CASES) == ["low", "high"]


@pytest.mark.parametrize("workers", [0, 9])
def test_training_rejects_invalid_worker_count(workers: int) -> None:
    with pytest.raises(ValueError, match="split worker count"):
        c50.train(NAMES, TRAINING, split_workers=workers)


def test_dense_array_shapes_are_validated() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        c50.train_dense(NAMES, DENSE_TRAINING.ravel(), DENSE_CLASSES)
    with pytest.raises(ValueError, match="inconsistent row counts"):
        c50.train_dense(NAMES, DENSE_TRAINING, DENSE_CLASSES[:-1])


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


@pytest.mark.parametrize("kind", [c50.ModelKind.TREE, c50.ModelKind.RULES])
def test_attribute_usage_counts_tested_cases(kind: c50.ModelKind) -> None:
    options = c50.Options()
    options.minimum_cases = 1
    model = c50.Model.train(NAMES, TRAINING, kind, options)

    usage = model.attribute_usage(TRAINING)

    assert usage == [8, 0]
    assert model.attribute_usage_dense(DENSE_TRAINING) == usage
    assert model.attribute_usage("?, alpha, ?\n") == [0, 0]
    assert model.attribute_usage("") == [0, 0]
    assert model.attribute_usage_dense(np.empty((0, 2))) == [0, 0]
    with pytest.raises(ValueError, match="two-dimensional"):
        model.attribute_usage_dense(np.zeros(2))


def test_empty_prediction_batch() -> None:
    model = c50.train(NAMES, TRAINING)

    assert model.predict("") == []
    assert model.predict_scores("") == []
    assert len(model.predict_details("")) == 0


@pytest.mark.parametrize("factory", [c50.train_dense, c50.Model.train_dense])
@pytest.mark.parametrize(
    "labels",
    [
        np.asarray([0, 0, 0, 0, 1, 1, 1, 1.5]),
        np.asarray([-0.2, 0, 0, 0, 1, 1, 1, 1]),
        np.asarray([-1, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64),
        np.asarray([2**63, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint64),
        np.asarray([np.nan, 0, 0, 0, 1, 1, 1, 1]),
    ],
)
def test_dense_class_indices_reject_lossy_conversion(
    factory: Callable[..., c50.Model], labels: np.ndarray
) -> None:
    with pytest.raises(ValueError, match="class_indices"):
        factory(NAMES, DENSE_TRAINING, labels)


@pytest.mark.parametrize("dtype", [np.int8, np.int32, np.uint32, np.uint64])
def test_dense_class_indices_accept_integer_dtypes(dtype: type[np.generic]) -> None:
    labels = DENSE_CLASSES.astype(dtype)
    model = c50.train_dense(NAMES, DENSE_TRAINING, labels)
    assert model.predict_details_dense(DENSE_CASES).class_indices == [0, 1]
