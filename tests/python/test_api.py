# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Public API tests for the Python bindings."""

from __future__ import annotations

import pickle
from collections.abc import Callable
from decimal import Decimal
from fractions import Fraction
from typing import Any

import c50
import numpy as np
import pytest
from model_text import without_date

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

    assert without_date(repeated.serialized_data) == without_date(
        first.serialized_data
    )
    assert without_date(parallel.serialized_data) == without_date(
        first.serialized_data
    )
    assert len(first.predict_details_dense(values)) == len(values)


def test_stable_ties_agree_across_text_and_dense_training() -> None:
    options = c50.Options()
    options.ties = c50.TieOrder.STABLE
    text_model = c50.train(NAMES, TRAINING, options=options)
    dense_model = c50.train_dense(
        NAMES, DENSE_TRAINING, DENSE_CLASSES, options=options
    )

    assert without_date(text_model.serialized_data) == without_date(
        dense_model.serialized_data
    )
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

    assert without_date(dense_model.serialized_data) == without_date(
        text_model.serialized_data
    )
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
        model = c50.Model.train_dense(NAMES, values, DENSE_CLASSES, copy=False)
        assert without_date(model.serialized_data) == without_date(
            expected.serialized_data
        )
    assert without_date(
        c50.train_dense(
            NAMES, DENSE_TRAINING, DENSE_CLASSES, copy=False
        ).serialized_data
    ) == without_date(expected.serialized_data)


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_training_worker_count_preserves_model(workers: int) -> None:
    text_model = c50.train(NAMES, TRAINING, split_workers=workers)
    dense_model = c50.train_dense(
        NAMES, DENSE_TRAINING, DENSE_CLASSES, split_workers=workers
    )

    assert without_date(text_model.serialized_data) == without_date(
        dense_model.serialized_data
    )
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


def test_unused_costs_are_not_retained() -> None:
    model = c50.train(NAMES, TRAINING)
    loaded = c50.load(NAMES, model.serialized_data, costs="low, high: 5\n")

    assert loaded.costs_data == ""
    restored = pickle.loads(pickle.dumps(loaded))
    assert restored.costs_data == ""
    assert restored.predict(CASES) == ["low", "high"]


def test_missing_required_costs_is_a_value_error() -> None:
    model = c50.train(NAMES, TRAINING, costs="low, high: 5\n")
    assert model.costs_data

    with pytest.raises(ValueError, match="requires costs"):
        c50.load(NAMES, model.serialized_data)
    assert c50.load(NAMES, model.serialized_data, costs=model.costs_data)


def test_native_failures_become_python_exceptions() -> None:
    options = c50.Options()
    options.trials = 0
    with pytest.raises(ValueError, match="trials"):
        c50.train(NAMES, TRAINING, options=options)

    model = c50.train(NAMES, TRAINING)
    with pytest.raises(ValueError, match="bad value"):
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


@pytest.mark.parametrize("encoded", [bytes, bytearray])
def test_bytes_like_text_matches_str(
    encoded: type[bytes] | type[bytearray],
) -> None:
    # The stub declares str, but the binding also accepts bytes-like text. It
    # reads bytes in place and copies a bytearray when it is converted.
    def text(value: str) -> Any:
        return encoded(value, "ascii")

    expected = c50.train(NAMES, TRAINING)
    model = c50.train(text(NAMES), text(TRAINING), costs=text(""))
    loaded = c50.load(text(NAMES), text(expected.serialized_data))

    assert without_date(model.serialized_data) == without_date(
        expected.serialized_data
    )
    assert loaded.serialized_data == expected.serialized_data
    assert model.predict(text(CASES)) == ["low", "high"]
    assert model.predict_scores(text(CASES)) == expected.predict_scores(CASES)
    assert model.attribute_usage(text(TRAINING)) == expected.attribute_usage(
        TRAINING
    )
    details = model.prepare_predictor().predict_details(text(CASES))
    assert details.scores == expected.predict_details(CASES).scores


@pytest.mark.parametrize("cases", [memoryview(b""), 0, None])
def test_text_arguments_reject_other_objects(cases: object) -> None:
    model = c50.train(NAMES, TRAINING)

    with pytest.raises(TypeError):
        model.predict_details(cases)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        model.prepare_predictor().predict_details(cases)  # type: ignore[arg-type]


def test_text_arguments_reject_unencodable_text_as_value_errors() -> None:
    # A lone surrogate is a str with an invalid value, so it raises
    # UnicodeEncodeError, a ValueError, and not an unsuitable-argument
    # TypeError.
    model = c50.train(NAMES, TRAINING)
    bad = "\ud800"
    assert issubclass(UnicodeEncodeError, ValueError)

    with pytest.raises(UnicodeEncodeError):
        model.predict(bad)
    with pytest.raises(UnicodeEncodeError):
        model.predict_details(bad)
    with pytest.raises(UnicodeEncodeError):
        model.prepare_predictor().predict_details(bad)
    with pytest.raises(UnicodeEncodeError):
        model.attribute_usage(bad)
    with pytest.raises(UnicodeEncodeError):
        c50.train(NAMES, TRAINING + bad)
    with pytest.raises(UnicodeEncodeError):
        c50.train(NAMES + bad, TRAINING)
    with pytest.raises(UnicodeEncodeError):
        c50.train(NAMES, TRAINING, costs=bad)
    with pytest.raises(UnicodeEncodeError):
        c50.load(bad, model.serialized_data)


_TRAIN_CALLS: dict[str, Callable[[Any], c50.Model]] = {
    "train": lambda w: c50.train(NAMES, TRAINING, split_workers=w),
    "train_dense": lambda w: c50.train_dense(
        NAMES, DENSE_TRAINING, DENSE_CLASSES, split_workers=w
    ),
    "Model.train": lambda w: c50.Model.train(NAMES, TRAINING, split_workers=w),
    "Model.train_dense": lambda w: c50.Model.train_dense(
        NAMES, DENSE_TRAINING, DENSE_CLASSES, split_workers=w
    ),
}


@pytest.mark.parametrize("call", sorted(_TRAIN_CALLS))
@pytest.mark.parametrize("workers", [-1, 2**32, 2**64, 0, 9])
def test_split_workers_out_of_range_is_value_error(
    call: str, workers: int
) -> None:
    with pytest.raises(ValueError, match=r"split_workers|split worker"):
        _TRAIN_CALLS[call](workers)


_NOT_COUNTS = [
    True,
    np.True_,
    Fraction(3, 1),
    Decimal(3),
    1.5,
    "2",
    None,
]


@pytest.mark.parametrize("call", sorted(_TRAIN_CALLS))
@pytest.mark.parametrize(
    "workers", _NOT_COUNTS, ids=lambda value: type(value).__name__
)
def test_split_workers_of_the_wrong_type_is_type_error(
    call: str, workers: object
) -> None:
    with pytest.raises(TypeError):
        _TRAIN_CALLS[call](workers)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("trials", -1),
        ("trials", 2**32),
        ("random_seed", -1),
        ("random_seed", 2**32),
        ("minimum_cases", 10**400),
        ("confidence_factor", 10**400),
        ("sample_fraction", -(10**400)),
    ],
    ids=lambda value: "huge" if value in (10**400, -(10**400)) else str(value),
)
def test_options_setters_name_the_field_of_an_unrepresentable_number(
    field: str, value: int
) -> None:
    options = c50.Options()
    before = getattr(options, field)

    with pytest.raises(ValueError, match=field):
        setattr(options, field, value)
    assert getattr(options, field) == before


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("trials", 1.5),
        ("trials", "2"),
        ("random_seed", None),
        ("minimum_cases", "2"),
        ("confidence_factor", None),
        ("sample_fraction", "0"),
    ],
)
def test_options_setters_reject_other_types(field: str, value: object) -> None:
    with pytest.raises(TypeError):
        setattr(c50.Options(), field, value)


@pytest.mark.parametrize("field", ["trials", "random_seed"])
@pytest.mark.parametrize(
    "value",
    [True, np.True_, Fraction(3, 1), Decimal(3)],
    ids=lambda value: type(value).__name__,
)
def test_options_counts_reject_bool_and_non_index_numbers(
    field: str, value: object
) -> None:
    options = c50.Options()
    before = getattr(options, field)

    with pytest.raises(TypeError):
        setattr(options, field, value)
    assert getattr(options, field) == before


@pytest.mark.parametrize("call", sorted(_TRAIN_CALLS))
def test_split_workers_accepts_numpy_integers(call: str) -> None:
    assert without_date(
        _TRAIN_CALLS[call](np.int64(2)).serialized_data
    ) == without_date(_TRAIN_CALLS[call](2).serialized_data)


@pytest.mark.parametrize("field", ["trials", "random_seed"])
def test_options_counts_accept_numpy_integers(field: str) -> None:
    options = c50.Options()
    setattr(options, field, np.int64(2))
    assert getattr(options, field) == 2


def test_options_setters_accept_in_range_numbers() -> None:
    options = c50.Options()
    options.trials = 2**32 - 1
    options.random_seed = np.int64(7)  # type: ignore[assignment]
    options.minimum_cases = 3
    options.confidence_factor = np.float32(0.5)  # type: ignore[assignment]
    options.sample_fraction = 0.5

    assert options.trials == 2**32 - 1
    assert options.random_seed == 7
    assert options.minimum_cases == 3.0
    assert options.confidence_factor == 0.5
    assert options.sample_fraction == 0.5


def test_bytearray_text_is_copied_before_later_arguments() -> None:
    # pybind11 converts split_workers after training_data. Training must read
    # the bytearray's contents as they were when it was converted. Overwriting
    # them before clearing makes a stale view visible even when freed storage
    # keeps its old bytes.
    data = bytearray(TRAINING, "ascii")

    class ReplacesData:
        def __index__(self) -> int:
            data[:] = bytes(len(data))
            data.clear()
            return 1

    model = c50.train(NAMES, data, split_workers=ReplacesData())  # type: ignore[arg-type]

    assert data == bytearray()
    assert without_date(model.serialized_data) == without_date(
        c50.train(NAMES, TRAINING).serialized_data
    )


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


@pytest.mark.parametrize("factory", [c50.train_dense, c50.Model.train_dense])
@pytest.mark.parametrize("copy", [True, False])
@pytest.mark.parametrize(
    "values",
    [
        np.full((8, 2), "word"),
        np.full((8, 2), "word", dtype=object),
        [[1.0, 2.0]] * 7 + [[1.0]],
    ],
    ids=["words", "object-words", "ragged"],
)
def test_dense_values_that_cannot_convert_raise_value_error(
    factory: Callable[..., c50.Model], copy: bool, values: object
) -> None:
    with pytest.raises(ValueError, match="values cannot be converted"):
        factory(NAMES, values, DENSE_CLASSES, copy=copy)


def test_dense_values_keep_the_casts_that_the_documentation_names() -> None:
    expected = c50.train_dense(NAMES, DENSE_TRAINING, DENSE_CLASSES)
    numeric_text = DENSE_TRAINING.astype(str)
    for values in (
        numeric_text,
        DENSE_TRAINING.astype(np.float32),
        DENSE_TRAINING.tolist(),
        np.asfortranarray(DENSE_TRAINING),
    ):
        model = c50.train_dense(NAMES, values, DENSE_CLASSES)
        assert without_date(model.serialized_data) == without_date(
            expected.serialized_data
        )


@pytest.mark.parametrize("dtype", [np.int8, np.int32, np.uint32, np.uint64])
def test_dense_class_indices_accept_integer_dtypes(
    dtype: type[np.generic],
) -> None:
    labels = DENSE_CLASSES.astype(dtype)
    model = c50.train_dense(NAMES, DENSE_TRAINING, labels)
    assert model.predict_details_dense(DENSE_CASES).class_indices == [0, 1]


def test_dense_methods_raise_c50_error_for_unsupported_attributes() -> None:
    # A label attribute is valid in text but unsupported by the dense
    # interface. The docstrings and stubs document C50Error for it.
    names = "low, high.\n\nsignal: continuous.\nid: label.\n"
    training = "0, a, low\n1, b, low\n2, c, high\n3, d, high\n"
    model = c50.train(names, training)
    values = np.asarray([[0.0, 1.0], [3.0, 0.0]])

    with pytest.raises(c50.C50Error, match="label"):
        model.predict_details_dense(values)
    with pytest.raises(c50.C50Error, match="label"):
        model.attribute_usage_dense(values)
    with pytest.raises(c50.C50Error, match="label"):
        model.prepare_predictor().predict_details_dense(values)
    for method in (
        c50.Model.predict_details_dense,
        c50.Model.attribute_usage_dense,
    ):
        assert "C50Error" in (method.__doc__ or "")
    assert "C50Error" in (c50.Predictor.predict_details_dense.__doc__ or "")


# Each schema is valid for text data but unsupported by the dense interface.
# The tuple holds the names data, the training data, and the feature count.
UNSUPPORTED_DENSE_SCHEMAS = {
    "implicit": ("a, b.\nx: continuous.\nd := x * 2.\n", "0,a\n1,b\n", 2),
    "ignored": ("a, b.\nx: continuous.\ny: ignore.\n", "0,1,a\n1,2,b\n", 2),
    "label": ("a, b.\nx: continuous.\nid: label.\n", "0,p,a\n1,q,b\n", 2),
    "date": (
        "a, b.\nx: continuous.\nd: date.\n",
        "0,2020/01/01,a\n1,2020/01/02,b\n",
        2,
    ),
    "time": (
        "a, b.\nx: continuous.\nt: time.\n",
        "0,10:00:00,a\n1,10:00:01,b\n",
        2,
    ),
    "timestamp": (
        "a, b.\nx: continuous.\nt: timestamp.\n",
        "0,2020/01/01 10:00:00,a\n1,2020/01/01 10:00:01,b\n",
        2,
    ),
    "discrete": (
        "a, b.\nx: continuous.\ng: discrete 2.\n",
        "0,u,a\n1,v,b\n",
        2,
    ),
    "class": ("cls.\nx: continuous.\ncls: a, b.\n", "0,a\n1,b\n", 1),
}


@pytest.mark.parametrize("kind", sorted(UNSUPPORTED_DENSE_SCHEMAS))
@pytest.mark.parametrize("extra", [None, 0, 1])
@pytest.mark.parametrize("rows", [0, 2])
@pytest.mark.parametrize("copy", [True, False])
def test_unsupported_dense_schema_raises_c50_error_at_any_shape(
    kind: str, extra: int | None, rows: int, copy: bool
) -> None:
    # The support checks need only the schema, so neither the rows nor the
    # width of the array, which is zero for extra=None, may decide which
    # exception the caller sees.
    names, training, width = UNSUPPORTED_DENSE_SCHEMAS[kind]
    model = c50.train(names, training)
    values = np.zeros((rows, 0 if extra is None else width + extra))

    with pytest.raises(c50.C50Error):
        model.predict_details_dense(values)
    with pytest.raises(c50.C50Error):
        model.attribute_usage_dense(values)
    with pytest.raises(c50.C50Error):
        model.prepare_predictor().predict_details_dense(values)
    with pytest.raises(c50.C50Error):
        c50.train_dense(
            names, values, np.zeros(rows, dtype=np.intp), copy=copy
        )
    with pytest.raises(c50.C50Error):
        c50.Model.train_dense(
            names, values, np.zeros(rows, dtype=np.intp), copy=copy
        )


@pytest.mark.parametrize("width", [0, 1, 3])
@pytest.mark.parametrize("copy", [True, False])
def test_supported_dense_schema_rejects_a_mismatched_width(
    width: int, copy: bool
) -> None:
    names = "a, b.\nx: continuous.\ny: continuous.\n"
    model = c50.train(names, "0,1,a\n1,2,b\n")
    values = np.zeros((2, width))

    with pytest.raises(ValueError, match="feature count"):
        model.predict_details_dense(values)
    with pytest.raises(ValueError, match="feature count"):
        model.attribute_usage_dense(values)
    with pytest.raises(ValueError, match="feature count"):
        model.prepare_predictor().predict_details_dense(values)
    with pytest.raises(ValueError, match="feature count"):
        c50.train_dense(names, values, np.zeros(2, dtype=np.intp), copy=copy)
    with pytest.raises(ValueError, match="feature count"):
        c50.Model.train_dense(
            names, values, np.zeros(2, dtype=np.intp), copy=copy
        )
