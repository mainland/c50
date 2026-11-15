# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pickle format tests for the low-level ``Model``."""

from __future__ import annotations

import copy
import pickle
import pickletools
from typing import Any

import c50
import pytest

NAMES = "low, high.\n\nsignal: continuous.\n"
TRAINING = "0, low\n1, low\n2, high\n3, high\n"
CASES = "0.5, ?\n2.5, ?\n"
COSTS = "low, high: 5\n"
_PROTOCOLS = range(pickle.HIGHEST_PROTOCOL + 1)


def _strings(data: bytes) -> list[str]:
    """Return every string operand of a pickle."""
    found: list[str] = []
    for _, argument, _ in pickletools.genops(data):
        if isinstance(argument, str):
            found.append(argument)
    return found


@pytest.mark.parametrize("protocol", _PROTOCOLS)
@pytest.mark.parametrize("kind", [c50.ModelKind.TREE, c50.ModelKind.RULES])
def test_pickle_names_only_the_public_module(
    protocol: int, kind: c50.ModelKind
) -> None:
    model = c50.train(NAMES, TRAINING, kind=kind, costs=COSTS)
    data = pickle.dumps(model, protocol=protocol)

    assert b"_c50" not in data
    for name in ("ModelKind", "TieOrder", "NodeKind", "ConditionKind"):
        assert name.encode() not in data
    assert b"pybind11" not in data
    strings = _strings(data)
    assert "c50 _restore_model" in strings or (
        "c50" in strings and "_restore_model" in strings
    )
    restored = pickle.loads(data)
    assert isinstance(restored, c50.Model)
    assert restored.kind == model.kind
    assert restored.names_data == model.names_data
    assert restored.serialized_data == model.serialized_data
    assert restored.costs_data == model.costs_data
    assert restored.predict(CASES) == model.predict(CASES)


_ENUMERATION_MEMBERS = [
    member
    for enumeration in (
        c50.ModelKind,
        c50.TieOrder,
        c50.NodeKind,
        c50.ConditionKind,
    )
    for member in enumeration.__members__.values()
]


@pytest.mark.parametrize("protocol", _PROTOCOLS)
@pytest.mark.parametrize(
    "member",
    _ENUMERATION_MEMBERS,
    ids=lambda member: f"{type(member).__name__}.{member.name}",
)
def test_enumeration_pickle_names_the_public_module(
    protocol: int, member: Any
) -> None:
    data = pickle.dumps(member, protocol=protocol)

    assert b"_c50" not in data
    assert b"pybind11" not in data
    strings = _strings(data)
    assert "c50 _restore_enum" in strings or (
        "c50" in strings and "_restore_enum" in strings
    )
    assert member.name in strings
    assert pickle.loads(data) is member
    assert copy.copy(member) is member
    assert copy.deepcopy(member) is member


# Pickles of c50.TieOrder.STABLE that the release with pybind11 enumerations
# wrote, at every protocol. Every 1.x release must load them, as
# docs/stability.md promises.
_RECORDED_ENUMERATION_PICKLES: list[bytes] = [
    b"cc50\n_restore_enum\np0\n(VTieOrder\np1\nVSTABLE\np2\ntp3\nRp4\n.",
    (
        b"cc50\n_restore_enum\nq\x00(X\x08\x00\x00\x00TieOrderq\x01"
        b"X\x06\x00\x00\x00STABLEq\x02tq\x03Rq\x04."
    ),
    (
        b"\x80\x02cc50\n_restore_enum\nq\x00X\x08\x00\x00\x00TieOrderq\x01"
        b"X\x06\x00\x00\x00STABLEq\x02\x86q\x03Rq\x04."
    ),
    (
        b"\x80\x03cc50\n_restore_enum\nq\x00X\x08\x00\x00\x00TieOrderq\x01"
        b"X\x06\x00\x00\x00STABLEq\x02\x86q\x03Rq\x04."
    ),
    (
        b"\x80\x04\x951\x00\x00\x00\x00\x00\x00\x00\x8c\x03c50\x94"
        b"\x8c\r_restore_enum\x94\x93\x94\x8c\x08TieOrder\x94\x8c\x06STABLE"
        b"\x94\x86\x94R\x94."
    ),
    (
        b"\x80\x05\x951\x00\x00\x00\x00\x00\x00\x00\x8c\x03c50\x94"
        b"\x8c\r_restore_enum\x94\x93\x94\x8c\x08TieOrder\x94\x8c\x06STABLE"
        b"\x94\x86\x94R\x94."
    ),
]


@pytest.mark.parametrize("data", _RECORDED_ENUMERATION_PICKLES)
def test_recorded_enumeration_pickles_load(data: bytes) -> None:
    assert pickle.loads(data) is c50.TieOrder.STABLE


def test_current_pickles_equal_the_recorded_pickles() -> None:
    # The pickle bytes are stable across the supported Python versions for
    # these protocols, so a change in what the enumerations pickle shows here.
    current = [
        pickle.dumps(c50.TieOrder.STABLE, protocol) for protocol in _PROTOCOLS
    ]
    assert current == _RECORDED_ENUMERATION_PICKLES


@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        (("Options", "trials"), ValueError),
        (("TieOrder", "NO_SUCH_MEMBER"), ValueError),
        (("TieOrder", 1), TypeError),
        ((1, "STABLE"), TypeError),
    ],
)
def test_restore_enum_rejects_unknown_or_malformed_state(
    arguments: tuple[Any, Any], error: type[Exception]
) -> None:
    with pytest.raises(error):
        c50._restore_enum(*arguments)


def test_pickle_retains_required_costs() -> None:
    model = c50.train(NAMES, TRAINING, costs=COSTS)
    assert model.costs_data
    restored = pickle.loads(pickle.dumps(model))

    assert restored.costs_data == model.costs_data


@pytest.mark.parametrize(
    "costs",
    [
        "",
        "\n",
        "  \n\n",
        "\t \n",
        "| comment\n",
        "low, high: 1\n",
        "low, low: 5\n",
    ],
    ids=["empty", "newline", "blank", "tab", "comment", "cost-1", "diagonal"],
)
def test_load_rejects_costs_that_set_no_cost_of_a_cost_model(
    costs: str,
) -> None:
    model = c50.train(NAMES, TRAINING, costs=COSTS)
    with pytest.raises(ValueError, match="requires costs"):
        c50.load(NAMES, model.serialized_data, costs=costs)


def test_load_accepts_a_comment_beside_an_entry_of_a_cost_model() -> None:
    model = c50.train(NAMES, TRAINING, costs=COSTS)
    loaded = c50.load(
        NAMES, model.serialized_data, costs="| comment\n" + COSTS
    )
    assert loaded.predict(CASES) == model.predict(CASES)


def test_copy_and_deepcopy_use_the_same_state() -> None:
    model = c50.train(NAMES, TRAINING)

    for duplicate in (copy.copy(model), copy.deepcopy(model)):
        assert duplicate is not model
        assert duplicate.serialized_data == model.serialized_data


def _reduce(model: c50.Model) -> tuple[Any, tuple[Any, ...]]:
    callable_, arguments = model.__reduce__()[:2]
    assert isinstance(arguments, tuple)
    return callable_, arguments


def test_reconstruction_callable_is_public() -> None:
    callable_, _ = _reduce(c50.train(NAMES, TRAINING))

    assert callable_.__module__ == "c50"
    assert getattr(c50, callable_.__name__) is callable_


def test_unknown_version_is_a_value_error() -> None:
    callable_, arguments = _reduce(c50.train(NAMES, TRAINING))

    for version in (0, 2, 99, -1):
        with pytest.raises(ValueError, match="version"):
            callable_(version, *arguments[1:])
    for version in ("1", None, 1.0, True):
        with pytest.raises((ValueError, TypeError)):
            callable_(version, *arguments[1:])


@pytest.mark.parametrize("position", [1, 2, 3, 4])
@pytest.mark.parametrize("value", [None, 7, b"bytes", ["x"], 1.5])
def test_wrong_element_types_are_not_runtime_errors(
    position: int, value: object
) -> None:
    callable_, arguments = _reduce(c50.train(NAMES, TRAINING))
    state = list(arguments)
    state[position] = value

    with pytest.raises((ValueError, TypeError)) as caught:
        callable_(*state)
    assert not isinstance(caught.value, RuntimeError)


def test_unknown_kind_and_wrong_arity_are_rejected() -> None:
    callable_, arguments = _reduce(c50.train(NAMES, TRAINING))

    with pytest.raises(ValueError, match="kind"):
        callable_(arguments[0], "forest", *arguments[2:])
    with pytest.raises(TypeError):
        callable_(*arguments[:-1])
    with pytest.raises(TypeError):
        callable_(*arguments, "extra")


def test_corrupt_text_is_a_value_error() -> None:
    callable_, arguments = _reduce(c50.train(NAMES, TRAINING))
    state = list(arguments)
    state[3] = "not a model"

    with pytest.raises(ValueError, match="unexpected eof"):
        callable_(*state)
