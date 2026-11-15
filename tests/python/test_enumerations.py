# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests of the enumerations as ``enum.IntEnum`` types."""

from __future__ import annotations

import copy
import enum
import json
import operator
import pickle
from typing import Any

import c50
import numpy as np
import pytest

NAMES = "low, high.\n\nsignal: continuous.\n"
TRAINING = "0, low\n1, low\n2, high\n3, high\n"
VALUES = np.asarray([[0.0], [1.0], [2.0], [3.0]])
CLASSES = np.asarray([0, 0, 1, 1], dtype=np.uintp)

# Each enumeration with its members in definition order.
_MEMBERS: dict[type[enum.IntEnum], list[str]] = {
    c50.ModelKind: ["TREE", "RULES"],
    c50.TieOrder: ["REFERENCE", "STABLE"],
    c50.NodeKind: ["LEAF", "DISCRETE", "THRESHOLD", "SUBSET"],
    c50.ConditionKind: [
        "NOT_APPLICABLE",
        "EQUALS",
        "LESS_EQUAL",
        "GREATER",
        "IN_SUBSET",
    ],
}
_ENUMERATIONS = list(_MEMBERS)
_ALL_MEMBERS = [
    member
    for enumeration in _MEMBERS
    for member in enumeration.__members__.values()
]


def _id(value: Any) -> str:
    return getattr(value, "__name__", None) or str(value)


@pytest.mark.parametrize("enumeration", _ENUMERATIONS, ids=_id)
def test_enumeration_is_an_int_enum(enumeration: type[enum.IntEnum]) -> None:
    assert issubclass(enumeration, enum.IntEnum)
    assert issubclass(enumeration, enum.Enum)
    assert issubclass(enumeration, int)
    assert enumeration.__module__ == "c50._c50"


@pytest.mark.parametrize("enumeration", _ENUMERATIONS, ids=_id)
def test_iteration_lookup_and_length(
    enumeration: type[enum.IntEnum],
) -> None:
    names = _MEMBERS[enumeration]

    assert [member.name for member in enumeration] == names
    assert len(enumeration) == len(names)
    assert list(enumeration.__members__) == names
    for value, name in enumerate(names):
        member = enumeration[name]
        assert member is enumeration(value)
        assert member is enumeration.__members__[name]
        assert member.name == name
        assert member.value == value
        assert type(member.value) is int
        assert int(member) == value
        assert operator.index(member) == value
        assert hash(member) == hash(value)


@pytest.mark.parametrize("enumeration", _ENUMERATIONS, ids=_id)
def test_unknown_names_and_values_are_rejected(
    enumeration: type[enum.IntEnum],
) -> None:
    with pytest.raises(KeyError):
        enumeration["NO_SUCH_MEMBER"]
    with pytest.raises(ValueError, match="is not a valid"):
        enumeration(len(enumeration))


@pytest.mark.parametrize("member", _ALL_MEMBERS, ids=str)
def test_members_compare_equal_to_their_integers(member: Any) -> None:
    assert member == member.value
    assert member.value == member
    assert member != member.value + 1
    assert member in {member.value: None}
    assert {member: None}[member.value] is None
    assert member < len(type(member))
    assert "%d" % member == str(member.value)  # noqa: UP031


def test_members_of_different_enumerations_compare_by_value() -> None:
    assert c50.TieOrder.REFERENCE == c50.ModelKind.TREE
    assert c50.TieOrder.STABLE == c50.ModelKind.RULES
    assert c50.TieOrder.STABLE != c50.ModelKind.TREE


# The text of a member must not depend on the Python version. enum.IntEnum
# prints the integer on Python 3.11 and later and the qualified name before.
@pytest.mark.parametrize("member", _ALL_MEMBERS, ids=str)
def test_text_forms_do_not_depend_on_the_python_version(member: Any) -> None:
    type_name = type(member).__name__
    qualified = f"{type_name}.{member.name}"

    assert str(member) == qualified
    assert format(member) == qualified
    assert f"{member}" == qualified
    assert f"{member!s:>40}" == qualified.rjust(40)
    assert "%s" % member == qualified  # noqa: UP031
    assert repr(member) == f"<{qualified}: {member.value}>"


@pytest.mark.parametrize("member", _ALL_MEMBERS, ids=str)
def test_copy_pickle_and_deepcopy_return_the_member(member: Any) -> None:
    assert copy.copy(member) is member
    assert copy.deepcopy(member) is member
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        assert pickle.loads(pickle.dumps(member, protocol)) is member


def test_native_boundary_accepts_members() -> None:
    options = c50.Options()
    options.ties = c50.TieOrder.STABLE
    assert options.ties is c50.TieOrder.STABLE
    assert type(options.ties) is c50.TieOrder

    for kind in c50.ModelKind:
        model = c50.train(NAMES, TRAINING, kind)
        assert model.kind is kind
        dense = c50.train_dense(NAMES, VALUES, CLASSES, kind)
        assert dense.kind is kind
        loaded = c50.load(NAMES, model.serialized_data, kind)
        assert loaded.kind is kind
        assert c50.train(NAMES, TRAINING, kind=kind, options=options)


def test_native_boundary_returns_members() -> None:
    model = c50.train(NAMES, TRAINING, c50.ModelKind.RULES)
    snapshot = model.inspect()
    assert type(snapshot.kind) is c50.ModelKind
    condition = snapshot.rulesets[0].rules[0].conditions[0]
    assert type(condition.kind) is c50.ConditionKind
    assert condition.kind is c50.ConditionKind.LESS_EQUAL

    tree = c50.train(NAMES, TRAINING).inspect().trees[0]
    assert type(tree.nodes[0].kind) is c50.NodeKind
    assert tree.nodes[0].kind is c50.NodeKind.THRESHOLD
    assert isinstance(tree.nodes[0].kind, enum.IntEnum)


# The boundary accepts only members of the expected enumeration. A plain
# integer, a member of another enumeration with the same value, a name, and
# None raise TypeError, as they did for the pybind11 enumerations.
_NOT_A_MODEL_KIND = [
    1,
    True,
    1.0,
    None,
    "rules",
    "RULES",
    c50.TieOrder.STABLE,
    c50.NodeKind.DISCRETE,
]


@pytest.mark.parametrize("value", _NOT_A_MODEL_KIND, ids=repr)
def test_native_boundary_rejects_other_values_for_a_kind(value: Any) -> None:
    model = c50.train(NAMES, TRAINING)

    with pytest.raises(TypeError):
        c50.train(NAMES, TRAINING, value)
    with pytest.raises(TypeError):
        c50.train(NAMES, TRAINING, kind=value)
    with pytest.raises(TypeError):
        c50.train_dense(NAMES, VALUES, CLASSES, value)
    with pytest.raises(TypeError):
        c50.load(NAMES, model.serialized_data, value)
    with pytest.raises(TypeError):
        c50.Model.train(NAMES, TRAINING, value)


@pytest.mark.parametrize(
    "value",
    [1, True, 1.0, None, "stable", c50.ModelKind.RULES, c50.NodeKind.DISCRETE],
    ids=repr,
)
def test_native_boundary_rejects_other_values_for_ties(value: Any) -> None:
    options = c50.Options()

    with pytest.raises(TypeError):
        options.ties = value
    assert options.ties == c50.TieOrder.REFERENCE


@pytest.mark.parametrize("member", _ALL_MEMBERS, ids=str)
def test_json_encodes_a_member_as_its_integer(member: Any) -> None:
    assert json.dumps(member) == str(member.value)


def test_estimator_parameters_carry_no_enumeration_members() -> None:
    from c50.sklearn import C50Classifier
    from sklearn.base import clone

    classifier = C50Classifier(model_kind="rules", ties="reference")
    parameters = classifier.get_params()

    assert not any(isinstance(v, enum.Enum) for v in parameters.values())
    assert clone(classifier).get_params() == parameters
    classifier.set_params(ties="stable", model_kind="tree")
    assert classifier.get_params()["ties"] == "stable"

    features = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    labels = np.asarray(["low", "low", "high", "high"])
    fitted = classifier.fit(features, labels)
    assert fitted.model_.kind is c50.ModelKind.TREE
    restored = pickle.loads(pickle.dumps(fitted))
    assert restored.model_.kind is c50.ModelKind.TREE
    assert list(restored.predict(features)) == list(fitted.predict(features))
