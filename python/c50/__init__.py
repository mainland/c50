# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Python interface to the C5.0 GPL core.

Exceptions: the functions and methods raise ``ValueError`` for invalid
arguments, options, schemas, data, costs, and model text, including shape and
range errors in dense arrays. They raise ``OSError`` for stream failures,
``MemoryError`` when memory is exhausted, and ``C50Error`` for a schema feature
or operation that the library does not support and for internal failures.
Arguments of the wrong type raise ``TypeError``, as pybind11 reports it. A
count argument (``split_workers``, ``max_tree_depth``, ``Options.trials``,
``Options.random_seed``, and the ``max_depth`` of ``export_text`` and of
``C50Classifier.export_text``) accepts an ``int`` or any other object with
``__index__``, such as a NumPy integer, and raises ``TypeError`` for a
``bool``, a NumPy boolean, or a number without ``__index__``, such as a
``float``, ``Fraction``, or ``Decimal``. An exception that the type check or
``__index__`` raises propagates. The estimator's integer parameters follow a
different rule, which ``docs/estimator.md`` states. A number that its native
type cannot hold, such as a negative ``split_workers`` or an ``Options`` field
above the unsigned range, raises ``ValueError``, and text that UTF-8 cannot
encode raises ``UnicodeEncodeError``, a ``ValueError``. A training call with
``split_workers`` above one raises a plain ``RuntimeError``, not ``C50Error``,
when the system cannot start a worker thread. The text of a message is not part
of the stable interface.

Text inputs end a call at the first bad value, such as a discrete value that
the names file does not declare or a nonfinite number, and the call raises
``ValueError`` with the line number. The ``c5.0`` program instead reports each
bad value, replaces it with ``?``, and continues until ten errors.
"""

from ._c50 import (
    C50Error,
    ConditionKind,
    ContinuousThreshold,
    Model,
    ModelInspection,
    ModelKind,
    NodeKind,
    Options,
    Predictions,
    Predictor,
    RuleInspection,
    RulesetInspection,
    SplitCondition,
    TieOrder,
    TreeBranch,
    TreeInspection,
    TreeNode,
    load,
    train,
    train_dense,
)
from .inspection import export_json, export_text

# Version of the state that ``Model`` pickles. Increase it when the state
# changes, and keep ``_restore_model`` loading every earlier version.
_MODEL_PICKLE_VERSION = 1


def _restore_model(
    version: int, kind: str, names: str, serialized: str, costs: str
) -> Model:
    """Rebuild a pickled ``Model`` from its serialized text.

    Pickles name this function, not ``c50._c50``, so that they do not depend on
    the extension module's layout or on pybind11's enumeration protocol.

    Args:
        version: The pickle format version.
        kind: ``"tree"`` or ``"rules"``.
        names: The retained names-file contents.
        serialized: The retained tree or rules-file contents.
        costs: The retained costs-file contents, or an empty string.

    Returns:
        The reconstructed model.

    Raises:
        ValueError: If the version or kind is unknown, or the text is not a
            valid classifier.
        TypeError: If an element has the wrong type.
    """
    if type(version) is not int:
        raise TypeError("model pickle version must be an integer")
    if version != _MODEL_PICKLE_VERSION:
        raise ValueError(
            f"unsupported model pickle version {version}; "
            f"this version of c50 reads version {_MODEL_PICKLE_VERSION}"
        )
    for name, value in (
        ("kind", kind),
        ("names", names),
        ("serialized model", serialized),
        ("costs", costs),
    ):
        if not isinstance(value, str):
            raise TypeError(f"model pickle {name} must be a string")
    if kind == "tree":
        model_kind = ModelKind.TREE
    elif kind == "rules":
        model_kind = ModelKind.RULES
    else:
        raise ValueError(f"unknown model pickle kind {kind!r}")
    # The pickle holds text that a trained or loaded model wrote. Loading it
    # runs trusted code anyway, so it does not limit the tree depth.
    return load(names, serialized, model_kind, costs, max_tree_depth=None)


_ENUMERATIONS: dict[
    str, type[ModelKind | TieOrder | NodeKind | ConditionKind]
] = {
    enumeration.__name__: enumeration
    for enumeration in (ModelKind, TieOrder, NodeKind, ConditionKind)
}


def _restore_enum(
    type_name: str, member: str
) -> ModelKind | TieOrder | NodeKind | ConditionKind:
    """Return an enumeration member from its type name and member name.

    Pickles of the enumerations name this function, not ``c50._c50``, and
    identify the member by name, so that they do not depend on the extension
    module's layout or on the integer values of the members. Unpickling returns
    the member itself.

    Args:
        type_name: ``"ModelKind"``, ``"TieOrder"``, ``"NodeKind"``, or
            ``"ConditionKind"``.
        member: The name of a member of that enumeration.

    Returns:
        The member.

    Raises:
        ValueError: If the enumeration or the member is unknown.
        TypeError: If an argument is not a string.
    """
    # The annotations describe valid arguments, and a pickle may hold others.
    supplied: tuple[object, object] = (type_name, member)
    if not all(isinstance(name, str) for name in supplied):
        raise TypeError("enumeration pickle names must be strings")
    enumeration = _ENUMERATIONS.get(type_name)
    if enumeration is None:
        raise ValueError(f"unknown enumeration {type_name!r} in pickle")
    try:
        return enumeration.__members__[member]
    except KeyError:
        raise ValueError(
            f"{type_name} has no member {member!r}; the pickle comes from a "
            "different version of c50"
        ) from None


__all__ = [
    "C50Error",
    "ConditionKind",
    "ContinuousThreshold",
    "Model",
    "ModelInspection",
    "ModelKind",
    "NodeKind",
    "Options",
    "Predictions",
    "Predictor",
    "RuleInspection",
    "RulesetInspection",
    "SplitCondition",
    "TieOrder",
    "TreeBranch",
    "TreeInspection",
    "TreeNode",
    "export_json",
    "export_text",
    "load",
    "train",
    "train_dense",
]
