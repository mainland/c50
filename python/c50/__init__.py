"""Python interface to the C5.0 GPL core.

Exceptions: the functions and methods raise ``ValueError`` for invalid
arguments, options, schemas, data, costs, and model text, including shape and
range errors in dense arrays. They raise ``OSError`` for stream failures,
``MemoryError`` when memory is exhausted, and ``C50Error`` for a schema feature
or operation that the library does not support and for internal failures.
Arguments of the wrong type raise ``TypeError``, as pybind11 reports it. A
number that its native type cannot hold, such as a negative ``split_workers``
or an ``Options`` field above the unsigned range, raises ``ValueError``, and
text that UTF-8 cannot encode raises ``UnicodeEncodeError``, a ``ValueError``.
A training call with ``split_workers`` above one raises a plain
``RuntimeError``, not ``C50Error``, when the system cannot start a worker
thread. The text of a message is not part of the stable interface.

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
    return load(names, serialized, model_kind, costs)


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
