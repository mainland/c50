"""Python interface to the C5.0 GPL core."""

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
    SplitCondition,
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
    "SplitCondition",
    "TreeBranch",
    "TreeInspection",
    "TreeNode",
    "export_json",
    "export_text",
    "load",
    "train",
    "train_dense",
]
