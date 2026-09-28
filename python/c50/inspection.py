"""Readable exports of owned native model-inspection snapshots."""

from __future__ import annotations

import json
from collections.abc import Sequence

from ._c50 import ConditionKind, ModelInspection, NodeKind, SplitCondition


def _names(supplied: Sequence[str] | None, native: list[str]) -> list[str]:
    names = native if supplied is None else list(supplied)
    if len(names) != len(native) or any(not isinstance(name, str) for name in names):
        raise ValueError("display names must be strings matching the schema length")
    return names


def _condition_data(condition: SplitCondition) -> dict[str, object]:
    return {
        "feature": condition.feature,
        "kind": condition.kind.name.lower(),
        "cut": condition.cut,
        "values": condition.values,
        "includes_not_applicable": condition.includes_not_applicable,
    }


def export_json(
    snapshot: ModelInspection,
    *,
    feature_names: Sequence[str] | None = None,
    class_names: Sequence[str] | None = None,
) -> str:
    """Export structural metadata as deterministic JSON, not a loadable model.

    Args:
        snapshot: Owned result of ``Model.inspect()``.
        feature_names: Optional display names in schema order.
        class_names: Optional display names in class-score order.

    Returns:
        Versioned JSON containing all nodes, branches, and structural statistics.

    Raises:
        ValueError: If supplied display names do not match the schema.
    """
    features = _names(feature_names, snapshot.feature_names)
    classes = _names(class_names, snapshot.class_names)
    trees = []
    for tree in snapshot.trees:
        nodes = []
        for node in tree.nodes:
            threshold = node.threshold
            nodes.append(
                {
                    "kind": node.kind.name.lower(),
                    "feature": node.feature,
                    "predicted_class": node.predicted_class,
                    "case_weight": node.case_weight,
                    "class_weights": node.class_weights,
                    "threshold": None if threshold is None else {
                        "cut": threshold.cut,
                        "lower": threshold.lower,
                        "midpoint": threshold.midpoint,
                        "upper": threshold.upper,
                    },
                    "branches": [
                        {"child": branch.child, "condition": _condition_data(branch.condition)}
                        for branch in node.branches
                    ],
                }
            )
        trees.append(
            {
                "nodes": nodes,
                "leaf_count": tree.leaf_count,
                "supported_leaf_count": tree.supported_leaf_count,
                "depth": tree.depth,
                "supported_depth": tree.supported_depth,
                "feature_use": tree.feature_use,
            }
        )
    return json.dumps(
        {
            "format_version": 1,
            "kind": snapshot.kind.name.lower(),
            "feature_names": features,
            "class_names": classes,
            "trees": trees,
        },
        indent=2, ensure_ascii=True, allow_nan=False,
    ) + "\n"


def _condition_text(condition: SplitCondition, names: list[str]) -> str:
    feature = repr(names[condition.feature])
    if condition.kind == ConditionKind.NOT_APPLICABLE:
        return f"{feature} is N/A"
    if condition.kind == ConditionKind.EQUALS:
        return f"{feature} == {condition.values[0]!r}"
    if condition.kind == ConditionKind.IN_SUBSET:
        values = [repr(value) for value in condition.values]
        if condition.includes_not_applicable:
            values.insert(0, "N/A")
        return f"{feature} in {{{', '.join(values)}}}"
    comparison = "<=" if condition.kind == ConditionKind.LESS_EQUAL else ">"
    return f"{feature} {comparison} {condition.cut:.9g}"


def export_text(
    snapshot: ModelInspection,
    *,
    feature_names: Sequence[str] | None = None,
    class_names: Sequence[str] | None = None,
    max_depth: int | None = None,
    include_empty: bool = True,
) -> str:
    """Render an inspected tree or ensemble without parsing serialized text.

    Args:
        snapshot: Owned result of ``Model.inspect()``.
        feature_names: Optional display names in schema order.
        class_names: Optional display names in class-score order.
        max_depth: Maximum displayed edge depth, or ``None`` for all nodes.
        include_empty: Whether to display leaves with zero training support.

    Returns:
        Deterministic text with conditions, stored classes, and weighted support.
        Soft-threshold bounds are reported at their split nodes.

    Raises:
        ValueError: If display names or the depth limit are invalid.
    """
    features = _names(feature_names, snapshot.feature_names)
    classes = _names(class_names, snapshot.class_names)
    if max_depth is not None and (
        isinstance(max_depth, bool)
        or not isinstance(max_depth, int)
        or max_depth < 0
    ):
        raise ValueError("max_depth must be a nonnegative integer or None")
    lines = []
    for number, tree in enumerate(snapshot.trees):
        nodes = tree.nodes
        lines.append(
            f"tree {number}: {len(nodes)} nodes, {tree.supported_leaf_count} supported "
            f"leaves, {tree.leaf_count - tree.supported_leaf_count} empty leaves, "
            f"depth {tree.depth}"
        )
        pending = [(0, 0, "")]
        while pending:
            index, depth, condition = pending.pop()
            node = nodes[index]
            if not include_empty and node.kind == NodeKind.LEAF and node.case_weight == 0:
                continue
            prefix = "  " * depth + (condition + " -> " if condition else "")
            description = (
                f"[{index}] class={classes[node.predicted_class]!r}, "
                f"support={node.case_weight:g}"
            )
            if node.threshold is not None:
                t = node.threshold
                if t.lower != t.upper:
                    description += f", soft=({t.lower:.9g}, {t.midpoint:.9g}, {t.upper:.9g})"
            lines.append(prefix + description)
            if max_depth is not None and depth >= max_depth:
                if node.branches:
                    lines.append("  " * (depth + 1) + "... children omitted")
                continue
            for branch in reversed(node.branches):
                pending.append(
                    (branch.child, depth + 1, _condition_text(branch.condition, features))
                )
    return "\n".join(lines) + "\n"
