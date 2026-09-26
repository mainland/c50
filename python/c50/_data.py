"""Encode array data for the C5.0 native interfaces."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray


type Array = NDArray[Any]
type CategoryKey = tuple[type[object], object]
type UnknownCategoryPolicy = Literal["error", "missing"]


@dataclass(frozen=True)
class FeatureSchema:
    """Encoding state for one input feature."""

    categorical: bool
    categories: tuple[object, ...]
    lookup: dict[CategoryKey, int]


@dataclass(frozen=True)
class Schema:
    """Fitted conversion between array values and C5.0 case fields."""

    features: tuple[FeatureSchema, ...]

    @property
    def categorical_indices(self) -> tuple[int, ...]:
        """Return the indices of categorical features."""
        return tuple(
            index
            for index, feature in enumerate(self.features)
            if feature.categorical
        )

    @property
    def categories(self) -> tuple[tuple[object, ...], ...]:
        """Return fitted categories, with empty tuples for continuous features."""
        return tuple(feature.categories for feature in self.features)

    def names_data(self, class_count: int) -> str:
        """Build a names buffer using internal, delimiter-safe identifiers."""
        if class_count < 2:
            raise ValueError("C5.0 classification requires at least two classes")

        lines = [", ".join(_class_token(index) for index in range(class_count)) + ".", ""]
        for index, feature in enumerate(self.features):
            if feature.categorical:
                value_count = max(2, len(feature.categories))
                values = ", ".join(
                    _category_token(value_index)
                    for value_index in range(value_count)
                )
                lines.append(f"feature_{index}: {values}.")
            else:
                lines.append(f"feature_{index}: continuous.")
        return "\n".join(lines) + "\n"

    def dense_data(
        self,
        X: Array,
        unknown_categories: UnknownCategoryPolicy,
    ) -> NDArray[np.float64]:
        """Encode features as the native dense matrix representation."""
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array")
        if X.shape[1] != len(self.features):
            raise ValueError(
                f"X has {X.shape[1]} features, but the fitted schema expects "
                f"{len(self.features)}"
            )

        encoded = np.empty(X.shape, dtype=np.float64, order="C")
        for feature_index, feature in enumerate(self.features):
            column = X[:, feature_index]
            if not feature.categorical and column.dtype.kind in "iuf":
                converted = np.asarray(column, dtype=np.float64)
                if np.any(np.isinf(converted)):
                    raise ValueError(
                        f"feature {feature_index} contains a non-finite value"
                    )
                encoded[:, feature_index] = converted
                continue

            for row_index, value in enumerate(column):
                encoded[row_index, feature_index] = _encode_dense_value(
                    feature,
                    value,
                    feature_index,
                    unknown_categories,
                )
        return encoded


def fit_schema(
    X: Array,
    categorical_features: Sequence[int | str] | None,
    feature_names: Sequence[str] | None = None,
) -> Schema:
    """Infer or apply categorical feature selection and fit value mappings."""
    selected = _resolve_categorical_features(
        categorical_features,
        X.shape[1],
        feature_names,
    )

    features = []
    for index in range(X.shape[1]):
        column = X[:, index]
        categorical = (
            _infer_categorical(column) if selected is None else index in selected
        )
        if categorical:
            categories, lookup = _fit_categories(column, index)
            features.append(FeatureSchema(True, categories, lookup))
        else:
            _validate_continuous(column, index)
            features.append(FeatureSchema(False, (), {}))
    return Schema(tuple(features))


def encode_costs(class_count: int, cost_matrix: object | None) -> str:
    """Convert a predicted-by-actual cost matrix to C5.0 costs data."""
    if cost_matrix is None:
        return ""

    costs = np.asarray(cost_matrix, dtype=float)
    expected_shape = (class_count, class_count)
    if costs.shape != expected_shape:
        raise ValueError(
            f"cost_matrix has shape {costs.shape}, expected {expected_shape}"
        )
    if not np.all(np.isfinite(costs)) or np.any(costs < 0):
        raise ValueError("cost_matrix entries must be finite and nonnegative")
    if not np.all(costs.diagonal() == 0):
        raise ValueError("cost_matrix diagonal entries must be zero")

    lines = []
    for predicted in range(class_count):
        for actual in range(class_count):
            if predicted == actual or costs[predicted, actual] == 1:
                continue
            lines.append(
                f"{_class_token(predicted)}, {_class_token(actual)}: "
                f"{_format_number(costs[predicted, actual])}"
            )
    return "\n".join(lines) + ("\n" if lines else "")


def _resolve_categorical_features(
    categorical_features: Sequence[int | str] | None,
    feature_count: int,
    feature_names: Sequence[str] | None,
) -> set[int] | None:
    if categorical_features is None:
        return None
    if isinstance(categorical_features, (str, bytes)):
        raise TypeError("categorical_features must be a sequence of indices or names")

    name_to_index = (
        {name: index for index, name in enumerate(feature_names)}
        if feature_names is not None
        else None
    )
    selected: set[int] = set()
    for feature in categorical_features:
        if isinstance(feature, str):
            if name_to_index is None:
                raise ValueError(
                    "categorical feature names require input with feature names"
                )
            try:
                index = name_to_index[feature]
            except KeyError as exc:
                raise ValueError(f"unknown categorical feature name {feature!r}") from exc
        elif isinstance(feature, Integral) and not isinstance(feature, bool):
            index = int(feature)
            if index < 0 or index >= feature_count:
                raise ValueError(
                    f"categorical feature index {index} is outside [0, {feature_count})"
                )
        else:
            raise TypeError(
                "categorical_features entries must be integer indices or names"
            )
        if index in selected:
            raise ValueError(f"categorical feature {feature!r} is repeated")
        selected.add(index)
    return selected


def _infer_categorical(column: Array) -> bool:
    if column.dtype.kind in "iuf":
        return False
    if column.dtype.kind == "b":
        return True
    values = [_normalize_scalar(value) for value in column if not _is_missing(value)]
    return bool(values) and not all(
        isinstance(value, Real) and not isinstance(value, bool) for value in values
    )


def _fit_categories(
    column: Array,
    feature_index: int,
) -> tuple[tuple[object, ...], dict[CategoryKey, int]]:
    categories: list[object] = []
    lookup: dict[CategoryKey, int] = {}
    for raw_value in column:
        if _is_missing(raw_value):
            continue
        value = _normalize_scalar(raw_value)
        key = _category_key(value, feature_index)
        if key not in lookup:
            lookup[key] = len(categories)
            categories.append(value)
    return tuple(categories), lookup


def _validate_continuous(column: Array, feature_index: int) -> None:
    if column.dtype.kind in "iuf":
        if np.any(np.isinf(np.asarray(column, dtype=np.float64))):
            raise ValueError(f"feature {feature_index} contains a non-finite value")
        return
    for value in column:
        if _is_missing(value):
            continue
        _continuous_value(value, feature_index)


def _encode_dense_value(
    feature: FeatureSchema,
    raw_value: object,
    feature_index: int,
    unknown_categories: UnknownCategoryPolicy,
) -> float:
    if _is_missing(raw_value):
        return math.nan
    if not feature.categorical:
        return _continuous_value(raw_value, feature_index)

    value = _normalize_scalar(raw_value)
    value_index = feature.lookup.get(_category_key(value, feature_index))
    if value_index is not None:
        return float(value_index)
    if unknown_categories == "missing":
        return math.nan
    raise ValueError(
        f"X contains unseen category {value!r} in feature {feature_index}"
    )


def _continuous_value(value: object, feature_index: int) -> float:
    value = _normalize_scalar(value)
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(
            f"feature {feature_index} is continuous but contains {value!r}"
        )
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(
            f"feature {feature_index} contains non-finite value {value!r}"
        )
    return result


def _category_key(value: object, feature_index: int) -> CategoryKey:
    try:
        hash(value)
    except TypeError as exc:
        raise TypeError(
            f"categorical feature {feature_index} contains unhashable value {value!r}"
        ) from exc
    return type(value), value


def _normalize_scalar(value: object) -> object:
    return value.item() if isinstance(value, np.generic) else value


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        missing = np.isnan(cast(Any, value))
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _format_number(value: float | np.floating[Any]) -> str:
    return format(float(value), ".17g")


def _class_token(index: int) -> str:
    return f"class_{index}"


def _category_token(index: int) -> str:
    return f"value_{index}"
