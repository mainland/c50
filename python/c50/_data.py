"""Encode array data for the C5.0 native interfaces."""

from __future__ import annotations

import datetime
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Literal, Self, cast

import numpy as np
from numpy.typing import NDArray

type Array = NDArray[Any]
type CategoryKey = tuple[type[object], object]
type UnknownCategoryPolicy = Literal["error", "missing"]


@dataclass(frozen=True)
class Columns:
    """Two-dimensional input stored as one array per feature.

    A column keeps its own dtype, so numeric columns of a mixed table can be
    encoded without converting their values to Python objects.
    """

    arrays: tuple[Array, ...]
    rows: int

    @property
    def shape(self) -> tuple[int, int]:
        """Return the number of rows and features."""
        return self.rows, len(self.arrays)

    def __getitem__(self, rows: slice) -> Self:
        """Return a range of rows from every column."""
        start, stop, step = rows.indices(self.rows)
        return type(self)(
            tuple(array[rows] for array in self.arrays),
            len(range(start, stop, step)),
        )


def dataframe_columns(frame: Any) -> Columns:
    """Convert pandas DataFrame columns without a common dtype.

    NumPy numeric and Boolean columns keep their dtype. Other columns become
    object arrays, as they would in a conversion of the whole frame.
    """
    arrays = []
    for position in range(frame.shape[1]):
        column = frame.iloc[:, position]
        if isinstance(column.dtype, np.dtype) and column.dtype.kind in "iufb":
            arrays.append(column.to_numpy())
        else:
            arrays.append(column.to_numpy(dtype=object))
    return Columns(tuple(arrays), frame.shape[0])


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
        """Return fitted categories, empty for continuous features."""
        return tuple(feature.categories for feature in self.features)

    def names_data(self, class_count: int) -> str:
        """Build a names buffer using internal, delimiter-safe identifiers."""
        if class_count < 2:
            raise ValueError(
                "C5.0 classification requires at least two classes"
            )

        lines = [
            ", ".join(_class_token(index) for index in range(class_count))
            + ".",
            "",
        ]
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
        X: Array | Columns,
        unknown_categories: UnknownCategoryPolicy,
        first_row: int = 0,
    ) -> NDArray[np.float64]:
        """Encode features as the native dense matrix representation.

        ``first_row`` is the position of the first row of ``X`` in the caller's
        data, so that a range error names the row there.
        """
        columns = _as_columns(X)
        if len(columns.arrays) != len(self.features):
            raise ValueError(
                f"X has {len(columns.arrays)} features, but the fitted schema "
                f"expects {len(self.features)}"
            )

        encoded = np.empty(columns.shape, dtype=np.float64, order="C")
        for feature_index, feature in enumerate(self.features):
            column = columns.arrays[feature_index]
            if not feature.categorical and column.dtype.kind in "iuf":
                converted = np.asarray(column, dtype=np.float64)
                _check_continuous_range(converted, feature_index, first_row)
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
    X: Array | Columns,
    categorical_features: Sequence[int | str] | None,
    feature_names: Sequence[str] | None = None,
) -> Schema:
    """Infer or apply categorical feature selection and fit value mappings."""
    columns = _as_columns(X)
    selected = _resolve_categorical_features(
        categorical_features,
        len(columns.arrays),
        feature_names,
    )

    features = []
    for index, column in enumerate(columns.arrays):
        categorical = (
            _infer_categorical(column)
            if selected is None
            else index in selected
        )
        if categorical:
            categories, lookup = _fit_categories(column, index)
            features.append(FeatureSchema(True, categories, lookup))
        else:
            _validate_continuous(column, index)
            features.append(FeatureSchema(False, (), {}))
    return Schema(tuple(features))


# The types of object elements that a cost matrix refuses. NumPy converts a
# `numpy.datetime64` or `numpy.timedelta64` to a float without complaint, as a
# count of units, and a `numpy.complexfloating` to its real part with a
# warning. A `datetime.date`, a `datetime.timedelta`, and a Python `complex`
# fail the conversion, but a subclass or an object that defines `__float__`
# would not, so the scan lists them too. `datetime.date` covers
# `datetime.datetime` and `pandas.Timestamp`, and `datetime.timedelta` covers
# `pandas.Timedelta`.
_NOT_REAL = (
    np.datetime64,
    np.timedelta64,
    datetime.date,
    datetime.timedelta,
    complex,
    np.complexfloating,
)


def encode_costs(class_count: int, cost_matrix: object | None) -> str:
    """Convert a predicted-by-actual cost matrix to C5.0 costs data."""
    if cost_matrix is None:
        return ""

    try:
        array = np.asarray(cost_matrix)
        kind = array.dtype.kind
        costs = None if kind in "cMm" else np.asarray(cost_matrix, dtype=float)
    except MemoryError:
        raise
    except Exception as exc:
        # NumPy raises TypeError, ValueError, or another exception, depending
        # on the object that fails to convert, such as a complex number in an
        # object array, a missing pandas value, or a dict.
        raise ValueError(
            f"cost_matrix must hold real numbers that convert to float: {exc}"
        ) from exc
    # An object array holds elements that the conversion above accepts but that
    # are not costs: a time scalar becomes a count of units, and a complex
    # scalar of NumPy becomes its real part. Refuse them as the arrays of those
    # dtypes are refused. Other elements of the types in `_NOT_REAL` already
    # failed to convert. The matrix holds at most the number of classes squared
    # elements.
    if costs is None or (
        kind == "O" and any(isinstance(x, _NOT_REAL) for x in array.flat)
    ):
        raise ValueError(
            "cost_matrix must hold real numbers, not complex, datetime, or "
            "timedelta values"
        )
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


def _as_columns(X: Array | Columns) -> Columns:
    if isinstance(X, Columns):
        return X
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array")
    return Columns(
        tuple(X[:, index] for index in range(X.shape[1])),
        X.shape[0],
    )


def _resolve_categorical_features(
    categorical_features: Sequence[int | str] | None,
    feature_count: int,
    feature_names: Sequence[str] | None,
) -> set[int] | None:
    if categorical_features is None:
        return None
    if isinstance(categorical_features, (str, bytes)):
        raise TypeError(
            "categorical_features must be a sequence of indices or names"
        )

    name_to_index = (
        {name: index for index, name in enumerate(feature_names)}
        if feature_names is not None
        else None
    )
    selected: set[int] = set()
    # The annotation describes valid input. Entries of any other type reach the
    # checks below.
    for feature in cast("Sequence[object]", categorical_features):
        if isinstance(feature, str):
            if name_to_index is None:
                raise ValueError(
                    "categorical feature names require input with feature "
                    "names"
                )
            try:
                index = name_to_index[feature]
            except KeyError as exc:
                raise ValueError(
                    f"unknown categorical feature name {feature!r}"
                ) from exc
        elif not isinstance(feature, bool) and isinstance(feature, Integral):
            index = int(feature)
            if index < 0 or index >= feature_count:
                raise ValueError(
                    f"categorical feature index {index} is outside "
                    f"[0, {feature_count})"
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
    values = [
        _normalize_scalar(value) for value in column if not _is_missing(value)
    ]
    return bool(values) and not all(_is_number(value) for value in values)


def _fit_categories(
    column: Array,
    feature_index: int,
) -> tuple[tuple[object, ...], dict[CategoryKey, int]]:
    categories: list[object] = []
    lookup: dict[CategoryKey, int] = {}
    for raw_value in column:
        if _is_missing(raw_value):
            continue
        _add_category(
            categories, lookup, _normalize_scalar(raw_value), feature_index
        )
    return tuple(categories), lookup


def _add_category(
    categories: list[object],
    keys: dict[CategoryKey, int],
    value: object,
    feature_index: int,
) -> None:
    """Append a category unless an equal one is already present.

    Equal numbers of different types share a key, so the first of them
    stays.
    """
    key = _category_key(value, feature_index)
    if key not in keys:
        keys[key] = len(categories)
        categories.append(value)


def _check_continuous_range(
    values: NDArray[np.float64], feature_index: int, first_row: int = 0
) -> None:
    """Reject infinite values and values beyond the single-precision range.

    C5.0 stores continuous values as single-precision floats. The native dense
    path rejects a value whose conversion overflows, so this check uses the
    same conversion and fails before the native call with the feature and the
    first offending row, counted from ``first_row``. NaN is a missing value and
    passes.
    """
    with np.errstate(over="ignore"):
        narrowed = values.astype(np.float32)
    bad = np.flatnonzero(np.isinf(narrowed))
    if bad.size == 0:
        return
    position = int(bad[0])
    row = first_row + position
    if math.isinf(values[position]):
        raise ValueError(
            f"feature {feature_index} contains a non-finite value in row {row}"
        )
    raise ValueError(
        f"feature {feature_index} contains a value outside the "
        f"single-precision range in row {row}"
    )


def _validate_continuous(column: Array, feature_index: int) -> None:
    if column.dtype.kind in "iuf":
        _check_continuous_range(
            np.asarray(column, dtype=np.float64), feature_index
        )
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
    if not _is_number(value):
        raise TypeError(
            f"feature {feature_index} is continuous but contains {value!r}"
        )
    try:
        result = float(cast(Real, value))
    except OverflowError:
        raise ValueError(
            f"feature {feature_index} contains a value outside the "
            "single-precision range"
        ) from None
    if not math.isfinite(result):
        raise ValueError(
            f"feature {feature_index} contains non-finite value {value!r}"
        )
    with np.errstate(over="ignore"):
        beyond_single = bool(np.isinf(np.float32(result)))
    if beyond_single:
        raise ValueError(
            f"feature {feature_index} contains a value outside the "
            "single-precision range"
        )
    return result


def _category_key(value: object, feature_index: int) -> CategoryKey:
    try:
        hash(value)
    except TypeError as exc:
        raise TypeError(
            f"categorical feature {feature_index} contains unhashable "
            f"value {value!r}"
        ) from exc
    return _category_group(value), value


def _category_group(value: object) -> type[object]:
    # Real numbers other than bool match by value, whatever their type, so that
    # 1 and 1.0 are one category as they are for scikit-learn. A float32 value
    # equals only the double with the same value.
    if _is_number(value):
        return float
    return type(value)


def _is_number(value: object) -> bool:
    """Return whether ``value`` is a real number that is not a duration.

    ``numpy.timedelta64`` is registered as an integer, but it is a duration, so
    it is categorical like ``datetime.timedelta``.
    """
    return isinstance(value, Real) and not isinstance(
        value, (bool, np.timedelta64)
    )


# The units in which NumPy converts a date or duration to ``int`` when it
# converts to ``object``, so that ``item()`` and ``astype(object)`` lose the
# type. The other units become ``datetime.date``, ``datetime.datetime``, and
# ``datetime.timedelta``. A duration in years or months has no
# ``datetime.timedelta``, so it becomes an ``int`` too.
_INTEGER_TIME_UNITS: dict[str, frozenset[str]] = {
    "M": frozenset({"ns", "ps", "fs", "as"}),
    "m": frozenset({"Y", "M", "ns", "ps", "fs", "as"}),
}


def is_integer_time_dtype(dtype: np.dtype[Any]) -> bool:
    """Return whether NumPy converts ``dtype`` to ``int`` objects.

    This holds for ``datetime64`` in nanoseconds and finer, and for
    ``timedelta64`` in nanoseconds and finer, years, and months. The estimator
    keeps such values as ``numpy.datetime64`` and ``numpy.timedelta64``
    scalars, so that the feature stays categorical.
    """
    units = _INTEGER_TIME_UNITS.get(dtype.kind)
    return units is not None and np.datetime_data(dtype)[0] in units


def _normalize_scalar(value: object) -> object:
    if isinstance(value, (np.datetime64, np.timedelta64)) and (
        is_integer_time_dtype(value.dtype)
    ):
        # item() would return an int, which makes the value continuous.
        return value
    return value.item() if isinstance(value, np.generic) else value


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, (str, bytes)):
        # np.isnan() would reject these by raising, at a much higher cost.
        return False
    if type(value).__module__.startswith("pandas."):
        # Only pandas creates these values, so it is already imported.
        missing = sys.modules["pandas"].isna(value)
        return isinstance(missing, (bool, np.bool_)) and bool(missing)
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
