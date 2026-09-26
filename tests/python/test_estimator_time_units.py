# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Test time scalar normalization in the feature encoder."""

from __future__ import annotations

import numpy as np
import pytest
from c50._data import fit_schema, is_integer_time_dtype

ALL_UNITS = (
    *("Y", "M", "W", "D", "h", "m", "s", "ms", "us"),
    *("ns", "ps", "fs", "as"),
)

# The units of each kind that NumPy converts to `int` under `astype(object)`.
# `test_the_scalar_units_are_the_ones_that_numpy_converts_to_int` checks them
# against NumPy, and the other units are the ones that were categorical before.
SCALAR_UNITS = {
    "date": ("ns", "ps", "fs", "as"),
    "duration": ("Y", "M", "ns", "ps", "fs", "as"),
}


@pytest.mark.parametrize("kind", ["M", "m"])
@pytest.mark.parametrize("unit", ALL_UNITS)
def test_the_scalar_units_are_the_ones_that_numpy_converts_to_int(
    kind: str, unit: str
) -> None:
    dtype = np.dtype(f"{kind}8[{unit}]")
    converted = np.zeros(1, dtype=dtype).astype(object)[0]
    name = "date" if kind == "M" else "duration"
    assert isinstance(converted, int) == (unit in SCALAR_UNITS[name])
    assert is_integer_time_dtype(dtype) == (unit in SCALAR_UNITS[name])


@pytest.mark.parametrize("dtype", ["datetime64[ns]", "timedelta64[M]"])
def test_schema_preserves_time_scalars_and_missing_values(dtype: str) -> None:
    X = np.array([[1], [2], ["NaT"]], dtype=dtype)

    schema = fit_schema(X, categorical_features=None)

    assert schema.categorical_indices == (0,)
    assert all(type(value) is type(X[0, 0]) for value in schema.categories[0])
    np.testing.assert_array_equal(
        schema.dense_data(X, "error").ravel(), [0, 1, np.nan]
    )
    with pytest.raises(TypeError, match="feature 0 is continuous"):
        fit_schema(X, categorical_features=())
