# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Dense calls give the same result for unaligned arrays as for aligned ones.

An array from ``numpy.frombuffer`` at an odd offset is C-contiguous and has the
native dtype, but its data does not start at a multiple of the element size.
The dense calls convert such an array once instead of reading it through a
typed pointer. Without a sanitizer a misaligned read gives the same result on
x86-64, so these tests show the equality and the sanitizer builds show the
absence of the misaligned access.
"""

from __future__ import annotations

import c50
import numpy as np
import pytest
from model_text import without_date
from numpy.typing import DTypeLike, NDArray

NAMES = "no, yes.\n\nvalue: continuous.\n"
VALUES = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
CLASSES = np.array([0, 0, 0, 1, 1, 1])


def unaligned(
    source: NDArray[np.generic], dtype: DTypeLike
) -> NDArray[np.generic]:
    """Return a C-contiguous copy of source that starts at an odd address."""
    shape = source.shape
    flat = np.asarray(source, dtype=dtype).ravel()
    buffer = bytearray(1 + flat.nbytes)
    result = np.frombuffer(buffer, dtype=dtype, offset=1, count=flat.size)
    result = result.reshape(shape)
    result.flags.writeable = True
    result[...] = source
    assert result.flags.c_contiguous
    assert not result.flags.aligned
    return result


def stable(model: c50.Model) -> str:
    return without_date(model.serialized_data)


@pytest.mark.parametrize("copy", [True, False])
@pytest.mark.parametrize("class_dtype", [np.int64, np.uint64])
def test_train_dense_accepts_unaligned_arrays(
    copy: bool, class_dtype: type[np.generic]
) -> None:
    expected = c50.train_dense(NAMES, VALUES, CLASSES.astype(class_dtype))
    model = c50.train_dense(
        NAMES,
        unaligned(VALUES, np.float64),
        unaligned(CLASSES, class_dtype),
        copy=copy,
    )
    assert stable(model) == stable(expected)


@pytest.mark.parametrize("copy", [True, False])
def test_train_dense_class_indices_alone_may_be_unaligned(copy: bool) -> None:
    expected = c50.train_dense(NAMES, VALUES, CLASSES)
    model = c50.train_dense(
        NAMES, VALUES, unaligned(CLASSES, np.int64), copy=copy
    )
    assert stable(model) == stable(expected)


def test_model_train_dense_accepts_unaligned_arrays() -> None:
    expected = c50.Model.train_dense(NAMES, VALUES, CLASSES)
    model = c50.Model.train_dense(
        NAMES,
        unaligned(VALUES, np.float64),
        unaligned(CLASSES, np.int64),
        copy=False,
    )
    assert stable(model) == stable(expected)


def test_prediction_and_usage_accept_unaligned_values() -> None:
    model = c50.train_dense(NAMES, VALUES, CLASSES)
    queries = np.array([[0.0], [2.5], [5.0], [np.nan]])
    odd = unaligned(queries, np.float64)
    predictor = model.prepare_predictor()
    expected = model.predict_details_dense(queries)
    for actual in (
        model.predict_details_dense(odd),
        predictor.predict_details_dense(odd),
    ):
        assert actual.class_indices == expected.class_indices
        assert actual.confidences == expected.confidences
        assert actual.scores == expected.scores
    assert model.attribute_usage_dense(odd) == model.attribute_usage_dense(
        queries
    )


def test_one_row_unaligned_values() -> None:
    model = c50.train_dense(NAMES, VALUES, CLASSES)
    one = np.array([[4.0]])
    assert (
        model.predict_details_dense(unaligned(one, np.float64)).class_indices
        == model.predict_details_dense(one).class_indices
    )
