# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""``c50.NOT_APPLICABLE`` is one object that nothing else equals."""

from __future__ import annotations

import copy
import numbers
import operator
import pickle
import pickletools

import c50
import c50.sklearn
import numpy as np
import pandas as pd

NA = c50.NOT_APPLICABLE


def test_name_and_repr() -> None:
    assert repr(NA) == "c50.NOT_APPLICABLE"
    assert str(NA) == "c50.NOT_APPLICABLE"
    assert "NOT_APPLICABLE" in c50.__all__
    assert "NOT_APPLICABLE" in c50.sklearn.__all__
    assert c50.sklearn.NOT_APPLICABLE is NA


def test_there_is_one_instance() -> None:
    assert type(NA)() is NA
    assert copy.copy(NA) is NA
    assert copy.deepcopy(NA) is NA
    assert copy.deepcopy([NA, {"k": NA}])[1]["k"] is NA


def test_pickle_names_the_public_attribute() -> None:
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        data = pickle.dumps(NA, protocol)
        assert pickle.loads(data) is NA
        assert pickle.loads(pickle.dumps([NA, NA], protocol))[1] is NA
        strings = [
            argument
            for _, argument, _ in pickletools.genops(data)
            if isinstance(argument, str)
        ]
        # The pickle stays valid when private modules move.
        assert not any("_sentinel" in text for text in strings)
        assert not any(text.startswith("c50._") for text in strings)
        assert "NOT_APPLICABLE" in " ".join(strings)


def test_equality_and_hashing() -> None:
    assert operator.eq(NA, NA)
    others = (
        None,
        np.nan,
        0,
        1,
        0.0,
        False,
        "N/A",
        "NOT_APPLICABLE",
        object(),
    )
    for other in others:
        assert operator.ne(NA, other)
        assert operator.ne(other, NA)
    assert hash(NA) == hash(NA)
    assert {NA: 1}[NA] == 1
    assert len({NA, NA, None}) == 2


def test_not_missing_and_not_a_number() -> None:
    assert not pd.isna(NA)
    assert not pd.isna(pd.Series([NA, 1.0], dtype=object)).any()
    assert not isinstance(NA, numbers.Number)
    assert not isinstance(NA, numbers.Real)
