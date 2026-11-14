# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Native operations must not depend on the host locale."""

from __future__ import annotations

import locale
import re
from collections.abc import Iterator

import pytest
from c50 import Model, ModelKind, Options

NAMES = "low, high.\n\nsignal: continuous.\nweight: continuous.\n"
DATA = (
    "0.25, 1.5, low\n0.5, 1.25, low\n0.75, 2.5, low\n1.25, 0.5, low\n"
    "1.5, 3.25, low\n1.75, 2.75, low\n2.25, 0.75, low\n2.5, 3.5, low\n"
    "3.25, 1.25, high\n3.5, 2.25, high\n3.75, 0.25, high\n4.25, 3.75, high\n"
    "4.5, 1.75, high\n4.75, 2.5, high\n5.25, 0.5, high\n5.5, 3.25, high\n"
)
CASES = "0.5, 1.5, ?\n5.25, 2.5, ?\n2.75, 0.75, ?\n"
CANDIDATES = (
    "de_DE.UTF-8",
    "de_DE.utf8",
    "fr_FR.UTF-8",
    "fr_FR.utf8",
    "de_DE",
    "fr_FR",
    "de_DE.ISO-8859-1",
    "fr_FR.ISO-8859-1",
)


def _decimal_comma_locale() -> str | None:
    saved = locale.setlocale(locale.LC_ALL)
    try:
        for name in CANDIDATES:
            try:
                locale.setlocale(locale.LC_ALL, name)
            except locale.Error:
                continue
            if locale.localeconv()["decimal_point"] == ",":
                return name
    finally:
        locale.setlocale(locale.LC_ALL, saved)
    return None


@pytest.fixture
def decimal_comma() -> Iterator[str]:
    name = _decimal_comma_locale()
    if name is None:
        pytest.skip("no decimal-comma locale is installed")
    saved = locale.setlocale(locale.LC_ALL)
    try:
        yield name
    finally:
        locale.setlocale(locale.LC_ALL, saved)


def _cuts(text: str) -> list[str]:
    return re.findall(r'cut="([^"]*)"', text)


@pytest.mark.parametrize("kind", [ModelKind.TREE, ModelKind.RULES])
def test_operations_ignore_the_process_locale(
    decimal_comma: str, kind: ModelKind
) -> None:
    options = Options()
    options.trials = 2
    options.confidence_factor = 0.375
    options.sample_fraction = 0.75
    locale.setlocale(locale.LC_ALL, "C")
    expected = Model.train(NAMES, DATA, kind, options, split_workers=4)
    expected_labels = expected.predict_details(CASES).labels
    expected_inspection = expected.inspect()

    locale.setlocale(locale.LC_ALL, decimal_comma)
    trained = Model.train(NAMES, DATA, kind, options, split_workers=4)
    assert trained.serialized_data == expected.serialized_data
    if kind is ModelKind.TREE:
        cuts = _cuts(trained.serialized_data)
        assert cuts
        assert all("," not in cut for cut in cuts)

    loaded = Model.load(NAMES, expected.serialized_data, kind)
    assert loaded.serialized_data == expected.serialized_data
    assert loaded.predict_details(CASES).labels == expected_labels
    assert trained.predict_details(CASES).labels == expected_labels
    assert (
        loaded.prepare_predictor().predict_details(CASES).labels
        == expected_labels
    )
    assert loaded.attribute_usage(CASES) == expected.attribute_usage(CASES)
    inspection = loaded.inspect()
    assert [
        tree.nodes[0].threshold.cut
        for tree in inspection.trees
        if tree.nodes[0].threshold
    ] == [
        tree.nodes[0].threshold.cut
        for tree in expected_inspection.trees
        if tree.nodes[0].threshold
    ]
    assert locale.localeconv()["decimal_point"] == ","
