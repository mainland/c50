# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Test the log encoding of the slow property hunt."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import c50
import numpy as np
import pytest
from test_dense_properties import Dataset

SCRIPT = Path(__file__).resolve().parents[1] / "slow" / "property_hunt.py"

# The script imports the strategy module by name, so it needs this directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
SPECIFICATION = importlib.util.spec_from_file_location("property_hunt", SCRIPT)
assert SPECIFICATION is not None
assert SPECIFICATION.loader is not None
property_hunt = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(property_hunt)


def make_dataset() -> Dataset:
    """Build a dataset with a missing value in each dense array."""
    training = np.asarray(
        [[1.5, np.nan], [np.nan, 2.0], [0.5, 1.0]], dtype=np.float64
    )
    return Dataset(
        names="a, b.\n\nf0: continuous.\nf1: continuous.\n",
        text_training="1.5, ?, a\n?, 2, b\n0.5, 1, a\n",
        text_cases="1.5, ?, ?\n?, 2, ?\n0.5, 1, ?\n",
        dense_training=training,
        dense_cases=training.copy(),
        class_indices=np.asarray([0, 1, 0], dtype=np.uintp),
        kind=c50.ModelKind.TREE,
        subset_splits=False,
        winnow=False,
        global_pruning=True,
        probabilistic_thresholds=False,
        trials=1,
        minimum_cases=2.0,
        confidence_factor=0.25,
        sample_fraction=0.0,
        random_seed=0,
        costs="",
    )


def reject_constant(token: str) -> Any:
    raise ValueError(f"nonstandard JSON constant {token}")


def test_record_with_a_missing_value_is_standard_json(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    dataset = make_dataset()
    with path.open("w") as log:
        property_hunt.record(log, dataset)
    line = path.read_text().splitlines()[-1]
    fields = json.loads(line, parse_constant=reject_constant)
    assert fields["dense_training"] == [[1.5, None], [None, 2.0], [0.5, 1.0]]


def test_load_last_reads_null_and_nan_to_the_same_dataset(
    tmp_path: Path,
) -> None:
    dataset = make_dataset()
    path = tmp_path / "log.jsonl"
    with path.open("w") as log:
        property_hunt.record(log, dataset)
    new = property_hunt.load_last(path)

    old_path = tmp_path / "old.jsonl"
    old_line = path.read_text().replace("null", "NaN")
    assert "NaN" in old_line
    old_path.write_text(old_line)
    old = property_hunt.load_last(old_path)

    for loaded in (new, old):
        assert loaded.dense_training.dtype == np.float64
        np.testing.assert_array_equal(
            loaded.dense_training, dataset.dense_training
        )
        np.testing.assert_array_equal(loaded.dense_cases, dataset.dense_cases)
        np.testing.assert_array_equal(
            loaded.class_indices, dataset.class_indices
        )
        assert loaded.kind is dataset.kind


@pytest.mark.parametrize("flag", ["-O", "-OO"])
@pytest.mark.parametrize("mode", ["hunt", "replay"])
def test_optimized_python_is_refused_before_a_log_is_opened(
    tmp_path: Path, flag: str, mode: str
) -> None:
    log = tmp_path / "log.jsonl"
    if mode == "hunt":
        arguments = ["--log", str(log), "--examples", "1"]
    else:
        log.write_text("")
        arguments = ["--replay", str(log)]
    environment = dict(os.environ)
    package_parent = str(Path(c50.__file__).resolve().parents[1])
    environment["PYTHONPATH"] = os.pathsep.join(
        [package_parent, environment.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [sys.executable, flag, str(SCRIPT), *arguments],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert result.returncode == 2, result.stderr
    assert "optimized" in result.stderr
    if mode == "hunt":
        assert not log.exists()
    else:
        assert log.read_text() == ""
