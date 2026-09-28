# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pickle every public binding class at every protocol.

Pickling a pybind11 object that defines no ``__reduce__`` aborts the process at
protocols 0 and 1, so each class runs in its own subprocess. An abort then
fails one test instead of ending the test run.
"""

from __future__ import annotations

import inspect
import pickle
import subprocess
import sys
import textwrap

import c50._c50 as native
import pytest

_CLASSES = sorted(
    name
    for name, value in vars(native).items()
    if inspect.isclass(value) and value.__module__ == native.__name__
)

# Classes whose objects pickle and compare equal after a round trip.
_ROUND_TRIP = {"Model", "ModelKind", "NodeKind", "ConditionKind"}

_SCRIPT = textwrap.dedent(
    """
    import pickle
    import sys

    import c50
    import c50._c50 as native

    NAMES = "low, high.\\n\\nsignal: continuous.\\n"
    TRAINING = "0, low\\n1, low\\n2, high\\n3, high\\n"
    tree = c50.train(NAMES, TRAINING, c50.ModelKind.TREE)
    rules = c50.train(NAMES, TRAINING, c50.ModelKind.RULES)
    tree_node = tree.inspect().trees[0].nodes[0]
    samples = {
        "Model": tree,
        "ModelKind": native.ModelKind.RULES,
        "NodeKind": native.NodeKind.THRESHOLD,
        "ConditionKind": native.ConditionKind.IN_SUBSET,
        "Options": native.Options(),
        "Predictions": tree.predict_details("0.5, ?\\n"),
        "ModelInspection": tree.inspect(),
        "TreeInspection": tree.inspect().trees[0],
        "TreeNode": tree_node,
        "TreeBranch": tree_node.branches[0],
        "SplitCondition": tree_node.branches[0].condition,
        "ContinuousThreshold": tree_node.threshold,
        "RulesetInspection": rules.inspect().rulesets[0],
        "RuleInspection": rules.inspect().rulesets[0].rules[0],
    }
    name = sys.argv[1]
    sample = samples[name]
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        try:
            data = pickle.dumps(sample, protocol)
        except TypeError as error:
            print(protocol, "TypeError", error)
            continue
        restored = pickle.loads(data)
        same = type(restored) is type(sample)
        if name == "Model":
            same = same and restored.serialized_data == sample.serialized_data
        elif name.endswith("Kind") or name == "TieOrder":
            same = same and restored == sample
        print(protocol, "ok" if same else "mismatch")
    """
)


def test_every_class_has_a_sample() -> None:
    """A new binding class must be added to the sample table."""
    probe = subprocess.run(
        [sys.executable, "-c", "import c50._c50"],
        capture_output=True,
        check=True,
    )
    assert probe.returncode == 0
    known = set(_ROUND_TRIP) | {
        "Options",
        "Predictions",
        "ModelInspection",
        "TreeInspection",
        "TreeNode",
        "TreeBranch",
        "SplitCondition",
        "ContinuousThreshold",
        "RulesetInspection",
        "RuleInspection",
        "C50Error",
    }
    assert set(_CLASSES) == known


@pytest.mark.parametrize("name", [c for c in _CLASSES if c != "C50Error"])
def test_pickle_protocols(name: str) -> None:
    """Pickling raises TypeError or round-trips, and never aborts."""
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT, name],
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert len(lines) == pickle.HIGHEST_PROTOCOL + 1
    for protocol, line in enumerate(lines):
        fields = line.split(maxsplit=2)
        assert fields[0] == str(protocol)
        if name in _ROUND_TRIP:
            assert fields[1] == "ok", line
        else:
            assert fields[1] == "TypeError", line
            assert name in fields[2], line


def test_exception_pickles() -> None:
    """The exception class pickles through the ordinary protocol."""
    error = native.C50Error("failed")
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        restored = pickle.loads(pickle.dumps(error, protocol))
        assert type(restored) is native.C50Error
        assert restored.args == ("failed",)
