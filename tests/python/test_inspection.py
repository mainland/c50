"""Contracts for owned inspection metadata and readable exports."""

from __future__ import annotations

import gc
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from c50 import ConditionKind, Model, NodeKind, export_json, export_text


NAMES = "low, high.\nsignal: continuous.\n"
DATA = "0, low\n1, low\n2, low\n3, high\n4, high\n5, high\n"


def test_snapshot_lifetime_and_readonly_metadata() -> None:
    model = Model.train(NAMES, DATA)
    snapshot = model.inspect()
    del model
    gc.collect()

    tree = snapshot.trees[0]
    assert snapshot.class_names == ["low", "high"]
    assert snapshot.feature_names == ["signal"]
    assert len(tree.nodes) == 4
    assert tree.supported_leaf_count == 2
    assert tree.leaf_count == 3
    assert tree.depth == tree.supported_depth == 1
    assert tree.feature_use == [1]
    root = tree.nodes[0]
    assert root.kind == NodeKind.THRESHOLD
    assert root.feature == 0
    assert root.class_weights == [3, 3]
    assert root.case_weight == 6
    assert root.threshold is not None
    assert [branch.child for branch in root.branches] == [1, 2, 3]
    assert root.branches[0].condition.kind == ConditionKind.NOT_APPLICABLE
    copied_names = snapshot.feature_names
    copied_names[0] = "changed"
    assert snapshot.feature_names == ["signal"]
    with pytest.raises(AttributeError):
        root.case_weight = 99


def test_json_and_text_preserve_structure_and_escape_display_names() -> None:
    model = Model.train(NAMES, DATA)
    before = model.predict_proba("0, ?\n5, ?\n?, ?\n")
    snapshot = model.inspect()
    exported = export_json(snapshot, feature_names=['received "power"\n'])
    document = json.loads(exported)
    assert document["format_version"] == 1
    assert document["feature_names"] == ['received "power"\n']
    assert len(document["trees"][0]["nodes"]) == 4
    assert document["trees"][0]["nodes"][0]["branches"][1]["condition"]["kind"] == "less_equal"
    assert export_json(snapshot) == export_json(model.inspect())
    assert "empty leaves" in export_text(snapshot)
    assert "is N/A" in export_text(snapshot)
    assert "is N/A" not in export_text(snapshot, include_empty=False)
    assert "children omitted" in export_text(snapshot, max_depth=0)
    assert "'received power' <=" in export_text(snapshot, feature_names=["received power"])
    assert model.predict_proba("0, ?\n5, ?\n?, ?\n") == before
    with pytest.raises(ValueError, match="schema length"):
        export_json(snapshot, feature_names=[])
    with pytest.raises(ValueError, match="max_depth"):
        export_text(snapshot, max_depth=-1)


def test_concurrent_inspection_matches_loaded_model() -> None:
    model = Model.train(NAMES, DATA)
    loaded = Model.load(NAMES, model.serialized_data)
    expected = export_json(loaded.inspect())
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: export_json(model.inspect()), range(8)))
    assert results == [expected] * 8
