# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The tree depth limit of ``load`` and the models that skip it."""

from __future__ import annotations

import copy
import pickle

import c50
import pytest

NAMES = "no, yes.\nx: continuous.\n"
HEADER = 'id="See5/C5.0 2.07 GPL Edition"\nentries="1"\n'
NO_LEAF = 'type="0" class="no" freq="2,0"\n'
YES_LEAF = 'type="0" class="yes" freq="0,2"\n'
NODE = 'type="2" class="no" att="x" forks="3" cut="0" freq="1,1"\n'

DEFAULT_LIMIT = 1024
DEEP = DEFAULT_LIMIT + 76


def tree_text(levels: int) -> str:
    """Return a tree of ``levels`` levels.

    Every node tests ``x`` against 0. A case with ``x = -1`` reaches the "yes"
    leaf on the deepest level, and a case with ``x = 5`` stops at the "no" leaf
    under the root.
    """
    internal = levels - 1
    return HEADER + (NODE + NO_LEAF) * internal + YES_LEAF + NO_LEAF * internal


def check_usable(model: c50.Model, levels: int) -> None:
    """Predict, inspect, and prepare a predictor, each of which parses."""
    assert model.predict("-1, ?\n") == ["yes"]
    assert model.predict("5, ?\n") == ["no"]
    assert len(model.inspect().trees[0].nodes) == 3 * (levels - 1) + 1
    predictor = model.prepare_predictor()
    assert predictor.predict_details("-1, ?\n").labels == ["yes"]


def test_default_limit_counts_levels() -> None:
    model = c50.load(NAMES, tree_text(DEFAULT_LIMIT))
    assert model.predict("-1, ?\n") == ["yes"]
    with pytest.raises(ValueError, match="max_tree_depth"):
        c50.load(NAMES, tree_text(DEFAULT_LIMIT + 1))
    with pytest.raises(ValueError, match="max_tree_depth"):
        c50.Model.load(NAMES, tree_text(DEFAULT_LIMIT + 1))


def test_explicit_limit() -> None:
    model = c50.load(NAMES, tree_text(3), max_tree_depth=3)
    assert model.predict("-1, ?\n") == ["yes"]
    with pytest.raises(ValueError, match="max_tree_depth"):
        c50.load(NAMES, tree_text(4), max_tree_depth=3)
    with pytest.raises(ValueError, match="max_tree_depth"):
        c50.Model.load(NAMES, tree_text(4), max_tree_depth=3)


@pytest.mark.parametrize("loader", [c50.load, c50.Model.load])
def test_deep_tree_loads_without_a_limit(loader: object) -> None:
    text = tree_text(DEEP)
    with pytest.raises(ValueError, match="max_tree_depth"):
        c50.load(NAMES, text)
    model = loader(NAMES, text, max_tree_depth=None)  # type: ignore[operator]
    check_usable(model, DEEP)
    # A limit that the tree fits keeps the model usable on later parses.
    sized = loader(NAMES, text, max_tree_depth=DEEP)  # type: ignore[operator]
    check_usable(sized, DEEP)
    with pytest.raises(ValueError, match="max_tree_depth"):
        loader(NAMES, text, max_tree_depth=DEEP - 1)  # type: ignore[operator]


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_deep_model_pickles(protocol: int) -> None:
    model = c50.load(NAMES, tree_text(DEEP), max_tree_depth=None)
    restored = pickle.loads(pickle.dumps(model, protocol))
    assert restored.serialized_data == model.serialized_data
    check_usable(restored, DEEP)


def test_deep_model_copies() -> None:
    model = c50.load(NAMES, tree_text(DEEP), max_tree_depth=DEEP)
    # Unpickling runs trusted code, so a copy has no limit, not the original's.
    check_usable(copy.deepcopy(model), DEEP)
    check_usable(copy.copy(model), DEEP)


def test_unpickling_ignores_the_default_limit() -> None:
    model = c50.load(NAMES, tree_text(DEEP), max_tree_depth=None)
    restored = c50._restore_model(
        c50._MODEL_PICKLE_VERSION,
        "tree",
        model.names_data,
        model.serialized_data,
        "",
    )
    check_usable(restored, DEEP)


@pytest.mark.parametrize("value", [0, -1, -(2**70), 2**70])
def test_unusable_limit_is_a_value_error(value: int) -> None:
    for loader in (c50.load, c50.Model.load):
        with pytest.raises(ValueError, match="max_tree_depth"):
            loader(NAMES, tree_text(1), max_tree_depth=value)


@pytest.mark.parametrize("value", [True, False, 1.0, "3", b"3", [3], 3 + 0j])
def test_limit_must_be_an_int(value: object) -> None:
    for loader in (c50.load, c50.Model.load):
        with pytest.raises(TypeError):
            loader(NAMES, tree_text(1), max_tree_depth=value)  # type: ignore[arg-type]


def test_limit_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        c50.load(  # type: ignore[call-arg, misc]
            NAMES, tree_text(1), c50.ModelKind.TREE, "", None
        )


def test_zero_is_rejected_for_rules_models_too() -> None:
    with pytest.raises(ValueError, match="max_tree_depth"):
        c50.load(NAMES, tree_text(1), c50.ModelKind.RULES, max_tree_depth=0)


def test_trained_model_round_trips() -> None:
    names = "cls.\nx: continuous.\ncls: no, yes.\n"
    data = "0, no\n1, yes\n2, no\n3, yes\n4, no\n5, yes\n"
    model = c50.train(names, data)
    restored = pickle.loads(pickle.dumps(model))
    assert restored.predict("1, ?\n") == model.predict("1, ?\n")
