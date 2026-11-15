# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Dense input with a not-applicable mask matches text input with N/A."""

from __future__ import annotations

from typing import Any

import c50
import numpy as np
import pytest
from numpy.typing import NDArray

NAMES = """lo, mid, hi.
x: continuous.
y: continuous.
g: a, b, c.
h: p, q, r, s, t.
"""
CLASSES = ("lo", "mid", "hi")
G_LEVELS = ("a", "b", "c")
H_LEVELS = ("p", "q", "r", "s", "t")
COSTS = "mid, lo: 3\nhi, lo: 2\n"


def table(
    rows: int, seed: int, with_na: bool
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.intp]]:
    """Return values, an N/A mask, and class indices.

    NaN marks missing values. The class depends on N/A as well as on the
    values, so that the learner has a reason to split on N/A.
    """
    rng = np.random.default_rng(seed)
    values = np.empty((rows, 4))
    values[:, 0] = rng.integers(0, 80, rows) / 4
    values[:, 1] = rng.integers(0, 80, rows) / 4
    values[:, 2] = rng.integers(0, 3, rows)
    values[:, 3] = rng.integers(0, 5, rows)
    mask = np.zeros((rows, 4), dtype=np.bool_)
    if with_na:
        for column, period in enumerate((4, 6, 5, 4)):
            mask[:, column] = rng.integers(0, period, rows) == 0
        mask[3::17, :] = True  # Rows with every feature N/A.
    values[rng.integers(0, 11, rows) == 0, 0] = np.nan
    values[rng.integers(0, 9, rows) == 0, 3] = np.nan
    labels = np.where(values[:, 0] > 12, 1, 0)
    labels = np.where(
        (values[:, 3] == 3) | (values[:, 3] == 4),
        1 + (values[:, 0] <= 10),
        labels,
    )
    labels = np.where(mask[:, 2], 2, labels)
    labels = np.where(mask[:, 0], 0, labels)
    labels = np.where(mask[:, 3], (labels + 1) % 3, labels)
    return values, mask, labels.astype(np.intp)


def hide(
    values: NDArray[np.float64], mask: NDArray[np.bool_], kind: str
) -> NDArray[np.float64]:
    """Replace the values under the mask."""
    result = values.copy()
    if kind == "nan":
        result[mask] = np.nan
    elif kind == "invalid":
        invalid = np.array([np.inf, -np.inf, -3.0, 99.0])
        result = np.where(mask, np.broadcast_to(invalid, result.shape), result)
    return result


def text_of(
    values: NDArray[np.float64],
    mask: NDArray[np.bool_],
    classes: NDArray[np.intp] | None,
) -> str:
    """Return the data-file text of a table, with N/A under the mask."""
    lines = []
    for row, flags in zip(values, mask, strict=True):
        fields = []
        for column, (value, na) in enumerate(zip(row, flags, strict=True)):
            if na:
                fields.append("N/A")
            elif np.isnan(value):
                fields.append("?")
            elif column < 2:
                fields.append(repr(float(value)))
            elif column == 2:
                fields.append(G_LEVELS[int(value)])
            else:
                fields.append(H_LEVELS[int(value)])
        lines.append(",".join(fields))
    if classes is None:
        return "".join(f"{line},?\n" for line in lines)
    return "".join(
        f"{line},{CLASSES[int(k)]}\n"
        for line, k in zip(lines, classes, strict=True)
    )


def body(model: c50.Model) -> list[str]:
    """Return the serialized model without the line that has the time stamp."""
    return [
        line
        for line in model.serialized_data.splitlines()
        if not line.startswith("id=")
    ]


def same_details(a: c50.Predictions, b: c50.Predictions) -> None:
    assert list(a.labels) == list(b.labels)
    assert np.array_equal(a.confidences, b.confidences)
    assert np.array_equal(a.scores, b.scores)


def options(**fields: Any) -> c50.Options:
    result = c50.Options()
    for name, value in fields.items():
        setattr(result, name, value)
    return result


SCENARIOS: dict[str, dict[str, Any]] = {
    "tree": {},
    "rules": {"kind": c50.ModelKind.RULES},
    "boosted tree": {"options": options(trials=4)},
    "boosted rules": {
        "kind": c50.ModelKind.RULES,
        "options": options(trials=4),
    },
    "subsets": {"options": options(subset_splits=True)},
    "subset rules": {
        "kind": c50.ModelKind.RULES,
        "options": options(subset_splits=True),
    },
    "winnow": {"options": options(winnow=True)},
    "winnow rules": {
        "kind": c50.ModelKind.RULES,
        "options": options(winnow=True),
    },
    "everything": {
        "options": options(
            trials=3, subset_splits=True, winnow=True, global_pruning=False
        )
    },
    "costs": {"costs": COSTS},
    "costs rules": {"kind": c50.ModelKind.RULES, "costs": COSTS},
}


@pytest.mark.parametrize("hidden", ["real", "nan", "invalid"])
@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_mask_matches_text_not_applicable(scenario: str, hidden: str) -> None:
    settings = SCENARIOS[scenario]
    values, mask, classes = table(160, 12345, True)
    case_values, case_mask, _ = table(70, 777, True)
    text_model = c50.train(NAMES, text_of(values, mask, classes), **settings)
    assert any(
        "N/A" in line for line in text_model.serialized_data.splitlines()
    )
    dense_model = c50.train_dense(
        NAMES,
        hide(values, mask, hidden),
        classes,
        **settings,
        not_applicable=mask,
    )
    assert body(dense_model) == body(text_model)

    cases = text_of(case_values, case_mask, None)
    hidden_cases = hide(case_values, case_mask, hidden)
    expected = text_model.predict_details(cases)
    same_details(
        expected,
        dense_model.predict_details_dense(
            hidden_cases, not_applicable=case_mask
        ),
    )
    same_details(
        expected,
        dense_model.prepare_predictor().predict_details_dense(
            hidden_cases, not_applicable=case_mask
        ),
    )
    assert dense_model.attribute_usage_dense(
        hidden_cases, not_applicable=case_mask
    ) == text_model.attribute_usage(cases)


@pytest.mark.parametrize("kind", [c50.ModelKind.TREE, c50.ModelKind.RULES])
def test_not_applicable_only_at_prediction(kind: c50.ModelKind) -> None:
    values, mask, classes = table(160, 12345, False)
    assert not mask.any()
    model = c50.train_dense(NAMES, values, classes, kind=kind)
    case_values, case_mask, _ = table(70, 777, True)
    cases = text_of(case_values, case_mask, None)
    hidden = hide(case_values, case_mask, "invalid")
    same_details(
        model.predict_details(cases),
        model.predict_details_dense(hidden, not_applicable=case_mask),
    )
    assert model.attribute_usage_dense(
        hidden, not_applicable=case_mask
    ) == model.attribute_usage(cases)
    # N/A is not missing: the two encodings can classify differently.
    missing = case_values.copy()
    missing[case_mask] = np.nan
    assert text_of(missing, np.zeros_like(case_mask), None) != cases


def test_all_false_mask_equals_no_mask() -> None:
    values, mask, classes = table(120, 99, False)
    plain = c50.train_dense(NAMES, values, classes)
    masked = c50.train_dense(NAMES, values, classes, not_applicable=mask)
    assert body(plain) == body(masked)
    same_details(
        plain.predict_details_dense(values),
        plain.predict_details_dense(values, not_applicable=mask),
    )
    assert plain.attribute_usage_dense(values) == plain.attribute_usage_dense(
        values, not_applicable=mask
    )


def test_nan_without_mask_stays_missing() -> None:
    values, mask, classes = table(120, 5, True)
    nan_values = hide(values, mask, "nan")
    text_missing = text_of(nan_values, np.zeros_like(mask), classes)
    assert "N/A" not in text_missing
    model = c50.train_dense(NAMES, nan_values, classes)
    assert body(model) == body(c50.train(NAMES, text_missing))
    assert body(model) != body(
        c50.train_dense(NAMES, nan_values, classes, not_applicable=mask)
    )


def test_none_mask_and_keyword_only() -> None:
    values, _, classes = table(60, 1, False)
    assert body(
        c50.train_dense(NAMES, values, classes, not_applicable=None)
    ) == body(c50.train_dense(NAMES, values, classes))
    mask = np.zeros(values.shape, dtype=np.bool_)
    positional: Any = (
        NAMES,
        values,
        classes,
        c50.ModelKind.TREE,
        options(),
        "",
        1,
        True,
        mask,
    )
    with pytest.raises(TypeError):
        c50.train_dense(*positional)
    model = c50.train_dense(NAMES, values, classes)
    predict: Any = model.predict_details_dense
    with pytest.raises(TypeError):
        predict(values, mask)


def entry_points(
    values: NDArray[np.float64], classes: NDArray[np.intp]
) -> dict[str, Any]:
    model = c50.train_dense(NAMES, values, classes)
    return {
        "module train_dense": lambda m: c50.train_dense(
            NAMES, values, classes, not_applicable=m
        ),
        "Model.train_dense": lambda m: c50.Model.train_dense(
            NAMES, values, classes, not_applicable=m
        ),
        "Model.predict_details_dense": lambda m: model.predict_details_dense(
            values, not_applicable=m
        ),
        "Predictor.predict_details_dense": lambda m: (
            model.prepare_predictor().predict_details_dense(
                values, not_applicable=m
            )
        ),
        "Model.attribute_usage_dense": lambda m: model.attribute_usage_dense(
            values, not_applicable=m
        ),
    }


@pytest.mark.parametrize(
    "entry",
    [
        "module train_dense",
        "Model.train_dense",
        "Model.predict_details_dense",
        "Predictor.predict_details_dense",
        "Model.attribute_usage_dense",
    ],
)
def test_mask_validation(entry: str) -> None:
    values, mask, classes = table(40, 2, True)
    call = entry_points(values, classes)[entry]
    call(mask)
    for bad in (mask[:-1], mask[:, :-1], mask.ravel(), mask[None]):
        with pytest.raises(ValueError, match="same shape as values"):
            call(bad)
    for dtype in (np.uint8, np.int8, np.int64, np.float64):
        with pytest.raises(ValueError, match="bool dtype"):
            call(mask.astype(dtype))
    with pytest.raises(ValueError, match="bool dtype"):
        call(np.where(mask, "x", ""))


def test_mask_must_be_an_array() -> None:
    values, mask, classes = table(40, 2, True)
    listed: Any = mask.tolist()
    with pytest.raises(TypeError):
        c50.train_dense(NAMES, values, classes, not_applicable=listed)


def test_mask_layouts() -> None:
    values, mask, classes = table(80, 3, True)
    expected = body(
        c50.train_dense(NAMES, values, classes, not_applicable=mask)
    )
    fortran = np.asfortranarray(mask)
    assert not fortran.flags.c_contiguous
    strided = np.zeros((80, 8), dtype=np.bool_)
    strided[:, ::2] = mask
    view = strided[:, ::2]
    assert not view.flags.c_contiguous
    readonly = mask.copy()
    readonly.flags.writeable = False
    for candidate in (fortran, view, readonly, mask.T.copy().T):
        for copy in (True, False):
            model = c50.train_dense(
                NAMES, values, classes, copy=copy, not_applicable=candidate
            )
            assert body(model) == expected
    model = c50.train_dense(NAMES, values, classes, not_applicable=mask)
    expected_labels = list(
        model.predict_details_dense(values, not_applicable=mask).labels
    )
    for candidate in (fortran, view, readonly):
        assert (
            list(
                model.predict_details_dense(
                    values, not_applicable=candidate
                ).labels
            )
            == expected_labels
        )


def test_mask_bytes_other_than_one_are_not_applicable() -> None:
    values, mask, classes = table(80, 4, True)
    expected = body(
        c50.train_dense(NAMES, values, classes, not_applicable=mask)
    )
    bytes_view = mask.view(np.uint8).copy()
    bytes_view[bytes_view == 1] = 7
    odd = bytes_view.view(np.bool_)
    assert (
        body(
            c50.train_dense(
                NAMES, values, classes, copy=False, not_applicable=odd
            )
        )
        == expected
    )


def test_copy_false_reads_the_callers_mask() -> None:
    values, mask, classes = table(80, 5, True)
    expected = body(
        c50.train_dense(NAMES, values, classes, not_applicable=mask)
    )
    # copy=False and copy=True read the same bytes when nothing changes them.
    assert (
        body(
            c50.train_dense(
                NAMES, values, classes, copy=False, not_applicable=mask
            )
        )
        == expected
    )
    # A caller that modifies the mask only after the call does not change the
    # model, which owns its serialized text.
    model = c50.train_dense(
        NAMES, values, classes, copy=False, not_applicable=mask
    )
    mask[:] = False
    assert body(model) == expected


def test_empty_and_single_row_masks() -> None:
    values, mask, classes = table(60, 6, True)
    model = c50.train_dense(NAMES, values, classes, not_applicable=mask)
    empty = model.predict_details_dense(
        np.empty((0, 4)), not_applicable=np.empty((0, 4), dtype=np.bool_)
    )
    assert len(empty.labels) == 0
    one = model.predict_details_dense(
        np.full((1, 4), np.nan), not_applicable=np.ones((1, 4), dtype=np.bool_)
    )
    text = model.predict_details("N/A,N/A,N/A,N/A,?\n")
    same_details(text, one)
