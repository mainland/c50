"""Load model files written by the imported C5.0 Release 2.07 program."""

from __future__ import annotations

from pathlib import Path

import pytest

import c50

TESTS = Path(__file__).resolve().parents[1]

# Fixture, expected file, and the command-line options that produced it, as in
# tests/test_cli.sh. The expected files hold output of the imported program,
# with the date in the id line normalized.
CASES = [
    ("basic", "tree.tree", {}),
    ("basic", "rules.rules", {"rules": True}),
    ("basic", "subsets.tree", {"subset_splits": True}),
    ("basic", "winnow.tree", {"winnow": True}),
    ("basic", "soft-thresholds.tree", {"probabilistic_thresholds": True}),
    ("soft-threshold-ties", "soft-thresholds.tree", {"probabilistic_thresholds": True}),
    ("basic", "sample.tree", {"sample_fraction": 0.7, "random_seed": 17}),
    ("basic", "costs.tree", {"costs": True}),
    ("boost", "boost.tree", {"trials": 5}),
    ("case-weight", "case-weight.tree", {}),
    ("implicit", "implicit.tree", {}),
    ("multiclass", "tree.tree", {}),
    ("multiclass", "rules.rules", {"rules": True}),
    ("multiclass", "sample.tree", {"sample_fraction": 0.3, "random_seed": 17}),
]


def _body(serialized: str) -> list[str]:
    """Return a serialized classifier without its dated id line."""
    lines = serialized.splitlines()
    assert lines[0].startswith('id="See5/C5.0 2.07 GPL Edition ')
    return lines[1:]


@pytest.mark.parametrize(("fixture", "filename", "settings"), CASES)
def test_imported_program_models_load_and_predict(
    fixture: str, filename: str, settings: dict[str, object]
) -> None:
    directory = TESTS / "fixtures" / fixture
    names = (directory / f"{fixture}.names").read_text()
    data = (directory / f"{fixture}.data").read_text()
    costs = (directory / f"{fixture}.costs").read_text() if settings.get("costs") else ""
    kind = c50.ModelKind.RULES if settings.get("rules") else c50.ModelKind.TREE
    options = c50.Options()
    for name, value in settings.items():
        if name not in ("rules", "costs"):
            setattr(options, name, value)

    legacy = c50.load(
        names, (TESTS / "expected" / fixture / filename).read_text(), kind, costs
    )
    trained = c50.train(names, data, kind, options, costs)

    assert _body(legacy.serialized_data) == _body(trained.serialized_data)
    cases = data
    if (directory / f"{fixture}.test").exists():
        cases += (directory / f"{fixture}.test").read_text()
    expected = trained.predict_details(cases)
    actual = legacy.predict_details(cases)
    assert actual.labels == expected.labels
    assert actual.scores == expected.scores
