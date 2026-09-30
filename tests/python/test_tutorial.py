"""Run the tutorial's code blocks and check the results that its text reports."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import pytest

TUTORIAL = Path(__file__).resolve().parents[2] / "docs" / "tutorial.md"


def _blocks(language: str) -> list[str]:
    text = TUTORIAL.read_text(encoding="utf-8")
    return re.findall(rf"^```{language}\n(.*?)^```", text, flags=re.S | re.M)


@pytest.mark.skipif(
    os.environ.get("C50_NETWORK_TESTS") != "1",
    reason="set C50_NETWORK_TESTS=1 to download the tutorial data from OpenML",
)
def test_tutorial_code_runs_and_matches_its_text(
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    namespace: dict[str, Any] = {"__name__": "tutorial"}
    for index, block in enumerate(_blocks("python")):
        exec(compile(block, f"{TUTORIAL.name} block {index}", "exec"), namespace)
    printed = capsys.readouterr().out

    tree_text, rules_text = _blocks("text")
    assert namespace["tree"].export_text(include_empty=False) == tree_text
    assert namespace["rules"].export_text() == rules_text
    assert "trials=1: 0.968" in printed
    assert "trials=10: 0.977" in printed
    assert "plain: accuracy 0.717, total cost 273" in printed
    assert "cost-sensitive: accuracy 0.567, total cost 174" in printed
