# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Test the Hypothesis profiles that `conftest.py` registers."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

CONFTEST_DIRECTORY = Path(__file__).resolve().parent

PROBE = """
import sys
sys.path.insert(0, sys.argv[1])
import conftest
from hypothesis import settings
settings.load_profile(sys.argv[2])
print(settings.default.derandomize)
"""


def derandomize_of(profile: str, ci: str | None) -> bool:
    """Load a profile in a fresh interpreter and return `derandomize`.

    Args:
        profile: Name of a profile that `conftest.py` registers.
        ci: Value of the `CI` environment variable, or None to remove it.

    Returns:
        The profile's `derandomize` setting.
    """
    environment = dict(os.environ)
    environment.pop("CI", None)
    if ci is not None:
        environment["CI"] = ci
    result = subprocess.run(
        [sys.executable, "-c", PROBE, str(CONFTEST_DIRECTORY), profile],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip() == "True"


@pytest.mark.parametrize("ci", [None, "true"])
def test_explore_profile_draws_new_examples(ci: str | None) -> None:
    assert not derandomize_of("explore", ci)


@pytest.mark.parametrize("ci", [None, "true"])
def test_reproducible_profile_repeats_examples(ci: str | None) -> None:
    assert derandomize_of("reproducible", ci)
