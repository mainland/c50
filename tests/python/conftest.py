# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Configure Hypothesis for the Python tests.

Every run draws the same examples unless it asks for others, so that a failure
in CI, CTest, or a plain `pytest` can be reproduced and bisected.
"""

from __future__ import annotations

import pytest
from hypothesis import settings

# The default profile derandomizes each test from a hash of its source, so the
# examples depend neither on a seed nor on the example database.
settings.register_profile(
    "reproducible",
    derandomize=True,
    database=None,
    print_blob=True,
)
settings.register_profile("explore", print_blob=True)


def pytest_configure(config: pytest.Config) -> None:
    """Load the default profile unless the command line names one.

    The Hypothesis plugin loads a profile named by `--hypothesis-profile` after
    this hook runs, so the option takes effect over the default.

    Args:
        config: The pytest configuration.
    """
    if config.getoption("--hypothesis-profile", default=None) is None:
        settings.load_profile("reproducible")
