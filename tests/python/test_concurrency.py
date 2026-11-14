# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Concurrency checks for independent native Python operations."""

from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import c50
import pytest

NAMES = """low, high.

signal: continuous.
"""

ASCENDING = """0, low
1, low
2, low
3, high
4, high
5, high
"""

DESCENDING = """0, high
1, high
2, high
3, low
4, low
5, low
"""


def train_and_predict(training_data: str, expected: str) -> None:
    for _ in range(20):
        model = c50.train(NAMES, training_data)
        assert model.predict("0, ?\n") == [expected]


def test_concurrent_training_and_prediction() -> None:
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(train_and_predict, ASCENDING, "low"),
            executor.submit(train_and_predict, DESCENDING, "high"),
        ]
        for future in futures:
            future.result()


# Training with split workers needs at least 10,000 cases. The child imports
# NumPy first, because its thread pool cannot start under the process limit.
THREAD_FAILURE_SCRIPT = """
import resource
import sys

import numpy as np

import c50

names = "no, yes.\\n\\nx0: continuous.\\nx1: continuous.\\nx2: continuous.\\n"
values = np.random.default_rng(0).random((12000, 3))
classes = (values[:, 0] + values[:, 1] > 1).astype(np.uintp)
soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
resource.setrlimit(resource.RLIMIT_NPROC, (1, hard))
try:
    c50.train_dense(names, values, classes, split_workers=4)
except BaseException as error:
    resource.setrlimit(resource.RLIMIT_NPROC, (soft, hard))
    print(type(error).__name__, isinstance(error, c50.C50Error))
    sys.exit(0)
print("trained")
"""


@pytest.mark.skipif(sys.platform == "win32", reason="requires RLIMIT_NPROC")
def test_thread_creation_failure_is_runtime_error() -> None:
    result = subprocess.run(
        [sys.executable, "-c", THREAD_FAILURE_SCRIPT],
        capture_output=True,
        text=True,
        check=True,
    )
    outcome = result.stdout.strip()
    if outcome == "trained":
        pytest.skip("the process may exceed RLIMIT_NPROC")
    # A std::system_error is not a c50::exception, so it is not a C50Error.
    assert outcome == "RuntimeError False"
