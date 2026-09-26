"""Concurrency checks for independent native Python operations."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import c50


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
