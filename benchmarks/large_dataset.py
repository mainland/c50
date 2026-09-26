#!/usr/bin/env python3
"""Measure dense C5.0 training and prediction time and peak memory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from typing import Any

import numpy as np

from c50.sklearn import C50Classifier


def parse_args() -> argparse.Namespace:
    """Parse benchmark parameters."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--features", type=int, default=50)
    parser.add_argument("--categorical-features", type=int, default=5)
    parser.add_argument("--categories", type=int, default=8)
    parser.add_argument("--classes", type=int, default=3)
    parser.add_argument("--missing-fraction", type=float, default=0.01)
    parser.add_argument("--prediction-rows", type=int, default=10_000)
    parser.add_argument("--prediction-batch-size", type=int, default=65_536)
    parser.add_argument("--minimum-cases", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=1729)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Reject invalid or misleading workload dimensions."""
    if args.rows < 1 or args.features < 1:
        raise ValueError("rows and features must be positive")
    if not 0 <= args.categorical_features <= args.features:
        raise ValueError("categorical-features must be between zero and features")
    if args.categories < 2:
        raise ValueError("categories must be at least two")
    if not 2 <= args.classes <= 10:
        raise ValueError("classes must be between two and ten")
    if not 0 <= args.missing_fraction < 1:
        raise ValueError("missing-fraction must be in [0, 1)")
    if args.prediction_rows < 0:
        raise ValueError("prediction-rows must be nonnegative")


def peak_rss_bytes() -> int | None:
    """Return peak resident bytes where the standard resource API exists."""
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def generate_data(
    args: argparse.Namespace,
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.int64]]]:
    """Generate a deterministic mixed-feature classification workload."""
    rng = np.random.default_rng(args.seed)
    X = rng.standard_normal((args.rows, args.features), dtype=np.float64)
    categorical_start = args.features - args.categorical_features
    if args.categorical_features:
        X[:, categorical_start:] = rng.integers(
            0,
            args.categories,
            size=(args.rows, args.categorical_features),
        )

    informative = min(8, args.features)
    weights = np.linspace(1.0, 0.25, informative)
    signal = X[:, :informative] @ weights
    thresholds = np.quantile(
        signal,
        np.linspace(0, 1, args.classes + 1)[1:-1],
    )
    y = np.searchsorted(thresholds, signal).astype(np.int64, copy=False)

    if args.missing_fraction:
        missing = rng.random(X.shape) < args.missing_fraction
        X[missing] = np.nan
    return X, y


def stable_model_digest(serialized: str) -> str:
    """Hash serialized classifier content after volatile identification data.

    Args:
        serialized: Complete serialized C5.0 classifier.

    Returns:
        SHA-256 digest of the stable classifier content.

    Raises:
        ValueError: If the classifier has no header line.
    """
    _, separator, content = serialized.partition("\n")
    if not separator:
        raise ValueError("serialized classifier has no header line")
    return hashlib.sha256(content.encode()).hexdigest()


def prediction_digest(probabilities: np.ndarray[Any, Any]) -> str:
    """Hash prediction scores in a platform-independent byte order.

    Args:
        probabilities: Two-dimensional prediction score matrix.

    Returns:
        SHA-256 digest of contiguous little-endian float64 scores.
    """
    canonical = np.ascontiguousarray(probabilities, dtype="<f8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def main() -> None:
    """Run one workload and print a machine-readable result."""
    args = parse_args()
    validate_args(args)

    generation_wall_start = time.perf_counter()
    generation_cpu_start = time.process_time()
    X, y = generate_data(args)
    generation_cpu_seconds = time.process_time() - generation_cpu_start
    generation_wall_seconds = time.perf_counter() - generation_wall_start
    generation_peak_rss = peak_rss_bytes()
    categorical_start = args.features - args.categorical_features
    categorical = tuple(range(categorical_start, args.features))

    classifier = C50Classifier(
        categorical_features=categorical,
        minimum_cases=args.minimum_cases,
        prediction_batch_size=args.prediction_batch_size,
    )
    train_wall_start = time.perf_counter()
    train_cpu_start = time.process_time()
    classifier.fit(X, y)
    train_cpu_seconds = time.process_time() - train_cpu_start
    train_wall_seconds = time.perf_counter() - train_wall_start
    training_peak_rss = peak_rss_bytes()

    prediction_rows = min(args.rows, args.prediction_rows)
    predict_wall_start = time.perf_counter()
    predict_cpu_start = time.process_time()
    probabilities = classifier.predict_proba(X[:prediction_rows])
    predict_cpu_seconds = time.process_time() - predict_cpu_start
    predict_wall_seconds = time.perf_counter() - predict_wall_start
    prediction_peak_rss = peak_rss_bytes()

    serialized = classifier.model_.serialized_data
    result = {
        "environment": {
            "machine": platform.machine(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "processor_count": os.cpu_count(),
            "python": platform.python_version(),
        },
        "model": {
            "kind": "tree",
            "serialized_bytes": len(serialized.encode()),
            "stable_serialized_sha256": stable_model_digest(serialized),
            "serialized_tree_records": serialized.count("\ntype=\""),
            "trials": 1,
        },
        "resources": {
            "peak_rss_bytes": prediction_peak_rss,
            "peak_rss_checkpoints_bytes": {
                "after_data_generation": generation_peak_rss,
                "after_prediction": prediction_peak_rss,
                "after_training": training_peak_rss,
            },
        },
        "timing_seconds": {
            "data_generation_cpu": generation_cpu_seconds,
            "data_generation_wall": generation_wall_seconds,
            "prediction_cpu": predict_cpu_seconds,
            "prediction_wall": predict_wall_seconds,
            "training_cpu": train_cpu_seconds,
            "training_wall": train_wall_seconds,
        },
        "workload": {
            "categorical_features": args.categorical_features,
            "categories": args.categories,
            "classes": args.classes,
            "features": args.features,
            "minimum_cases": args.minimum_cases,
            "missing_fraction": args.missing_fraction,
            "prediction_batch_size": args.prediction_batch_size,
            "prediction_sha256": prediction_digest(probabilities),
            "prediction_rows": prediction_rows,
            "probability_checksum": float(probabilities.sum()),
            "rows": args.rows,
            "seed": args.seed,
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
