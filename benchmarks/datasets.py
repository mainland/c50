"""Prepare benchmark datasets as C5.0 ``.names`` and ``.data`` files.

Every dataset is written once into a cache directory and reused. Categorical
values and classes are written as generated tokens, such as ``v3`` and ``c1``,
so that no value can change the C5.0 input grammar. Both the imported C5.0
program and this library read the same files.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

# Increase when the written files change, so that cached files are rewritten.
WRITER_VERSION = 2

# OpenML datasets: name, dataset ID, and version, which the ID fixes.
OPENML = {
    "adult": (1590, 2),
    "covertype": (1596, 4),
}


@dataclass(frozen=True)
class SyntheticSpec:
    """A deterministic mixed continuous and categorical workload."""

    rows: int
    features: int
    categorical_features: int
    categories: int = 8
    classes: int = 3
    missing_fraction: float = 0.01
    seed: int = 1729

    @property
    def name(self) -> str:
        return (
            f"synthetic-{self.rows}x{self.features}"
            f"-c{self.categorical_features}"
        )


@dataclass(frozen=True)
class Dataset:
    """A prepared dataset and the files that hold it."""

    name: str
    stem: Path
    rows: int
    attributes: int
    classes: int
    source: dict[str, Any]
    names_sha256: str
    data_sha256: str

    @property
    def names_path(self) -> Path:
        return self.stem.with_suffix(".names")

    @property
    def data_path(self) -> Path:
        return self.stem.with_suffix(".data")

    def record(self) -> dict[str, Any]:
        """Return the dataset description recorded with results."""
        return {
            "name": self.name,
            "rows": self.rows,
            "attributes": self.attributes,
            "classes": self.classes,
            "source": self.source,
            "names_sha256": self.names_sha256,
            "data_sha256": self.data_sha256,
        }


def load_frame(stem: Path) -> tuple[Any, Any]:
    """Read prepared ``.names`` and ``.data`` files back as estimator input.

    Discrete attributes become pandas categorical columns with the declared
    values as categories, in declaration order, continuous attributes become
    float64 columns, and ``?`` becomes a missing value. ``C50Classifier``
    keeps a categorical dtype's order, so it declares the values to C5.0 in
    the same order as the ``.names`` file.

    Args:
        stem: Path without suffix of a prepared dataset's files.

    Returns:
        The feature frame and an array of class tokens.
    """
    import pandas as pd

    names = stem.with_suffix(".names").read_text().splitlines()
    dtypes: dict[str, Any] = {}
    for line in names[2:]:
        attribute, _, rest = line.partition(":")
        values = rest.split("|")[0].strip().rstrip(".")
        if values == "continuous":
            dtypes[attribute] = np.float64
        else:
            dtypes[attribute] = pd.CategoricalDtype(
                [value.strip() for value in values.split(",")]
            )
    frame = pd.read_csv(
        stem.with_suffix(".data"),
        header=None,
        names=[*dtypes, "class"],
        dtype={**dtypes, "class": object},
        na_values=["?"],
        keep_default_na=False,
    )
    return frame.drop(columns="class"), frame["class"].to_numpy()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(frame: Any, target: Any, stem: Path) -> tuple[int, int, int]:
    """Write a pandas frame and target as C5.0 files.

    Args:
        frame: Feature columns. Categorical, object, and Boolean columns are
            discrete. Numeric columns are continuous.
        target: Class labels.
        stem: Path without suffix for the ``.names`` and ``.data`` files.

    Returns:
        Row, attribute, and class counts.
    """
    import pandas as pd

    columns = {}
    lines = []
    classes = pd.Categorical(target)
    width = len(str(len(classes.categories) - 1))
    class_names = [f"c{i:0{width}}" for i in range(len(classes.categories))]
    lines.append(", ".join(class_names) + ".")
    lines.append("")
    for index, (name, column) in enumerate(frame.items()):
        attribute = f"a{index}"
        if isinstance(column.dtype, pd.CategoricalDtype) or column.dtype in (
            object,
            bool,
        ):
            codes = pd.Categorical(column).codes
            count = int(codes.max()) + 1 if len(codes) and codes.max() >= 0 else 1
            values = np.array([f"v{i}" for i in range(count)], dtype=object)
            text = np.where(codes >= 0, values[np.maximum(codes, 0)], "?")
            columns[attribute] = text
            lines.append(
                f"{attribute}: " + ", ".join(values) + f".  | {name}"
            )
        else:
            # Keep the column's dtype so float32 values print briefly.
            columns[attribute] = pd.to_numeric(column).to_numpy()
            lines.append(f"{attribute}: continuous.  | {name}")
    columns["class"] = np.array(class_names, dtype=object)[classes.codes]
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".names").write_text("\n".join(lines) + "\n")
    pd.DataFrame(columns).to_csv(
        stem.with_suffix(".data"),
        header=False,
        index=False,
        na_rep="?",
        lineterminator="\n",
    )
    return len(frame), frame.shape[1], len(classes.categories)


def _openml(name: str, cache: Path) -> Callable[[Path], tuple[int, int, int]]:
    data_id, version = OPENML[name]

    def write(stem: Path) -> tuple[int, int, int]:
        from sklearn.datasets import fetch_openml

        bunch = fetch_openml(
            data_id=data_id, as_frame=True, data_home=str(cache / "openml")
        )
        if int(bunch.details["version"]) != version:
            raise RuntimeError(f"OpenML dataset {data_id} is not version {version}")
        return _write(bunch.data, bunch.target, stem)

    return write


def _synthetic(spec: SyntheticSpec) -> Callable[[Path], tuple[int, int, int]]:
    def write(stem: Path) -> tuple[int, int, int]:
        import pandas as pd

        rng = np.random.default_rng(spec.seed)
        values = rng.standard_normal((spec.rows, spec.features))
        start = spec.features - spec.categorical_features
        informative = min(8, spec.features)
        signal = values[:, :informative] @ np.linspace(1.0, 0.25, informative)
        cuts = np.quantile(signal, np.linspace(0, 1, spec.classes + 1)[1:-1])
        target = np.searchsorted(cuts, signal)
        frame = pd.DataFrame(
            {f"x{i}": values[:, i].astype(np.float32) for i in range(start)}
        )
        for i in range(start, spec.features):
            frame[f"x{i}"] = pd.Categorical(
                rng.integers(0, spec.categories, spec.rows),
                categories=range(spec.categories),
            )
        if spec.missing_fraction:
            missing = rng.random(frame.shape) < spec.missing_fraction
            frame = frame.mask(missing)
        return _write(frame, target, stem)

    return write


def prepare(name: str | SyntheticSpec, cache: Path) -> Dataset:
    """Prepare one dataset, reusing cached files.

    Args:
        name: An OpenML dataset name from ``OPENML`` or a synthetic workload.
        cache: Directory for downloaded data and generated files.

    Returns:
        The prepared dataset.
    """
    if isinstance(name, SyntheticSpec):
        label = name.name
        source: dict[str, Any] = {
            "kind": "synthetic",
            "writer": WRITER_VERSION,
            **name.__dict__,
        }
        write = _synthetic(name)
    else:
        label = name
        data_id, version = OPENML[name]
        source = {
            "kind": "openml",
            "data_id": data_id,
            "version": version,
            "writer": WRITER_VERSION,
        }
        write = _openml(name, cache)
    stem = cache / "files" / label / label
    manifest = stem.with_suffix(".json")
    if manifest.exists():
        recorded = json.loads(manifest.read_text())
        if (
            recorded["source"] == source
            and _sha256(stem.with_suffix(".names")) == recorded["names_sha256"]
            and _sha256(stem.with_suffix(".data")) == recorded["data_sha256"]
        ):
            return Dataset(stem=stem, **recorded)
    rows, attributes, classes = write(stem)
    record: dict[str, Any] = {
        "name": label,
        "rows": rows,
        "attributes": attributes,
        "classes": classes,
        "source": source,
        "names_sha256": _sha256(stem.with_suffix(".names")),
        "data_sha256": _sha256(stem.with_suffix(".data")),
    }
    manifest.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return Dataset(stem=stem, **record)
