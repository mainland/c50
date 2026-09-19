"""Python interface to the C5.0 GPL core."""

from ._c50 import (
    C50Error,
    Model,
    ModelKind,
    Options,
    Predictions,
    load,
    train,
)

__all__ = [
    "C50Error",
    "Model",
    "ModelKind",
    "Options",
    "Predictions",
    "load",
    "train",
]
