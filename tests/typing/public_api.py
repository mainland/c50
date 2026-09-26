"""Static type assertions for the installed public Python interfaces."""

from __future__ import annotations

from typing import Any, assert_type

import numpy as np
from numpy.typing import NDArray

import c50


options = c50.Options()
assert_type(options.trials, int)
assert_type(options.confidence_factor, float)

model = c50.train(
    "no, yes.\n\nvalue: continuous.\n",
    "0, no\n1, yes\n",
    options=options,
)
assert_type(model, c50.Model)
assert_type(model.predict("0, ?\n"), list[str])
assert_type(model.predict_proba("0, ?\n"), list[list[float]])

