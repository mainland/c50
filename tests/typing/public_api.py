"""Static type assertions for the installed public Python interfaces."""

from __future__ import annotations

from typing import assert_type

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

# The enumerations are pybind11 enums, not enum.Enum: members and the
# attributes below exist, but iteration and subscripting fail at run time.
assert_type(c50.ModelKind.TREE, c50.ModelKind)
assert_type(c50.ModelKind.TREE.name, str)
assert_type(c50.ModelKind.TREE.value, int)
assert_type(c50.ModelKind.__members__, dict[str, c50.ModelKind])
for _member in c50.ModelKind:  # type: ignore[attr-defined]
    pass
