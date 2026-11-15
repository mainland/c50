# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Static type assertions for the installed public Python interfaces."""

from __future__ import annotations

import enum
import sys
from typing import Any

import c50
import numpy as np
from c50.sklearn import C50Classifier
from numpy.typing import NDArray

if sys.version_info >= (3, 11):
    from typing import assert_type
else:
    from typing_extensions import assert_type


options = c50.Options()
assert_type(options.trials, int)
assert_type(options.confidence_factor, float)
# The counts accept any object with __index__, which includes NumPy integers.
options.trials = np.int64(2)
options.random_seed = np.int64(7)
assert_type(options.random_seed, int)

model = c50.train(
    "no, yes.\n\nvalue: continuous.\n",
    "0, no\n1, yes\n",
    options=options,
)
assert_type(model, c50.Model)
assert_type(
    c50.train(
        "no, yes.\n\nvalue: continuous.\n",
        "0, no\n1, yes\n",
        split_workers=np.int64(1),
    ),
    c50.Model,
)
assert_type(
    c50.load(
        model.names_data,
        model.serialized_data,
        max_tree_depth=np.int64(8),
    ),
    c50.Model,
)
assert_type(model.predict("0, ?\n"), list[str])
assert_type(model.predict_scores("0, ?\n"), list[list[float]])

X: NDArray[np.float64] = np.asarray([[0.0], [1.0], [2.0], [3.0]])
y: NDArray[np.str_] = np.asarray(["no", "no", "yes", "yes"])
classifier = C50Classifier(minimum_cases=1).fit(X, y)
assert_type(classifier, C50Classifier)

labels: NDArray[Any] = classifier.predict(X)
probabilities: NDArray[np.float64] = classifier.predict_proba(X)
assert_type(classifier.export_text(max_depth=np.int64(1)), str)

inspection = model.inspect()
assert_type(inspection, c50.ModelInspection)
assert_type(inspection.trees[0].nodes[0].feature, int | None)
assert_type(c50.export_text(inspection), str)
assert_type(c50.export_text(inspection, max_depth=np.int64(2)), str)
assert_type(c50.export_json(inspection), str)

predictor = model.prepare_predictor()
assert_type(predictor.predict_details(""), c50.Predictions)

# The enumerations are enum.IntEnum types: members iterate, subscript, and
# compare equal to their integers.
assert_type(c50.ModelKind["TREE"], c50.ModelKind)
assert_type(c50.ModelKind(1), c50.ModelKind)
assert_type(list(c50.TieOrder), list[c50.TieOrder])
assert_type(model.inspect().kind, c50.ModelKind)
assert issubclass(c50.NodeKind, enum.IntEnum)
assert issubclass(c50.ConditionKind, int)


def _check_member(kind: c50.ModelKind, number: int) -> None:
    assert_type(kind.name, str)
    assert_type(kind.value, int)
    assert_type(int(kind), int)
    assert_type(kind == number, bool)
    assert_type(kind + 1, int)
