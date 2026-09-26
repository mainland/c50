"""Static type assertions for the installed public Python interfaces."""

from __future__ import annotations

from typing import Any, assert_type

import numpy as np
from numpy.typing import NDArray

import c50
from c50.sklearn import C50Classifier


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

X: NDArray[np.float64] = np.asarray([[0.0], [1.0], [2.0], [3.0]])
y: NDArray[np.str_] = np.asarray(["no", "no", "yes", "yes"])
classifier = C50Classifier(minimum_cases=1).fit(X, y)
assert_type(classifier, C50Classifier)

labels: NDArray[Any] = classifier.predict(X)
probabilities: NDArray[np.float64] = classifier.predict_proba(X)
