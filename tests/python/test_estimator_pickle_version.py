"""The estimator pickle records the c50 version that saved it."""

from __future__ import annotations

import copy
import pickle
import re
from importlib.metadata import version

import c50.sklearn
import numpy as np
import pytest
from c50.sklearn import C50Classifier
from sklearn.base import clone

X = np.asarray([[0.0], [0.5], [1.0], [1.5], [2.5], [3.0], [3.5], [4.0]])
Y = np.asarray(["low", "low", "low", "low", "high", "high", "high", "high"])


def test_state_records_the_version() -> None:
    state = C50Classifier().fit(X, Y).__getstate__()

    assert state["_c50_version"] == version("c50")


def test_same_version_round_trips_fitted_and_unfitted() -> None:
    fitted = C50Classifier().fit(X, Y)

    restored = pickle.loads(pickle.dumps(fitted))
    assert restored.predict(X).tolist() == fitted.predict(X).tolist()
    assert "_c50_version" not in restored.__dict__
    assert pickle.loads(pickle.dumps(C50Classifier(trials=3))).trials == 3
    assert (
        copy.deepcopy(fitted).predict(X).tolist() == fitted.predict(X).tolist()
    )


def test_clone_is_unaffected() -> None:
    cloned = clone(C50Classifier(trials=2).fit(X, Y))

    assert cloned.trials == 2
    assert not hasattr(cloned, "model_")


def test_other_version_is_rejected_naming_both() -> None:
    state = C50Classifier().fit(X, Y).__getstate__()
    state["_c50_version"] = "0.0.1"

    with pytest.raises(
        ValueError, match=r"0\.0\.1.*" + re.escape(version("c50"))
    ):
        C50Classifier().__setstate__(state)


def test_missing_version_is_rejected() -> None:
    state = C50Classifier().fit(X, Y).__getstate__()
    del state["_c50_version"]

    with pytest.raises(ValueError, match="no c50 version"):
        C50Classifier().__setstate__(state)


def test_pickle_from_another_version_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = pickle.dumps(C50Classifier().fit(X, Y))
    monkeypatch.setattr(c50.sklearn, "version", lambda name: "99.0.0")

    with pytest.raises(ValueError, match=r"99\.0\.0"):
        pickle.loads(data)
