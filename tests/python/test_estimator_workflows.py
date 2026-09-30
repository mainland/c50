"""End-to-end scikit-learn workflows that use the C5.0 estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    StackingClassifier,
)
from sklearn.exceptions import UnsetMetadataPassedError
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from c50.sklearn import C50Classifier


def mixed_frame() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return a mixed frame, labels, and sample weights."""
    rng = np.random.default_rng(41)
    count = 360
    color = rng.choice(["red", "green", "blue"], size=count)
    frame = pd.DataFrame(
        {
            "signal": rng.normal(size=count),
            "noise": rng.normal(size=count),
            "gap": np.where(rng.random(count) < 0.2, np.nan, rng.normal(size=count)),
            "color": pd.Categorical(color),
            "label_text": pd.array(rng.choice(["x", "y"], size=count), dtype="string"),
        }
    )
    score = frame["signal"] + np.where(color == "red", 1.0, -0.3)
    labels = np.where(score + rng.normal(scale=0.7, size=count) > 0.2, "yes", "no")
    weights = rng.uniform(0.5, 2.0, size=count)
    return frame, labels, weights


def model_data(estimator: C50Classifier) -> str:
    return str(estimator.model_.serialized_data)


def test_column_transformer_pipeline_keeps_pandas_columns() -> None:
    frame, labels, _ = mixed_frame()
    categorical = ["color", "label_text"]
    preprocess = ColumnTransformer(
        [("impute", SimpleImputer(strategy="median"), ["gap"])],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    pipeline = Pipeline(
        [
            ("preprocess", preprocess),
            ("c50", C50Classifier(categorical_features=categorical)),
        ]
    ).fit(frame, labels)

    transformed = preprocess.transform(frame)
    direct = C50Classifier(categorical_features=categorical).fit(transformed, labels)
    classifier = pipeline.named_steps["c50"]

    assert classifier.feature_names_in_.tolist() == [
        "gap",
        "signal",
        "noise",
        "color",
        "label_text",
    ]
    assert classifier.categorical_features_.tolist() == [3, 4]
    assert isinstance(transformed["color"].dtype, pd.CategoricalDtype)
    assert model_data(classifier) == model_data(direct)
    np.testing.assert_array_equal(
        pipeline.predict_proba(frame), direct.predict_proba(transformed)
    )


def test_grid_search_routes_sample_weight() -> None:
    frame, labels, weights = mixed_frame()
    grid = {"minimum_cases": [2, 8], "class_weight": [None, "balanced"]}

    with sklearn.config_context(enable_metadata_routing=True):
        # GridSearchCV routes sample_weight to both fitting and scoring.
        estimator = (
            C50Classifier()
            .set_fit_request(sample_weight=True)
            .set_score_request(sample_weight=True)
        )
        search = GridSearchCV(estimator, grid, cv=StratifiedKFold(3)).fit(
            frame, labels, sample_weight=weights
        )
        expected = C50Classifier(**search.best_params_).fit(
            frame, labels, sample_weight=weights
        )
        assert model_data(search.best_estimator_) == model_data(expected)

        with pytest.raises(UnsetMetadataPassedError):
            GridSearchCV(C50Classifier(), grid, cv=3).fit(
                frame, labels, sample_weight=weights
            )


def test_pipeline_passes_sample_weight_without_routing() -> None:
    frame, labels, weights = mixed_frame()

    pipeline = Pipeline([("c50", C50Classifier())]).fit(
        frame, labels, c50__sample_weight=weights
    )
    expected = C50Classifier().fit(frame, labels, sample_weight=weights)

    assert model_data(pipeline.named_steps["c50"]) == model_data(expected)


def test_parallel_cross_validation_matches_serial() -> None:
    frame, labels, _ = mixed_frame()
    estimator = C50Classifier(sample_fraction=0.8, random_state=5)

    results = [
        cross_validate(
            estimator,
            frame,
            labels,
            cv=StratifiedKFold(4),
            scoring=["accuracy", "neg_log_loss"],
            return_estimator=True,
            n_jobs=n_jobs,
        )
        for n_jobs in (1, 2)
    ]

    for name in ("test_accuracy", "test_neg_log_loss"):
        np.testing.assert_array_equal(results[0][name], results[1][name])
    assert [model_data(fitted) for fitted in results[0]["estimator"]] == [
        model_data(fitted) for fitted in results[1]["estimator"]
    ]


def test_permutation_importance_ranks_the_signal() -> None:
    frame, labels, _ = mixed_frame()
    classifier = C50Classifier().fit(frame, labels)

    result = permutation_importance(
        classifier, frame, labels, n_repeats=5, random_state=0
    )

    assert frame.columns[np.argmax(result.importances_mean)] == "signal"
    assert classifier.feature_names_in_[
        np.argmax(classifier.feature_importances_)
    ] in ("signal", "color")


def test_calibrated_probabilities() -> None:
    frame, labels, _ = mixed_frame()

    calibrated = CalibratedClassifierCV(C50Classifier(), cv=3).fit(frame, labels)
    probabilities = calibrated.predict_proba(frame)

    assert calibrated.classes_.tolist() == ["no", "yes"]
    assert probabilities.shape == (len(labels), 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)


@pytest.mark.parametrize(
    "ensemble",
    [
        BaggingClassifier(C50Classifier(), n_estimators=4, random_state=0),
        AdaBoostClassifier(C50Classifier(), n_estimators=3, random_state=0),
        StackingClassifier(
            [("c50", C50Classifier()), ("rules", C50Classifier(model_kind="rules"))],
            final_estimator=LogisticRegression(),
            cv=3,
        ),
    ],
    ids=["bagging", "adaboost", "stacking"],
)
def test_ensembles_fit_and_predict(ensemble: object) -> None:
    frame, labels, _ = mixed_frame()
    # AdaBoostClassifier rejects missing values before its estimators see them.
    values = frame[["signal", "noise"]].to_numpy()

    fitted = sklearn.base.clone(ensemble).fit(values, labels)

    predictions = fitted.predict(values)
    assert set(predictions) <= {"no", "yes"}
    assert np.mean(predictions == labels) > 0.7
