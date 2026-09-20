"""Scikit-learn-compatible estimator backed by the C5.0 core."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

from ._c50 import Model, ModelKind, Options, train
from ._data import Schema, encode_costs, fit_schema


type ModelKindName = Literal["tree", "rules"]
type UnknownCategoryPolicy = Literal["error", "missing"]


class C50Classifier(ClassifierMixin, BaseEstimator):
    """Classify dense array-like data with C5.0.

    Numeric features are continuous by default. Non-numeric and Boolean
    features are categorical by default. Set ``categorical_features`` to a
    sequence of column indices or names to override inference. When it is set,
    every unlisted feature is continuous.

    The estimator converts arrays to the in-memory C5.0 names and case formats,
    then delegates training and prediction to the low-level binding. It does
    not implement a separate learning algorithm.

    Args:
        model_kind: Build a decision tree or a rule set.
        trials: Number of boosting trials. One disables boosting.
        subset_splits: Allow subset splits for categorical features.
        winnow: Enable attribute winnowing.
        global_pruning: Enable global tree pruning.
        probabilistic_thresholds: Use probabilistic thresholds for continuous
            features.
        ignore_costs: Ignore costs when choosing the predicted class while
            retaining their training effect.
        minimum_cases: Minimum cases associated with a branch of a split.
        confidence_factor: Pruning confidence factor.
        sample_fraction: Fraction of cases used for training. Zero disables
            sampling.
        random_seed: Seed used by native sampling.
        categorical_features: Categorical column indices or, for inputs with
            string feature names, column names. ``None`` infers feature types.
        unknown_categories: Raise an error for an unseen prediction-time
            category or pass it to C5.0 as a missing value.
        cost_matrix: Optional predicted-by-actual misclassification-cost
            matrix. Its shape must be ``(n_classes, n_classes)``.

    Attributes:
        classes_: Original class labels in probability-column order.
        model_: Fitted low-level :class:`c50.Model`.
        n_features_in_: Number of input features seen by :meth:`fit`.
        feature_names_in_: String feature names when supplied by the input.
        categorical_features_: Indices of fitted categorical features.
        categories_: Categories for every feature. Continuous features have an
            empty category array.
        cost_matrix_: Validated copy of the fitted cost matrix, or ``None``.
    """

    def __init__(
        self,
        *,
        model_kind: ModelKindName = "tree",
        trials: int = 1,
        subset_splits: bool = False,
        winnow: bool = False,
        global_pruning: bool = True,
        probabilistic_thresholds: bool = False,
        ignore_costs: bool = False,
        minimum_cases: float = 2.0,
        confidence_factor: float = 0.25,
        sample_fraction: float = 0.0,
        random_seed: int = 0,
        categorical_features: Sequence[int | str] | None = None,
        unknown_categories: UnknownCategoryPolicy = "error",
        cost_matrix: ArrayLike | None = None,
    ) -> None:
        self.model_kind = model_kind
        self.trials = trials
        self.subset_splits = subset_splits
        self.winnow = winnow
        self.global_pruning = global_pruning
        self.probabilistic_thresholds = probabilistic_thresholds
        self.ignore_costs = ignore_costs
        self.minimum_cases = minimum_cases
        self.confidence_factor = confidence_factor
        self.sample_fraction = sample_fraction
        self.random_seed = random_seed
        self.categorical_features = categorical_features
        self.unknown_categories = unknown_categories
        self.cost_matrix = cost_matrix

    def fit(self, X: ArrayLike, y: ArrayLike) -> C50Classifier:
        """Fit a C5.0 classifier.

        Args:
            X: Dense two-dimensional training data.
            y: One-dimensional class labels.

        Returns:
            This fitted estimator.

        Raises:
            TypeError: If a continuous feature contains a non-numeric value.
            ValueError: If the data, schema, options, or cost matrix is invalid.
        """
        self._validate_configuration()
        X_checked, y_checked = validate_data(
            self,
            X,
            y,
            reset=True,
            dtype=object,
            ensure_all_finite=False,
        )
        check_classification_targets(y_checked)

        label_encoder = LabelEncoder().fit(y_checked)
        classes = label_encoder.classes_
        if classes.shape[0] < 2:
            raise ValueError("C5.0 classification requires at least two classes")
        class_indices = label_encoder.transform(y_checked)

        feature_names = getattr(self, "feature_names_in_", None)
        schema = fit_schema(
            X_checked,
            self.categorical_features,
            feature_names,
        )
        names_data = schema.names_data(classes.shape[0])
        training_data = schema.training_data(X_checked, class_indices)
        costs_data = encode_costs(classes.shape[0], self.cost_matrix)
        options = self._native_options()
        model = train(
            names_data,
            training_data,
            self._native_model_kind(),
            options,
            costs_data,
        )

        self.classes_ = classes
        self.model_ = model
        self.categorical_features_ = np.asarray(
            schema.categorical_indices,
            dtype=np.intp,
        )
        self.categories_ = tuple(
            np.asarray(categories, dtype=object) for categories in schema.categories
        )
        self.cost_matrix_ = (
            None
            if self.cost_matrix is None
            else np.asarray(self.cost_matrix, dtype=float).copy()
        )
        self._label_encoder = label_encoder
        self._schema = schema
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.generic]:
        """Predict a class label for each row in ``X``.

        Args:
            X: Dense two-dimensional prediction data.

        Returns:
            An array of labels with the fitted class-label dtype.

        Raises:
            ValueError: If the input shape or a feature value is invalid.
        """
        details = self._predict_details(X)
        indices = np.asarray(details.class_indices, dtype=np.intp)
        return self.classes_[indices]

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return class scores in ``classes_`` order.

        Args:
            X: Dense two-dimensional prediction data.

        Returns:
            A two-dimensional array with one row per input and one column per
            fitted class.

        Raises:
            ValueError: If the input shape or a feature value is invalid.
        """
        details = self._predict_details(X)
        return np.asarray(details.scores, dtype=np.float64)

    def __sklearn_is_fitted__(self) -> bool:
        """Return whether native model state has been fitted."""
        return hasattr(self, "model_")

    def __sklearn_tags__(self):  # type: ignore[no-untyped-def]
        """Declare input capabilities to scikit-learn."""
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.input_tags.categorical = True
        tags.input_tags.string = True
        return tags

    def _predict_details(self, X: ArrayLike):  # type: ignore[no-untyped-def]
        check_is_fitted(self)
        X_checked = validate_data(
            self,
            X,
            reset=False,
            dtype=object,
            ensure_all_finite=False,
        )
        cases = self._schema.prediction_data(
            X_checked,
            self.unknown_categories,
        )
        return self.model_.predict_details(cases)

    def _validate_configuration(self) -> None:
        if self.model_kind not in ("tree", "rules"):
            raise ValueError("model_kind must be 'tree' or 'rules'")
        if self.unknown_categories not in ("error", "missing"):
            raise ValueError("unknown_categories must be 'error' or 'missing'")

    def _native_model_kind(self) -> ModelKind:
        return ModelKind.TREE if self.model_kind == "tree" else ModelKind.RULES

    def _native_options(self) -> Options:
        options = Options()
        options.trials = self.trials
        options.subset_splits = self.subset_splits
        options.winnow = self.winnow
        options.global_pruning = self.global_pruning
        options.probabilistic_thresholds = self.probabilistic_thresholds
        options.ignore_costs = self.ignore_costs
        options.minimum_cases = self.minimum_cases
        options.confidence_factor = self.confidence_factor
        options.sample_fraction = self.sample_fraction
        options.random_seed = self.random_seed
        return options


__all__ = ["C50Classifier"]
