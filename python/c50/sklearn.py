"""Scikit-learn-compatible estimator backed by the C5.0 core."""

from __future__ import annotations

import sys
from collections.abc import Iterator, Sequence
from numbers import Integral
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import Tags
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import (
    check_consistent_length,
    check_is_fitted,
    validate_data,
)

from ._c50 import Model, ModelKind, Options, TieOrder
from ._data import Columns, Schema, dataframe_columns, encode_costs, fit_schema


type ModelKindName = Literal["tree", "rules"]
type TieOrderName = Literal["reference", "stable"]
type UnknownCategoryPolicy = Literal["error", "missing"]


class C50Classifier(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """Classify dense array-like data with C5.0.

    Numeric features are continuous by default. Non-numeric and Boolean
    features are categorical by default. Set ``categorical_features`` to a
    sequence of column indices or names to override inference. When it is set,
    every unlisted feature is continuous.

    The estimator converts arrays to the native dense representation, then
    delegates training and prediction to the low-level binding. It does not
    implement a separate learning algorithm.

    Args:
        model_kind: Build a decision tree or a rule set.
        trials: Number of boosting trials. One disables boosting.
        subset_splits: Allow subset splits for categorical features.
        winnow: Enable attribute winnowing.
        global_pruning: Enable global tree pruning.
        probabilistic_thresholds: Use probabilistic thresholds for continuous
            features.
        ignore_costs: Ignore supplied costs during training and prediction.
        minimum_cases: Minimum cases associated with a branch of a split.
        confidence_factor: Pruning confidence factor.
        sample_fraction: Fraction of cases used for training. Zero disables
            sampling.
        random_seed: Seed used by native sampling.
        ties: Order of equal continuous values when evaluating splits.
            ``"reference"`` reproduces the reference C5.0 learner.
            ``"stable"`` is faster and deterministic on every platform, but
            may build a classifier that differs from the reference learner.
        split_workers: Number of native split-evaluation workers,
            from 1 through 8.
        categorical_features: Categorical column indices or, for inputs with
            string feature names, column names. ``None`` infers feature types.
        unknown_categories: Raise an error for an unseen prediction-time
            category or pass it to C5.0 as a missing value.
        prediction_batch_size: Maximum rows converted and predicted in one
            native call. ``None`` processes all prediction rows together.
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

    classes_: NDArray[Any]
    model_: Model
    n_features_in_: int
    feature_names_in_: NDArray[Any]
    categorical_features_: NDArray[np.signedinteger[Any]]
    categories_: tuple[NDArray[Any], ...]
    cost_matrix_: NDArray[np.float64] | None
    _label_encoder: LabelEncoder
    _schema: Schema

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
        ties: TieOrderName = "reference",
        split_workers: int = 1,
        categorical_features: Sequence[int | str] | None = None,
        unknown_categories: UnknownCategoryPolicy = "error",
        prediction_batch_size: int | None = 65536,
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
        self.ties = ties
        self.split_workers = split_workers
        self.categorical_features = categorical_features
        self.unknown_categories = unknown_categories
        self.prediction_batch_size = prediction_batch_size
        self.cost_matrix = cost_matrix

    def fit(self, X: ArrayLike, y: ArrayLike) -> C50Classifier:
        """Fit a C5.0 classifier.

        A failed fit clears all fitted state, including any previous model.

        Args:
            X: Dense two-dimensional training data.
            y: One-dimensional class labels.

        Returns:
            This fitted estimator.

        Raises:
            TypeError: If a continuous feature contains a non-numeric value.
            ValueError: If the data, schema, options, or cost matrix is invalid.
        """
        self._clear_fitted_state()
        fitted = False
        try:
            self._validate_configuration()
            X_checked: NDArray[Any] | Columns
            if _is_column_frame(X):
                y_checked = validate_data(self, y=y, reset=True)
                validate_data(self, X, reset=True, skip_check_array=True)
                check_consistent_length(X, y_checked)
                X_checked = dataframe_columns(X)
            else:
                X_checked, y_checked = validate_data(
                    self,
                    X,
                    y,
                    reset=True,
                    dtype=_validation_dtype(X),
                    ensure_all_finite=False,
                )
            check_classification_targets(y_checked)

            label_encoder = LabelEncoder().fit(y_checked)
            classes = label_encoder.classes_
            if classes.shape[0] < 2:
                raise ValueError(
                    "C5.0 classification requires at least two classes; one class "
                    "was provided"
                )
            class_indices = label_encoder.transform(y_checked)

            feature_names = getattr(self, "feature_names_in_", None)
            schema = fit_schema(
                X_checked,
                self.categorical_features,
                feature_names,
            )
            names_data = schema.names_data(classes.shape[0])
            training_values = schema.dense_data(X_checked, "error")
            native_class_indices = np.asarray(class_indices, dtype=np.uintp)
            costs_data = encode_costs(classes.shape[0], self.cost_matrix)
            options = self._native_options()
            model = Model.train_dense(
                names_data,
                training_values,
                native_class_indices,
                self._native_model_kind(),
                options,
                costs_data,
                self.split_workers,
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
            fitted = True
            return self
        finally:
            if not fitted:
                self._clear_fitted_state()

    def _clear_fitted_state(self) -> None:
        for name in (
            "classes_",
            "model_",
            "n_features_in_",
            "feature_names_in_",
            "categorical_features_",
            "categories_",
            "cost_matrix_",
            "_label_encoder",
            "_schema",
        ):
            self.__dict__.pop(name, None)

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        """Predict a class label for each row in ``X``.

        Args:
            X: Dense two-dimensional prediction data.

        Returns:
            An array of labels with the fitted class-label dtype.

        Raises:
            ValueError: If the input shape or a feature value is invalid.
        """
        X_checked = self._validate_prediction_input(X)
        indices = np.empty(X_checked.shape[0], dtype=np.intp)
        predictor = self.model_.prepare_predictor()
        for row_slice, values in self._prediction_batches(X_checked):
            details = predictor.predict_details_dense(values)
            indices[row_slice] = np.asarray(details.class_indices, dtype=np.intp)
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
        X_checked = self._validate_prediction_input(X)
        scores = np.empty(
            (X_checked.shape[0], self.classes_.shape[0]),
            dtype=np.float64,
        )
        predictor = self.model_.prepare_predictor()
        for row_slice, values in self._prediction_batches(X_checked):
            details = predictor.predict_details_dense(values)
            scores[row_slice] = np.asarray(details.scores, dtype=np.float64)
        return scores

    def __sklearn_is_fitted__(self) -> bool:
        """Return whether native model state has been fitted."""
        return hasattr(self, "model_")

    def __sklearn_tags__(self) -> Tags:
        """Declare input capabilities to scikit-learn."""
        tags = super().__sklearn_tags__()
        tags.array_api_support = False
        tags.input_tags.allow_nan = True
        tags.input_tags.categorical = True
        tags.input_tags.string = True
        return tags

    def _validate_prediction_input(self, X: ArrayLike) -> NDArray[Any] | Columns:
        check_is_fitted(self)
        # set_params() may change prediction parameters after fitting.
        self._validate_prediction_configuration()
        if _is_column_frame(X):
            validate_data(self, X, reset=False, skip_check_array=True)
            return dataframe_columns(X)
        return cast(
            NDArray[Any],
            validate_data(
                self,
                X,
                reset=False,
                dtype=_validation_dtype(X),
                ensure_all_finite=False,
            ),
        )

    def _prediction_batches(
        self,
        X: NDArray[Any] | Columns,
    ) -> Iterator[tuple[slice, NDArray[np.float64]]]:
        batch_size = (
            max(1, X.shape[0])
            if self.prediction_batch_size is None
            else self.prediction_batch_size
        )
        for start in range(0, X.shape[0], batch_size):
            stop = min(start + batch_size, X.shape[0])
            yield (
                slice(start, stop),
                self._schema.dense_data(
                    X[start:stop],
                    self.unknown_categories,
                ),
            )

    def _validate_configuration(self) -> None:
        if self.model_kind not in ("tree", "rules"):
            raise ValueError("model_kind must be 'tree' or 'rules'")
        if self.ties not in ("reference", "stable"):
            raise ValueError("ties must be 'reference' or 'stable'")
        if (
            not isinstance(self.split_workers, Integral)
            or isinstance(self.split_workers, bool)
            or not 1 <= self.split_workers <= 8
        ):
            raise ValueError("split_workers must be an integer from 1 through 8")
        self._validate_prediction_configuration()

    def _validate_prediction_configuration(self) -> None:
        if self.unknown_categories not in ("error", "missing"):
            raise ValueError("unknown_categories must be 'error' or 'missing'")
        if (
            self.prediction_batch_size is not None
            and (
                not isinstance(self.prediction_batch_size, Integral)
                or isinstance(self.prediction_batch_size, bool)
                or self.prediction_batch_size < 1
            )
        ):
            raise ValueError("prediction_batch_size must be a positive integer or None")

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
        options.ties = (
            TieOrder.STABLE if self.ties == "stable" else TieOrder.REFERENCE
        )
        return options


def _is_column_frame(X: object) -> bool:
    """Return whether ``X`` is a DataFrame to convert column by column.

    Empty and sparse frames use the whole-array conversion, which reports
    their errors.
    """
    pandas = sys.modules.get("pandas")
    return (
        pandas is not None
        and isinstance(X, pandas.DataFrame)
        and 0 not in X.shape
        and not any(isinstance(dtype, pandas.SparseDtype) for dtype in X.dtypes)
    )


def _validation_dtype(X: ArrayLike) -> type[object] | None:
    """Preserve numeric arrays while retaining mixed Python scalar types."""
    dtype = getattr(X, "dtype", None)
    if dtype is not None and np.dtype(dtype).kind in "iufb":
        return None
    return object


__all__ = ["C50Classifier"]
