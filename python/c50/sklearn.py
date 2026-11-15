# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Scikit-learn-compatible estimator backed by the C5.0 core."""

from __future__ import annotations

import dataclasses
import json
import sys
from collections.abc import Iterator, Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from numbers import Integral, Real
from typing import Any, Literal, SupportsIndex, TypeAlias, cast

import numpy as np
from joblib import effective_n_jobs
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import Tags, check_random_state
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import (
    check_consistent_length,
    check_is_fitted,
    validate_data,
)

from ._c50 import Model, ModelKind, Options, TieOrder
from ._data import (
    Columns,
    DenseData,
    Schema,
    dataframe_columns,
    encode_costs,
    fit_schema,
)
from ._sentinel import NOT_APPLICABLE
from .inspection import export_json as _export_json
from .inspection import export_text as _export_text

ModelKindName: TypeAlias = Literal["tree", "rules"]
TieOrderName: TypeAlias = Literal["reference", "stable"]
UnknownCategoryPolicy: TypeAlias = Literal["error", "missing"]
ClassWeight: TypeAlias = Mapping[Any, float] | Literal["balanced"]

# Native limits on the boosting trials, sampling seed, and split workers.
_SEED_COUNT = 4096
_MAX_TRIALS = 1000
_MAX_WORKERS = 8
# Key of the saving package version in the pickled estimator state.
_VERSION_KEY = "_c50_version"
_BOOLEAN_PARAMETERS = (
    "subset_splits",
    "winnow",
    "global_pruning",
    "probabilistic_thresholds",
    "ignore_costs",
)
_REAL_PARAMETERS = ("minimum_cases", "confidence_factor", "sample_fraction")


class C50Classifier(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """Classify dense array-like data with C5.0.

    Numeric features are continuous by default. Non-numeric and Boolean
    features are categorical by default. Set ``categorical_features`` to a
    sequence of column indices or names to override inference. When it is set,
    every unlisted feature is continuous.

    A cell that holds :data:`c50.NOT_APPLICABLE` reaches C5.0 as ``N/A``. Type
    inference ignores such cells, so a column of numbers and markers is
    continuous. See ``docs/estimator.md``, section "Not-applicable values".

    The estimator converts arrays to the native dense representation, then
    delegates training and prediction to the low-level binding. It does not
    implement a separate learning algorithm.

    Args:
        model_kind: Build a decision tree or a rule set.
        trials: Number of boosting trials, from 1 through 1000. One disables
            boosting. A value that is not an integer raises ``ValueError`` at
            ``fit``, as the estimator documentation describes. That includes a
            ``bool``, ``numpy.bool_``, and ``numpy.timedelta64``.
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
        random_state: Controls native sampling when ``sample_fraction`` is
            positive. An integer or ``numpy.random.RandomState`` draws the
            native seed from 0 through 4095. ``None`` uses NumPy's global
            random state.
        ties: Order of equal continuous values when evaluating splits.
            ``"stable"``, the default, is faster and deterministic on every
            platform, but may build a classifier that differs from the
            reference learner. ``"reference"`` reproduces the reference C5.0
            learner. The low-level ``c50.train`` and ``c5.0`` default to the
            reference order.
        n_jobs: Number of native split-evaluation workers. ``None`` means one
            unless a ``joblib.parallel_config`` context sets another value, and
            ``-1`` means all processors. At most 8 workers are used. The worker
            count does not change the classifier. Zero and a value that is not
            an integer, including a ``bool``, ``numpy.bool_``, and
            ``numpy.timedelta64``, raise ``ValueError`` at ``fit``.
        categorical_features: Categorical column indices or, for inputs with
            string feature names, column names. ``None`` infers feature types.
            ``"from_dtype"`` selects exactly the DataFrame columns with a
            pandas ``category`` dtype.
        unknown_categories: Raise an error for an unseen prediction-time
            category or pass it to C5.0 as a missing value.
        prediction_batch_size: Maximum rows converted and predicted in one
            native call. ``None`` processes all prediction rows together. A
            value below one and a value that is not an integer, including a
            ``bool``, ``numpy.bool_``, and ``numpy.timedelta64``, raise
            ``ValueError`` at ``fit``, and at prediction if ``set_params``
            changes the value after ``fit``.
        cost_matrix: Optional predicted-by-actual misclassification-cost
            matrix. Its shape must be ``(n_classes, n_classes)``.
        class_weight: Optional weights for classes, as a mapping from class
            label to weight or ``"balanced"``, which weights each class by
            ``n_samples / (n_classes * n_class_samples)``. Class weights
            multiply ``sample_weight``, and C5.0 uses the products relative to
            their mean.

    Attributes:
        classes_: Original class labels in probability-column order.
        model_: Fitted low-level :class:`c50.Model`.
        n_features_in_: Number of input features seen by :meth:`fit`.
        feature_names_in_: String feature names when supplied by the input.
        categorical_features_: Indices of fitted categorical features.
        categories_: Categories for every feature. Continuous features have an
            empty category array.
        cost_matrix_: Validated copy of the fitted cost matrix, or ``None``.
        attribute_usage_: Fraction of training samples whose classification
            tests each feature with a known value. These are the values of
            C5.0's "Attribute usage" report divided by 100. Samples with zero
            weight are not counted.
        feature_importances_: ``attribute_usage_`` divided by its sum, or zeros
            when the classifier tests no feature.
    """

    classes_: NDArray[Any]
    model_: Model
    n_features_in_: int
    feature_names_in_: NDArray[Any]
    categorical_features_: NDArray[np.signedinteger[Any]]
    categories_: tuple[NDArray[Any], ...]
    cost_matrix_: NDArray[np.float64] | None
    attribute_usage_: NDArray[np.float64]
    feature_importances_: NDArray[np.float64]
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
        random_state: int | np.random.RandomState | None = None,
        ties: TieOrderName = "stable",
        n_jobs: int | None = None,
        categorical_features: Sequence[int | str]
        | Literal["from_dtype"]
        | None = None,
        unknown_categories: UnknownCategoryPolicy = "error",
        prediction_batch_size: int | None = 65536,
        cost_matrix: ArrayLike | None = None,
        class_weight: ClassWeight | None = None,
    ) -> None:
        """Store the parameters, which the class docstring describes."""
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
        self.random_state = random_state
        self.ties = ties
        self.n_jobs = n_jobs
        self.categorical_features = categorical_features
        self.unknown_categories = unknown_categories
        self.prediction_batch_size = prediction_batch_size
        self.cost_matrix = cost_matrix
        self.class_weight = class_weight

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike | None = None,
    ) -> C50Classifier:
        """Fit a C5.0 classifier.

        A failed fit clears all fitted state, including any previous model.

        Args:
            X: Dense two-dimensional training data.
            y: One-dimensional class labels.
            sample_weight: Optional nonnegative weight for each sample. C5.0
                uses weights relative to their mean, so multiplying every
                weight by a constant does not change the classifier. Samples
                with zero weight are removed before fitting.

        Returns:
            This fitted estimator.

        Raises:
            TypeError: If a continuous feature contains a non-numeric value
                other than :data:`c50.NOT_APPLICABLE`.
            ValueError: If the data, schema, options, or cost matrix is
                invalid.
        """
        self._clear_fitted_state()
        fitted = False
        try:
            self._validate_configuration()
            X = _host_array(X, "X")
            y = _host_array(y, "y")
            _reject_not_applicable_labels(y)
            sample_weight = _host_array(sample_weight, "sample_weight")
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

            weights = _relative_sample_weight(
                sample_weight,
                _class_sample_weight(self.class_weight, y_checked),
                y_checked.shape[0],
            )
            if weights is not None and not np.all(weights > 0):
                present = weights > 0
                X_checked = X_checked[present]
                y_checked = y_checked[present]
                weights = weights[present]

            label_encoder = LabelEncoder().fit(y_checked)
            classes = label_encoder.classes_
            if classes.shape[0] < 2:
                raise ValueError(
                    "C5.0 classification requires at least two classes; "
                    "one class was provided"
                )
            class_indices = label_encoder.transform(y_checked)

            feature_names = getattr(self, "feature_names_in_", None)
            schema = fit_schema(
                X_checked,
                self.categorical_features,
                feature_names,
            )
            if weights is not None:
                schema = dataclasses.replace(schema, case_weight=True)
            names_data = schema.names_data(classes.shape[0])
            training = schema.dense_data(X_checked, "error", weights)
            native_class_indices = np.asarray(class_indices, dtype=np.uintp)
            costs_data = encode_costs(classes.shape[0], self.cost_matrix)
            options = self._native_options()
            model = Model.train_dense(
                names_data,
                training.values,
                native_class_indices,
                self._native_model_kind(),
                options,
                costs_data,
                self._native_workers(),
                # training is private to this call, so no thread can modify its
                # arrays while training runs without the GIL.
                copy=False,
                not_applicable=training.not_applicable,
            )

            usage = self._attribute_usage(model, training)[
                : len(schema.features)
            ]
            total = usage.sum()

            self.classes_ = classes
            self.model_ = model
            self.attribute_usage_ = usage / training.values.shape[0]
            self.feature_importances_ = (
                usage / total if total else np.zeros_like(usage)
            )
            self.categorical_features_ = np.asarray(
                schema.categorical_indices,
                dtype=np.intp,
            )
            self.categories_ = tuple(
                np.asarray(categories, dtype=object)
                for categories in schema.categories
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
            "attribute_usage_",
            "feature_importances_",
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
        for row_slice, batch in self._prediction_batches(X_checked):
            details = predictor.predict_details_dense(
                batch.values, not_applicable=batch.not_applicable
            )
            indices[row_slice] = np.asarray(
                details.class_indices, dtype=np.intp
            )
        return self.classes_[indices]

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return class probabilities in ``classes_`` order.

        C5.0's native class scores need not sum to one: a ruleset averages the
        votes of the rules that match a case, and soft thresholds drop small
        branch weights. Each row is divided by its sum, which keeps the order
        of the classes. A row without any score assigns probability one to the
        predicted class. ``Model.predict_details`` returns the native scores.

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
        predicted = np.empty(X_checked.shape[0], dtype=np.intp)
        predictor = self.model_.prepare_predictor()
        for row_slice, batch in self._prediction_batches(X_checked):
            details = predictor.predict_details_dense(
                batch.values, not_applicable=batch.not_applicable
            )
            scores[row_slice] = np.asarray(details.scores, dtype=np.float64)
            predicted[row_slice] = np.asarray(
                details.class_indices, dtype=np.intp
            )
        return _normalize_scores(scores, predicted)

    def predict_log_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return the logarithms of :meth:`predict_proba`.

        Args:
            X: Dense two-dimensional prediction data.

        Returns:
            An array with the shape of :meth:`predict_proba`. A zero score
            becomes ``-inf``.

        Raises:
            ValueError: If the input shape or a feature value is invalid.
        """
        probabilities = self.predict_proba(X)
        with np.errstate(divide="ignore"):
            return np.log(probabilities)

    def export_text(
        self,
        *,
        max_depth: SupportsIndex | None = None,
        include_empty: bool = True,
    ) -> str:
        """Render the fitted classifier with the original names and labels.

        Features use ``feature_names_in_`` when available and otherwise
        ``feature_<index>``. Classes and categories use the string forms of
        ``classes_`` and ``categories_``.

        Args:
            max_depth: Maximum displayed tree depth, or ``None`` for all nodes.
            include_empty: Whether to display tree leaves without training
                support.

        Returns:
            The text produced by :func:`c50.export_text`.

        Raises:
            TypeError: If ``max_depth`` is neither ``None`` nor a count, as
                :func:`c50.export_text` raises.
            ValueError: If ``max_depth`` is negative.
        """
        check_is_fitted(self)
        return _export_text(
            self.model_.inspect(),
            **self._display_names(),
            max_depth=max_depth,
            include_empty=include_empty,
        )

    def export_json(self) -> str:
        """Export the fitted classifier's structure with original names.

        Names follow :meth:`export_text`. The result is metadata, not a
        loadable model. Persist the estimator or ``model_`` instead. Every
        per-feature list has ``n_features_in_`` entries, whether or not the fit
        used sample weights. The case-weight attribute that a weighted fit adds
        to the native schema does not appear, because no condition tests it.

        Returns:
            The JSON produced by :func:`c50.export_json`, without the
            case-weight entry.
        """
        check_is_fitted(self)
        text = _export_json(self.model_.inspect(), **self._display_names())
        if not self._schema.case_weight:
            return text
        # The native schema ends with the case-weight attribute, which no split
        # tests. Drop its entry from each per-feature list.
        count = self.n_features_in_
        document = json.loads(text)
        document["feature_names"] = document["feature_names"][:count]
        for group in (*document["trees"], *document["rulesets"]):
            group["feature_use"] = group["feature_use"][:count]
        return (
            json.dumps(document, indent=2, ensure_ascii=True, allow_nan=False)
            + "\n"
        )

    def _display_names(self) -> dict[str, Any]:
        """Map native schema names to the estimator's original names."""
        feature_names = (
            [str(name) for name in self.feature_names_in_]
            if hasattr(self, "feature_names_in_")
            else [f"feature_{index}" for index in range(self.n_features_in_)]
        )
        value_names: list[dict[str, str] | None] = [
            {
                f"value_{index}": str(value)
                for index, value in enumerate(categories)
            }
            for categories in self.categories_
        ]
        if self._schema.case_weight:
            feature_names.append("case weight")
            value_names.append(None)
        return {
            "feature_names": feature_names,
            "class_names": [str(label) for label in self.classes_],
            "value_names": value_names,
        }

    def __getstate__(self) -> dict[str, Any]:
        """Return the pickle state, which records the ``c50`` version."""
        state: dict[str, Any] = super().__getstate__()
        state[_VERSION_KEY] = _package_version()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore pickled state saved by the same ``c50`` version.

        The state holds private encoding state, so only the version that saved
        it can load it.

        Raises:
            ValueError: If the state records no version or another version than
                the running one.
        """
        state = dict(state)
        saved = state.pop(_VERSION_KEY, None)
        current = _package_version()
        if saved is None:
            raise ValueError(
                "this C50Classifier pickle records no c50 version; load it "
                f"with the version of c50 that saved it, not {current}"
            )
        if saved != current:
            raise ValueError(
                f"this C50Classifier pickle was saved by c50 {saved} and "
                f"cannot be loaded by c50 {current}"
            )
        super().__setstate__(state)

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

    def _validate_prediction_input(
        self, X: ArrayLike
    ) -> NDArray[Any] | Columns:
        check_is_fitted(self)
        # set_params() may change prediction parameters after fitting.
        self._validate_prediction_configuration()
        X = _host_array(X, "X")
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

    def _attribute_usage(
        self,
        model: Model,
        data: DenseData,
    ) -> NDArray[np.float64]:
        """Count, per C5.0 attribute, the training rows that use it."""
        rows, width = data.values.shape
        batch_size = self._batch_size() or max(1, rows)
        usage = np.zeros(width, dtype=np.float64)
        for start in range(0, rows, batch_size):
            batch = slice(start, start + batch_size)
            usage += model.attribute_usage_dense(
                data.values[batch],
                not_applicable=(
                    None
                    if data.not_applicable is None
                    else data.not_applicable[batch]
                ),
            )
        return usage

    def _prediction_batches(
        self,
        X: NDArray[Any] | Columns,
    ) -> Iterator[tuple[slice, DenseData]]:
        batch_size = self._batch_size() or max(1, X.shape[0])
        for start in range(0, X.shape[0], batch_size):
            stop = min(start + batch_size, X.shape[0])
            yield (
                slice(start, stop),
                self._schema.dense_data(
                    X[start:stop],
                    self.unknown_categories,
                    first_row=start,
                ),
            )

    def _validate_configuration(self) -> None:
        # Check types here so that errors name the estimator parameter. The
        # native options validate numeric ranges.
        for name in _BOOLEAN_PARAMETERS:
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise ValueError(f"{name} must be a Boolean")
        # Parameters hold whatever `set_params` stored, whatever their
        # annotations say.
        self._trials()
        for name in _REAL_PARAMETERS:
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, Real
            ):
                raise ValueError(f"{name} must be a real number")
            try:
                float(value)
            except (OverflowError, ValueError, TypeError) as error:
                raise ValueError(
                    f"{name} cannot be represented as a float"
                ) from error
        if self.model_kind not in ("tree", "rules"):
            raise ValueError("model_kind must be 'tree' or 'rules'")
        if self.ties not in ("reference", "stable"):
            raise ValueError("ties must be 'reference' or 'stable'")
        self._n_jobs()
        self._validate_prediction_configuration()

    def _validate_prediction_configuration(self) -> None:
        if self.unknown_categories not in ("error", "missing"):
            raise ValueError("unknown_categories must be 'error' or 'missing'")
        self._batch_size()

    # The three integer parameters are read only through these methods. Each
    # checks the rule in the estimator documentation and returns a Python int,
    # because the batch arithmetic and joblib follow NumPy's scalar promotion
    # for a NumPy integer. A method must not store the result in the parameter,
    # because an estimator must not change its parameters.

    def _trials(self) -> int:
        trials = _integral(self.trials)
        if trials is None or not 1 <= trials <= _MAX_TRIALS:
            raise ValueError(
                f"trials must be an integer from 1 through {_MAX_TRIALS}"
            )
        return trials

    def _n_jobs(self) -> int | None:
        value = self.n_jobs
        if value is None:
            return None
        n_jobs = _integral(value)
        if not n_jobs:
            raise ValueError("n_jobs must be None or a nonzero integer")
        return n_jobs

    def _batch_size(self) -> int | None:
        value = self.prediction_batch_size
        if value is None:
            return None
        batch_size = _integral(value)
        if batch_size is None or batch_size < 1:
            raise ValueError(
                "prediction_batch_size must be a positive integer or None"
            )
        return batch_size

    def _native_workers(self) -> int:
        return min(int(effective_n_jobs(self._n_jobs())), _MAX_WORKERS)

    def _native_model_kind(self) -> ModelKind:
        return ModelKind.TREE if self.model_kind == "tree" else ModelKind.RULES

    def _native_options(self) -> Options:
        options = Options()
        options.trials = self._trials()
        options.subset_splits = bool(self.subset_splits)
        options.winnow = bool(self.winnow)
        options.global_pruning = bool(self.global_pruning)
        options.probabilistic_thresholds = bool(self.probabilistic_thresholds)
        options.ignore_costs = bool(self.ignore_costs)
        options.minimum_cases = float(self.minimum_cases)
        options.confidence_factor = float(self.confidence_factor)
        options.sample_fraction = float(self.sample_fraction)
        random_state = check_random_state(self.random_state)
        # Sampling is the only native use of the seed. Without it, leave a
        # supplied RandomState unchanged.
        if self.sample_fraction:
            options.random_seed = int(random_state.randint(_SEED_COUNT))
        options.ties = (
            TieOrder.STABLE if self.ties == "stable" else TieOrder.REFERENCE
        )
        return options


def _integral(value: object) -> int | None:
    """Return a Python int for an integer parameter, or None if invalid.

    A parameter is an instance of ``numbers.Integral`` other than a ``bool``, a
    NumPy boolean, or a ``numpy.timedelta64``, as ``docs/estimator.md`` states.
    NumPy registers ``timedelta64`` as an integer type, although ``int``
    rejects a value with a unit.
    """
    if isinstance(value, (bool, np.bool_, np.timedelta64)) or not isinstance(
        value, Integral
    ):
        return None
    return int(value)


def _package_version() -> str:
    """Return the installed ``c50`` version, or ``"unknown"``."""
    try:
        return version("c50")
    except PackageNotFoundError:
        return "unknown"


def _normalize_scores(
    scores: NDArray[np.float64], predicted: NDArray[np.intp]
) -> NDArray[np.float64]:
    """Scale each row of native class scores to sum to one, in place."""
    totals = scores.sum(axis=1)
    scored = totals > 0
    scores[scored] /= totals[scored, np.newaxis]
    unscored = np.flatnonzero(~scored)
    scores[unscored] = 0.0
    scores[unscored, predicted[unscored]] = 1.0
    return scores


def _class_sample_weight(
    class_weight: ClassWeight | None,
    y: NDArray[Any],
) -> NDArray[np.float64] | None:
    """Return the class weight of each sample, or ``None`` without weights."""
    if class_weight is None:
        return None
    weights = np.asarray(
        compute_sample_weight(class_weight, y), dtype=np.float64
    )
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("class_weight values must be finite and nonnegative")
    return weights


def _relative_sample_weight(
    sample_weight: ArrayLike | None,
    class_weights: NDArray[np.float64] | None,
    sample_count: int,
) -> NDArray[np.float64] | None:
    """Validate sample weights, apply class weights, and divide by the mean.

    C5.0 stores case weights in single precision and uses only their ratios.
    Scaling first keeps uniform large or small weights representable.
    """
    if sample_weight is None:
        if class_weights is None:
            return None
        weights = class_weights
    else:
        weights = np.asarray(sample_weight, dtype=np.float64)
        if weights.ndim == 0:
            weights = np.full(sample_count, weights)
        if weights.shape != (sample_count,):
            raise ValueError(
                f"sample_weight has shape {weights.shape}, "
                f"expected ({sample_count},)"
            )
        if not np.all(np.isfinite(weights)):
            raise ValueError("sample_weight must contain only finite values")
        if np.any(weights < 0):
            raise ValueError("sample_weight must be nonnegative")
    nonzero = weights > 0
    factors = class_weights if sample_weight is not None else None
    if factors is not None:
        nonzero &= factors > 0
    if not np.any(nonzero):
        raise ValueError("sample_weight cannot be zero for every sample")

    # Keep ordinary arithmetic unchanged. Rescale mantissas and exponents only
    # when the product or mean becomes subnormal or overflows.
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        combined = weights if factors is None else weights * factors
        mean = combined[nonzero].mean()
    if not np.isfinite(mean) or np.any(
        combined[nonzero] < np.finfo(np.float64).tiny
    ):
        mantissas, exponents = np.frexp(weights[nonzero])
        if factors is not None:
            factor_mantissas, factor_exponents = np.frexp(factors[nonzero])
            mantissas *= factor_mantissas
            exponents += factor_exponents
        with np.errstate(under="ignore"):
            scaled = np.ldexp(mantissas, exponents - exponents.max())
        combined = np.zeros(sample_count, dtype=np.float64)
        combined[nonzero] = scaled
        mean = scaled.mean()
    with np.errstate(under="ignore"):
        relative: NDArray[np.float64] = combined / mean
    limits = np.finfo(np.float32)
    positive = relative[nonzero]
    if positive.min() < limits.tiny or positive.max() > limits.max:
        raise ValueError(
            "sample_weight ratios exceed the single-precision range of C5.0 "
            "case weights"
        )
    return relative


def _host_array(values: Any, name: str) -> Any:
    """Convert an array from another Array API library to NumPy.

    C5.0 runs on the CPU, so such arrays are converted through NumPy's array
    protocol. The array's library decides whether that is possible, which
    excludes arrays in GPU or other device memory.
    """
    if isinstance(values, np.ndarray) or not hasattr(
        values, "__array_namespace__"
    ):
        return values
    try:
        return np.asarray(values)
    except (BufferError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} cannot be converted to a NumPy array: {exc} C5.0 runs on "
            "the CPU, so move the array to CPU memory first."
        ) from exc


def _reject_not_applicable_labels(y: object) -> None:
    """Raise ``ValueError`` if ``y`` holds ``NOT_APPLICABLE``.

    A class label is always applicable. Numeric and string arrays cannot hold
    the marker, so only other inputs are searched.
    """
    dtype = getattr(y, "dtype", None)
    if isinstance(dtype, np.dtype) and dtype.kind != "O":
        return
    try:
        labels = np.asarray(y, dtype=object).ravel()
    except (TypeError, ValueError):
        # Let the input check report an input that is not an array of labels.
        return
    if any(label is NOT_APPLICABLE for label in labels):
        raise ValueError(
            "y contains c50.NOT_APPLICABLE; class labels cannot be not "
            "applicable"
        )


def _is_column_frame(X: object) -> bool:
    """Return whether ``X`` is a DataFrame to convert column by column.

    Empty and sparse frames use the whole-array conversion, which reports their
    errors.
    """
    pandas = sys.modules.get("pandas")
    return (
        pandas is not None
        and isinstance(X, pandas.DataFrame)
        and 0 not in X.shape
        and not any(
            isinstance(dtype, pandas.SparseDtype) for dtype in X.dtypes
        )
    )


def _validation_dtype(X: ArrayLike) -> type[object] | None:
    """Preserve numeric arrays while retaining mixed Python scalar types."""
    dtype = getattr(X, "dtype", None)
    if dtype is not None and np.dtype(dtype).kind in "iufb":
        return None
    return object


__all__ = ["NOT_APPLICABLE", "C50Classifier"]
