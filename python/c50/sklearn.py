"""Scikit-learn-compatible estimator backed by the C5.0 core."""

from __future__ import annotations

import dataclasses
import json
import sys
from collections.abc import Iterator, Mapping, Sequence
from decimal import Decimal
from numbers import Integral, Real
from typing import Any, Literal, cast

import numpy as np
from joblib import effective_n_jobs
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import Tags, check_random_state
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import (
    check_array,
    check_consistent_length,
    check_is_fitted,
    validate_data,
)

from ._c50 import Model, ModelKind, Options, TieOrder
from ._data import (
    Columns,
    Schema,
    dataframe_columns,
    encode_costs,
    fit_schema,
    is_integer_time_dtype,
)
from .inspection import export_json as _export_json
from .inspection import export_text as _export_text

type ModelKindName = Literal["tree", "rules"]
type TieOrderName = Literal["reference", "stable"]
type UnknownCategoryPolicy = Literal["error", "missing"]
type ClassWeight = Mapping[Any, float] | Literal["balanced"]

# Native limits on the boosting trials, sampling seed, and split workers.
_SEED_COUNT = 4096
_MAX_TRIALS = 1000
_MAX_WORKERS = 8
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

    The estimator converts arrays to the native dense representation, then
    delegates training and prediction to the low-level binding. It does not
    implement a separate learning algorithm.

    Args:
        model_kind: Build a decision tree or a rule set.
        trials: Number of boosting trials, from 1 through 1000. One disables
            boosting.
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
            ``"reference"`` reproduces the reference C5.0 learner.
            ``"stable"`` is faster and deterministic on every platform, but
            may build a classifier that differs from the reference learner.
        n_jobs: Number of native split-evaluation workers. ``None`` means
            one unless a ``joblib.parallel_config`` context sets another
            value, and ``-1`` means all processors. At most 8 workers are
            used. The worker count does not change the classifier.
        categorical_features: Categorical column indices or, for inputs with
            string feature names, column names. ``None`` infers feature types.
            ``"from_dtype"`` selects exactly the DataFrame columns with a
            pandas ``category`` dtype.
        unknown_categories: Raise an error for an unseen prediction-time
            category or pass it to C5.0 as a missing value.
        prediction_batch_size: Maximum rows converted and predicted in one
            native call. ``None`` processes all prediction rows together.
        cost_matrix: Optional predicted-by-actual misclassification-cost
            matrix. Its shape must be ``(n_classes, n_classes)``. It must hold
            real numbers. A complex, datetime, or timedelta matrix raises
            ``ValueError``.
        class_weight: Optional weights for classes, as a mapping from class
            label to weight or ``"balanced"``, which weights each class by
            ``n_samples / (n_classes * n_class_samples)``. Class weights
            multiply ``sample_weight``, and C5.0 uses the products relative
            to their mean.

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
        feature_importances_: ``attribute_usage_`` divided by its sum, or
            zeros when the classifier tests no feature.
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
        ties: TieOrderName = "reference",
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
            X: Dense two-dimensional training data. ``X``, ``y``, and
                ``sample_weight`` may be NumPy arrays, pandas objects, or
                arrays such as PyTorch CPU tensors that implement
                ``__array_namespace__`` or ``__array__``. The estimator
                converts such tensors and Array API arrays with
                :func:`numpy.asarray`.
            y: One-dimensional class labels.
            sample_weight: Optional nonnegative weight for each sample. C5.0
                uses weights relative to their mean, so multiplying every
                weight by a constant does not change the classifier. Samples
                with zero weight are removed before fitting. A weight is a real
                number, including a ``bool`` or ``decimal.Decimal``, or a
                string that parses as one. Arrays of other kinds and objects
                such as ``None``, ``pandas.NA``, dates, times, timedeltas, and
                complex numbers are rejected.

        Returns:
            This fitted estimator.

        Raises:
            TypeError: If a continuous feature contains a non-numeric value, or
                if ``sample_weight`` is an array of datetime, timedelta,
                complex, or structured kind or holds an object other than a
                real number, ``decimal.Decimal``, or a string.
            ValueError: If the data, schema, options, or cost matrix is
                invalid, or if an input conversion to NumPy raises an
                ``Exception`` other than ``MemoryError``, as for a tensor in
                device memory. An invalid numeric string in ``sample_weight``
                also raises ``ValueError``.
            MemoryError: If an input conversion exhausts memory. Other
                exceptions outside the ``Exception`` hierarchy also propagate.
        """
        self._clear_fitted_state()
        fitted = False
        try:
            self._validate_configuration()
            X = _host_array(X, "X")
            y = _host_array(y, "y")
            sample_weight = _host_array(sample_weight, "sample_weight")
            X_checked: NDArray[Any] | Columns
            if _is_column_frame(X):
                y_checked = validate_data(self, y=y, reset=True)
                validate_data(self, X, reset=True, skip_check_array=True)
                check_consistent_length(X, y_checked)
                X_checked = dataframe_columns(X)
            elif (stamps := _time_frame(X)) is not None:
                y_checked = validate_data(self, y=y, reset=True)
                validate_data(self, X, reset=True, skip_check_array=True)
                X_checked = _check_time_frame(self, stamps)
                check_consistent_length(X, y_checked)
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
            training_values = schema.dense_data(X_checked, "error", weights)
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
                self._native_workers(),
                # training_values is private to this call, so no thread can
                # modify it while training runs without the GIL.
                copy=False,
            )

            usage = self._attribute_usage(model, training_values)[
                : len(schema.features)
            ]
            total = usage.sum()

            self.classes_ = classes
            self.model_ = model
            self.attribute_usage_ = usage / training_values.shape[0]
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
            X: Dense two-dimensional prediction data, as for :meth:`fit`.

        Returns:
            An array of labels with the fitted class-label dtype.

        Raises:
            ValueError: If the input shape or a feature value is invalid, or if
                an array cannot be converted to NumPy.
        """
        X_checked = self._validate_prediction_input(X)
        indices = np.empty(X_checked.shape[0], dtype=np.intp)
        predictor = self.model_.prepare_predictor()
        for row_slice, values in self._prediction_batches(X_checked):
            details = predictor.predict_details_dense(values)
            indices[row_slice] = np.asarray(
                details.class_indices, dtype=np.intp
            )
        return self.classes_[indices]

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return class scores in ``classes_`` order.

        Args:
            X: Dense two-dimensional prediction data, as for :meth:`fit`.

        Returns:
            A two-dimensional array with one row per input and one column per
            fitted class.

        Raises:
            ValueError: If the input shape or a feature value is invalid, or if
                an array cannot be converted to NumPy.
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

    def predict_log_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return the logarithms of :meth:`predict_proba`.

        Args:
            X: Dense two-dimensional prediction data, as for :meth:`fit`.

        Returns:
            An array with the shape of :meth:`predict_proba`. A zero score
            becomes ``-inf``.

        Raises:
            ValueError: If the input shape or a feature value is invalid, or if
                an array cannot be converted to NumPy.
        """
        probabilities = self.predict_proba(X)
        with np.errstate(divide="ignore"):
            return np.log(probabilities)

    def export_text(
        self,
        *,
        max_depth: int | None = None,
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
        if (stamps := _time_frame(X)) is not None:
            validate_data(self, X, reset=False, skip_check_array=True)
            return _check_time_frame(self, stamps)
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
        values: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Count, per C5.0 attribute, the training rows that use it."""
        # A NumPy batch size would wrap or become a float in the slicing below.
        batch_size = int(self.prediction_batch_size or max(1, values.shape[0]))
        usage = np.zeros(values.shape[1], dtype=np.float64)
        for start in range(0, values.shape[0], batch_size):
            usage += model.attribute_usage_dense(
                values[start : start + batch_size]
            )
        return usage

    def _prediction_batches(
        self,
        X: NDArray[Any] | Columns,
    ) -> Iterator[tuple[slice, NDArray[np.float64]]]:
        batch_size = (
            max(1, X.shape[0])
            if self.prediction_batch_size is None
            # A NumPy integer would take part in NumPy's scalar arithmetic,
            # which wraps or becomes a float when added to the start row.
            else int(self.prediction_batch_size)
        )
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
        trials: object = self.trials
        if (
            isinstance(trials, (bool, np.bool_))
            or not isinstance(trials, Integral)
            or not 1 <= int(trials) <= _MAX_TRIALS
        ):
            raise ValueError(
                f"trials must be an integer from 1 through {_MAX_TRIALS}"
            )
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
        # Parameters hold whatever `set_params` stored, whatever their
        # annotations say.
        n_jobs: object = self.n_jobs
        if n_jobs is not None and (
            isinstance(n_jobs, bool)
            or not isinstance(n_jobs, Integral)
            or n_jobs == 0
        ):
            raise ValueError("n_jobs must be None or a nonzero integer")
        self._validate_prediction_configuration()

    def _validate_prediction_configuration(self) -> None:
        if self.unknown_categories not in ("error", "missing"):
            raise ValueError("unknown_categories must be 'error' or 'missing'")
        batch_size: object = self.prediction_batch_size
        if batch_size is not None and (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, Integral)
            or batch_size < 1
        ):
            raise ValueError(
                "prediction_batch_size must be a positive integer or None"
            )

    def _native_workers(self) -> int:
        # joblib adds a NumPy n_jobs to the processor count in NumPy's
        # arithmetic, so it receives a Python int.
        n_jobs = None if self.n_jobs is None else int(self.n_jobs)
        return min(int(effective_n_jobs(n_jobs)), _MAX_WORKERS)

    def _native_model_kind(self) -> ModelKind:
        return ModelKind.TREE if self.model_kind == "tree" else ModelKind.RULES

    def _native_options(self) -> Options:
        options = Options()
        options.trials = int(self.trials)
        options.subset_splits = bool(self.subset_splits)
        options.winnow = bool(self.winnow)
        options.global_pruning = bool(self.global_pruning)
        options.probabilistic_thresholds = bool(self.probabilistic_thresholds)
        options.ignore_costs = bool(self.ignore_costs)
        options.minimum_cases = float(self.minimum_cases)
        options.confidence_factor = float(self.confidence_factor)
        options.sample_fraction = float(self.sample_fraction)
        random_state = check_random_state(self.random_state)
        # Sampling is the only native use of the seed. Without it, leave
        # a supplied RandomState unchanged.
        if self.sample_fraction:
            options.random_seed = int(random_state.randint(_SEED_COUNT))
        options.ties = (
            TieOrder.STABLE if self.ties == "stable" else TieOrder.REFERENCE
        )
        return options


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


def _is_weight_type(kind: type) -> bool:
    """Return whether objects of a type are acceptable weights.

    A weight is a real number, a ``decimal.Decimal``, or a string.
    ``numpy.timedelta64`` subclasses ``numpy.signedinteger``, so it needs an
    explicit exclusion.
    """
    if issubclass(kind, np.timedelta64):
        return False
    return issubclass(kind, (Real, Decimal, np.bool_, str, bytes))


def _is_weight_number(element: object) -> bool:
    """Return whether an object element of ``sample_weight`` is acceptable."""
    return _is_weight_type(type(element))


def _checked_weight_objects(values: NDArray[Any]) -> NDArray[Any]:
    """Return an object array of weights after checking each element.

    Checking each distinct type once keeps the check fast for a long list of
    one type. A failure reports the first element of an unacceptable type.
    """
    flat = values.ravel()
    if all(_is_weight_type(kind) for kind in set(map(type, flat))):
        return values
    for element in flat:
        if not _is_weight_number(element):
            raise TypeError(
                "sample_weight must hold numbers or strings, not "
                f"{type(element).__name__}"
            )
    return values


def _unwrap_zero_dimensional_elements(values: NDArray[Any]) -> NDArray[Any]:
    """Return an object array with each 0-d array element replaced.

    The scalar that a 0-d array holds keeps its dtype, so that the element
    check judges it like any other element. A sequence or an array with
    dimensions stays in place, and the element check rejects it.
    """
    flat = values.ravel()
    if not any(issubclass(kind, np.ndarray) for kind in set(map(type, flat))):
        return values
    unwrapped = flat.copy()
    for index, element in enumerate(flat):
        if isinstance(element, np.ndarray) and element.ndim == 0:
            unwrapped[index] = element[()]
    return unwrapped.reshape(values.shape)


def _checked_weight_values(sample_weight: ArrayLike) -> NDArray[Any]:
    """Return the weights as an array, or raise unless they are numbers.

    The check runs before any conversion to ``float64``, which would turn
    dates, ``None``, and other objects into numbers or NaN, and would drop the
    imaginary part of complex numbers. The caller converts the returned array,
    never the original object, so that the checked elements are the converted
    ones.

    Only a NumPy or pandas object has a ``dtype`` that this function reads. A
    NumPy array keeps its dtype. Arrays of boolean, integer, unsigned, float,
    and string kinds pass. A variable-width string array is read as an object
    array of its elements. In an object array, each element must be a real
    number, a ``decimal.Decimal``, or a string. A 0-d array element is replaced
    by the scalar that it holds, so that its dtype decides. Any other sequence
    or array element has no accepted type.

    A pandas object with an extension dtype, such as ``Int64``, converts its
    missing values to NaN under a default ``numpy.asarray``. It is converted to
    an object array instead, so that ``pandas.NA`` is rejected like any other
    object. A list, a tuple, or a scalar becomes an object array of its
    original elements, because NumPy's dtype inference turns a mixture of
    strings and complex numbers into strings.
    """
    pandas = sys.modules.get("pandas")
    dtype = (
        getattr(sample_weight, "dtype", None)
        if isinstance(sample_weight, (np.ndarray, np.generic))
        or _is_pandas_object(sample_weight)
        else None
    )
    values: NDArray[Any]
    if pandas is not None and isinstance(
        dtype, pandas.api.extensions.ExtensionDtype
    ):
        # An extension dtype belongs to a pandas object, which has `to_numpy`.
        values = np.asarray(cast(Any, sample_weight).to_numpy(dtype=object))
    elif isinstance(dtype, np.dtype):
        values = np.asarray(sample_weight)
    else:
        values = np.asarray(sample_weight, dtype=object)
    kind = values.dtype.kind
    if kind == "T":
        values = values.astype(object)
        kind = "O"
    if kind in "biufUS":
        return values
    if kind == "O":
        return _checked_weight_objects(
            _unwrap_zero_dimensional_elements(values)
        )
    raise TypeError(
        f"sample_weight must hold numbers or strings, not dtype {values.dtype}"
    )


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
        weights = np.asarray(
            _checked_weight_values(sample_weight), dtype=np.float64
        )
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

    # Keep ordinary arithmetic unchanged. Rescale mantissas and exponents
    # only when the product or mean becomes subnormal or overflows.
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
    """Convert a CPU array from another library to NumPy.

    C5.0 runs on the CPU, so arrays that implement ``__array_namespace__`` or
    ``__array__`` are converted through NumPy's array protocol. The array's
    library decides whether conversion is possible, which excludes arrays in
    device memory, arrays that require gradients, and dtypes that NumPy lacks.
    NumPy arrays and scalars and pandas objects keep their usual validation.
    The ``__array__`` path leaves data frames for frame validation.

    Args:
        values: The input.
        name: The name of the parameter, for error messages.

    Returns:
        The input, or a NumPy array that holds it.

    Raises:
        ValueError: If conversion raises an ``Exception`` other than
            ``MemoryError``, whatever its type.
        MemoryError: If the conversion exhausts memory. It propagates, as does
            a ``BaseException`` that is not an ``Exception``.
    """
    if isinstance(values, (np.ndarray, np.generic)) or _is_pandas_object(
        values
    ):
        return values
    if not hasattr(values, "__array_namespace__") and (
        not hasattr(values, "__array__") or _is_data_frame(values)
    ):
        return values
    try:
        return np.asarray(values)
    except MemoryError:
        # Exhausted memory is not a problem of the input, as the package
        # documentation promises.
        raise
    except Exception as exc:
        raise _conversion_error(name, exc) from exc


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
        and not any(
            isinstance(dtype, pandas.SparseDtype) for dtype in X.dtypes
        )
    )


def _is_pandas_object(values: object) -> bool:
    """Return whether ``values`` is a pandas container.

    pandas is looked up, not imported, so that it stays optional.
    """
    pandas = sys.modules.get("pandas")
    return pandas is not None and isinstance(
        values,
        (
            pandas.DataFrame,
            pandas.Series,
            pandas.Index,
            pandas.api.extensions.ExtensionArray,
        ),
    )


def _is_data_frame(values: object) -> bool:
    """Return whether ``values`` has ``__dataframe__`` or ``columns``."""
    return hasattr(values, "__dataframe__") or hasattr(values, "columns")


def _conversion_error(name: str, exc: Exception) -> ValueError:
    """Return the ``ValueError`` for an input that NumPy cannot convert."""
    return ValueError(
        f"{name} cannot be converted to a NumPy array: {exc} C5.0 runs on "
        "the CPU, so the array must be in CPU memory and have a dtype "
        "that NumPy supports."
    )


def _validation_dtype(X: ArrayLike) -> type[object] | None:
    """Preserve numeric arrays while retaining mixed Python scalar types.

    An array of dates or durations whose values NumPy converts to ``int``
    objects also keeps its dtype, because scikit-learn's conversion to
    ``object`` would turn the values into integers and make the feature
    continuous. Iterating the array yields ``numpy.datetime64`` and
    ``numpy.timedelta64`` scalars, which the encoder keeps.
    """
    if isinstance(X, np.ndarray) or _is_pandas_object(X):
        dtype = getattr(X, "dtype", None)
        if dtype is not None and np.dtype(dtype).kind in "iufb":
            return None
    if isinstance(X, np.ndarray) and is_integer_time_dtype(X.dtype):
        return None
    return object


def _check_time_frame(
    estimator: C50Classifier, stamps: NDArray[Any]
) -> NDArray[Any]:
    """Check the sample and feature counts of a table of dates or durations.

    ``validate_data`` skips its array check for such a table, so that it can
    read the feature names of the table, and this check restores the rest of
    it. A table without rows or columns raises ``ValueError``, as it does in
    any other container.
    """
    return cast(
        NDArray[Any],
        check_array(
            stamps, dtype=None, ensure_all_finite=False, estimator=estimator
        ),
    )


def _time_frame(X: object) -> NDArray[Any] | None:
    """Return a table of integer-converted dates or durations as an array.

    Only a data frame that is not a pandas object and whose conversion to a
    NumPy array holds dates or durations of the units of
    ``is_integer_time_dtype`` qualifies, as a polars table or a pyarrow table
    of nanosecond columns does. scikit-learn converts such a frame through
    ``object``, which turns the values into integers. For any other input the
    result is ``None``.

    Raises:
        ValueError: If the conversion raises an ``Exception`` other than
            ``MemoryError``, as ``_host_array`` does. A conversion that fails
            once is not retried, because a retry that succeeds would discard
            the error.
        MemoryError: If the conversion exhausts memory. It propagates, as does
            a ``BaseException`` that is not an ``Exception``.
    """
    if (
        isinstance(X, np.ndarray)
        or _is_pandas_object(X)
        or not _is_data_frame(X)
    ):
        return None
    try:
        array = np.asarray(X)
    except MemoryError:
        raise
    except Exception as exc:
        raise _conversion_error("X", exc) from exc
    if array.ndim != 2 or not is_integer_time_dtype(array.dtype):
        return None
    return array


__all__ = ["C50Classifier"]
