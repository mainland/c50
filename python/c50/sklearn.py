# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Scikit-learn-compatible estimator backed by the C5.0 core."""

from __future__ import annotations

import dataclasses
import decimal
import json
import re
import sys
import warnings
from collections.abc import Iterator, Mapping, Sequence
from decimal import Decimal, InvalidOperation
from fractions import Fraction
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
    check_array,
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
    is_integer_time_dtype,
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
        subset_splits: Allow subset splits for categorical features. Memory,
            and the time to fill the tables, grow roughly with the square of
            the largest number of categories in any categorical feature. The
            search for the best subsets can take time that grows with the cube
            of the number of categories of the feature under test.
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
            matrix. Its shape must be ``(n_classes, n_classes)``. It is
            converted like ``X``, and it must hold real numbers: a complex,
            datetime, or timedelta matrix raises ``ValueError``.
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
            X: Dense two-dimensional training data. ``X``, ``y``, and
                ``sample_weight`` may be NumPy arrays, lists, pandas objects,
                objects with ``__dataframe__`` or ``columns``, such as polars
                and pyarrow tables, or any other object that implements an
                array protocol (``__array__``, ``__array_interface__``,
                ``__array_struct__``, or the buffer protocol), such as a
                PyTorch CPU tensor or a polars or pyarrow array. Array API
                arrays and the objects of an array protocol are converted with
                :func:`numpy.asarray`. A ``numpy.ma.MaskedArray`` is read by
                its data values, without its mask.
            y: One-dimensional class labels.
            sample_weight: Optional weight for each sample, either one finite,
                nonnegative scalar or one such value per sample. C5.0 uses
                weights relative to their mean, so multiplying every weight by
                a constant does not change the classifier, provided that every
                weight stays within the range of ``float64`` after rounding to
                it. Samples with zero weight are removed before fitting. A
                weight is a ``bool``, ``int``, or ``float``, a NumPy boolean,
                integer, or floating-point number, a ``decimal.Decimal``, a
                ``fractions.Fraction``, or a string or bytes that parses as a
                number. Each must be of exactly one of these types, so convert
                a subclass, such as an ``enum.IntEnum`` member, with ``int``,
                ``float``, or ``str`` first. NumPy's own scalar types are
                accepted, including ``numpy.str_`` and ``numpy.bytes_``, which
                subclass ``str`` and ``bytes``. Convert any other real number,
                such as one from an arbitrary-precision library, with ``float``
                first. An array of any other kind, or an object of any other
                type, is rejected, including ``None``, ``pandas.NA``, dates,
                times, timedeltas, complex numbers, and the marker. A
                ``numpy.ma.MaskedArray`` is read without its mask, so set a
                weight to zero to remove its sample.

        Returns:
            This fitted estimator.

        Raises:
            TypeError: If a continuous feature contains a non-numeric value
                other than :data:`c50.NOT_APPLICABLE`, if a categorical value
                cannot be hashed, if ``X`` or ``y`` is a sparse matrix or a
                ``numpy.matrix``, or if ``sample_weight`` is an array of
                datetime, timedelta, complex, or structured kind or contains an
                object whose type is not exactly ``bool``, ``int``, ``float``,
                a NumPy boolean, integer, or floating-point number,
                ``decimal.Decimal``, ``fractions.Fraction``, ``str``, or
                ``bytes``, including a subclass of one of these other than
                ``numpy.str_`` and ``numpy.bytes_``, another real number, and
                the marker.
            ValueError: If the data, schema, options, or cost matrix is
                invalid, if an input that implements an array protocol cannot
                be converted to NumPy, as for a tensor in device memory or a
                128-bit polars integer array, or if ``sample_weight`` contains
                a string that does not parse as a number, a NaN, an infinity,
                or a negative number. A finite weight also raises it when
                rounding to ``float64`` makes it infinite, or makes it zero
                although it is not. Any ``Exception`` that the array's library
                raises during that conversion gives this ``ValueError``,
                chained to it, except ``MemoryError``.
            MemoryError: If the conversion of an input exhausts memory. It
                propagates, as does an exception that is not an ``Exception``,
                such as the ``PanicException`` that polars 2.0 raises for a
                128-bit integer array, and an error that scikit-learn raises
                while it converts a data frame of another library.
        """
        self._clear_fitted_state()
        fitted = False
        try:
            self._validate_configuration()
            X = _host_array(X, "X")
            y = _host_array(y, "y")
            _reject_not_applicable_labels(y)
            sample_weight = _host_array(sample_weight, "sample_weight")
            cost_matrix = _host_array(self.cost_matrix, "cost_matrix")
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
            training = schema.dense_data(X_checked, "error", weights)
            native_class_indices = np.asarray(class_indices, dtype=np.uintp)
            costs_data = encode_costs(classes.shape[0], cost_matrix)
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
                if cost_matrix is None
                else np.asarray(cost_matrix, dtype=float).copy()
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
            TypeError: If a continuous feature contains a non-numeric value
                other than :data:`c50.NOT_APPLICABLE`, if a categorical value
                cannot be hashed, or if ``X`` is a sparse matrix or a
                ``numpy.matrix``.
            ValueError: If the input shape or a feature value is invalid, or if
                an array cannot be converted to NumPy.
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
            X: Dense two-dimensional prediction data, as for :meth:`fit`.

        Returns:
            A two-dimensional array with one row per input and one column per
            fitted class.

        Raises:
            TypeError: If a continuous feature contains a non-numeric value
                other than :data:`c50.NOT_APPLICABLE`, if a categorical value
                cannot be hashed, or if ``X`` is a sparse matrix or a
                ``numpy.matrix``.
            ValueError: If the input shape or a feature value is invalid, or if
                an array cannot be converted to NumPy.
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
            X: Dense two-dimensional prediction data, as for :meth:`fit`.

        Returns:
            An array with the shape of :meth:`predict_proba`. A zero score
            becomes ``-inf``.

        Raises:
            TypeError: If a continuous feature contains a non-numeric value
                other than :data:`c50.NOT_APPLICABLE`, if a categorical value
                cannot be hashed, or if ``X`` is a sparse matrix or a
                ``numpy.matrix``.
            ValueError: If the input shape or a feature value is invalid, or if
                an array cannot be converted to NumPy.
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


_PYTHON_WEIGHT_TYPES: frozenset[type] = frozenset(
    {bool, int, float, Decimal, Fraction, str, bytes}
)
# NumPy's own scalar types: booleans, signed and unsigned integers, and
# floating-point numbers of every width, and its string and bytes scalars. Some
# codes name one type on a platform and another elsewhere, so the types come
# from the dtypes. ``numpy.timedelta64`` is not among them.
_NUMPY_WEIGHT_TYPES: frozenset[type] = frozenset(
    {np.dtype(code).type for code in "?" + np.typecodes["AllInteger"] + "efdg"}
    | {np.str_, np.bytes_}
)


def _is_weight_type(kind: type) -> bool:
    """Return whether objects of exactly one type are acceptable weights.

    The accepted types are a closed list: ``bool``, ``int``, ``float``,
    ``decimal.Decimal``, ``fractions.Fraction``, ``str``, and ``bytes``, and
    NumPy's own scalar types for booleans, integers, floating-point numbers,
    strings, and bytes. A subclass is not accepted, because NumPy converts an
    object through its own ``__float__``, which can disagree with the value
    that the exact checks read. Another real number, even one registered as a
    ``numbers.Real``, has no finiteness or zero test that the estimator can
    rely on, so it is not accepted either.
    """
    return kind in _PYTHON_WEIGHT_TYPES or kind in _NUMPY_WEIGHT_TYPES


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

    The weights have passed ``_host_array``, so only a NumPy array or a pandas
    object has a ``dtype`` that this function reads. A NumPy array keeps its
    dtype. A NumPy scalar of a type outside NumPy's own scalar types, such as a
    user subclass of ``numpy.float64``, is an object like any other. Arrays of
    boolean, integer, unsigned, float, and string kinds pass. A variable-width
    string array is read as an object array of its elements. In an object
    array, each element must be of an accepted type. A 0-d array element is
    replaced by the scalar that it holds, so that its dtype decides. Any other
    sequence or array element has no accepted type.

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
    elif isinstance(sample_weight, np.generic) and (
        type(sample_weight) not in _NUMPY_WEIGHT_TYPES
    ):
        # NumPy would replace the scalar by its base type in an object array.
        values = np.empty((), dtype=object)
        values[()] = sample_weight
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


def _weight_text(element: str | bytes) -> str | None:
    """Return the text of a string weight, or ``None`` if it cannot decode.

    NumPy's converter ignores the trailing NULs of a NumPy string or bytes
    scalar, as it does for the elements of a ``U`` or ``S`` array. The exact
    checks must read the same text, so they drop them from such a scalar. A
    Python string keeps them, and ``Decimal`` and ``float`` reject it.
    """
    if isinstance(element, np.str_):
        return str(element).rstrip("\x00")
    if isinstance(element, np.bytes_):
        element = bytes(element).rstrip(b"\x00")
    if isinstance(element, str):
        return element
    try:
        return element.decode()
    except UnicodeDecodeError:
        return None


# The context under which a string weight is parsed. ``Decimal`` signals an
# unparsable string through the calling thread's context, so a caller who has
# untrapped ``InvalidOperation`` would get a NaN instead of the exception that
# the checks rely on, and every unparsable string would set the caller's flag.
# Every setting is explicit, so that a change to ``decimal.DefaultContext``
# does not reach it.
_PARSE_CONTEXT = decimal.Context(
    prec=28,
    rounding=decimal.ROUND_HALF_EVEN,
    Emin=-999999,
    Emax=999999,
    capitals=1,
    clamp=0,
    flags=[],
    traps=[InvalidOperation],
)


def _weight_is_finite(element: Any) -> bool:
    """Return whether the exact value of an accepted weight is finite.

    The element has an accepted type. Decimal and string weights can hold
    values beyond ``float64``, so only a floating-point number needs a NumPy
    test, which handles ``longdouble`` without rounding. A string that
    ``Decimal`` cannot parse has an exponent too large for it. Such a string is
    not NaN or infinite, because ``Decimal`` parses those.
    """
    if isinstance(element, Decimal):
        return element.is_finite()
    if isinstance(element, (str, bytes)):
        text = _weight_text(element)
        if text is None:
            return True
        with decimal.localcontext(_PARSE_CONTEXT):
            try:
                return Decimal(text).is_finite()
            except InvalidOperation:
                return True
    if isinstance(element, (float, np.floating)):
        return bool(np.isfinite(element))
    return True


def _weight_is_zero(element: Any) -> bool:
    """Return whether the exact value of an accepted weight is zero.

    ``Decimal`` cannot parse a string with an exponent of very large magnitude,
    about ``10**18`` or more. Such a string is zero exactly when its mantissa
    is zero, whatever the exponent.
    """
    if isinstance(element, (str, bytes)):
        text = _weight_text(element)
        if text is None:
            return False
        with decimal.localcontext(_PARSE_CONTEXT):
            try:
                return Decimal(text) == 0
            except InvalidOperation:
                pass
            try:
                return Decimal(re.split("[eE]", text, maxsplit=1)[0]) == 0
            except InvalidOperation:
                return False
    return bool(element == 0)


def _float64_holds(kind: type) -> bool:
    """Return whether an accepted type cannot lose a weight to ``float64``.

    The type is exactly one of the accepted types. Such a weight cannot round
    to infinity, and it rounds to zero only when it is zero. That includes
    every integer, because an integer that is too large for ``float64`` raises
    ``OverflowError`` during conversion. Checking the exact value of these
    weights one by one is slow for a long list.
    """
    if kind in (bool, int, float):
        return True
    if issubclass(kind, np.floating):
        return np.dtype(kind).itemsize <= 8
    return issubclass(kind, (np.bool_, np.integer))


def _weight_is_lost(element: Any, overflowed: bool) -> bool:
    """Return whether rounding an accepted weight to ``float64`` loses it.

    A weight that rounded to infinity is lost if its exact value is finite. One
    that rounded to zero is lost if its exact value is not zero.
    """
    if _float64_holds(type(element)):
        return False
    if overflowed:
        return _weight_is_finite(element)
    return not _weight_is_zero(element)


def _convert_weights(values: NDArray[Any]) -> NDArray[np.float64]:
    """Round checked weights to ``float64``, or raise ``ValueError``.

    Each weight is rounded to the nearest ``float64``. A finite weight that
    rounds to infinity, or a nonzero weight that rounds to zero, makes the
    ratios between weights unrepresentable, so it raises ``ValueError``. This
    includes the ``OverflowError`` of a Python integer or a fraction that is
    too large. Weights that are already ``float64`` or narrower cannot leave
    the range, and a NaN or infinite weight is left to the caller's check.
    """
    message = (
        "sample_weight must stay finite, and nonzero where it is nonzero, "
        "when rounded to float64"
    )
    try:
        with np.errstate(over="ignore", under="ignore"):
            weights = np.asarray(values, dtype=np.float64)
    except OverflowError as exc:
        raise ValueError(message) from exc
    kind = values.dtype.kind
    if kind in "biu" or (kind == "f" and values.dtype.itemsize <= 8):
        return weights
    flat = weights.ravel()
    infinite = np.isinf(flat)
    suspect = np.flatnonzero((flat == 0) | infinite)
    elements = values.ravel()[suspect]
    overflows = infinite[suspect]
    if values.dtype.kind in "US":
        # Iterating a million NumPy strings is slower than the checks.
        elements, first = np.unique(elements, return_index=True)
        overflows = overflows[first]
    if all(_float64_holds(kind) for kind in set(map(type, elements))):
        return weights
    # A weight column holds few distinct zeros and infinities, so decide each
    # distinct one once. The key holds the type, so that equal objects of
    # different types, such as ``0`` and ``False`` or ``"0"`` and ``b"0"``,
    # keep their own answers. Equal objects of one type hold one value.
    lost: dict[tuple[bool, type, object], bool] = {}
    for element, overflowed in zip(elements, overflows.tolist(), strict=True):
        key = (overflowed, type(element), element)
        if key not in lost:
            lost[key] = _weight_is_lost(element, overflowed)
        if lost[key]:
            raise ValueError(message)
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
        weights = _convert_weights(_checked_weight_values(sample_weight))
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


# Inputs that need no conversion at the boundary: NumPy arrays and scalars,
# Python scalars, text, lists, tuples, and ``None``.
_PASSED_UNCHANGED = (
    np.ndarray,
    np.generic,
    bool,
    int,
    float,
    complex,
    str,
    bytes,
    list,
    tuple,
    type(None),
)

# The attributes through which NumPy reads an object as an array. The buffer
# protocol has no attribute, so ``_exports_buffer`` tests it.
_ARRAY_PROTOCOLS = (
    "__array__",
    "__array_interface__",
    "__array_struct__",
    "__array_namespace__",
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
    if hasattr(values, "__dataframe__"):
        return True
    with warnings.catch_warnings():
        # A polars LazyFrame warns about its performance when asked for its
        # columns, and the question here is only whether the attribute exists.
        warnings.simplefilter("ignore")
        return hasattr(values, "columns")


def _exports_buffer(values: object) -> bool:
    """Return whether ``values`` supports the buffer protocol."""
    try:
        memoryview(values)  # type: ignore[arg-type]
    except TypeError:
        return False
    return True


def _host_array(values: Any, name: str) -> Any:
    """Return ``values`` for validation, converting arrays that NumPy can read.

    Every array-like input of ``fit`` and the prediction methods passes through
    this function, which sorts it by what it is and never by what its
    attributes claim. After it, only a NumPy array or a pandas object is asked
    for its ``dtype``. In this order:

    1. NumPy arrays, including masked arrays, NumPy and Python scalars, text,
       lists, tuples, and ``None`` are returned unchanged.
    2. pandas objects are returned unchanged, so that columns, feature names,
       and ``pandas.NA`` keep their own handling.
    3. Objects with ``__dataframe__`` or a ``columns`` attribute, such as
       polars and pyarrow tables, are returned unchanged, so that scikit-learn
       reads their feature names.
    4. Arrays from an Array API library are converted with ``numpy.asarray``.
    5. Any other object that implements an array protocol, ``__array__``,
       ``__array_interface__``, ``__array_struct__``, or the buffer protocol,
       is converted with ``numpy.asarray``, which keeps the dtype that the
       array's library hands over, such as ``datetime64[ns]``. Any other
       object, such as a sparse matrix, is returned unchanged for
       scikit-learn to judge.

    C5.0 runs on the CPU, so the array's library decides whether the conversion
    is possible, which excludes arrays in GPU or other device memory, arrays
    that require gradients, and dtypes that NumPy lacks. A NumPy scalar is
    already on the CPU, and converting it would replace a subclass by its base
    type.

    Args:
        values: The input.
        name: The name of the parameter, for error messages.

    Returns:
        The input, or a NumPy array that holds it.

    Raises:
        ValueError: If an array's library, or the probe of the input, raises an
            ``Exception`` other than ``MemoryError``, whatever its type.
        MemoryError: If the conversion exhausts memory. It propagates, as does
            a ``BaseException`` that is not an ``Exception``, such as the
            ``PanicException`` of polars, because catching it would also catch
            ``KeyboardInterrupt``.
    """
    if isinstance(values, _PASSED_UNCHANGED) or _is_pandas_object(values):
        return values
    try:
        if _is_data_frame(values):
            return values
        if not (
            any(hasattr(values, protocol) for protocol in _ARRAY_PROTOCOLS)
            or _exports_buffer(values)
        ):
            return values
        return np.asarray(values)
    except MemoryError:
        # Exhausted memory is not a problem of the input, as the package
        # documentation promises.
        raise
    except Exception as exc:
        raise ValueError(
            f"{name} cannot be converted to a NumPy array: {exc} C5.0 runs on "
            "the CPU, so the array must be in CPU memory and have a dtype "
            "that NumPy supports."
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
    """
    if (
        isinstance(X, np.ndarray)
        or _is_pandas_object(X)
        or not _is_data_frame(X)
    ):
        return None
    try:
        array = np.asarray(X)
    except Exception:
        # The conversion that scikit-learn performs reports the error.
        return None
    if array.ndim != 2 or not is_integer_time_dtype(array.dtype):
        return None
    return array


__all__ = ["NOT_APPLICABLE", "C50Classifier"]
