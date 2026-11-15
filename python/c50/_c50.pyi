# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import enum
from typing import Any, SupportsIndex, final

import numpy as np
from numpy.typing import NDArray

class C50Error(RuntimeError): ...

class ModelKind(enum.IntEnum):
    """Serialized classifier representations supported by C5.0."""

    TREE = 0
    """A decision tree or boosted tree ensemble."""
    RULES = 1
    """A ruleset or boosted ruleset ensemble."""

class TieOrder(enum.IntEnum):
    """Order of equal continuous values when evaluating splits."""

    REFERENCE = 0
    """Reproduce the tie order of the reference C5.0 learner."""
    STABLE = 1
    """Keep equal values in case order with a faster sort.

    Classifiers are deterministic on every platform but may differ from the
    reference learner.
    """

class NodeKind(enum.IntEnum):
    """Tree node representation in an inspection snapshot."""

    LEAF = 0
    """Terminal class distribution."""
    DISCRETE = 1
    """One branch per categorical value, including N/A."""
    THRESHOLD = 2
    """Continuous cutoff, possibly with soft thresholds."""
    SUBSET = 3
    """Branches containing subsets of categorical values."""

class ConditionKind(enum.IntEnum):
    """A tree branch or rule condition, independent of node encodings."""

    NOT_APPLICABLE = 0
    """The feature has the distinct N/A value."""
    EQUALS = 1
    """The feature equals the one named category."""
    LESS_EQUAL = 2
    """The feature is at or below the cutoff."""
    GREATER = 3
    """The feature is above the cutoff."""
    IN_SUBSET = 4
    """The feature belongs to the named subset."""

@final
class SplitCondition:
    @property
    def feature(self) -> int: ...
    @property
    def kind(self) -> ConditionKind: ...
    @property
    def cut(self) -> float: ...
    @property
    def values(self) -> list[str]: ...
    @property
    def includes_not_applicable(self) -> bool: ...

@final
class ContinuousThreshold:
    @property
    def cut(self) -> float: ...
    @property
    def lower(self) -> float: ...
    @property
    def midpoint(self) -> float: ...
    @property
    def upper(self) -> float: ...

@final
class TreeBranch:
    @property
    def condition(self) -> SplitCondition: ...
    @property
    def child(self) -> int: ...

@final
class TreeNode:
    @property
    def kind(self) -> NodeKind: ...
    @property
    def feature(self) -> int | None: ...
    @property
    def predicted_class(self) -> int: ...
    @property
    def case_weight(self) -> float: ...
    @property
    def class_weights(self) -> list[float]: ...
    @property
    def threshold(self) -> ContinuousThreshold | None: ...
    @property
    def branches(self) -> list[TreeBranch]: ...

@final
class TreeInspection:
    @property
    def nodes(self) -> list[TreeNode]: ...
    @property
    def leaf_count(self) -> int: ...
    @property
    def supported_leaf_count(self) -> int: ...
    @property
    def depth(self) -> int: ...
    @property
    def supported_depth(self) -> int: ...
    @property
    def feature_use(self) -> list[int]: ...

@final
class RuleInspection:
    @property
    def conditions(self) -> list[SplitCondition]: ...
    @property
    def predicted_class(self) -> int: ...
    @property
    def cover(self) -> float: ...
    @property
    def correct(self) -> float: ...
    @property
    def prior(self) -> float: ...
    @property
    def vote(self) -> int: ...

@final
class RulesetInspection:
    @property
    def default_class(self) -> int: ...
    @property
    def rules(self) -> list[RuleInspection]: ...
    @property
    def feature_use(self) -> list[int]: ...

@final
class ModelInspection:
    @property
    def kind(self) -> ModelKind: ...
    @property
    def class_names(self) -> list[str]: ...
    @property
    def feature_names(self) -> list[str]: ...
    @property
    def trees(self) -> list[TreeInspection]: ...
    @property
    def rulesets(self) -> list[RulesetInspection]: ...

@final
class Options:
    def __init__(self) -> None: ...
    @property
    def trials(self) -> int: ...
    @trials.setter
    def trials(self, value: SupportsIndex) -> None: ...
    subset_splits: bool
    winnow: bool
    global_pruning: bool
    probabilistic_thresholds: bool
    ignore_costs: bool
    minimum_cases: float
    confidence_factor: float
    sample_fraction: float
    @property
    def random_seed(self) -> int: ...
    @random_seed.setter
    def random_seed(self, value: SupportsIndex) -> None: ...
    ties: TieOrder

@final
class Predictions:
    def __len__(self) -> int: ...
    @property
    def class_names(self) -> list[str]: ...
    @property
    def class_indices(self) -> list[int]: ...
    @property
    def labels(self) -> list[str]: ...
    @property
    def confidences(self) -> list[float]: ...
    @property
    def scores(self) -> list[list[float]]: ...

@final
class Predictor:
    def predict_details(self, cases: str) -> Predictions: ...
    def predict_details_dense(
        self,
        values: NDArray[np.float64],
        *,
        not_applicable: NDArray[np.bool_] | None = None,
    ) -> Predictions:
        """Predict a dense matrix.

        A true element of ``not_applicable`` marks that value as not
        applicable, which is distinct from NaN (missing).

        Raises:
            C50Error: If the model's schema has an implicit, ignored, label,
                date, time, or timestamp attribute, a class attribute, or a
                discrete attribute without explicit values, which the dense
                interface does not support. The number of columns does not
                matter.
        """

@final
class Model:
    @staticmethod
    def train(
        names: str,
        training_data: str,
        kind: ModelKind = ModelKind.TREE,
        options: Options = ...,
        costs: str = "",
        split_workers: SupportsIndex = 1,
    ) -> Model: ...
    @staticmethod
    def train_dense(
        names: str,
        values: NDArray[np.float64],
        class_indices: NDArray[np.integer[Any]],
        kind: ModelKind = ModelKind.TREE,
        options: Options = ...,
        costs: str = "",
        split_workers: SupportsIndex = 1,
        copy: bool = True,
        *,
        not_applicable: NDArray[np.bool_] | None = None,
    ) -> Model: ...
    @staticmethod
    def load(
        names: str,
        serialized_data: str,
        kind: ModelKind = ModelKind.TREE,
        costs: str = "",
        *,
        max_tree_depth: SupportsIndex | None = 1024,
    ) -> Model: ...
    @property
    def kind(self) -> ModelKind: ...
    @property
    def names_data(self) -> str: ...
    @property
    def serialized_data(self) -> str: ...
    @property
    def costs_data(self) -> str: ...
    @property
    def class_names(self) -> list[str]: ...
    def prepare_predictor(self) -> Predictor: ...
    def inspect(self) -> ModelInspection: ...
    def predict_details(self, cases: str) -> Predictions: ...
    def predict_details_dense(
        self,
        values: NDArray[np.float64],
        *,
        not_applicable: NDArray[np.bool_] | None = None,
    ) -> Predictions:
        """Predict a dense matrix.

        A true element of ``not_applicable`` marks that value as not
        applicable, which is distinct from NaN (missing).

        Raises:
            C50Error: If the model's schema has an implicit, ignored, label,
                date, time, or timestamp attribute, a class attribute, or a
                discrete attribute without explicit values, which the dense
                interface does not support. The number of columns does not
                matter.
        """
    def attribute_usage(self, cases: str) -> list[int]: ...
    def attribute_usage_dense(
        self,
        values: NDArray[np.float64],
        *,
        not_applicable: NDArray[np.bool_] | None = None,
    ) -> list[int]:
        """Count attribute usage for a dense matrix.

        A true element of ``not_applicable`` marks that value as not
        applicable, which is distinct from NaN (missing).

        Raises:
            C50Error: If the model's schema has an implicit, ignored, label,
                date, time, or timestamp attribute, a class attribute, or a
                discrete attribute without explicit values, which the dense
                interface does not support. The number of columns does not
                matter.
        """
    def predict(self, cases: str) -> list[str]: ...
    def predict_scores(self, cases: str) -> list[list[float]]: ...

def train(
    names: str,
    training_data: str,
    kind: ModelKind = ModelKind.TREE,
    options: Options = ...,
    costs: str = "",
    split_workers: SupportsIndex = 1,
) -> Model: ...
def train_dense(
    names: str,
    values: NDArray[np.float64],
    class_indices: NDArray[np.integer[Any]],
    kind: ModelKind = ModelKind.TREE,
    options: Options = ...,
    costs: str = "",
    split_workers: SupportsIndex = 1,
    copy: bool = True,
    *,
    not_applicable: NDArray[np.bool_] | None = None,
) -> Model: ...
def load(
    names: str,
    serialized_data: str,
    kind: ModelKind = ModelKind.TREE,
    costs: str = "",
    *,
    max_tree_depth: SupportsIndex | None = 1024,
) -> Model: ...
def _count_index(value: object, name: str) -> int: ...
