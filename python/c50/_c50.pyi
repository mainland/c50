from enum import Enum
from typing import Any, final

import numpy as np
from numpy.typing import NDArray


class C50Error(RuntimeError): ...


@final
class ModelKind(Enum):
    TREE = 0
    RULES = 1


@final
class TieOrder(Enum):
    REFERENCE = 0
    STABLE = 1


@final
class NodeKind(Enum):
    LEAF = 0
    DISCRETE = 1
    THRESHOLD = 2
    SUBSET = 3


@final
class ConditionKind(Enum):
    NOT_APPLICABLE = 0
    EQUALS = 1
    LESS_EQUAL = 2
    GREATER = 3
    IN_SUBSET = 4


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
    trials: int
    subset_splits: bool
    winnow: bool
    global_pruning: bool
    probabilistic_thresholds: bool
    ignore_costs: bool
    minimum_cases: float
    confidence_factor: float
    sample_fraction: float
    random_seed: int
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
    def predict_details_dense(self, values: NDArray[np.float64]) -> Predictions: ...


@final
class Model:
    @staticmethod
    def train(
        names: str,
        training_data: str,
        kind: ModelKind = ModelKind.TREE,
        options: Options = ...,
        costs: str = "",
        split_workers: int = 1,
    ) -> Model: ...
    @staticmethod
    def train_dense(
        names: str,
        values: NDArray[np.float64],
        class_indices: NDArray[np.integer[Any]],
        kind: ModelKind = ModelKind.TREE,
        options: Options = ...,
        costs: str = "",
        split_workers: int = 1,
    ) -> Model: ...
    @staticmethod
    def load(
        names: str,
        serialized_data: str,
        kind: ModelKind = ModelKind.TREE,
        costs: str = "",
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
    def classes_(self) -> list[str]: ...
    def prepare_predictor(self) -> Predictor: ...
    def inspect(self) -> ModelInspection: ...
    def predict_details(self, cases: str) -> Predictions: ...
    def predict_details_dense(
        self, values: NDArray[np.float64]
    ) -> Predictions: ...
    def predict(self, cases: str) -> list[str]: ...
    def predict_proba(self, cases: str) -> list[list[float]]: ...


def train(
    names: str,
    training_data: str,
    kind: ModelKind = ModelKind.TREE,
    options: Options = ...,
    costs: str = "",
    split_workers: int = 1,
) -> Model: ...


def train_dense(
    names: str,
    values: NDArray[np.float64],
    class_indices: NDArray[np.integer[Any]],
    kind: ModelKind = ModelKind.TREE,
    options: Options = ...,
    costs: str = "",
    split_workers: int = 1,
) -> Model: ...


def load(
    names: str,
    serialized_data: str,
    kind: ModelKind = ModelKind.TREE,
    costs: str = "",
) -> Model: ...
