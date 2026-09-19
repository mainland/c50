from enum import Enum
from typing import final


class C50Error(RuntimeError): ...


@final
class ModelKind(Enum):
    TREE = 0
    RULES = 1


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
class Model:
    @staticmethod
    def train(
        names: str,
        training_data: str,
        kind: ModelKind = ModelKind.TREE,
        options: Options = ...,
        costs: str = "",
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
    def predict_details(self, cases: str) -> Predictions: ...
    def predict(self, cases: str) -> list[str]: ...
    def predict_proba(self, cases: str) -> list[list[float]]: ...


def train(
    names: str,
    training_data: str,
    kind: ModelKind = ModelKind.TREE,
    options: Options = ...,
    costs: str = "",
) -> Model: ...


def load(
    names: str,
    serialized_data: str,
    kind: ModelKind = ModelKind.TREE,
    costs: str = "",
) -> Model: ...
