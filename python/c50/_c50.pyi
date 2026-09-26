from typing import Any, ClassVar, final

import numpy as np
from numpy.typing import NDArray

class C50Error(RuntimeError): ...

@final
class ModelKind:
    """A pybind11 enumeration, not an ``enum.Enum``.

    It cannot be iterated or subscripted. Use the members, ``__members__``,
    ``name``, and ``value``.
    """

    TREE: ClassVar[ModelKind]
    RULES: ClassVar[ModelKind]
    __members__: ClassVar[dict[str, ModelKind]]
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...
    def __int__(self) -> int: ...
    def __index__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

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
