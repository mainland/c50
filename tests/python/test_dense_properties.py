"""Property tests for equivalent text and dense native input."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

import c50


@dataclass(frozen=True)
class Dataset:
    """One generated C5.0 schema and its equivalent input encodings."""

    names: str
    text_training: str
    text_cases: str
    dense_training: NDArray[np.float64]
    dense_cases: NDArray[np.float64]
    class_indices: NDArray[np.uintp]
    kind: c50.ModelKind
    subset_splits: bool
    winnow: bool
    global_pruning: bool
    probabilistic_thresholds: bool
    trials: int
    minimum_cases: float
    confidence_factor: float
    sample_fraction: float
    random_seed: int
    costs: str


@st.composite
def datasets(draw: st.DrawFn) -> Dataset:
    """Generate a small supported mixed-feature classification problem."""
    row_count = draw(st.integers(min_value=6, max_value=24))
    continuous_count = draw(st.integers(min_value=1, max_value=3))
    categorical_count = draw(st.integers(min_value=0, max_value=2))
    class_count = draw(st.integers(min_value=2, max_value=min(4, row_count)))
    category_counts = draw(
        st.lists(
            st.integers(min_value=2, max_value=4),
            min_size=categorical_count,
            max_size=categorical_count,
        )
    )

    continuous_values = [
        draw(
            st.lists(
                st.one_of(
                    st.none(),
                    st.integers(min_value=-8, max_value=8).map(
                        lambda value: value / 2
                    ),
                ),
                min_size=row_count,
                max_size=row_count,
            )
        )
        for _ in range(continuous_count)
    ]
    categorical_values = [
        draw(
            st.lists(
                st.one_of(
                    st.none(),
                    st.integers(min_value=0, max_value=value_count - 1),
                ),
                min_size=row_count,
                max_size=row_count,
            )
        )
        for value_count in category_counts
    ]
    class_indices = np.asarray(
        [row % class_count for row in range(row_count)],
        dtype=np.uintp,
    )

    columns = continuous_values + categorical_values
    dense_training = np.asarray(
        [
            [np.nan if column[row] is None else column[row] for column in columns]
            for row in range(row_count)
        ],
        dtype=np.float64,
    )

    class_names = [f"class_{index}" for index in range(class_count)]
    names_lines = [", ".join(class_names) + ".", ""]
    for feature_index in range(continuous_count):
        names_lines.append(f"feature_{feature_index}: continuous.")
    for category_index, value_count in enumerate(category_counts):
        feature_index = continuous_count + category_index
        values = ", ".join(
            f"value_{value_index}" for value_index in range(value_count)
        )
        names_lines.append(f"feature_{feature_index}: {values}.")
    names = "\n".join(names_lines) + "\n"

    training_rows = []
    case_rows = []
    for row in range(row_count):
        fields = []
        for continuous_column in continuous_values:
            value = continuous_column[row]
            fields.append("?" if value is None else format(value, ".17g"))
        for categorical_column in categorical_values:
            category = categorical_column[row]
            fields.append("?" if category is None else f"value_{category}")
        training_rows.append(
            ", ".join([*fields, class_names[int(class_indices[row])]])
        )
        case_rows.append(", ".join([*fields, "?"]))

    use_costs = draw(st.booleans())
    costs = ""
    if use_costs:
        predicted, actual = draw(st.sampled_from([(0, 1), (1, 0)]))
        cost = draw(st.integers(min_value=2, max_value=5))
        costs = f"{class_names[predicted]}, {class_names[actual]}: {cost}\n"

    sample_fraction = 0.0
    random_seed = 0
    if row_count >= 12:
        sample_fraction = draw(st.sampled_from([0.0, 0.5, 0.75]))
        if sample_fraction:
            random_seed = draw(st.integers(min_value=0, max_value=4095))

    return Dataset(
        names=names,
        text_training="\n".join(training_rows) + "\n",
        text_cases="\n".join(case_rows) + "\n",
        dense_training=dense_training,
        dense_cases=dense_training.copy(),
        class_indices=class_indices,
        kind=draw(st.sampled_from([c50.ModelKind.TREE, c50.ModelKind.RULES])),
        subset_splits=draw(st.booleans()),
        winnow=draw(st.booleans()),
        global_pruning=draw(st.booleans()),
        probabilistic_thresholds=draw(st.booleans()),
        trials=draw(st.integers(min_value=1, max_value=3)),
        minimum_cases=draw(st.sampled_from([1.0, 2.0, 3.0])),
        confidence_factor=draw(st.sampled_from([0.05, 0.25, 0.5])),
        sample_fraction=sample_fraction,
        random_seed=random_seed,
        costs=costs,
    )


def stable_model_data(model: c50.Model) -> str:
    """Remove the date-bearing identifier and retain classifier content."""
    _, separator, content = model.serialized_data.partition("\n")
    assert separator
    return content


def assert_same_predictions(
    left: c50.Predictions,
    right: c50.Predictions,
) -> None:
    """Require identical labels, confidences, and class-score vectors.

    Args:
        left: First prediction result.
        right: Prediction result expected to match the first.
    """
    assert left.class_names == right.class_names
    assert left.class_indices == right.class_indices
    assert np.isfinite(left.confidences).all()
    assert np.isfinite(left.scores).all()
    assert np.isfinite(right.confidences).all()
    assert np.isfinite(right.scores).all()
    np.testing.assert_array_equal(left.confidences, right.confidences)
    np.testing.assert_array_equal(left.scores, right.scores)


@given(dataset=datasets())
@settings(
    deadline=None,
    max_examples=40,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_text_and_dense_interfaces_are_equivalent(dataset: Dataset) -> None:
    options = c50.Options()
    options.subset_splits = dataset.subset_splits
    options.winnow = dataset.winnow
    options.global_pruning = dataset.global_pruning
    options.probabilistic_thresholds = dataset.probabilistic_thresholds
    options.trials = dataset.trials
    options.minimum_cases = dataset.minimum_cases
    options.confidence_factor = dataset.confidence_factor
    options.sample_fraction = dataset.sample_fraction
    options.random_seed = dataset.random_seed

    text_model = c50.train(
        dataset.names,
        dataset.text_training,
        dataset.kind,
        options,
        dataset.costs,
    )
    dense_model = c50.train_dense(
        dataset.names,
        dataset.dense_training,
        dataset.class_indices,
        dataset.kind,
        options,
        dataset.costs,
    )

    assert stable_model_data(dense_model) == stable_model_data(text_model)

    text_predictions = text_model.predict_details(dataset.text_cases)
    dense_predictions = dense_model.predict_details_dense(dataset.dense_cases)
    assert_same_predictions(dense_predictions, text_predictions)

    loaded_text_model = c50.load(
        text_model.names_data,
        text_model.serialized_data,
        text_model.kind,
        text_model.costs_data,
    )
    loaded_dense_model = c50.load(
        dense_model.names_data,
        dense_model.serialized_data,
        dense_model.kind,
        dense_model.costs_data,
    )
    assert_same_predictions(
        loaded_text_model.predict_details(dataset.text_cases),
        text_predictions,
    )
    assert_same_predictions(
        loaded_dense_model.predict_details_dense(dataset.dense_cases),
        dense_predictions,
    )
