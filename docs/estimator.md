# Scikit-learn estimator

The optional `C50Classifier` adapts dense array-like data to the low-level C5.0
text interface. Install it with NumPy and scikit-learn support:

```sh
python -m pip install 'c50-gpl[sklearn]'
```

Importing `c50` does not import NumPy or scikit-learn. Applications that use
the estimator import it from the optional module:

```python
import numpy as np

from c50.sklearn import C50Classifier

X = np.asarray([[0.0], [0.5], [2.5], [3.0]])
y = np.asarray(["low", "low", "high", "high"])

classifier = C50Classifier(minimum_cases=1).fit(X, y)
labels = classifier.predict([[0.25], [2.75]])
probabilities = classifier.predict_proba([[0.25], [2.75]])
```

`C50Classifier` follows the scikit-learn estimator protocol. Constructor
arguments remain unchanged until `fit`, learned public attributes use trailing
underscores, and fitted instances work with cloning, pipelines, grid search,
and pickling.

## Input contract

`X` must be a dense, two-dimensional array-like object. Sparse matrices are not
supported. `y` must be a one-dimensional classification target containing at
least two classes. `classes_` records the original labels in the same order as
the columns returned by `predict_proba`.

With `categorical_features=None`, numeric columns are continuous, and Boolean
or non-numeric columns are categorical. Set `categorical_features` to a
sequence of zero-based indices to override inference. String column names may
also be used when the input exposes string feature names. When an explicit
sequence is present, every unlisted column is continuous.

`None` and numeric NaN values represent missing feature values. Infinite
continuous values are rejected. An unseen prediction-time category raises
`ValueError` by default. Set `unknown_categories="missing"` to pass unseen
categories to C5.0 as missing values.

The adapter maps class labels and categorical values to internal tokens before
constructing the C5.0 names and case buffers. Delimiters and punctuation in
Python values therefore cannot change the generated input grammar. The
original labels and categories remain available through `classes_` and
`categories_`.

## Native options and costs

The estimator exposes the low-level training options as cloneable constructor
parameters. `model_kind="tree"` builds a tree, and `model_kind="rules"` builds
a rule set. The remaining parameters control boosting, subset splits,
winnowing, pruning, probabilistic thresholds, sampling, and their corresponding
native numeric settings.

`cost_matrix` accepts an optional square array in predicted-by-actual order.
Entry `[predicted, actual]` is the cost of predicting the row class when the
column class is correct. Entries must be finite and nonnegative, and diagonal
entries must be zero.

## Ownership and concurrency

`fit` stores the low-level immutable model in `model_`. Prediction delegates to
that model and creates independent native operation state. Independent fitted
estimators and concurrent prediction operations may run in different threads.
Do not call `fit` on an estimator while another thread is using the same
estimator.
