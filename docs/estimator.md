# Scikit-learn estimator

The optional `C50Classifier` adapts dense array-like data to the native dense
C5.0 interface. Install it with NumPy and scikit-learn support:

```sh
python -m pip install 'c50[sklearn]'
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
sequence is present, every unlisted column is continuous. Set
`categorical_features="from_dtype"` to treat exactly the DataFrame columns with
a pandas `category` dtype as categorical, as `HistGradientBoostingClassifier`
does. Every other column, and every column of non-DataFrame input, is then
continuous.

A categorical column with an ordered pandas `category` dtype becomes a C5.0
ordered discrete attribute. Its categories keep the dtype's order, and C5.0
tests it with one three-way split at a cut between consecutive categories
instead of one branch per category. Use an unordered dtype to treat the
categories as unordered. C5.0 treats an ordered attribute with fewer than three
categories in the training data as unordered.

`None`, numeric NaN, `pd.NA`, and `pd.NaT` represent missing feature values.
Infinite continuous values are rejected. An unseen prediction-time category
raises `ValueError` by default. Set `unknown_categories="missing"` to pass
unseen categories to C5.0 as missing values. A category that a pandas
`category` dtype declared during fitting, but that no training row used, is
always passed as a missing value, as in `HistGradientBoostingClassifier`.
Such categories are not in `categories_` and do not change the classifier, so
a filtered DataFrame that keeps unused categories trains the same classifier
as one without them.

The adapter maps class labels and categorical values to internal tokens before
constructing the C5.0 names buffer and dense feature matrix. Delimiters and
punctuation in Python values therefore cannot change the generated input
grammar. The original labels and categories remain available through
`classes_` and `categories_`.

Prediction is processed in batches of 65,536 rows by default so native case
storage and conversion scratch space do not grow with the complete prediction
input. Set `prediction_batch_size` to a different positive integer, or to
`None` for one batch. Training remains in-memory and retains both the encoded
float64 matrix and native case data while the classifier is built. See
{doc}`large-datasets` for the scale target, limitations, and benchmark.

## pandas and Array API support

DataFrames with string column names preserve `feature_names_in_`. Numeric,
Boolean, string, and categorical columns use the same encoding rules as NumPy
arrays. Prediction requires the fitted column names in their original order.
Nullable pandas columns need no conversion because `pd.NA` is a missing value.

The estimator converts a DataFrame one column at a time. Numeric and Boolean
columns with NumPy dtypes keep those dtypes, so numeric columns are encoded as
quickly as a numeric NumPy array. Nullable numeric columns, such as `Int64`
and `Float64`, are encoded the same way when their features are continuous.
`category` and `string` columns are encoded from their codes, so each distinct
value is examined once. Prediction matches values rather than codes, so a
prediction frame may order or extend its categories differently. Other columns, including
object columns, become object arrays and are encoded one value at a time.
Empty and sparse DataFrames are converted as whole arrays, so they raise the
same errors as before.

The order in which categories are declared to C5.0 can change the classifier,
as {doc}`compatibility` describes. The estimator therefore derives the order
from the training values, not from the order of the rows. A pandas `category`
column keeps its dtype's order, without unused categories. Other categorical
values are grouped by type, with types ordered by module and name, and sorted
within each type. `categories_` lists each feature's categories in this order.
Values of a type that cannot be sorted keep their order of first occurrence.
On the OpenML adult data, four random permutations of the training rows each
trained the same classifier as the original order.

pandas is a test dependency, not a runtime requirement. The estimator uses CPU
NumPy arrays and returns NumPy arrays. It declares `array_api_support=False`.
It does not promise preservation of another array namespace, device, or dtype.

C5.0 runs on the CPU. Arrays from other Array API libraries, such as PyTorch
CPU tensors, are converted to NumPy arrays with `numpy.asarray` for `X`, `y`,
and `sample_weight`, with or without `array_api_dispatch`. Results are NumPy
arrays. The array's library decides whether it can be converted. An array in
GPU or other device memory raises `ValueError`, so move it to CPU memory
first. See the
[scikit-learn Array API contract](https://scikit-learn.org/stable/modules/array_api.html).

CI installs pandas and sets `SCIPY_ARRAY_API=1` before importing SciPy or
scikit-learn so the available NumPy Array API estimator checks run as well.

## Native options and costs

The estimator exposes the low-level training options as cloneable constructor
parameters. `model_kind="tree"` builds a tree, and `model_kind="rules"` builds
a rule set. The remaining parameters control boosting, subset splits,
winnowing, pruning, probabilistic thresholds, sampling, and their corresponding
native numeric settings.

`n_jobs` sets the maximum number of native split-evaluation workers. `None`
means one worker unless a `joblib.parallel_config` context sets another value,
and negative values count back from the number of processors, so `-1` uses all
of them. At most 8 workers are used, and only eligible large nodes use more
than one. The worker count does not change the classifier.

`random_state` controls training-row sampling, which is the only native use
of randomness. When `sample_fraction` is positive, the estimator draws the
native seed, an integer from 0 through 4095, from
`check_random_state(random_state)`. An integer therefore gives reproducible
sampling, and `None` draws from NumPy's global random state. Without sampling,
no seed is drawn, and a supplied `RandomState` is not advanced.

`cost_matrix` accepts an optional square array in predicted-by-actual order.
Entry `[predicted, actual]` is the cost of predicting the row class when the
column class is correct. Entries must be finite and nonnegative, and diagonal
entries must be zero. Costs must fit the native single-precision representation.
Each actual-class column must have a positive, representable sum of error costs.
Consequently, binary off-diagonal costs must be positive. Individual zero costs
are supported in multiclass matrices whose columns satisfy this requirement.

`ignore_costs=True` ignores costs during both training and prediction. The
model retains no costs data in this mode.

## Class probabilities

`predict_proba` returns rows that sum to one. C5.0's native class scores need
not: a ruleset averages the Laplace accuracies of the rules that match a case,
so its scores sum to less than one, and soft thresholds drop branches that
receive less than 1% of a case. The estimator divides each row by its sum,
which keeps the order of the classes, so the class with the highest
probability is the class that `predict` returns unless a cost matrix is in
use. A case with no score at all receives probability one for its predicted
class. The low-level `Model.predict_details` and `predict_details_dense`
methods return the native scores. See {doc}`algorithm` for how C5.0 computes
them.

## Feature importances

After `fit`, `attribute_usage_` holds, for each feature, the fraction of
training samples whose classification tests that feature with a known value.
These are the values of C5.0's "Attribute usage" report divided by 100. The
root attribute of a tree has usage one, and the values need not sum to one.
Samples with zero weight are not counted.

`feature_importances_` divides `attribute_usage_` by its sum, so it sums to
one, or is all zeros when the classifier tests no feature. It measures how
often C5.0 consults a feature, not how much the feature reduces impurity, so
it is not comparable with the impurity-based importances of scikit-learn
trees. Tools such as `SelectFromModel` use it for relative ranking. Usage is
computed after training on the training rows, in batches of
`prediction_batch_size`.

## Sample weights

`fit` accepts `sample_weight`, either one finite, nonnegative weight per sample
or a scalar. The estimator passes the weights to C5.0 as its case-weight
attribute. C5.0 uses case weights when it counts cases but never tests them in
a classifier. Without `sample_weight`, the names buffer has no case-weight
attribute, and the classifier is unchanged.

C5.0 divides each case weight by the mean weight, so only the ratios between
weights matter. Multiplying every weight by a constant does not change the
classifier. The estimator divides the weights by their mean before training.
The resulting ratios must fit C5.0's single-precision case weights.

A weight of 2 is therefore not equivalent to repeating a sample. Repetition
increases the total weight, while C5.0 rescales the weights to the number of
samples. Case-count thresholds such as `minimum_cases` and pruning estimates
see different totals. For this reason, scikit-learn's
`check_sample_weight_equivalence_on_dense_data` check fails with the default
options.

C5.0 counts a zero or negative case weight as a weight of one. The estimator
instead follows scikit-learn and removes samples with zero weight before
fitting, so their labels and categories do not appear in `classes_` or
`categories_`. Negative weights are rejected.

`class_weight` weights samples by class. It accepts a mapping from class label
to weight, in which unlisted classes have weight one, or `"balanced"`, which
gives each class the weight `n_samples / (n_classes * n_class_samples)`. The
estimator computes the class weights with scikit-learn's
`compute_sample_weight` and multiplies them by `sample_weight`, so the rules
above apply to the products. Only the ratios between class weights matter, and
the samples of a class with weight zero are removed. Class weights change the training
distribution, while `cost_matrix` changes the cost of each error. Both may be
used together.

## Scikit-learn workflows

Tests fit the estimator in pipelines with a pandas `ColumnTransformer`, grid
search, cross-validation with worker processes, permutation importance,
probability calibration, and bagging, AdaBoost, and stacking ensembles.

With metadata routing enabled, request `sample_weight` for fitting with
`set_fit_request(sample_weight=True)`. Search and cross-validation tools also
pass `sample_weight` to scoring, so request or decline it there with
`set_score_request`. Without routing, pass weights to a pipeline step as
`step__sample_weight`.

`BaggingClassifier` passes each bootstrap sample to estimators that accept
`sample_weight` as weights, the number of times each sample was drawn. C5.0
rescales those weights to their mean, so a sample drawn twice is not the same
as a repeated sample, as described above. `AdaBoostClassifier`
rejects missing values in its own input validation, before C5.0 receives them.

## Split selection and evaluation scoring

`C50Classifier` uses the native C5.0 split-selection policy: gain ratio with
minimum-gain filtering, the native MDL adjustment, and C5.0 tie handling. It has
no `criterion` parameter. The C++ and Python interfaces use the same learner
and preserve the reference classifier behavior.

Scikit-learn compatibility does not require the parameters accepted by
`DecisionTreeClassifier`. Alternative split criteria would define different
training modes. A future alternative must be implemented and validated in the
C++ core before it is exposed through Python. Reference C5.0 behavior must
remain the default. No alternative criterion is implemented.

Choose an evaluation metric through scikit-learn's `scoring` interface. For
example, reuse `X` and `y` from the example above to select native training
parameters by balanced accuracy while also recording macro-averaged F1:

```python
from sklearn.model_selection import GridSearchCV

search = GridSearchCV(
    C50Classifier(),
    {"minimum_cases": [1, 2], "confidence_factor": [0.1, 0.25]},
    scoring={"balanced_accuracy": "balanced_accuracy", "f1_macro": "f1_macro"},
    refit="balanced_accuracy",
    cv=2,
)
search.fit(X, y)
classifier = search.best_estimator_
```

The scores evaluate predictions on held-out folds. `refit` chooses the metric
used to select the parameters and refit the estimator on all supplied data.
These settings do not change split selection within a fit. Without an explicit
scorer, the estimator's `score` method reports classification accuracy. See
[scikit-learn's scoring documentation](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter)
for named metrics and custom scorers.

## Ownership and concurrency

`fit` stores the low-level immutable model in `model_`. Prediction delegates to
that model and creates independent native operation state. Independent fitted
estimators and concurrent prediction operations may run in different threads.
A failed `fit` clears all learned attributes and leaves the estimator unfitted,
including when it previously held a fitted model. A later successful `fit` may
reuse the estimator.

Do not call `fit` on an estimator while another thread is using the same
estimator.
