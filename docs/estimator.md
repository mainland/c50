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
A `Decimal` NaN, quiet or signaling, is missing like any other numeric NaN.
Infinite continuous values, and finite values beyond the single-precision
range of about 3.4e38, are rejected with a `ValueError` that names the feature
and, for numeric columns, the first row. C5.0 stores continuous values as
single-precision floats. An unseen prediction-time category raises
`ValueError` by default. Set `unknown_categories="missing"` to pass
unseen categories to C5.0 as missing values. A category that a pandas
`category` dtype declared during fitting, but that no training row used, is
always passed as a missing value, as in `HistGradientBoostingClassifier`.
Such categories are not in `categories_` and do not change the classifier, so
a filtered DataFrame that keeps unused categories trains the same classifier
as one without them.

Real numbers match by value, and other values match only equal values of
their own type. An integer category matches an equal float, so a model fitted
on an integer column accepts the float column that results when a missing
value promotes the column to `float64`, and the reverse. An object column that
holds `1` and `1.0` declares one category, and `categories_` keeps the `int`
form. `Fraction(1)` joins them, because `Fraction` is a real number. `True`
and `1` stay distinct, as do strings and numbers. `Decimal` and `complex` are
not real numbers in this sense, so `Decimal(1)` is a separate category from
`1`, and a model fitted on the integers 1 and 2 rejects `Decimal(1)` as unseen
under `unknown_categories="error"`. Equal `Decimal` values, and equal `complex`
values, form one category whose `categories_` entry has a canonical form, such
as `Decimal("1")` for `Decimal("1.0")`. A `float32` value converts to the
double with the same value, so it matches a fitted `float64` category only
when the two are equal as doubles. `0.5` matches, while `float32(0.1)` does not
match `0.1`. A model fitted on `float32` data matches `float32` values and the
equal doubles.

Values of any other type keep the form of their first occurrence in
`categories_`. Equal aware `datetime` values in different time zones, such as
12:00 UTC and 14:00 at UTC+2, are one category, and `categories_` and the value
names in `export_text` show the form that occurs first in the training rows.
The same holds for the tuples `(1,)` and `(1.0,)`. The classifier and its
predictions do not depend on the form.

Dates and durations are categorical in every unit. NumPy converts most units
to Python objects when it converts an array to `object`, and the categories are
those objects: `datetime.date` for `datetime64` in `Y`, `M`, `W`, and `D`,
`datetime.datetime` for `datetime64` in `h`, `m`, `s`, `ms`, and `us`, and
`datetime.timedelta` for `timedelta64` in `W`, `D`, `h`, `m`, `s`, `ms`, and
`us`. NumPy converts `datetime64` in `ns`, `ps`, `fs`, and `as`, and
`timedelta64` in `Y`, `M`, `ns`, `ps`, `fs`, and `as`, to `int`. The estimator
keeps those values as `numpy.datetime64` and `numpy.timedelta64` scalars, so
they stay categories. The two forms are different types, so they never match. A
model fitted on `datetime64[ns]` data rejects the same instants in
`datetime64[us]` as unseen under `unknown_categories="error"`, and the reverse.
Cast the data to the unit of the training data first. Values of two units in
the scalar group, such as `ns` and `ps`, have one type and match by value.
Arrays, lists or object arrays of scalars, and polars and pyarrow tables whose
columns are all nanosecond dates or durations give the same classifier and the
same `categories_`. polars and pyarrow hold no other unit of the scalar group.
A polars or pyarrow table that mixes a nanosecond column with columns of
another type reaches C5.0 as the library's own conversion of the whole table
makes it, and polars converts a nanosecond date in a table with float columns
to floats, so cast those columns to `us` first. `NaT` is missing in every unit.

The estimator first converts a data frame, other than a pandas one, once
with `numpy.asarray` to find out whether it holds dates or durations. An
`Exception` from that conversion raises `ValueError`, and a `MemoryError`
propagates, even when a second conversion would succeed.

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
quickly as a numeric NumPy array. Nullable numeric columns, such as `Int64` and
`Float64`, are encoded the same way when their features are continuous.
`category` and `string` columns are encoded from their codes, so each distinct
value is examined once. Prediction matches values rather than codes, so a
prediction frame may order or extend its categories differently. Other columns,
including object columns, become object arrays and are encoded one value at a
time. Empty and sparse DataFrames are converted as whole arrays. A DataFrame
without rows raises `ValueError` from the scikit-learn input check, which
requires at least one sample. A DataFrame without columns raises `ValueError`
from NumPy, whose message reads "at least one array or dtype is required". A
DataFrame whose columns are all sparse raises `TypeError` and asks for dense
data. A DataFrame that mixes sparse and other columns is converted to a dense
array with scikit-learn's `UserWarning` about sparse columns, and then trains
and predicts. `fit` and `predict` raise the same exceptions.

The order in which categories are declared to C5.0 can change the classifier,
as {doc}`compatibility` describes. The estimator therefore derives the order
from the training values, not from the order of the rows. A pandas `category`
column keeps its dtype's order, without unused categories. Other categorical
values are grouped by type, with types ordered by module and name, and sorted
within each type. All real numbers other than `bool` form one group, so `int`
and `float` values sort together. `categories_` lists each feature's categories
in this order. Values of a type that cannot be sorted keep their order of first
occurrence. On the OpenML adult data, four random permutations of the training
rows each trained the same classifier as the original order.

pandas is a test dependency, not a runtime requirement. The estimator uses CPU
NumPy arrays and returns NumPy arrays. It declares `array_api_support=False`.
It does not promise preservation of another array namespace, device, or dtype.

C5.0 runs on the CPU. The estimator converts arrays from other Array API
libraries and CPU tensors, such as PyTorch tensors, with `numpy.asarray` for
`X`, `y`, and `sample_weight` in `fit` and for `X` in the prediction methods.
It accepts `__array_namespace__` and `__array__` with or without
`array_api_dispatch`. Results are NumPy arrays.

The array's library decides whether conversion is possible. An array in GPU
or other device memory, a tensor that requires gradients, and a `bfloat16`
tensor raise `ValueError` that names the input and chains the library's
exception. Other conversion failures that raise `Exception` give the same
`ValueError`, except `MemoryError`. A `MemoryError` and any exception outside
the `Exception` hierarchy propagate. Move an array to CPU memory, detach it,
or convert its dtype first. See the
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

`ties` selects the order of equal continuous values when the learner evaluates
splits. The estimator's default, `"stable"`, is faster than `"reference"` and
is deterministic on every platform, but may build a classifier that differs
from the one that Release 2.07 builds from the same data. Pass
`ties="reference"` for Release 2.07's order. `c50.train`, the C++ options, and
`c5.0` default to the reference order. See {doc}`compatibility`.

`random_state` controls training-row sampling, which is the only native use
of randomness. When `sample_fraction` is positive, the estimator draws the
native seed, an integer from 0 through 4095, from
`check_random_state(random_state)`. An integer therefore gives reproducible
sampling, and `None` draws from NumPy's global random state. Without sampling,
no seed is drawn, and a supplied `RandomState` is not advanced.

`cost_matrix` accepts an optional square array in predicted-by-actual order.
Entry `[predicted, actual]` is the cost of predicting the row class when the
column class is correct. The array must hold real numbers. An array of complex,
datetime, or timedelta values, and any object that does not convert to `float`,
such as a `dict`, a generator, `pandas.NA`, or a complex number in an object
array, raises `ValueError`. Entries must be finite and nonnegative, and
diagonal entries must be zero. Costs must fit the native single-precision
representation. Each actual-class column must have a positive, representable
sum of error costs. Consequently, binary off-diagonal costs must be positive.
Individual zero costs are supported in multiclass matrices whose columns
satisfy this requirement.

`ignore_costs=True` ignores costs during both training and prediction. The
model retains no costs data in this mode.

## Class probabilities

`predict_proba` returns rows that sum to one. C5.0's native class scores need
not: a ruleset averages the Laplace accuracies of the rules that match a case,
so its scores sum to less than one, and soft thresholds drop branches that
receive less than 1% of a case. The estimator divides each row by its sum,
which keeps the order of the classes. `predict` returns a class with the
highest probability unless a cost matrix is in use. Under exact ties C5.0
chooses a later class, so that class may differ from the one that
`classes_[predict_proba(X).argmax(axis=1)]` selects, because `argmax` returns
the first. A case with no score at all receives probability one for its predicted
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
or a scalar. A weight is a real number, including `bool` and `decimal.Decimal`,
or a string. The estimator checks the type before it converts any weight to
`float64`:

- Arrays of boolean, integer, unsigned, and floating-point kinds are accepted,
  as are Python and NumPy real numbers and `decimal.Decimal` in lists, object
  arrays, and pandas columns.
- A string that parses as a number is accepted, and one that does not raises
  `ValueError`.
- Anything else raises `TypeError`: arrays of datetime, timedelta, complex, and
  structured kinds, and in an object array `None`, `pandas.NA`, `pandas.NaT`,
  dates, times, timedeltas, complex numbers, and any other object.
- A weight element that is a 0-d array is judged by its dtype. The element
  check comes first, before the shape check. Sequence or array elements of one
  common length whose items are of accepted types make a weight array with more
  than one dimension, which raises `ValueError`. Items of any other type, such
  as `None`, raise `TypeError`. A list, tuple, range, `pandas.Series`, or array
  with one or more dimensions as an element also raises `TypeError`, in a
  ragged list or in an object array.
- A NaN or infinite number, including a `decimal.Decimal`, raises `ValueError`.
  A missing value in an array from another library raises the error for the
  value that the conversion produces. A null in a polars or pyarrow numeric
  array becomes NaN and raises `ValueError`. A null in a string or Boolean
  array becomes `None` and raises `TypeError`.

The estimator passes the weights to C5.0 as its case-weight attribute. C5.0
uses case weights when it counts cases but never tests them in a classifier.
Without `sample_weight`, the names buffer has no case-weight attribute, and the
classifier is unchanged.

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
and preserve the reference classifier behavior, except that `ties` defaults to
`"stable"` for `C50Classifier`, which may build a different classifier than the
reference learner. Pass `ties="reference"` for Release 2.07's order.

Scikit-learn compatibility does not require the parameters accepted by
`DecisionTreeClassifier`. Alternative split criteria would define different
training modes. A future alternative must be implemented and validated in the
C++ core before it is exposed through Python. Reference C5.0 split selection must
remain the only criterion. No alternative criterion is implemented.

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
