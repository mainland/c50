# Tutorial

This tutorial trains C5.0 classifiers with the scikit-learn estimator on two
public datasets. It covers categorical features, missing values, reading trees
and rules, boosting, tuning, misclassification costs, and saving models. For
the algorithm behind each step, see {doc}`algorithm`.

Install the estimator, pandas, and scikit-learn:

```sh
python -m pip install 'c50[sklearn]' pandas
```

The examples download their data from [OpenML](https://www.openml.org) with
scikit-learn's `fetch_openml`, which verifies each file's checksum and caches
it under `~/scikit_learn_data`. Only the first run needs network access. Run
the code blocks in order, in one Python session.

## Load the data

The [Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/) data
records 344 penguins of three species from three islands in the Palmer
Archipelago, Antarctica. OpenML dataset 42585 is a fixed version of it:

```python
from sklearn.datasets import fetch_openml

penguins = fetch_openml(data_id=42585, as_frame=True)
X, y = penguins.data, penguins.target
print(X.dtypes)
print(X.isna().sum())
```

`island` and `sex` are pandas `category` columns, and the four measurements
are `float64`. Two penguins have no measurements, and 10 have no recorded sex.
C5.0 handles these missing values itself, so they need no imputation.

One penguin's sex is recorded as `_`. It is not a real category, so turn it
into a missing value:

```python
X = X.assign(sex=X["sex"].cat.remove_categories("_"))
```

Without this step, the estimator would learn `_` as a category whenever that
penguin is in the training data.

## Fit a tree

Hold out a quarter of the penguins for testing, keeping the species
proportions the same in both parts:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=0
)
```

`categorical_features="from_dtype"` treats exactly the `category` columns as
categorical, and every other column as continuous:

```python
from c50.sklearn import C50Classifier

tree = C50Classifier(categorical_features="from_dtype")
tree.fit(X_train, y_train)
print(f"test accuracy: {tree.score(X_test, y_test):.3f}")
```

The tree classifies about 95% of the test penguins correctly.

## Read the tree

`export_text` prints the tree with the original feature names, categories, and
class labels:

```python
print(tree.export_text(include_empty=False))
```

```text
tree 0: 14 nodes, 6 supported leaves, 4 empty leaves, depth 3
[0] class='Adelie', support=258
  'flipper_length_mm' <= 205 -> [2] class='Adelie', support=157.611
    'culmen_length_mm' <= 42.2 -> [4] class='Adelie', support=106.412
    'culmen_length_mm' > 42.2 -> [5] class='Chinstrap', support=51.1984
      'island' in {'Biscoe', 'Torgersen'} -> [7] class='Adelie', support=5.19844
      'island' in {'Dream'} -> [8] class='Chinstrap', support=46
  'flipper_length_mm' > 205 -> [9] class='Gentoo', support=100.389
    'island' == 'Biscoe' -> [11] class='Gentoo', support=93
    'island' == 'Dream' -> [12] class='Chinstrap', support=6
    'island' == 'Torgersen' -> [13] class='Adelie', support=1.38911
```

Each line is a branch: the test, the node it leads to, the node's majority
class, and its support, the weight of training penguins that reach it. Several
details show how C5.0 works:

- Flipper length separates Gentoo penguins, and culmen length and island then
  separate Adelie from Chinstrap penguins. Culmen depth, body mass, and sex are
  not used.
- Supports such as 157.611 are fractional because the training penguin with
  no flipper length went down both branches of the first test, in proportion
  to their sizes.
- Every continuous test also has a branch for not-applicable values, and
  categorical tests have one branch per category. Branches that no training
  penguin reached are empty leaves, which `include_empty=False` hides.
- Node 5 tests island membership in a set. C5.0 merged the Biscoe and
  Torgersen branches because they led to leaves of the same class.

`feature_importances_` reports the share of attribute usage for each feature.
A feature's usage is the fraction of training penguins whose classification
tests it:

```python
import pandas as pd

usage = pd.Series(tree.attribute_usage_, index=tree.feature_names_in_)
print(usage.sort_values(ascending=False))
```

Flipper length is tested for almost every penguin, because it is at the root.

## Predict with missing values

Pass a missing value, and the tree follows every branch of a test on that
feature, weighting the branches by their training support. Hide the flipper
lengths of two test penguins:

```python
import numpy as np

unknown_flippers = X_test.iloc[:2].assign(flipper_length_mm=np.nan)
scores = tree.predict_proba(unknown_flippers)
print(pd.DataFrame(scores, columns=tree.classes_, index=unknown_flippers.index))
```

The first penguin is from Dream with a long culmen, which is Chinstrap on both
sides of the flipper test, so the prediction stays confident. The second is
from Biscoe with a long culmen, which leads to Adelie on the short-flipper side
and Gentoo on the long-flipper side. Its scores split in the ratio of the two
branches' support, 157.6 to 100.4.

## Rules

A ruleset is often easier to read than a tree. Set `model_kind="rules"`:

```python
rules = C50Classifier(model_kind="rules", categorical_features="from_dtype")
rules.fit(X_train, y_train)
print(f"test accuracy: {rules.score(X_test, y_test):.3f}")
print(rules.export_text())
```

```text
ruleset 0: 4 rules, default='Adelie'
  rule 0: 'island' == 'Torgersen' -> 'Adelie' (cover=39, correct=39, prior=0.4418603, vote=976/1000)
  rule 1: 'flipper_length_mm' <= 205 -> 'Adelie' (cover=157, correct=111, prior=0.44186163, vote=704/1000)
  rule 2: 'island' == 'Dream' and 'culmen_length_mm' > 42.2 -> 'Chinstrap' (cover=51, correct=50, prior=0.1976746, vote=962/1000)
  rule 3: 'island' == 'Biscoe' and 'flipper_length_mm' > 205 -> 'Gentoo' (cover=93, correct=93, prior=0.36046532, vote=989/1000)
```

Each rule lists the training penguins it covers and how many of them belong to
its class. Its vote is the Laplace estimate of its accuracy,
$(\mathit{correct} + 1) / (\mathit{cover} + 2)$, times 1000. A penguin can
satisfy several rules, so the rules vote, and a penguin that satisfies none
gets the default class. Rule 0 shows that every penguin from Torgersen in the
training data is an Adelie penguin, a pattern the tree expressed only deep in
its branches.

## Boosting

Boosting builds a sequence of classifiers, each concentrating on the cases
that the previous ones misclassified, and lets them vote. Compare a single
tree with ten boosted trials by 10-fold cross-validation on all the data:

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
for trials in (1, 10):
    model = C50Classifier(trials=trials, categorical_features="from_dtype")
    accuracy = cross_val_score(model, X, y, cv=folds)
    print(f"trials={trials}: {accuracy.mean():.3f} ± {accuracy.std():.3f}")
```

On these folds, boosting raises the mean accuracy from about 96.8% to 97.7%.
The penguin species are nearly separable, so there is little room for
improvement. Boosting usually helps more on harder problems.

## Tune the parameters

C5.0 has few parameters that matter. `confidence_factor` controls pruning:
smaller values prune more. `minimum_cases` sets the minimum weight of training
cases in at least two branches of every test. Search both with the same folds:

```python
from sklearn.model_selection import GridSearchCV

search = GridSearchCV(
    C50Classifier(categorical_features="from_dtype"),
    {"confidence_factor": [0.05, 0.1, 0.25, 0.5], "minimum_cases": [2, 5, 10]},
    cv=folds,
)
search.fit(X_train, y_train)
print(search.best_params_, f"{search.best_score_:.3f}")
```

The estimator works with the rest of scikit-learn in the same way, including
pipelines, calibration, and permutation importance. See {doc}`estimator`.

## Misclassification costs

Some errors cost more than others. The Statlog German Credit data classifies
1000 loan applicants as good or bad credit risks, from 7 numeric and 13
categorical attributes. Its documentation specifies that classifying a bad
risk as good costs 5, while classifying a good risk as bad costs 1.

```python
credit = fetch_openml(data_id=31, as_frame=True)
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    credit.data, credit.target, test_size=0.3, stratify=credit.target,
    random_state=0,
)
```

`cost_matrix` is indexed by predicted class, then actual class, in the order
of `classes_`, which is `['bad', 'good']` here. Entry `[1, 0]` is the cost of
predicting `good` for a `bad` applicant:

```python
from sklearn.metrics import confusion_matrix

costs = np.array([[0.0, 1.0],
                  [5.0, 0.0]])


def total_cost(model):
    predicted = model.predict(Xc_test)
    # confusion_matrix counts actual classes by row and predicted by column.
    counts = confusion_matrix(yc_test, predicted, labels=model.classes_)
    return (counts.T * costs).sum()


plain = C50Classifier(categorical_features="from_dtype")
plain.fit(Xc_train, yc_train)
weighted = C50Classifier(categorical_features="from_dtype", cost_matrix=costs)
weighted.fit(Xc_train, yc_train)

for name, model in [("plain", plain), ("cost-sensitive", weighted)]:
    accuracy = model.score(Xc_test, yc_test)
    print(f"{name}: accuracy {accuracy:.3f}, total cost {total_cost(model):.0f}")
```

The cost-sensitive tree is less accurate but reduces the total cost on the
test applicants from 273 to 174. It rejects many more good applicants to avoid
accepting bad ones. With two classes, C5.0 applies the costs by reweighting
the training cases. See {doc}`algorithm` for how it uses costs with more
classes.

## Save and load

Fitted estimators can be pickled:

```python
import pickle

restored = pickle.loads(pickle.dumps(tree))
assert (restored.predict(X_test) == tree.predict(X_test)).all()
```

Pickles are specific to the Python package. To exchange a classifier with the
`c5.0` program or the C++ library, use the low-level {doc}`API <api/python>`,
whose models serialize to C5.0's own `.tree` and `.rules` formats.

## Data sources

- Horst, A. M., Hill, A. P., and Gorman, K. B. (2020). *palmerpenguins: Palmer
  Archipelago (Antarctica) penguin data*.
  [doi:10.5281/zenodo.3960218](https://doi.org/10.5281/zenodo.3960218).
  Released under CC0. The data were collected by Kristen Gorman and the
  Palmer Station Long Term Ecological Research Program.
- Hofmann, H. (1994). *Statlog (German Credit Data)*. UCI Machine Learning
  Repository. [doi:10.24432/C5NC77](https://doi.org/10.24432/C5NC77).
  Licensed under CC BY 4.0.
