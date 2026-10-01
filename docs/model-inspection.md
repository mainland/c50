# Model inspection

C++ and Python expose owned tree, ruleset, and ensemble snapshots with
deterministic text and JSON exports.

## Ownership and operation boundary

`model.inspect(context)` returns an owned, copyable `model_inspection` value.
It parses the model through the same validated native loader used by prediction.
The snapshot owns its strings, vectors, and scalar data and remains valid after
the context or model is destroyed. Editing a snapshot cannot change the model.
Inspection uses an exclusive context and leaves it reusable after failures.
Separate contexts may inspect the same immutable model concurrently.

The snapshot describes the retained serialized classifier. Its counts and
thresholds therefore reflect serialization precision, including for newly
trained models. It does not reconstruct unavailable training history or expose
private parser, tree, rule, or allocator structures.

## Trees and metadata

The snapshot contains class names and attribute names in schema order.
All class, feature, child, and ensemble indices are zero-based. Attribute names
include ignored and derived attributes in their original schema positions.

Each tree contains a flat preorder node vector with root index zero. Each node
reports its kind, stored class, weighted case count, and class distribution.
Internal nodes identify the tested feature and ordered outgoing branches.
Branches report an explicit not-applicable, category-equality, less-or-equal,
greater-than, or subset condition and their child index. Subsets preserve
not-applicable membership separately from ordinary category names. Missing
values do not become a fictitious extra branch: the learner distributes their
weight according to its native prediction policy.

Continuous nodes retain the cutoff and lower, middle, and upper soft-threshold
points. Tree statistics distinguish all leaves from leaves with positive
training support. Depth is measured in edges from the root, with separate
maximum depths for all nodes and supported leaves. Per-feature counts count
split nodes in trees and conditions in rulesets. They are not statistical
feature importance. Stored node classes do not replace cost-sensitive or
missing-value prediction.

## Rules and ensembles

Ruleset snapshots preserve serialized rule and condition order, default class,
covered and correctly classified case weights, derived prior, and integer vote
in thousandths. A rules model stores each rule's lift, the ratio of its Laplace
accuracy to its class's prior probability, to six significant digits, and not
the prior. The loader derives the prior from the lift, so rules of the same
class can report priors that differ in about the sixth significant digit.
Predictions do not use the prior or the lift. Conditions use the same public representation as tree branches.
Tree and ruleset ensembles preserve serialized component order. Exactly one of
`trees` and `rulesets` is populated, according to the model kind. C5.0 combines
per-case component confidences, so inspection does not invent fixed boosting
weights or represent the ensemble as an additive weighted-tree predictor.

## Attribute usage

`model.attribute_usage(context, cases)` classifies each case and counts, for
every attribute, the cases whose classification tests that attribute with a
known value. For an implicit attribute, the attributes used by its definition
also count. The result has one count per attribute in schema order, like the
snapshot's feature names. Both data-file and dense overloads are available.
Python exposes them as `Model.attribute_usage` and
`Model.attribute_usage_dense`.

These are the counts behind the command-line program's "Attribute usage"
report, which divides them by the number of training cases and omits
attributes below one percent. A test trains the regression fixtures through
the library and requires the same report as the imported program for trees,
rules, subsets, winnowing, soft thresholds, boosting, implicit attributes,
case weights, and costs. Like prediction, the operation uses an exclusive
context, and separate contexts may run it on one model concurrently.

## Python and readable output

Python exposes the same native snapshot and deterministic text and JSON
exports. Optional feature and class display names allow an estimator to show
original labels instead of its generated encoding tokens. Exports describe the
model but are not an alternate predictor or a replacement for legacy model
serialization. The snapshot supports node counts, depth limits, and feature-use
reports without application-side parsing of `.tree` or `.rules` files.

## Inspect a model

```python
from c50 import Model, export_json, export_text

model = Model.train(
    "low, high.\nsignal: continuous.\n",
    "0, low\n1, low\n2, low\n3, high\n4, high\n5, high\n",
)
snapshot = model.inspect()
tree = snapshot.trees[0]
assert tree.supported_leaf_count == 2
print(export_text(snapshot, include_empty=False))
metadata = export_json(snapshot)
```

`export_text` prints thresholds, which C5.0 stores in single precision, with
the fewest digits that read back as the stored value. For rules it prints the
lift to the six significant digits that the model stores, as the command-line
program reports rules, rather than the derived prior. JSON exports keep the
full double-precision values of the snapshot, including the prior.
`export_text` accepts `max_depth` to truncate display and `include_empty=False`
to omit zero-support tree leaves. These two options do not filter rules. Both
exports accept `feature_names` and `class_names` sequences in schema order.
They also accept `value_names`, one mapping or `None` per schema attribute,
from native categorical value names to display names. Native tokens remain
available in the snapshot. Python fields are read-only, and list accessors
return copies.

A fitted estimator encodes names, labels, and categories as internal tokens.
Its `export_text()` and `export_json()` methods supply the original feature
names, class labels, and categories as display names. Features without string
names are shown as `feature_<index>`.
