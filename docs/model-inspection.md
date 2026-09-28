# Model inspection

Status: proposed API contract. The following interfaces are implemented and
validated in subsequent commits in this topic.

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
split nodes, not statistical feature importance. Stored node classes do not
replace cost-sensitive or missing-value prediction.

## Rules and ensembles

Ruleset snapshots preserve serialized rule and condition order, default class,
covered and correctly classified case weights, derived prior, and integer vote
in thousandths. Conditions use the same public representation as tree branches.
Tree and ruleset ensembles preserve serialized component order. C5.0 combines
per-case component confidences, so inspection does not invent fixed boosting
weights or represent the ensemble as an additive weighted-tree predictor.

## Python and readable output

Python exposes the same native snapshot and deterministic text and JSON
exports. Optional feature and class display names allow an estimator to show
original labels instead of its generated encoding tokens. Exports describe the
model but are not an alternate predictor or a replacement for legacy model
serialization. The snapshot supports node counts, depth limits, and feature-use
reports without application-side parsing of `.tree` or `.rules` files.
