# Stability and versioning

The `c50` Python package, the native library, and its CMake package share one
version number and follow [semantic versioning](https://semver.org). This page
states what each version promises. It applies from version 1.0.0 onward.
Prereleases such as `1.0.0a1` make no promises: interfaces may still change
before 1.0.0.

## Stable interfaces

Within a major version, such as 1.x, a later release keeps these interfaces
source compatible. Code that uses them as documented continues to work without
changes.

- The scikit-learn estimator, `c50.sklearn.C50Classifier`: its constructor
  parameters, methods, and fitted attributes.
- The low-level Python API exported by the `c50` package: `train`,
  `train_dense`, `load`, `Model`, `Options`, `Predictor`, `Predictions`, the
  enumerations, `C50Error`, and `export_text` and `export_json`.
- The inspection records, from `ModelInspection` to `SplitCondition`, and the
  fields of the JSON export for a given `format_version`.
- The C++ API declared in `<c50/c50.hpp>`, outside the `c50::detail`
  namespace, and the `C50::cpp` CMake target.
- The command-line options and output of the `c5.0` and `report` programs, as
  documented by C5.0 Release 2.07.

A minor release may add parameters, methods, attributes, record fields, and
enumeration members. Adding an enumeration member can affect code that handles
every member exhaustively. A major release may remove or change any of these
interfaces. A release deprecates an interface, with a `DeprecationWarning` in
Python, at least one minor release before a major release removes it.

These are not part of the stable interface:

- Python names that begin with an underscore, including the `c50._c50` and
  `c50._data` modules.
- The `c50::detail` namespace in C++.
- The text of error messages. Match the exception type, or the C++
  `c50::error_code`, rather than the message.
- The binary interface of the native library. Rebuild C++ consumers when the
  library changes, as {doc}`ownership-concurrency` describes.
- Text exports, which are meant for people, and the formatting of command-line
  output beyond what C5.0 Release 2.07 documents.

## Classifiers

For fixed training data, options, seed, and number of split workers, a
release trains the same classifier on every supported platform. A later
release may train a different classifier only to fix a defect, and its release
notes then identify the change. {doc}`compatibility` describes the deliberate
differences from the imported C5.0 program. Options that are documented to
change classifiers, such as `ties="stable"`, are exceptions by design.

## Saved models

- `.tree` and `.rules` files, and the `names_data`, `serialized_data`, and
  `costs_data` of a `Model`, use C5.0's own formats. Every 1.x release loads
  models saved by any earlier 1.x release and by the imported C5.0 program,
  and the test suite loads models that the imported program wrote. These
  formats are the recommended way to keep models.
- A pickled low-level `Model` holds only those formats, so the same promise
  applies to it.
- A pickled `C50Classifier` also holds private encoding state, so a pickle
  must be loaded with the version of `c50` that saved it, as scikit-learn
  recommends for its own estimators. To use a classifier across versions, save
  the estimator's `model_` and predict through the low-level API, or refit
  the estimator from the training data.
- Inspection JSON records its `format_version`. A release that changes the
  meaning of an existing field increments it.
