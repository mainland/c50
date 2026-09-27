# Compatibility and provenance

The initial source import is the single-threaded C5.0 Release 2.07 GPL Edition
from RuleQuest Research's `C50.tgz` archive. The imported archive has this
SHA-256 digest:

```text
309db588eda420c06701bf8ae74c06a6c923e9a06e714a598ea761bcadfc5e2e  C50.tgz
```

The modernization retains the file-oriented command-line programs and builds
the same learner as a compiled C++17 library. Regression tests
compare command output, serialized tree and rules models, and row-level
predictions with fixtures captured from the imported implementation.

The `reference-compatibility` test also exercises 32 deterministic,
tie-heavy scenarios across tree and rules models, boosting, subsets,
winnowing, pruning, probabilistic thresholds, costs, sampling, missing values,
and model reloads. It compares normalized serialized-model and exact
prediction digests captured from the C++ core with the imported RuleQuest
sorting algorithms.

Sorting must reproduce the RuleQuest order of records with equal values, not
only a valid sorted order. Soft-threshold bounds accumulate single-precision
case weights through groups of equal values, and missing values make those
weights fractional, so the order changes the rounded sums. A standard-library
sort passed the 32 scenarios but changed a soft-threshold bound on other data.
The `soft-threshold-ties` command-line fixture retains that case with output
from the unmodified imported program. The native sort properties compare each
permutation with a reference copy of the original algorithm.

Run the compatibility oracle with:

```sh
ctest --test-dir build -R reference-compatibility --output-on-failure
```

The test executable accepts `--emit-reference` to print replacement digest
initializers. Use that option only after independently establishing that a
deliberate classifier change defines a new compatibility baseline. Do not
regenerate the values merely to make a failure pass.

The in-memory interfaces preserve C5.0 names, data, costs, tree, and rules text
formats. Trained and loaded models expose their serialized classifier so it can
be stored or exchanged with compatible tools.

The native interface is C++ only. The former C header, handles, status codes,
reserved ABI fields, and header-only facade are removed. Native callers must
rebuild with a compatible C++ toolchain. This API change does not change model
formats, classifier construction, predictions, or the command-line interface.
Python callers retain their existing interface.

Split evaluation can use 1 through 8 workers for a node with at least 10,000
rows and multiple eligible attributes. The default remains one worker. The
experimental parallel path is disabled for verbose diagnostics and
legacy split-value subsampling. Attribute results are published in serial
queue order before selection. The cross-worker workload test checks exact
serialized classifier digests at the 10,000-row boundary. Training-row
sampling remains eligible; other training phases remain serial.

Parallel training must reproduce the classifier produced by the reference
GPL C5.0 implementation, except for explicitly documented floating-point
tie-breaking differences.

The project is licensed under GPL-3.0-or-later. C5.0 and RuleQuest are
associated with RuleQuest Research Pty Ltd. This project is not an official
RuleQuest distribution.

Malformed names and data tokens are rejected at the same length limit whether
or not diagnostics are printed. Oversized model-property diagnostics are
bounded. These safety fixes change rejection behavior for malformed input, not
classifiers or predictions for accepted input. The same bound applies to class
names generated from continuous-target thresholds. A `discrete N` declaration
must contain a complete integer from 2 through `INT_MAX - 3`. Dynamic value
dictionaries grow with the values read instead of reserving `N` pointer slots
from the declaration.

Implicit definitions reject missing and not-applicable literals without a
preceding operand that determines their type. They also reject nonfinite numeric
literals and expressions nested beyond 100 levels. Token matching and date and
time recognition remain within the definition buffer, and division precision
scaling does not overflow for large finite divisors. These checks make extreme
or malformed definitions reject instead of exhausting resources or invoking
undefined behavior. Accepted definitions within these limits retain their
existing evaluation behavior.

Division by an infinite value, which finite literals can produce through
exponentiation overflow, yields an unknown value, as division by zero does.
The imported implementation never finished evaluating such a definition, so
this change affects no definition that previously completed.

Serialized properties must use the quoted syntax emitted by C5.0. Unquoted
values, incomplete escapes, and trailing property text produce parse errors.
Short model identifiers remain accepted without date recovery.

Model loading rejects inconsistent branch counts, invalid attribute kinds,
missing required classes or rule fields, nonfinite numeric properties, and
frequency vectors that disagree with the class schema. Dynamic attribute
lists must fit their declared capacity. Serialized rule and condition arrays
grow as records are read instead of reserving space from untrusted declared
counts. Rules with duplicate conditions are rejected before rule-tree
construction can exhaust its distinct tests. A tree root must have enough
total class frequency to satisfy the predictor's `1e-4` case threshold;
unlike child nodes, it has no parent distribution to use below that threshold. Zero-case
child nodes remain valid and retain their parent-based prediction behavior.
These checks reject malformed artifacts before prediction can index their
internal structures.

Costs must be complete finite nonnegative numbers representable by the native
cost type. Each actual class requires a finite positive total error cost, and
training normalization must produce finite weights. Inputs that previously
produced invalid weights or unusable classifiers now raise a parse error.

Cost-based prediction now compares all finite expected costs, including values
above the former `1e38` initial bound. Such models select the least-cost class
instead of accidentally retaining the default class.

Implicit-attribute parsing keeps scratch buffers and pending string literals
owned during exception unwinding. Partially constructed definitions always
have a terminator, including when an allocation fails. Allocation failures
therefore raise `std::bad_alloc` without invalid cleanup reads or leaks, and
the context remains reusable. Accepted expressions and classifier output are
unchanged.

Rule construction retains ownership of incomplete conditions, rulesets, and
rule-tree nodes until publication. Cleanup also releases partially allocated
rule scratch buffers and normalized costs. Allocation failures during rule
training, loading, and prediction leave the context reusable without leaks.
These changes do not alter successful rule selection or classifier output.

Subset pruning releases partially allocated per-attribute value sets and local
scratch arrays if allocation fails. This affects failure cleanup only. Subset
selection, rule contents, and successful predictions are unchanged.

Branch compression completes its replacement-array allocations before changing
child ownership or branch counts. Allocation failure previously let cleanup
scan a smaller array using the original count and leaked recursive scratch.
The repaired failure path raises `std::bad_alloc` and permits context reuse.
The merge order and floating-point arithmetic of successful runs are unchanged.
