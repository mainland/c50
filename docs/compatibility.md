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
from the declaration. Input lines may exceed the legacy 9,999-byte read
buffer. Per-token length checks still apply.

Implicit definitions reject missing and not-applicable literals without a
preceding operand that determines their type. They also reject nonfinite numeric
literals and expressions nested beyond 100 levels. Token matching and date and
time recognition remain within the definition buffer. These checks make extreme
or malformed definitions reject instead of exhausting resources or invoking
undefined behavior. Accepted definitions within these limits retain their
existing evaluation behavior.

Division rounds its quotient to a precision that Release 2.07 keeps in a
32-bit `int` multiplier. The multiplier starts at 10^p, where p is the number
of decimals that the numerator needs up to a tolerance, at most 6, and gains a
factor of 10 for each digit that scaling strips from the divisor's magnitude
above 1. Once the product passes 2^31 - 1, Release 2.07's `int` wraps, which
is undefined behavior in C, and every compiler and optimization level tried
gives the two's-complement result. This project reproduces that result with
unsigned arithmetic, so a numerator that needs six decimals divided by more
than 1,000, or an integer divided by more than 10^9, rounds to a coarser grid
than the exact quotient needs, and trees, rules, and predictions match
Release 2.07. A multiplier that wraps to negative rounds as its magnitude
does. A multiplier that wraps to 0 yields a not-a-number value, as in Release
2.07. A top-level result is reported as a bad value that changes to `?`, and a
nested one propagates through the enclosing operators.

The multiplier is 10^(p + q) modulo 2^32, where p is the number of decimals
needed to represent the numerator to within 2e-6 of its magnitude, up to 6, and
q is the number of times that the divisor, stored as a 32-bit `float`, divides
by 10 before its magnitude is at most 1. The multiplier is 0 exactly when
p + q is at least 32, because 2^32 divides 10^n only for n of at least 32, and
the quotient is then not a number.

The tolerance scales with the numerator, so a numerator of 1 or more can have
a smaller p than the decimals it has. `123456.25` has two decimals but p = 1,
because rounding it to one decimal errs by 0.05, below the tolerance of 0.247.
`123456.25 / 1.1e29` therefore has p + q = 31 and a finite quotient, while
`0.25 / 1.1e29` has p = 2 and q = 30 and is not a number.

A numerator of six decimals below 1 over a divisor with q = 26 gives a
not-a-number quotient, as in `0.123456 / 1.1e25`, and so does an integer over
a divisor with q = 32, as in `3 / 1.1e31`. A sum of 31 wraps to -2^31 instead
and gives a finite quotient, as in `0.123456 / 1e25` and `3 / 1e31`, because
the nearest `float` to each of those literals lies below the power of ten, so
that q is one less than the literal suggests.

Division by an infinite value, which finite literals can produce through
exponentiation overflow, yields an unknown value, as division by zero does.
The imported implementation never finished evaluating such a definition, so
this change affects no definition that previously completed.

Serialized properties must use the quoted syntax emitted by C5.0. Unquoted
values, incomplete escapes, and trailing property text produce parse errors.
Short model identifiers remain accepted without date recovery.
Recovered timestamp base years must lie between 0 and 9999. Timestamp
conversion uses checked arithmetic and rejects offsets outside the native
integer range or at least one billion minutes after the base. Representable
timestamps retain their existing rounded-minute values.

Model loading rejects inconsistent branch counts, invalid attribute kinds,
missing required classes or rule fields, nonfinite numeric properties, and
frequency vectors that disagree with the class schema. Dynamic attribute lists
must fit their declared capacity. Serialized rule and condition arrays grow as
records are read instead of reserving space from untrusted declared counts.
Rules with duplicate conditions are rejected before rule-tree construction can
exhaust its distinct tests. A tree root must have enough total class frequency
to satisfy the predictor's `1e-4` case threshold; unlike child nodes, it has no
parent distribution to use below that threshold. Zero-case child nodes remain
valid and retain their parent-based prediction behavior. These checks reject
malformed artifacts before prediction can index their internal structures.

Rule indexes are limited to 512 levels. Deeper rulesets use iterative rule
matching, which preserves their predictions without recursive index traversal.
Internal tree nodes must have at least `1e-4` total class frequency. Empty
leaves remain valid and use their parent's distribution. If matching rules
have zero votes and the default class has no matching rule, prediction uses
the default class with confidence `0.5`, as it does when no rules match.

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

## Command-line option values

The command-line program parses numeric option values with `strtod` and
`strtol`, as C5.0 Release 2.07 does, and checks them more strictly. Integer
options (`-v`, `-t`, `-I`, `-u`, `-X`) accept a decimal integer within the
range of `int`. Floating-point options (`-m`, `-c`, `-S`) accept any text that
`strtod` reads in full, such as `1e2` or `0x10`, if the value is finite and
within the range of `float`.

An empty value is the number 0, as in Release 2.07, which only tests that
`strtod` or `strtol` stopped at the terminating null. `-I ''` sets seed 0 and
`-c ''` sets confidence 0, and an empty value for an option with a range that
excludes 0, such as `-m ''`, gets the "illegal value" diagnostic below. A value
that underflows is its rounded value, as in Release 2.07: `-c 1e-400` and the
subnormal `-c 1e-320` train with confidence 0.

A value that overflows its type or is not finite gets the diagnostic that
Release 2.07 prints for a value outside an option's range:
`** illegal value V -- should be between L and H`. This covers `-c inf`,
`-m nan`, `-c 1e999`, `-I 99999999999`, and `-t 99999999999`. The range
printed for `-I` and `-v`, which have no documented range, is that of `int`.
Release 2.07 converted such values without an overflow check. It truncated
`-I 99999999999` to the seed 1215752191, accepted `-m nan`, and reported
`-t 99999999999` as the wrapped value 1.21575e+09. The program now stops with
status 1 for all of them. It prints `Unrecognised option` followed by the
option summary only for a value that is not a number or has trailing text, such
as `-c 25x`, as Release 2.07 does.

## Command-line failure messages

The command-line program exits with status 1 after a failure and prints a
message on the output stream, standard output or the `-o` file. `Error`
and the option checks print their own text, as in C5.0 Release 2.07. The
program now also prints the message of every other failure that the library
raises, in the form `*** message`. These failures include a classifier that
cannot be written or closed, an internal error, and an allocation failure that
`Error` did not report. Earlier builds of this project exited 1 without a
message for them.

Release 2.07 ignored the result of writing and closing a `.tree` or `.rules`
file, so a write failure such as a full disk left a truncated model and exit
status 0. This project reports `*** could not close classifier` or
`*** could not write classifier` and exits 1. This difference is deliberate.

## Pruning with two-class costs

With two classes and a costs file, C5.0 weights each training case by a
multiplier for its class. The pessimistic error estimate of a leaf scales with
the other class's multiplier, so a subtree's estimate can exceed the number of
training cases. When Release 2.07 decides whether to replace a subtree with a
leaf or with its largest branch, it gives a missing branch an estimate of the
number of training cases. If no branch qualifies, because each is a leaf,
holds less than 10% of the cases, or tests the subtree's continuous attribute
again, and the subtree's estimate reaches that number, Release 2.07 replaces
the subtree with branch 0, a null pointer, and crashes with a segmentation
fault. Few training cases, a low confidence factor
(`-c`), and a high cost ratio make this likely. Winnowing's trial tree reaches
the same step.

This project compares the branch only when there is one, so such a subtree
becomes a leaf or stays. The change affects only runs that crashed: every
subtree that Release 2.07 prunes without crashing is pruned as before. The fix
was developed in this project, and the C50 R package has the same code as
Release 2.07. The `costs-pruning` and `winnow-costs-pruning` fixtures record
this project's output, because Release 2.07 has none.
