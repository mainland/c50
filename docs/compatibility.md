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

## Stable tie order

The library option `ties` selects how continuous split evaluation orders equal
values. The default of the C++ library, `c50.train`, and the command-line
program, `reference`, uses the RuleQuest order. `stable` keeps equal values in
case order and sorts with a radix sort. `C50Classifier` defaults to `stable`.
Pass `ties="reference"` to select the RuleQuest order. The stable order is
defined without reference to the sorting algorithm, so training remains
deterministic across platforms, standard libraries, and split-worker counts.
Soft-threshold bounds and the final cut values always use the reference
order. The command-line program always uses the reference order, so
`C50Classifier` with its default can train a different classifier than `c5.0`
from the same data.

The stable order may produce a classifier that differs from the reference
learner. Two mechanisms are known. The number of candidate cuts, which sets
the threshold cost, can differ by one when the minimum-split boundary falls on
the first case of a group of equal values. Sums of fractional case weights can
also round differently. Either changes a split only when attribute gains are
nearly tied. In 38,800 randomized trainings with ties, case weights, missing
values, boosting, soft thresholds, winnowing, subsets, sampling, and rules,
both orders produced identical classifiers. That evidence does not establish
equality for all data. Use `ties="reference"` when a classifier must match the
reference learner.

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
Library training discards progress and diagnostic output without opening a
temporary file. Text and dense in-memory operations do not require filesystem
access. The command-line program retains its diagnostic and progress files.

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

Each training operation starts from the state of a fresh C5.0 process, so
reusing a context does not change the classifier.

## Order of discrete values

The order in which a `.names` file declares the values of a discrete attribute
can change the classifier, in this library and in C5.0 Release 2.07. C5.0
partitions cases among the branches of a discrete test in declaration order.
Cases with a missing value contribute fractional weights to every branch, and
the declaration order sets the order of the single-precision sums of those
weights. When two class frequencies at a leaf are equal apart from rounding,
the order decides which class the leaf predicts. Branch compression then groups
that branch with different values.

On the OpenML adult data, the same cases with two declaration orders produced
trees that predicted 2 of 48,842 training cases differently. Three random
permutations of the data rows, with the declaration order fixed, produced
identical classifiers. Keep the `.names` file unchanged when a classifier must
be reproduced.

## Winnowing with case weights

This project deliberately differs from C5.0 Release 2.07 when it winnows data
with a case-weight attribute and no two-class costs file. The imported program
computes the average case weight only after winnowing, so winnowing divides
every positive case weight by zero. Its trial tree then makes no split, and
winnowing discards every attribute, which leaves a single-leaf classifier. The
command-line program and library compute the average before winnowing in that
case and use the relative case weights of classifier construction.

A two-class costs file sets the average to one before winnowing in the
imported program, which then winnows with unnormalized case weights. That
behavior is preserved. Classifiers without winnowing or without a case-weight
attribute are unchanged.

The fix was developed in this project. The C50 R package keeps the imported
order and does not reset the average between calls, so its winnowing result
also depends on the previous training in the same R session. The
`winnow-case-weights` fixture records output from the imported program with
only the average computed before winnowing. The `winnow-case-weight-costs`
fixture records unmodified imported output for the preserved costs case.

## Input validation and failure handling

This project adds checks to the imported parsers, loaders, and allocation
paths. They change how malformed input and allocation failures end, not the
classifiers or predictions for accepted input. The subsections describe each
group of checks.

### Bad values in text input

The C++ and Python text interfaces end a call at the first bad value. This
covers training and prediction data, such as a discrete value that the names
file does not declare, and a continuous value that is not finite. The call
throws `c50::exception` with `error_code::parse_error` in C++ and raises
`ValueError` in Python, and the message names the line, the value, and the
attribute.

The `c5.0` program behaves as Release 2.07 does. It prints each bad value with
its line number, uses `?` in its place, and continues. After ten errors it
prints `Error limit exceeded` and stops. It reports a nonfinite continuous
value as a change to `?` and also continues. The program exits 0 when it
continued past fewer than ten errors, and exits 1 at the error limit and on
failures that cannot continue. A library caller that needs the
program's behavior must replace bad values with `?` before the call. The
library has no option that returns the diagnostics and continues.

### Names and data tokens

Malformed names and data tokens are rejected at the same length limit whether
or not diagnostics are printed. Oversized model-property diagnostics are
bounded. These safety fixes change rejection behavior for malformed input, not
classifiers or predictions for accepted input. The same bound applies to class
names generated from continuous-target thresholds. A `discrete N` declaration
must contain a complete integer from 2 through `INT_MAX - 3`. Dynamic value
dictionaries grow with the values read instead of reserving `N` pointer slots
from the declaration. Input lines may exceed the legacy 9,999-byte read
buffer. Per-token length checks still apply.

A name that appears more than once in an `attributes included:` or
`attributes excluded:` list means what one occurrence means. Release 2.07
subtracts the exclusion bit once for each occurrence of a name in an
`attributes included:` list. A repeated name leaves the attribute excluded
with corrupted type bits, and training then dereferences a null pointer or
writes a model that cannot be loaded. This project clears the bit, so two
results differ from Release 2.07 for text that it accepts. An explicit
discrete attribute listed twice is now used, where Release 2.07 drops it. A
class attribute that the schema declares and the list names twice is now
accepted, where Release 2.07 reports that the target attribute is missing.

### Implicit definitions

Implicit definitions reject missing and not-applicable literals without a
preceding operand that determines their type. They also reject nonfinite
numeric literals and expressions nested beyond 100 levels. Token matching and
date and time recognition remain within the definition buffer. These checks
make extreme or malformed definitions reject instead of exhausting resources or
invoking undefined behavior. Accepted definitions within these limits retain
their existing evaluation behavior.

The evaluator's operand stack grows with the definition, so every definition
within the nesting limit evaluates, including definitions that leave several
operands pending at each level, such as `1 + 1 * 1 ^ (` repeated. Release 2.07
evaluates with a fixed stack. A definition of 34 levels of that pattern
overflows it, and the reference build used for this repository's comparisons
terminates with a stack-smashing report.

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

### Model loading

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

`model::load` rejects a serialized tree of more than `max_tree_depth` levels
with `error_code::parse_error` (`ValueError` in Python), to bound the stack
that parsing needs for untrusted text. The root is level 1, and the default is
1,024 levels. Release 2.07 has no such limit. Pass `std::nullopt` (`None` in
Python) to load deeper trees from text that you trust. The library applies no
limit to text that it wrote: a model from `model::train`, and a model that
unpickling restores, parse any depth that the learner can produce. A tree that
deep still needs stack in proportion to its depth to parse and to predict, as
in Release 2.07. See {doc}`ownership-concurrency` for the stack that each level
needs. A later 1.x release may raise the default limit but never lowers it.
Name it through `c50::default_max_tree_depth`.

Rule indexes are limited to 512 levels. Deeper rulesets use iterative rule
matching, which preserves their predictions without recursive index traversal.
Internal tree nodes must have at least `1e-4` total class frequency. Empty
leaves remain valid and use their parent's distribution. If matching rules
have zero votes and the default class has no matching rule, prediction uses
the default class with confidence `0.5`, as it does when no rules match.

### Costs

Costs must be complete finite nonnegative numbers representable by the native
cost type. Each actual class requires a finite positive total error cost, and
training normalization must produce finite weights. Inputs that previously
produced invalid weights or unusable classifiers now raise a parse error.

Cost-based prediction now compares all finite expected costs, including values
above the former `1e38` initial bound. Such models select the least-cost class
instead of accidentally retaining the default class.

Costs text that sets no cost other than the default of 1 counts as missing.
That includes empty text, text with only comments, text whose entries all have
cost 1, and text whose entries all name one class twice. Loading a model that
requires costs with such text throws `invalid_argument`, or raises `ValueError`
in Python. Release 2.07 ignores a missing or empty costs file, builds no cost
matrix, and predicts without costs. The library reports the error at load
instead, because a cost model without its costs would otherwise fail on every
later prediction. A comment beside a real entry is valid.

### Failure cleanup

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

## Held-out cases with sampling

With `-S` below 50 percent, the command-line program evaluates a random subset
of the cases that sampling did not select for training. C5.0 Release 2.07 draws
that subset with the C library's `drand48()` without seeding it. The GNU C
library starts that generator from state zero, but BSD C libraries, including
macOS, start from the POSIX default seed, so the evaluated cases and the
printed test results differed between Linux and macOS. The classifier does not
depend on this subset.

This project reproduces the GNU C library sequence on every platform and keeps
the generator state in each context. It resets the state before each read of
held-out cases, which matches the single read of the imported program. The
`multiclass` `sample` fixture was captured on Linux with the GNU C library.

## Timestamp base near New Year

This project deliberately differs from C5.0 Release 2.07 in the model text
that it writes for a `timestamp` attribute within a few hours of New Year.
The imported program takes the base year of the timestamp cuts from the UTC
calendar and writes the `id` date of the model file from the local calendar.
Reading a model takes the base from the year of the `id`. Outside UTC, the two
years differ for up to 14 hours around New Year. A classifier trained then
prints correct results, but its file holds cuts relative to one year and an
`id` that names another, so its cuts shift by a year when the file is read.

The command-line program and the library read the clock once per operation
and take both the base year and the `id` date from the local calendar of that
reading. A model trained in the window now carries cuts consistent with its
`id`. Model text differs from the imported program's only in that window,
in the `cut` values of `timestamp` attributes. The fix was developed in this
project.

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
message on the output stream, standard output or the `-o` file, except for a
failure of that stream itself, which goes to standard error. `Error` and the
option checks print their own text, as in C5.0 Release 2.07. The program now
also prints the message of every other failure that the library raises, in the
form `*** message`. These failures include a classifier that cannot be written
or closed, an internal error, and an allocation failure that `Error` did not
report. Earlier builds of this project exited 1 without a message for them.

Release 2.07 ignored the result of writing and closing a `.tree` or `.rules`
file, so a write failure such as a full disk left a truncated model and exit
status 0. This project reports `*** could not close classifier` or
`*** could not write classifier` and exits 1. This difference is deliberate.

Release 2.07 also ignored the result of opening the `-o` file and of writing
the report. When the file could not be opened, it printed the report on
standard output, created no file, and exited with status 0. When a write to the
`-o` file or to standard output failed, such as a write to a full device, it
lost the report and exited with status 0. This project reports these failures
on standard error, because the report stream is not available, and exits with
status 1:

```text
*** could not open <path>: <reason>
*** could not write <path>: <reason>
*** could not write standard output: <reason>
```

`<reason>` is the operating system's description of the error that the final
flush or close reports. When an earlier write failed and the final flush
succeeds, the C library keeps the error state of the stream but not the error
number, so `<reason>` is `write error`. The program opens the `-o` file before
it prints anything, so a failed open prints no report. It checks the final
flush and close of the `-o` file and the final flush and error state of
standard output, and it does not close standard output. Runs that succeed print
the same output and exit with the same status as before. This difference is
deliberate.

## Read errors

Release 2.07 treats a failed read of a `.names`, `.data`, `.test`, or
`.costs` file as the end of the input. A `basic.test` that is a directory, for
example, evaluates 0 cases and the program exits with status 0. This project
reports the failure. The command-line program prints
`*** could not read <file>: <reason>` on the output stream, where `<reason>`
is the operating system's description of the error, and exits with status 1.
An ordinary end of file and the limit of 10 data errors are unchanged. This
difference is deliberate.

The C++ and Python entry points read text from memory, so they do not read
files and cannot reach this failure. A read error in the library raises
`c50::exception` with `c50::error_code::io_error`, which Python raises as
`OSError`.

## Training on no cases

When the training data holds no cases, `c5.0` trains a classifier that has a
single leaf of the first class, as Release 2.07 does, and evaluates the test
cases with it. Release 2.07 crashes with a segmentation fault when a tree
classifier, trained on no cases, classifies a case. It dereferences the parent
of the root leaf, which is a null pointer. The same crash occurs with
boosting, winnowing, and subset tests. Rules classifiers and runs without test
cases do not crash.

This project returns no class distribution for that leaf, so every case is
classified as the first class, as the rule classifier does, and the program
exits with status 0. The change affects only runs that crashed. Rules
classifiers with a costs file also train on no cases, and on data in which
every case has one class, as Release 2.07 does. The rule cost normalization
skips the terms that divide by zero, which occur with no cases or when one
class has every case, and the program forms no rules. The library and Python
entry points still reject zero training cases with `ValueError`. Release 2.07
also copies zero bytes through null pointers in this case, which is undefined
behavior but has no visible effect. This project skips those copies.

Cross-validation (`-X`) is different when fewer than two cases remain to
validate: the data file holds none or one case, every case but one has an
unknown class, or `-S` selects none or one case. The attributes do not matter.
The option accepts at least two folds, and each fold needs a case to hold out
and a case to train on. Release 2.07 divides by zero while it sizes the folds
for no case and dies from `SIGFPE` (status 136). For one case, it reduces the
folds to one, trains on no case, prints a standard error of `-nan`, and exits
with status 0, or crashes when the tree classifies the held-out case. `c5.0`
instead prints `cannot cross-validate with no cases` for no case and
`cannot cross-validate with one case` for one, before any fold trains, and
exits with status 1. Runs with two or more remaining cases are unchanged. This
difference is deliberate.

## Cross-validation summaries

`c5.0 -X` and `report` compute the summary of a cross-validation in `float`, as
Release 2.07 does. The program adds the fold values and their squares and takes
the standard error from the difference of the sum of squares and the square of
the sum. The fold cost is also a `float` sum, divided by the fold size. The
output matches Release 2.07, including three inherited limitations:

- A fold value above about 1.8e19 makes the sum of squares infinite. The cost
  file accepts such costs, and the standard error prints as `inf` or `nan`.
- Fold values whose sum exceeds the largest `float` print a mean of `inf`.
  Error costs that sum past the largest `float` print a fold cost of `inf`
  although the cost per case is finite.
- The difference of two nearly equal sums can be negative. Equal fold values
  trigger this in ordinary runs, for example three folds of the same error
  rate, and some such columns print a standard error of `nan`.

The spelling of `nan` and `inf` depends on the C library, for example `nan`,
`-nan`, or `nan(ind)`. The summary arithmetic is not corrected, because the
printed numbers of a run would then differ from those of Release 2.07 for
ordinary inputs as well.

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

## Class count

The number of classes is not limited, as in C5.0 Release 2.07. The
`many-classes` fixture trains a schema of 4,097 classes and compares its trees,
rules, and console output with Release 2.07. Earlier builds of this project
rejected more than 4,096 classes.

Memory grows with the square of the class count, as it does in Release 2.07.
Evaluation allocates a confusion matrix of `(classes + 1)^2` integers, and
other per-class arrays add a term linear in the class count. The address space
that the program needed was measured with a Release build on x86-64 Linux, a
12-case training set of one continuous attribute, default options, and
`ulimit -v` bisection. It was about 744 MB for 8,000 classes, 2.9 GB for
16,000 classes, and 11.8 GB for 32,000 classes. That is about 12 bytes per
squared class, or four times the memory for twice the classes. Rule-based
classifiers with `-r` needed more resident memory than trees did, 176 MB
against 114 MB at 4,000 classes. Peak resident memory is lower than the
address space because the program does not touch every page. Plan for the
address-space figure.

When memory is not available, the command-line program reports
`*** unable to allocate sufficient memory` and exits 1, and the library
entry points raise `std::bad_alloc`.

Release 2.07 counts the elements of these tables in `int`. For a schema of
46,340 to 65,534 classes, `(classes + 1)^2` exceeds `INT_MAX` but stays below
2^32, the count wraps negative, and Release 2.07 reports out of memory
regardless of the memory available. At 65,535 classes, `(classes + 1)^2` is
exactly 2^32, so the count wraps to 0. Release 2.07 then receives a null
pointer for the zero-element allocation and can crash with a segmentation fault
instead of reporting out of memory. It crashed under a 60 GB address-space
limit and reported out of memory under a smaller one, so the outcome depends on
the available address space. The same wrap applies to boost votes (cases times
classes plus one) and discrete frequencies (attribute values times classes)
whose element counts lie in `[2^31, 2^32)`. This project counts and indexes
these tables in `size_t`, so such inputs train when memory allows. From 65,536
classes, Release 2.07 allocates a wrapped, too-small confusion matrix and
writes out of bounds. This project does not.

## License and affiliation

The project is licensed under GPL-3.0-or-later. C5.0 and RuleQuest are
associated with RuleQuest Research Pty Ltd. This project is not an official
RuleQuest distribution.
