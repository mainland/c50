# Compatibility and provenance

The initial source import is the single-threaded C5.0 Release 2.07 GPL Edition
from RuleQuest Research's `C50.tgz` archive. The imported archive has this
SHA-256 digest:

```text
309db588eda420c06701bf8ae74c06a6c923e9a06e714a598ea761bcadfc5e2e  C50.tgz
```

The modernization retains the file-oriented command line programs and builds
the same learner into the reusable C core. Regression tests compare command
output, serialized tree and rules models, and row-level predictions with
fixtures captured from the imported implementation.

The in-memory interfaces preserve C5.0 names, data, costs, tree, and rules text
formats. Trained and loaded models expose their serialized classifier so it can
be stored or exchanged with compatible tools.

Future multithreaded implementations must reproduce the classifier produced by
the reference GPL C5.0 implementation, except for explicitly documented
floating-point tie-breaking differences.

The project is licensed under GPL-3.0-or-later. C5.0 and RuleQuest are
associated with RuleQuest Research Pty Ltd. This project is not an official
RuleQuest distribution.
