# The C5.0 algorithm

This page describes how C5.0 Release 2.07 GPL Edition, as preserved by this
project, constructs and applies classifiers. It is derived from the source
code rather than from RuleQuest's published descriptions, and it names the
functions that implement each step so that readers can check the details.
Unless a section says otherwise, the command-line program, the C++ library, and
the Python bindings run the same code.

Logarithms are base 2 throughout, and $\log 0$ is taken as 0. "Weight" means a
case's current weight, which starts at 1 and changes with case weights,
misclassification costs, missing values, and boosting. Counts such as "cases"
are sums of weights unless stated otherwise.

## Data

### Attributes and values

A names file, or the equivalent in-memory names buffer, declares the class and
the attributes. `GetNames` and `ExplicitAtt` in `src/getnames.cpp` parse it.

- **Continuous** attributes hold real numbers. Dates, times, and timestamps are
  continuous internally: `DateToDay`, `TimeToSecs`, and `TStampToMins` convert
  them to days, seconds, and minutes.
- **Discrete** attributes hold one of an explicit list of values. A
  `discrete N` declaration instead collects up to `N` values from the data. An
  `[ordered]` list declares an ordered discrete attribute, whose values have a
  meaningful order.
- **Implicit** attributes are defined by a formula over other attributes
  (`att := expression.`). `src/implicitatt.cpp` compiles the formula and
  evaluates it for each case as the data is read.
- `label` attributes identify cases in output, `ignore` attributes are read
  and discarded, and a continuous attribute named `case weight` supplies case
  weights.

Every value may be unknown, written `?`, or not applicable, written `N/A`.
C5.0 keeps the two apart throughout. An unknown value is a value that exists but
was not recorded, so an unknown case is distributed over all branches of a
test. A not-applicable value means that the attribute does not apply to the
case, so it forms a branch of its own. Internally, value 1 of every discrete
attribute is reserved for `N/A`, so an attribute with $k$ declared values has
$k + 1$ possible values.

### Initial weights

`InitialiseWeights` in `src/construct.cpp` gives every case weight 1. It then
applies two adjustments:

- With a two-class costs file, every case of class $c$ gets the weight
  multiplier $m_c$ described in [Misclassification costs](#algorithm-costs).
- With a case-weight attribute, every case's weight is multiplied by its case
  weight divided by the mean positive case weight, which `SetAvCWt` computes.
  A case whose weight is unknown, not applicable, zero, or negative gets
  relative weight 1.

## Growing a tree

`FormTree` in `src/formtree.cpp` grows a tree recursively by divide and
conquer. At each node it considers the cases that reach the node, whose total
weight is $N$.

### When a node becomes a leaf

A node starts as a leaf labeled with its most frequent class. `FormTree` stops
there without evaluating any test when any of these conditions holds:

- The most frequent class holds at least 99.9% of the weight.
- $N$ is less than twice the minimum number of cases per branch, `-m` or
  `minimum_cases` (default 2).
- A boosting trial's leaf budget allows fewer than two leaves. See
  [Boosting](#algorithm-boosting).

### Information and gain

For a weighted class distribution $f_1, \ldots, f_K$ with total $n$,
`TotalInfo` in `src/info.cpp` computes

$$
I(f) = n \log n - \sum_{c=1}^{K} f_c \log f_c ,
$$

which is $n$ times the entropy of the distribution. The information at a node
is $I_0 = I(\text{class frequencies}) / N$.

A test partitions the cases with known values into branches with class
distributions $f^{(1)}, \ldots, f^{(B)}$. Let $u$ be the fraction of the
node's weight whose tested value is unknown, and let $N_k$ be the weight with a
known value. The gain of the test is

$$
\mathrm{Gain} = (1 - u) \left( I_\text{known} - \frac{1}{N_k} \sum_{b=1}^{B} I\!\left(f^{(b)}\right) \right),
$$

where $I_\text{known}$ is the information of the known cases alone. When no
value is unknown, $u = 0$ and $I_\text{known} = I_0$. `ComputeGain` returns 0
instead of a negative gain.

The split information of the test treats the unknown cases as one more
partition:

$$
\mathrm{Split} = \frac{1}{N} I(w_0, w_1, \ldots, w_B),
$$

where $w_0$ is the unknown weight and $w_b$ is the weight of branch $b$. C5.0
ranks tests by the gain ratio $\mathrm{Gain} / \mathrm{Split}$.

### Discrete attributes

`EvalDiscreteAtt` in `src/discr.cpp` evaluates a test with one branch per
value, including the `N/A` value. The test is rejected unless at least two
values have at least `minimum_cases` weight. A multiway test on an attribute
is not repeated below a node that already tests it: `tested_attributes`
records the attributes on the current path.

An ordered discrete attribute instead gets a three-way test. `EvalOrderedAtt`
places `N/A` in the first branch and tries each cut between consecutive
values, sending lower values to the second branch and higher values to the
third. Each branch on either side of the cut must hold at least
`minimum_cases`. The best cut's gain is reduced by $\log(T) / N$, where $T$ is
the number of cuts tried.

#### Value subsets

With `-s` or `subset_splits`, an unordered attribute with more than three
possible values, counting `N/A`, may branch on groups of values. `EvalSubset`
in `src/subset.cpp` builds the groups:

1. Each value that occurs at the node starts as its own block, and values that
   do not occur are set aside. `N/A` is never merged with another block.
2. Blocks with identical class distributions are merged.
3. Blocks are merged greedily, two at a time, choosing the pair whose merger
   gives the largest gain, until two blocks remain.

Every grouping along the way is a candidate, including the initial one. A
grouping of $b$ blocks formed from $B$ initial blocks is penalized by
$\log S(B, b)$ bits, where $S(B, b)$ is the Stirling number of the second
kind, the number of ways to partition $B$ values into $b$ nonempty groups.
`InitialiseBellNumbers` tabulates these numbers, and an approximation replaces
any that overflow. The grouping with the best penalized gain ratio wins, and
the values that did not occur at the node form one more branch.

### Continuous attributes

`EvalContinuousAtt` in `src/contin.cpp` evaluates binary thresholds. The test
has three branches: `N/A`, values at or below the threshold, and values above
it.

1. `PrepareForContin` removes unknown and `N/A` cases and sorts the remaining
   cases by value with `Cachesort` in `src/sort.cpp`.
2. Each branch must hold at least
   $\max(\texttt{minimum\_cases}, \min(25, 0.1 N_k / K))$ weight, where $K$ is
   the number of classes. `PrepareForScan` skips the cuts that violate this at
   either end.
3. The scan visits every cut between two distinct values. It skips a cut when
   the cases with the values on both sides of it all belong to the same class,
   because such a cut cannot be optimal. It stops early once the information
   of the lower branch alone reaches the best total found so far.
4. The best cut minimizes the information of the two branches. Its gain is
   reduced by a threshold cost of $\min(\log T, \log R) / N$ bits, where $T$
   is the number of candidate cuts and $R$ is the range of the attribute's
   values divided by the gap at the chosen cut.

The threshold is the midpoint of the two values on either side of the cut.
After pruning, `AdjustAllThresholds` replaces each threshold with the largest
value of the attribute in the training data that does not exceed it, so that
printed thresholds are values that appear in the data.

The order in which `Cachesort` leaves equal values determines which of several
tied cuts is chosen. The `ties` option selects a faster, stable sort, which may
produce a different classifier. See {doc}`compatibility`.

### Choosing a test

`ChooseSplit` evaluates every attribute that is not excluded, and `FindBestAtt`
chooses among them.

1. Attributes with gain below $10^{-4}$ are discarded. So are attributes with
   at least $0.3 N_\text{train}$ values, where $N_\text{train}$ is the number
   of training cases, unless every attribute has that many values.
2. The remaining $P$ attributes set a minimum gain. With $\bar g$ their mean
   gain and $M = \log(P) / N$, the minimum is $\bar g$ when there are at most
   500 training cases per class, $0.9 M$ when there are at least 1000, and a
   linear interpolation between the two otherwise. `SetMinGainThresh` computes
   the weights. Winnowing trees use no minimum.
3. Among attributes with at least 99.9% of the minimum gain, C5.0 chooses the
   highest gain ratio. A candidate within 0.1% of the best ratio replaces it
   when its test has fewer branches, or the same number and a higher gain.

If no attribute qualifies, the node remains a leaf.

The imported program evaluates attributes one at a time. This project can
evaluate eligible attributes in parallel: see `context::split_workers` in C++,
the `split_workers` argument in Python, and `n_jobs` in the estimator. Each
worker evaluates whole attributes, and the results are published in the serial
order, so the chosen test does not depend on the number of workers.

### Dividing the cases

`Divide` in `src/formtree.cpp` sends each case with a known value down the
matching branch and grows each branch with `FormTree`. A case with an unknown
value goes down every branch, with its weight multiplied by the fraction of
known weight that the branch receives. A branch whose weight, including these
fractions, is below 0.05 becomes an empty leaf labeled with the parent's
class.

After the branches are grown, `FormTree` compares the training errors of the
subtree with the errors of the node as a leaf. If the subtree does not reduce
the errors by more than 0.1%, it collapses the node back into a leaf.

## Pruning

`Prune` in `src/prune.cpp` simplifies the grown tree in several passes.

### Pessimistic error estimates

C5.0 estimates the error rate of a leaf as the upper limit of a binomial
confidence interval. For a leaf with $N$ cases and $E$ errors, `ExtraErrs` and
`RawExtraErrs` compute the extra errors $X$ added to $E$. With confidence
factor $\mathit{CF}$, `-c` or `confidence_factor` (default 25%), and $z$ the
corresponding normal deviate, interpolated from a table by
`InitialiseExtraErrs`:

- If $E = 0$: $X = N\,(1 - \mathit{CF}^{1/N})$, the exact binomial limit.
- If $0 < E < 1$ and $N > 1$: $X$ interpolates linearly between the values for
  $E = 0$ and $E = 1$.
- If $E + 0.5 \ge N$: $X = 0.67\,(N - E)$.
- Otherwise, $X = N p - E$, where $p$ is the upper limit of the normal
  approximation with a continuity correction:

$$
p = \frac{E + 0.5 + \frac{z^2}{2} + z \sqrt{(E + 0.5)\left(1 - \frac{E + 0.5}{N}\right) + \frac{z^2}{4}}}{N + z^2} .
$$

Smaller confidence factors increase the estimates and prune more heavily.

### Local pruning

`EstimateErrs` traverses the tree bottom up and recomputes each node's class
distribution from the training cases. At each internal node it compares three
estimates:

- **Leaf**: the node's errors as a leaf plus their extra errors.
- **Subtree**: the sum of the branches' estimates. Branches with fewer than
  `minimum_cases` cases are pooled and estimated as one leaf.
- **Largest branch**: the estimate for the largest internal branch when it
  classifies all of the node's cases. A branch qualifies only if it holds at
  least 10% of the cases and does not test the same continuous attribute.

The node becomes a leaf if the leaf estimate is within 0.1 of both others.
Otherwise the largest branch replaces the node if its estimate is within 0.1 of
the subtree's. This replacement is known as subtree raising. In the first
boosting trial and while winnowing, a raised branch that held less than 95% of
the node's cases is grown again from all of them.

### Global pruning

Unless `-g` is given or `global_pruning` is false, `GlobalPrune` then prunes
the whole tree by cost complexity. It repeatedly replaces the subtrees with the
lowest ratio of added errors to removed leaves with leaves, as long as the
total added errors stay within one standard error,
$\sqrt{E (1 - E / N_\text{train})}$, of the tree's errors $E$. Global pruning
is skipped when a costs file is in use and while winnowing.

### Final cleanup

`CheckSubsets` removes values from subset and ordered tests that cannot reach
them. Without `-s`, `CompressBranches` then merges the empty leaves of each
test with at least four branches into one subset branch. In the first trial of
a tree classifier, it also merges leaves of the same class.

## Soft thresholds

With `-p` or `probabilistic_thresholds`, `SoftenThresh` in
`src/p-thresh.cpp` gives each continuous test a lower bound, a midpoint, and an
upper bound. `FindBounds` sets the bounds where the errors of sending the
nearby cases to the other branch grow too large, using limits of half a
standard error and twice the errors of the correct branch after a Laplace
correction.

When classifying a value $v$, `Interpolate` in `src/classify.cpp` gives the
lower branch the weight 1 below the lower bound, 0 above the upper bound, and a
piecewise-linear value through 0.5 at the midpoint in between. Branches whose
share of the case falls below 0.01 are not followed.

## Rulesets

With `-r` or a rules model kind, C5.0 converts each pruned tree into a ruleset.
`FormRules` in `src/formrules.cpp` and `SiftRules` in `src/siftrules.cpp`
implement the conversion.

### Candidate rules

`Scan` forms a candidate rule from every node other than the root that holds
at least one case. The rule's conditions are the tests on the path from the
root, and its class is the node's class. A subset test on a single value becomes an equality test.

`PruneRule` then deletes conditions greedily. For each condition, it
determines the cases that satisfy every other remaining condition and computes
a pessimistic error rate $(e + 1) / (n + 2)$, where $n$ is their weight and
$e$ is the weight not of the rule's class. It deletes the condition whose
removal gives the lowest rate, and it continues while that rate does not
exceed the current rule's rate, or while the rule's coding cost is too high
relative to the information it provides:

- The **coding cost** is the sum of the conditions' costs in bits, from
  `CondBits`, less $\log(d!)$ for $d$ conditions because their order does
  not matter.
- The **information** is the reduction in bits of the class/not-class
  distribution of all training cases after splitting them into covered and
  uncovered cases.

A rule survives only if $0.23$ times its coding cost does not exceed its
information, and its Laplace accuracy $(\mathit{correct} + 1) /
(\mathit{cover} + 2)$ is at least 95% of its class's prior probability. The
constant 0.23 is `THEORYFRAC` in `src/defns.i`. Each rule's vote is its
Laplace accuracy times 1000, rounded. Duplicate rules keep the higher vote.

### Selecting rules

`SiftRules` chooses a subset of the candidates by minimum description length.

1. `SetInitialTheory` builds a starting set class by class. `CoverClass`
   repeatedly adds the unused rule with the highest vote, among rules with at
   least `minimum_cases` correct cases, when it covers more new cases of the
   class than new cases of other classes by more than `minimum_cases`.
2. `HillClimb` adds or deletes one rule at a time to reduce the message length

   $$
   L = 0.23 \max\!\left(0, \sum_{r} \mathit{bits}_r - \log(R!)\right)
       + e \log\frac{1}{\hat\epsilon}
       + (N_\text{train} - e) \log\frac{1}{1 - \hat\epsilon}
       + e \log(K - 1),
   $$

   where $R$ is the number of selected rules, $\mathit{bits}_r$ is a rule's
   coding cost plus $\log d$ bits for its number of conditions, $e$ is the
   weight of training errors of the selected rules, and $K$ is the number of
   classes. $\hat\epsilon = (E + K - 1) / (N_\text{train} + K)$ estimates
   the error rate from the pruned tree's estimated errors $E$, and 0.45
   replaces it when it exceeds 0.5. After the first change that would
   increase $L$, `HillClimb` only deletes rules.
3. `SetDefaultClass` chooses the class for cases that no rule covers. It
   maximizes $(u_c + 1) / (u + 2) + p_c$, where $u_c$ is the weight of
   uncovered cases of class $c$, $u$ is their total, and $p_c$ is the class's
   prior probability. With a costs file, it minimizes expected cost instead.
4. `OrderRules` sorts the rules by class and, within each class, by decreasing
   vote. With `-u`, `OrderByUtility` orders them by their contribution to
   accuracy instead, for the utility-band report.

`ConstructRuleTree` in `src/ruletree.cpp` indexes the selected rules by their
conditions so that classification does not test every rule.

(algorithm-boosting)=

## Boosting

With `-b`, `-t`, or `trials` greater than 1, `ConstructClassifiers` in
`src/construct.cpp` builds a sequence of classifiers, each from reweighted
training cases. The command-line `-b` option requests 10 trials. C5.0's
boosting differs from AdaBoost in several ways.

After each trial except the last, C5.0 classifies every training case and
adds the trial's confidence in its prediction to the case's vote for the
predicted class. Let $W_+$ and $W_-$ be the weights of the correctly and
incorrectly classified cases.

- If $W_- < 0.1$, boosting stops because the classifier is essentially
  perfect.
- If a trial after the first produced a classifier without structure, or any
  trial's weighted error rate is at least 0.49, boosting stops and discards
  that trial, unless it is the first.
- Otherwise C5.0 moves the incorrect cases' total weight halfway toward half
  of all the weight, which moves $\delta = (W_+ - W_-)/4$ from the correct
  cases to the incorrect ones. Correct cases are scaled by
  $(W_+ - \delta) / W_+$, and each of the $n_-$ incorrect cases gains
  $\delta / n_-$, an additive rather than multiplicative increase. All weights
  are then normalized so that they sum to the number of active cases, and no
  correct case falls below $10^{-3}$.

In the second half of the trials, a case is dropped when its vote for the
correct class can no longer catch up with the leading class in the remaining
trials, in the manner of Freund's BrownBoost. With more than 4000 cases, once
the smallest weight a case can have reaches 0.2 or less, cases at that weight
are also set aside while trees are grown.

After the first trial, each tree is limited to about 1.1 times as many leaves
per training case as the first tree had, so later trees remain comparable in
size.

If boosting ends with two or fewer trials, C5.0 abandons it and keeps only the
first classifier. The command line accepts 3 to 1000 trials with `-t`. The
library accepts 1 to 1000, so a request for 2 trials produces a single
classifier.

## Winnowing

With `-w` or `winnow`, `WinnowAtts` in `src/attwinnow.cpp` removes attributes
before any classifier is built.

1. It splits the training cases into two halves, alternating the cases of each
   class between the halves.
2. `TrialTreeCost` grows and prunes a tree from the first half with half the
   usual `minimum_cases`, at least 2, and no minimum gain. Attributes that the
   unpruned tree never tests are removed.
3. For each attribute that the pruned tree tests, it measures the error, or
   cost, on the second half when the attribute is ignored. Ignoring an
   attribute makes classification follow all branches of its tests. An
   attribute whose removal lowers the error is removed.
4. If any attribute was removed in step 3, it grows a second trial tree without
   those attributes. If that tree's error on the second half exceeds the first
   tree's, the step 3 attributes are restored.

The remaining attributes are then used for the real classifier. The command
line reports their estimated importance as the percentage increase in error
when each one is removed.

(algorithm-costs)=

## Misclassification costs

A costs file, or the estimator's `cost_matrix`, defines the cost of predicting
one class when the case belongs to another. `src/mcost.cpp` reads it.

With two classes, C5.0 folds the costs into the training weights. Each case of
class $c$ is weighted by $m_c = C_{\bar c c} / S$, where $C_{\bar c c}$ is the
cost of misclassifying class $c$ and $S$ is the class-frequency-weighted mean
of the two costs. `minimum_cases` is scaled by the smaller multiplier. Pruning
removes the multipliers when estimating errors, and `RestoreDistribs` removes
them from the final leaf distributions.

With more classes, the weights are unchanged. In both cases, leaf labels and
tree classification choose the class with the lowest expected cost instead of
the most likely class. Rule construction and selection measure errors with
normalized costs, and the default class minimizes expected cost, but ruleset
classification itself compares votes.

## Sampling and cross-validation

With `-S` or `sample_fraction`, C5.0 trains on a random sample of the cases.
It reads the cases in order and selects each with probability
$w / \ell$, where $w$ cases are still wanted and $\ell$ remain, so the sample
has exactly the requested size. The random numbers come from `KRandom` in
`src/utility.cpp`, a subtractive lagged-Fibonacci generator after Knuth.
`ResetKR` seeds it by discarding $1000 + s$ values, where the command line uses
only the low 12 bits of the seed $s$ given with `-I`. The command line then
evaluates the classifier on the cases that were not sampled. Below 50%, it
evaluates a random subset of them. See {doc}`compatibility` for how this
project draws that subset.

The command line's `-X` option performs cross-validation. `Prepare` in
`src/xval.cpp` shuffles the cases, groups them by class, and deals them into
folds so that each fold has about the same class distribution. `CrossVal`
trains on all folds but one, tests on the held-out fold, and reports the
results for each fold and the average.

## Classification

`Classify` in `src/classify.cpp` returns a class and fills a score for each
class. The C++ and Python prediction interfaces return these scores.

### Trees

`FindLeaf` passes a case down the tree with a weight of 1.

- A known value follows its branch. For continuous tests with soft thresholds,
  it may follow both branches with interpolated weights.
- An `N/A` value follows the `N/A` branch.
- An unknown value, or a discrete value that was not seen in training, follows
  every nonempty branch, with the case's weight split in proportion to the
  training weight of each branch.
- A value that belongs to none of a subset test's branches stops at that node
  and uses its class distribution.
- At a leaf, the weight is multiplied by the leaf's class distribution divided
  by its total. An empty leaf uses its parent's distribution instead.

The scores are the sums over the leaves that the case reaches. The predicted
class has the highest score, or, with a costs file, the lowest expected cost.

### Rulesets

`RuleClassify` finds every rule whose conditions the case satisfies. Each class
score is the sum of the votes of the matching rules for that class, divided by
1000 times the number of matching rules. The predicted class has the highest
score. A tie with the default class goes to the default class, and other ties
go to the class listed first. If no rule matches, every score is 0 and the
prediction is the default class.

### Boosted classifiers

`BoostClassify` classifies the case with each trial's classifier and adds the
trial's confidence to the vote for the class it predicts. A tree's confidence
is its score for the predicted class, or, with a costs file, one minus the
predicted class's share of the total expected cost. A ruleset's confidence is
the vote, divided by 1000, of the most specific matching rule for the predicted
class, which is the one that covers the fewest training cases. The scores are
the votes divided by their sum.

## References

- J. R. Quinlan, *C4.5: Programs for Machine Learning*, Morgan Kaufmann, 1993.
  C5.0 extends the gain ratio criterion, handling of unknown values,
  pessimistic pruning, and rule generation described there.
- J. R. Quinlan, "Improved use of continuous attributes in C4.5," *Journal of
  Artificial Intelligence Research* 4 (1996): 77–90. It introduces the
  threshold cost that `EvalContinuousAtt` subtracts from the gain.
- Y. Freund, "An adaptive version of the boost by majority algorithm,"
  *Machine Learning* 43 (2001): 293–318. `src/construct.cpp` cites this
  BrownBoost approach for dropping cases in later trials.
- RuleQuest Research, "C5.0: An Informal Tutorial,"
  <https://www.rulequest.com/see5-unix.html>. It describes the command-line
  options and output from a user's perspective.
