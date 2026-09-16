# Missing-Value Imputation Strategies

Once sentinels have been converted to real missing markers, the Wheel & Way trip log
still has genuine gaps: a rider age that was never collected for casual riders, a bike
type that a faulty station sensor failed to record, and a handful of end stations left
blank when a bike was picked up by a maintenance crew instead of returned normally.
Deciding what to do with each of these gaps is a separate question for every column, not
a single rule applied everywhere.

The simplest option is to drop rows with missing values, using `dropna()`. This is
defensible when missing values are rare and concentrated in a column you do not need for
the analysis at hand, but it becomes costly quickly: dropping every row with any missing
field at all can discard a large fraction of an otherwise usable dataset if several
columns each have a small, unrelated sliver of gaps.

For numeric columns you intend to keep, filling gaps with the column's mean or median is
a common baseline. The median is usually the safer default for a skewed column like
duration, since it is not pulled around by the same extreme values that make the mean
unreliable; for a roughly symmetric column, the mean and median will be close enough
that the choice matters less.

For categorical columns, filling with the most frequent category (the mode) is the
equivalent default, though it should be used carefully: if 'member' already accounts for
80 percent of rider types, filling every missing rider type with 'member' will further
exaggerate that imbalance rather than represent genuine uncertainty about the missing
rows.

A more transparent alternative, and often the best one, is to fill a missing value with
an explicit placeholder category such as 'unknown' rather than a guess, and to also add
a boolean column recording which rows were originally missing. This keeps every row
usable while preserving the information that a value was imputed, so anyone using the
table downstream can choose to treat those rows differently, or exclude them, if the
analysis is sensitive to the distinction.

Whichever strategy you choose, imputation should always be logged: which column, which
method, and how many rows were affected. A dataset where 60 percent of a column was
imputed with the mean tells a very different story from one where 2 percent was, and
that context is often more important to a downstream analyst than the imputed values
themselves.

## 1. Practice Questions

### Q1. (Multiple Choice)
For a skewed numeric column such as trip duration, why is the median often preferred over the mean as a fill value for missing entries?

- A) The median is less affected by extreme values than the mean, so it better represents a 'typical' duration in a skewed distribution. - **Correct answer.** This robustness to extreme values is exactly why the median is the safer default here.
- B) The median is always numerically equal to the mean. - **Incorrect.** The two are only close for roughly symmetric distributions, and diverge under skew.
- C) Pandas cannot compute a mean on a column containing missing values. - **Incorrect.** Pandas computes the mean over the non-missing values by default; it is not blocked by gaps.
- D) The median must always be an integer. - **Incorrect.** The median of a numeric column can be a fractional value, just like the mean.

- **Answer:** A

### Q2. (True/False)
Adding a boolean column that records which rows had an imputed value is a way to preserve information that would otherwise be lost by filling gaps silently.

- **Answer:** True
- **Hint:** Think about what a downstream analyst loses if imputed and genuine values look identical.

### Q3. (Open-Ended)
Give one reason why filling every missing `rider_type` with the most frequent category could be a poor choice if that category already dominates the column.

- **Answer:** It would further exaggerate an already imbalanced distribution and could hide the fact that the missing rows might disproportionately belong to the less common category, biasing any analysis that relies on rider-type proportions.
