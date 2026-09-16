# Outlier Detection with IQR and Z-Scores

Most Wheel & Way trips last somewhere between three and forty minutes, which matches a
short commute or errand around Bramford. The trip log also contains a small number of
durations over twelve hours, and a handful of durations of zero or a few seconds. Some
of these extreme values are real - a bike checked out overnight and returned late, or a
rider who docked and immediately undocked the same bike - but others are the result of a
clock glitch or a station sensor that briefly lost connection.

The interquartile range, or IQR, gives a way to flag unusually extreme values without
assuming the data follows any particular distribution. You compute the 25th percentile
(Q1) and 75th percentile (Q3) of the duration column, take their difference as the IQR,
and treat anything below `Q1 - 1.5 * IQR` or above `Q3 + 1.5 * IQR` as a candidate
outlier. This threshold is a convention, not a law of nature, but it is a well-known
starting point that adapts automatically to the spread of whichever column you apply it
to.

The z-score approach instead measures how many standard deviations a value sits from the
column's mean, and commonly flags anything beyond three standard deviations. Z-scores
assume something closer to a bell-shaped distribution, so they can behave oddly on a
duration column that is skewed by a long tail of overnight rentals; the IQR method tends
to be more robust for exactly that kind of skew, since percentiles are less sensitive to
extreme values than the mean and standard deviation are.

Flagging a value as a statistical outlier is not the same as knowing what to do with it.
An overnight rental of 700 minutes might be a perfectly real, if unusual, trip that
should stay in the dataset for anyone studying long rentals, while a duration of zero
minutes for a trip that has different start and end stations is almost certainly a
logging error and a strong candidate for removal or correction.

A practical habit is to keep an `is_outlier` boolean column produced by your chosen
method rather than deleting flagged rows immediately. That keeps the decision reversible
and visible, lets you compute summary statistics with and without outliers side by side,
and gives whoever reviews your cleaning code a clear record of exactly which rows were
treated as unusual and why.

Outlier thresholds should be set per column, not globally. A threshold appropriate for
trip duration in minutes would flag almost every value in a column measured in
kilometres, so always re-derive Q1, Q3, or the mean and standard deviation from the
specific column you are examining rather than reusing a number computed elsewhere.

## 1. Practice Questions

### Q1. (Multiple Choice)
Why might the IQR method be preferred over the z-score method for flagging outliers in a skewed duration column?

- A) The IQR method requires fewer lines of code to implement. - **Incorrect.** Code length is not the reason one statistical method is preferred over another.
- B) The IQR is based on percentiles, which are less sensitive to extreme values than the mean and standard deviation used by the z-score method. - **Correct answer.** This robustness to skew and extreme values is exactly why IQR often suits skewed data better.
- C) Z-scores can only be computed on integer columns. - **Incorrect.** Z-scores work on any numeric column, integer or float.
- D) The IQR method guarantees zero false positives. - **Incorrect.** No statistical thresholding method guarantees zero false positives.

- **Answer:** B

### Q2. (True/False)
Flagging a row as a statistical outlier automatically means it should be deleted from the dataset.

- **Answer:** False
- **Hint:** The lesson distinguishes between flagging and deciding what action to take.

### Q3. (Open-Ended)
Explain why it can be useful to keep an `is_outlier` column instead of immediately dropping rows flagged as outliers.

- **Answer:** Keeping the flag makes the decision reversible and auditable: you can compare statistics with and without those rows, and a reviewer can see exactly which rows were treated as unusual and by what rule, instead of the data silently disappearing.
