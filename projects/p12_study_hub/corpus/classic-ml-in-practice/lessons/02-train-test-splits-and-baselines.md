# Train/Test Splits and Baseline Models

A model's score on the exact data it was trained on tells you almost nothing about how
well it will predict tomorrow's ride count, because a sufficiently flexible model can
simply memorize the training rows rather than learn a pattern that generalizes. The fix
is to hold out a portion of the data the model never sees during training, and to
evaluate only on that held-out portion.

For the daily Wheel & Way ride counts, a simple random split is the wrong choice: rows
are ordered in time, and a random split would let the model train on some days after the
test days it is being evaluated on, which is a form of leakage that makes performance
look better than it would in a real forecasting setting. A time-based split - train on
the first ten months of the year, test on the last two - matches how the model would
actually be used, predicting the future from the past.

`scikit-learn`'s `train_test_split` has a `shuffle` argument specifically for this
reason: setting it to `False` on data already sorted by date preserves chronological
order across the split instead of randomly interleaving train and test rows.

Before fitting any real model, it is worth establishing a baseline: the simplest
possible prediction rule, measured with the same metric you will use for every model
after it. For daily ride counts, a reasonable baseline predicts each day's rides as the
mean of the training set, ignoring every feature entirely. Any real model that cannot
beat this baseline is not adding value, no matter how sophisticated it is.

Mean absolute error (MAE) is a natural first metric here: it is the average absolute
difference between predicted and actual rides, expressed in the same units as the target
itself, which makes it easy to explain to a non-technical stakeholder - 'the model is
off by about 40 rides a day on average' is a sentence anyone on the operations team can
act on.

Recording the baseline's MAE alongside every subsequent model's MAE, in the same table
or plot, turns model evaluation into a running comparison rather than a series of
disconnected numbers, and makes it immediately visible whether the added complexity of a
later lesson's model is actually earning its keep.

It is worth resisting the temptation to peek at test-set performance more than once per
genuinely new modelling idea. Repeatedly checking the test set and adjusting a model in
response, over many small iterations, slowly turns the test set into something the model
was effectively tuned against, which quietly erodes the very independence that made it a
trustworthy final check in the first place.

## 1. Practice Questions

### Q1. (Multiple Choice)
Why is a random train/test split generally inappropriate for the daily ride count data in this course?

- A) Random splitting is never valid for any machine learning problem. — **Incorrect.** Random splitting is valid and common for many problems; the issue here is specifically about time order.
- B) `train_test_split` cannot be used on numeric targets. — **Incorrect.** `train_test_split` works on numeric targets without restriction.
- C) The rows are ordered in time, so a random split could train on days after some test days, leaking future information the model would not have in real use. — **Correct answer.** This time-ordering leakage is exactly why a chronological split matches real forecasting use.
- D) A random split always produces exactly a 50/50 division. — **Incorrect.** The split proportion is a separate parameter from whether the split is random or ordered.

- **Answer:** C

### Q2. (True/False)
A baseline model that always predicts the training mean is a useful reference point even though it ignores every feature.

- **Answer:** True
- **Hint:** Consider what a baseline is meant to establish before evaluating more complex models.

### Q3. (Open-Ended)
Explain why reporting mean absolute error in the target's original units (rides per day) can be more useful for communicating with a non-technical stakeholder than a more abstract statistical metric.

- **Answer:** It translates model performance directly into a concrete, actionable statement - being off by about 40 rides a day - that someone without a statistics background can interpret and use to judge whether the model's errors matter for their decisions.
