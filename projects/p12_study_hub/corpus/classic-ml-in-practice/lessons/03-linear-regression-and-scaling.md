# Linear Regression and Feature Scaling

Linear regression is a natural next step after the mean baseline: it fits a weighted sum
of the features - temperature, a holiday indicator, day-of-week indicators - to predict
daily rides, and the fitted weights themselves are interpretable, which is valuable when
the operations team wants to understand why the model predicts what it predicts, not
just what number it produces.

Fitting `sklearn.linear_model.LinearRegression` on the Wheel & Way features is
straightforward mechanically: call `.fit(X_train, y_train)`, then `.predict(X_test)`.
The subtlety is in preparing `X` correctly first. Categorical features like day of week
need to become numeric - one-hot encoding turns a single seven-category column into
seven binary columns - before linear regression can use them at all, since the model
only understands numbers, and treating 'Monday' through 'Sunday' as an arbitrary
1-through-7 scale would falsely imply an ordering the days do not actually have.

Numeric features on very different scales - temperature in single or double digits
versus a station count in the hundreds - do not break plain linear regression the way
they break some other algorithms, but scaling them to a comparable range, typically with
`StandardScaler`, still matters once you start comparing the fitted coefficients
themselves to judge which feature matters most, since an unscaled coefficient's size
reflects the feature's units as much as its real importance.

A critical rule when scaling: fit the scaler only on the training set, then apply that
same fitted transformation to the test set with `.transform()`, never `.fit_transform()`
on the test set. Fitting the scaler on the full dataset, including test rows, leaks
information about the test set's distribution into training - a subtle form of the same
leakage problem the previous lesson raised for time-based splitting.

Once fit, a linear regression model's coefficients tell a story: a large positive weight
on the 'is_weekday' indicator says weekdays see substantially more rides than weekends,
holding other features fixed, while a small coefficient on temperature says temperature
matters less than day of week for this dataset - a concrete, checkable claim rather than
a vague impression.

Comparing this model's test-set MAE against the mean baseline from the previous lesson
is the real test of whether the added complexity was worth it. A linear model that only
marginally beats the baseline is a signal that either the chosen features are not very
predictive, or the true relationship between features and rides is not well described by
a straight-line combination of them - both are useful things to learn before reaching
for a more complex model.

## 1. Practice Questions

### Q1. (Multiple Choice)
Why should a `StandardScaler` be fit only on the training set and then applied to the test set with `.transform()`, rather than fit on the combined dataset?

- A) Because `StandardScaler` cannot process more than one dataset at a time. - **Incorrect.** A fitted scaler can be applied to any number of datasets via `.transform()`.
- B) Because scaling is only needed for categorical features. - **Incorrect.** Scaling concerns numeric features; categorical features are handled separately with encoding.
- C) Because linear regression cannot run on scaled data. - **Incorrect.** Linear regression runs perfectly well, and often more interpretably, on scaled data.
- D) Fitting on the combined dataset would leak information about the test set's distribution into training, similar to other forms of data leakage discussed earlier. - **Correct answer.** This leakage of test-set distribution information is exactly the risk being avoided.

- **Answer:** D

### Q2. (True/False)
One-hot encoding a day-of-week column avoids implying a false numeric ordering among the days that a single 1-through-7 column would suggest.

- **Answer:** True
- **Hint:** Consider what a plain integer scale would imply about the relationship between Monday and Sunday.

### Q3. (Open-Ended)
Explain what it would mean, practically, if a fitted linear regression model's test-set MAE is only marginally better than the mean-baseline MAE from the previous lesson.

- **Answer:** It suggests the chosen features carry limited predictive signal for rides, or that the true relationship between the features and rides is not well captured by a linear combination, and it is a cue to reconsider the features or try a model that can capture non-linear patterns.
