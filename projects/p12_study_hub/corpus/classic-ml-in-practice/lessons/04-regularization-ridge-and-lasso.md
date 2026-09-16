# Regularization: Ridge and Lasso

As more features get added to the Wheel & Way ride model - not just temperature and day
of week, but rolling averages of recent rides, interaction terms between rain and
weekday, and so on - plain linear regression can start to overfit: it fits the training
data closely, including its noise, at the cost of generalizing worse to the held-out
test set. Regularization is a family of techniques that discourage the model from
relying too heavily on any single feature, which tends to reduce this kind of
overfitting.

Ridge regression (`sklearn.linear_model.Ridge`) adds a penalty to the training objective
proportional to the sum of the squared coefficients, controlled by a strength parameter
conventionally called alpha. A larger alpha pushes every coefficient closer to zero
without necessarily reaching zero, which tends to produce a model that spreads its
reliance more evenly across correlated features rather than leaning heavily on just one
of them.

Lasso regression (`sklearn.linear_model.Lasso`) uses a different penalty, proportional
to the sum of the absolute values of the coefficients, and this different shape of
penalty has a distinctive practical effect: lasso can push some coefficients to exactly
zero, effectively removing those features from the model entirely. For the Wheel & Way
feature set, this makes lasso useful as an automatic, if somewhat blunt, feature
selection tool - if a rolling-average feature's coefficient goes to zero, that is a
signal it added little beyond what the other features already captured.

Choosing alpha is itself a model-selection problem, not something to guess once and
leave fixed. `RidgeCV` and `LassoCV` try a range of alpha values and pick the one that
performs best under cross-validation - repeatedly splitting the training data itself
into smaller train/validation folds - rather than requiring you to tune alpha by hand
against the test set, which would itself be a form of leakage: the test set is meant to
be touched only once, for the final evaluation.

It is worth checking regularized coefficients against the unregularized linear
regression from the previous lesson side by side. A feature whose coefficient shrinks
dramatically under ridge, or vanishes entirely under lasso, was likely contributing
mostly noise or redundant information in the unregularized model - useful to know both
for simplifying the model and for understanding the data itself.

As with every model in this course, the real test is the held-out MAE, not how elegant
the regularization story sounds. A ridge or lasso model that beats plain linear
regression on the test set has earned its added complexity; one that does not is a sign
the original model was not overfitting enough for regularization to help.

## 1. Practice Questions

### Q1. (Multiple Choice)
What distinguishes lasso regression's practical effect from ridge regression's?

- A) Lasso cannot be used with more than ten features. - **Incorrect.** Lasso places no such limit on the number of features it can handle.
- B) Lasso's penalty can push some coefficients to exactly zero, effectively removing those features, while ridge shrinks coefficients toward zero without typically eliminating them. - **Correct answer.** This exact-zero behaviour is the well-known distinguishing property of the lasso penalty.
- C) Ridge regression requires labeled data while lasso does not. - **Incorrect.** Both ridge and lasso are supervised methods that require a labeled target.
- D) Lasso always produces a lower training error than ridge. - **Incorrect.** Neither method guarantees a lower training error than the other; it depends on the data and alpha.

- **Answer:** B

### Q2. (True/False)
Tuning the regularization strength `alpha` by directly checking performance against the test set is a valid way to pick its value.

- **Answer:** False
- **Hint:** Recall the lesson's point about the test set being touched only once, for final evaluation.

### Q3. (Open-Ended)
Explain why a lasso model's coefficient shrinking to exactly zero for a rolling-average feature is useful information, beyond simplifying the model itself.

- **Answer:** It suggests that feature contributed little predictive value beyond what the other features already captured, which is useful both for simplifying future models and for understanding which aspects of the data genuinely drive rides.
