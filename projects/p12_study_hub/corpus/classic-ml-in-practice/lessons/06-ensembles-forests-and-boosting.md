# Ensembles: Random Forests and Gradient Boosting

The instability of a single decision tree, noted at the end of the previous lesson, has
a surprisingly simple remedy: train many trees instead of one, and combine their
predictions. `sklearn.ensemble.RandomForestRegressor` trains a configurable number of
trees, each on a random bootstrap sample of the training rows and considering only a
random subset of features at each split, then averages every tree's prediction for the
final output.

This randomness is the whole point, not an accident: because each tree in the forest
sees slightly different data and slightly different features at each split, their
individual mistakes tend not to line up with each other, and averaging many such trees
smooths those mistakes out in a way a single tree cannot smooth out its own. The trade-
off is that a forest's predictions are harder to explain than a single tree's, though
`feature_importances_` still gives a ranked summary of which features the forest relied
on most across all of its trees.

Gradient boosting takes a different approach to combining trees: instead of training
many trees independently and averaging them,
`sklearn.ensemble.GradientBoostingRegressor` trains trees one after another, where each
new tree is fit specifically to the errors the trees before it made, gradually
correcting the ensemble's mistakes rather than averaging over independent guesses.

Gradient boosting often reaches a lower test MAE than a random forest on tabular data
like the Wheel & Way ride counts, but it is more sensitive to its settings -
particularly the learning rate, which controls how strongly each new tree corrects the
previous ones' errors, and the number of trees, `n_estimators`. Too high a learning rate
or too many trees, and boosting starts overfitting in a way random forests are
comparatively resistant to; this sensitivity is the price for its usually stronger
performance.

Comparing random forest and gradient boosting against the single decision tree, and
against the linear and regularized models from earlier lessons, on the same held-out
test set is what actually settles which model is 'best' for this dataset - there is no
universal winner among these algorithm families, only a winner for a specific dataset
under a specific evaluation.

Both ensemble methods lose the direct interpretability of a single small tree's if-else
structure or a linear model's coefficients, which is a real cost worth weighing against
their typically better accuracy: an operations team that wants to understand exactly why
tomorrow's forecast is high might prefer a slightly less accurate but more explainable
model.

## 1. Practice Questions

### Q1. (Multiple Choice)
What is the key difference in how random forests and gradient boosting combine their individual trees?

- A) Random forests use only one tree, while boosting uses many. - **Incorrect.** Both methods use many trees; the difference is in how those trees are trained and combined.
- B) Gradient boosting trains all its trees on identical data with no differences. - **Incorrect.** Boosting trees are trained sequentially, each targeting the previous ensemble's errors, not on identical unchanged targets.
- C) Random forests train many trees independently on random samples and average them, while gradient boosting trains trees sequentially, each correcting the errors of the ones before it. - **Correct answer.** This independent-averaging versus sequential-correction distinction is exactly the key difference.
- D) Random forests cannot report which features mattered most. - **Incorrect.** `feature_importances_` is available for random forests as well as boosted models.

- **Answer:** C

### Q2. (True/False)
Gradient boosting is generally more sensitive to its hyperparameters, such as learning rate, than random forests are.

- **Answer:** True
- **Hint:** Consider the trade-off the lesson describes for boosting's usually stronger performance.

### Q3. (Open-Ended)
Explain one reason an operations team might prefer a single decision tree over a random forest, even if the forest has a lower test MAE.

- **Answer:** A single tree's splits can be read directly as an explainable if-else sequence, so a team that needs to understand and justify exactly why a forecast is high or low might value that transparency over the forest's typically better but harder-to-explain accuracy.
