# Building a scikit-learn Pipeline

Every model in this course so far has needed the same sequence of steps applied
correctly and in the same order every time: encode categorical features, scale numeric
features, fit the scaler only on training data, then fit the model. Repeating that
sequence by hand for every experiment is exactly the kind of place small, easy-to-miss
mistakes creep in - such as accidentally fitting a scaler on the full dataset instead of
just the training split, the leakage problem raised twice already in this course.

`sklearn.pipeline.Pipeline` packages a sequence of transformation and modelling steps
into a single object with a `.fit()`/`.predict()` interface identical to any individual
model. Calling `.fit(X_train, y_train)` on a pipeline fits every step in order, and
calling `.predict(X_test)` applies every transformation step's already-fitted state to
the test data before the final model makes its prediction - by construction, it becomes
much harder to accidentally leak test-set information into the fitted transformers.

`sklearn.compose.ColumnTransformer` extends this to datasets with a mix of column types,
which the Wheel & Way feature table is: it applies one-hot encoding only to the
categorical columns (day of week, holiday flag) and scaling only to the numeric columns
(temperature), combining the results into a single feature matrix that the final model
step then consumes - each column gets exactly the preprocessing appropriate to its type,
specified once, in one place.

Wrapping a `ColumnTransformer` and a model, such as `GradientBoostingRegressor` from
earlier, inside one `Pipeline` also makes model comparison and tuning far more
convenient: `GridSearchCV` or `RandomizedSearchCV` can search over hyperparameters for
the model step and the preprocessing step together, correctly refitting the entire
pipeline - preprocessing included - on each cross-validation fold, rather than fitting
preprocessing once outside the search loop and risking it seeing data it should not.

A pipeline is also what makes a model genuinely reusable: once fit, the whole object -
preprocessing and model together - can be saved with `joblib.dump` and loaded elsewhere
with `joblib.load`, ready to transform and predict on new raw data with the exact same
steps used during training, with no risk of someone downstream applying a slightly
different, hand-written preprocessing sequence that does not match what the model was
actually trained on.

Building the ride-count model as a pipeline from this point in the course onward is what
turns the individual pieces from earlier lessons - encoding, scaling, a regularized or
tree-based model - into a single, reproducible unit that can be evaluated, tuned, saved,
and handed off with far less room for the kind of quiet mistakes this lesson opened
with.

## 1. Practice Questions

### Q1. (Multiple Choice)
What is the main benefit of wrapping preprocessing and a model inside a single `Pipeline` object?

- A) It makes it much harder to accidentally leak test-set information into fitted transformers, since `.fit()`/`.predict()` apply the steps consistently and in the correct order. - **Correct answer.** Preventing this specific leakage failure mode by construction is exactly the pipeline's central benefit.
- B) It guarantees the model will have a lower test MAE than any unpipelined model. - **Incorrect.** A pipeline changes how steps are organized, not the fundamental accuracy ceiling of the underlying model.
- C) It removes the need for a train/test split entirely. - **Incorrect.** A pipeline still requires the same train/test discipline; it does not eliminate the need for a split.
- D) It only works with linear models, not trees or ensembles. - **Incorrect.** Pipelines work with any scikit-learn-compatible estimator, including trees and ensembles.

- **Answer:** A

### Q2. (True/False)
A `ColumnTransformer` can apply different preprocessing, such as encoding versus scaling, to different columns within a single combined step.

- **Answer:** True
- **Hint:** Consider how the Wheel & Way feature table mixes categorical and numeric columns.

### Q3. (Open-Ended)
Explain why using `GridSearchCV` over a full pipeline, rather than over a model fit on already-preprocessed data, better protects against test-set leakage during cross-validation.

- **Answer:** Searching over the full pipeline refits preprocessing steps such as scaling separately within each cross-validation fold, so the preprocessing never sees the fold's held-out validation rows, whereas preprocessing the whole dataset once beforehand would let every fold's validation data influence the fitted transformers.
