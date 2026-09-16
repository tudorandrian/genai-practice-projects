# Model Persistence and Simple Monitoring

A model that only exists inside a notebook session is not useful to anyone else at Wheel
& Way. This final lesson covers saving a fitted pipeline so it can be reloaded and
reused later, and setting up a lightweight way to notice when its predictions start
drifting away from reality.

`joblib.dump(pipeline, 'ride_model.joblib')` saves the entire fitted pipeline -
preprocessing steps and model together - to a single file;
`joblib.load('ride_model.joblib')` restores it in a completely separate process, ready
to call `.predict()` on new rows exactly as it did right after training, with no need to
refit or re-derive any scaler or encoder by hand.

Saving a model file is not enough on its own; it should be saved alongside the metrics
that describe how good it is and when it was trained - test MAE, the date range of
training data, and which features it expects - so that six months later, someone
deciding whether to trust its predictions does not have to guess whether it is still the
best available model or a stale one nobody has revisited.

Models degrade over time even without any code changing, a phenomenon usually called
drift: Wheel & Way might open a seventh station, change its pricing, or see a shift in
commuting patterns that the training data never saw, and the model's assumptions quietly
stop matching reality. A simple monitoring habit is to log the model's prediction
alongside the actual ride count once it becomes known each day, and periodically
recompute MAE over just the last few weeks of these logged pairs.

A rising recent-MAE trend, compared against the MAE recorded at training time, is the
signal that retraining is due - not a fixed calendar schedule, which might retrain a
perfectly good model unnecessarily or leave a genuinely stale one in place too long
between scheduled updates.

This closes the loop the course opened in its first lesson: from framing what to
predict, through baselines, linear and regularized models, trees and ensembles,
clustering, and a reusable pipeline, to a model that is not just accurate once but
saved, documented, and watched for the point where it stops being accurate - the full,
ordinary lifecycle of an applied machine learning model, not just the part that ends at
a single, one-time test-set score.

## 1. Practice Questions

### Q1. (Multiple Choice)
Why is it important to save training-time metadata, such as test MAE and the training data's date range, alongside the saved model file itself?

- A) `joblib.load` requires this metadata to load the file at all. - **Incorrect.** `joblib.load` restores the pipeline regardless of whether separate metadata is saved alongside it.
- B) Metadata makes the model file smaller. - **Incorrect.** Saving additional metadata adds information rather than reducing file size.
- C) Without it, someone reusing the model later has no way to judge whether it is still the best available model or a stale one that has not been revisited. - **Correct answer.** This is exactly the gap the metadata closes for anyone deciding whether to trust the model later.
- D) Metadata is required for the pipeline's `.predict()` method to function. - **Incorrect.** `.predict()` works from the fitted pipeline alone; metadata serves human judgement, not the prediction call itself.

- **Answer:** C

### Q2. (True/False)
Retraining a model on a fixed calendar schedule is generally preferable to monitoring recent prediction error and retraining when it rises.

- **Answer:** False
- **Hint:** Consider the lesson's argument for what should actually trigger a retrain.

### Q3. (Open-Ended)
Describe a concrete change at Wheel & Way that could cause the ride-count model to drift, and explain briefly why the model would not automatically account for it.

- **Answer:** Opening a seventh station would shift the overall ride pattern in a way the model never saw during training, and since the model only encodes relationships learned from historical data, it has no built-in mechanism to notice or adapt to a structural change like a new station unless it is retrained on data that includes it.
