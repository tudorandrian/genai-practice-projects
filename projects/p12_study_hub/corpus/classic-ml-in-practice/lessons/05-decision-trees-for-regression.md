# Decision Trees for Regression

Every model in this course so far assumes rides responds to each feature in a straight-
line way, which is a real limitation: rain might barely matter on a cold day when few
people would ride anyway, but might cut ridership sharply on a warm day when many people
otherwise would. A decision tree regressor can capture exactly this kind of interaction
without anyone having to hand-engineer an interaction feature for it.

`sklearn.tree.DecisionTreeRegressor` builds its predictions by repeatedly splitting the
training data on the feature and threshold that most reduces prediction error at each
step - for instance, first splitting on 'is_weekday', then within weekdays splitting
further on temperature - until it reaches a set of small groups, called leaves, each
predicted with the average rides of the training rows that landed in that leaf.

This flexibility is also the tree's main weakness: left unconstrained, a tree can keep
splitting until every leaf contains a single training row, perfectly memorizing the
training set and generalizing poorly to new days. `max_depth` limits how many splits the
tree can make along any path, and `min_samples_leaf` requires each leaf to contain at
least a certain number of training rows, both of which trade some training accuracy for
a model that is far less likely to overfit.

Plotting training MAE and test MAE against increasing `max_depth` makes overfitting
visible directly: training MAE keeps falling as depth increases, while test MAE
typically falls at first and then starts rising again once the tree is deep enough to be
fitting noise rather than signal. The depth where test MAE is lowest is a reasonable
choice, not the depth where training MAE is lowest.

One clear advantage of a tree over the linear and regularized models from earlier
lessons is that it needs no feature scaling at all: a tree only ever asks whether a
feature is above or below a threshold, so the actual numeric scale of a feature has no
bearing on how the tree splits. It also handles the interaction case from this lesson's
opening naturally, since nothing stops the tree from splitting on temperature
differently within the weekday branch than within the weekend branch.

A single tree, even a well-tuned one, tends to be somewhat unstable: a small change to
the training data can produce a noticeably different tree structure, because an early
split near the top of the tree affects every split beneath it. That instability is the
motivation for the ensemble methods covered next.

## 1. Practice Questions

### Q1. (Multiple Choice)
Why can a decision tree capture an interaction between rain and temperature more naturally than a plain linear regression model can?

- A) A tree can split on temperature differently within different branches (such as rainy versus dry days), effectively letting one feature's effect depend on another without an explicit interaction term. - **Correct answer.** This branch-dependent splitting is exactly how a tree captures interactions without hand-built features.
- B) Linear regression cannot use more than one feature at a time. - **Incorrect.** Linear regression readily combines many features in a single weighted sum.
- C) Decision trees always require fewer training rows than linear regression. - **Incorrect.** Data requirements are not the distinguishing factor being described here.
- D) Trees cannot be regularized in any way. - **Incorrect.** Trees can be constrained through parameters like `max_depth` and `min_samples_leaf`, which act as a form of regularization.

- **Answer:** A

### Q2. (True/False)
Choosing the tree depth that minimizes training MAE is generally the best way to select `max_depth`.

- **Answer:** False
- **Hint:** Consider what happens to test MAE once a tree becomes deep enough to fit noise.

### Q3. (Open-Ended)
Explain why a decision tree regressor does not require feature scaling the way linear regression's coefficient comparison does.

- **Answer:** A tree only compares a feature's value against a threshold to decide which branch to follow, so the absolute scale of the feature has no effect on how splits are chosen, unlike a linear model's coefficients, whose size reflects the feature's units.
