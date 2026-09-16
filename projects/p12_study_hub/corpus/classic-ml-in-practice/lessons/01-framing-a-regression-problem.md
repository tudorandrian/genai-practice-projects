# Framing a Regression Problem: Predicting Daily Rides

Wheel & Way's operations team wants to know, each morning, roughly how many rides the
network will see that day, so they can decide how many bikes to move to which stations
overnight. This course picks up where the cleaning course left off: the trip log from
that course, aggregated to one row per day, becomes the training data for a small
collection of classic machine learning models built across these ten lessons.

Framing the problem correctly matters more than any single model choice. Predicting
'daily rides' could mean predicting a raw count, which is what operations actually cares
about, or it could mean predicting rides per active station, which normalizes away the
effect of the network slowly growing new stations over time. This course predicts the
raw daily count, since that is the number operations plans around, but the distinction
is worth stating explicitly before writing a single line of modelling code.

A supervised regression problem needs a target column and a set of features believed to
explain it. The target here is `rides` (a positive integer per day); candidate features
include the day of week, whether it is a public holiday, the average temperature that
day, and whether it rained. Some of these will turn out to matter far more than others,
and part of the job of the next several lessons is figuring out which.

It is worth being honest about what a model like this can and cannot do. A regression
model trained on a year of daily counts can pick up strong, repeated patterns - weekdays
differ from weekends, rain suppresses ridership - but it has no way to anticipate a one-
off event such as a marathon closing half the city's streets, unless that kind of event
is explicitly encoded as a feature. Treating a model's predictions as certainties rather
than estimates with error bars is one of the most common mistakes in applied machine
learning.

Before fitting anything, it helps to plot the target over time and against each
candidate feature. A scatter plot of rides against temperature that shows a rough upward
trend, or a box plot of rides by day of week that shows weekdays clearly higher than
weekends, builds intuition for what a model should find - and gives you something to
check its results against later, rather than trusting a number just because a model
produced it.

The rest of this course builds up from a simple baseline to more capable models, always
asking the same question at each step: does this added complexity actually improve
predictions on data the model has not seen, or does it only look better on the data it
was trained on?

## 1. Practice Questions

### Q1. (Multiple Choice)
Why does this course choose to predict the raw daily ride count rather than rides per active station?

- A) Because the raw count is the number Wheel & Way's operations team actually plans around when deciding how many bikes to move. — **Correct answer.** Matching the target to what the business actually uses is exactly the stated reason.
- B) Because rides per active station cannot be computed from the available data. — **Incorrect.** It could be computed; the lesson simply chooses not to use it as the target.
- C) Because regression models cannot predict integer counts. — **Incorrect.** Regression models routinely predict count-like targets; nothing about the model forbids it.
- D) Because temperature and rainfall are not available as features. — **Incorrect.** Those features are explicitly listed as candidates for the model.

- **Answer:** A

### Q2. (True/False)
A regression model trained on daily ride counts can automatically anticipate a one-off event, like a marathon closing streets, without that event being encoded as a feature.

- **Answer:** False
- **Hint:** Consider what information the model has actually seen during training.

### Q3. (Open-Ended)
Explain why plotting the target against a candidate feature before modelling is a useful step, even though the model itself will eventually estimate that relationship.

- **Answer:** It builds intuition for what a reasonable relationship should look like and gives you a check to compare the model's learned relationship against, so an obviously wrong or nonsensical model output is easier to catch rather than accepted at face value.
