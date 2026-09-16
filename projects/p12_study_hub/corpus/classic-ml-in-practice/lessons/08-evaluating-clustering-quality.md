# Evaluating Clustering Quality

Choosing `k` for the Wheel & Way station clusters should not be a guess. Two
complementary tools help make that choice more systematic: the elbow method, based on
how tightly clusters group their members, and the silhouette score, based on how well
separated different clusters are from each other.

The elbow method plots k-means' inertia - the sum of squared distances from each station
to its assigned cluster center - against increasing values of `k`. Inertia always
decreases as `k` grows, since more clusters can always fit the data at least as tightly,
but the rate of decrease typically slows sharply after some point, forming a bend, or
'elbow', in the plot; the `k` at that bend is a reasonable candidate, since adding more
clusters beyond it buys comparatively little improvement in fit.

The silhouette score, computed with `sklearn.metrics.silhouette_score`, measures for
each point how much closer it is to its own cluster than to the nearest other cluster,
averaged over every point, on a scale from -1 to 1. A score near 1 means clusters are
well separated and each point clearly belongs where it was assigned; a score near 0
means clusters overlap substantially; a negative score suggests points may have been
assigned to the wrong cluster entirely.

With only six stations, both diagnostics should be read as guidance rather than as a
strict rule to follow blindly - a dataset this small can produce a noisy elbow plot or a
silhouette score that shifts noticeably with small changes to the features used. Cross-
checking the diagnostics against a direct look at which stations land in which cluster,
and whether that grouping makes business sense, matters as much as the numeric scores
themselves.

It is also worth trying the same clustering with a different, smaller set of features -
say, just trips per day and member share, dropping duration - to see whether the
resulting clusters stay roughly stable. A clustering that changes dramatically with
small changes to the input features is telling you the underlying grouping is not very
robust, which is itself useful information before recommending it to the operations team
as a basis for decisions.

Unlike the regression metrics from earlier lessons, there is no single held-out test set
to fall back on here, since clustering has no ground-truth labels to check predictions
against; the elbow and silhouette diagnostics, combined with a sanity check against
domain knowledge, are the closest equivalent this course covers for unsupervised
learning.

## 1. Practice Questions

### Q1. (Multiple Choice)
What does a silhouette score near 1 indicate about a clustering result?

- A) Every cluster contains exactly the same number of points. - **Incorrect.** Silhouette score reflects separation quality, not equal cluster sizes.
- B) Points are, on average, much closer to their own cluster than to the nearest other cluster, indicating well-separated clusters. - **Correct answer.** This closeness-to-own-cluster relative to other clusters is exactly what a high silhouette score reflects.
- C) The clustering used the maximum possible value of `k`. - **Incorrect.** Silhouette score does not track or require the maximum value of `k`.
- D) Inertia is guaranteed to be zero. - **Incorrect.** A high silhouette score does not imply zero inertia; the two metrics measure different things.

- **Answer:** B

### Q2. (True/False)
Inertia always decreases as `k` increases, which is why the elbow method looks for where that decrease slows down rather than for its lowest value.

- **Answer:** True
- **Hint:** Consider what happens to inertia in the extreme case where `k` equals the number of points.

### Q3. (Open-Ended)
Explain why re-running the clustering with a smaller set of features and checking whether the groupings stay stable is a useful sanity check.

- **Answer:** If the clustering changes substantially when a feature is dropped, that suggests the original grouping was sensitive to that specific choice of features rather than reflecting a robust, underlying pattern in station usage, which matters before recommending the clustering as a basis for operational decisions.
