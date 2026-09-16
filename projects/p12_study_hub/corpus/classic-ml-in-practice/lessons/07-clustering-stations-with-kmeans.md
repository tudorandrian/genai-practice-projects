# Clustering Stations with K-Means

Every model so far has predicted a number - daily rides - from labeled examples. This
lesson switches to unsupervised learning: grouping Wheel & Way's six stations into
clusters of similar usage pattern, with no target column to predict at all, only
features describing each station's typical day.

For each station, a small feature table can be built from the trip log: average trips
started per day, the fraction of those trips taken by members versus casual riders, and
the average trip duration originating from that station. K-means clustering
(`sklearn.cluster.KMeans`) groups stations by finding `k` cluster centers and assigning
each station to whichever center is closest in this feature space, then updating the
centers to the average of their assigned stations and repeating until the assignments
stop changing.

Because k-means measures closeness with ordinary distance, features must be scaled
before clustering, exactly as flagged for linear regression's coefficients: without
scaling, a feature measured in the hundreds (trips per day) would dominate the distance
calculation over a feature measured in fractions (member share), regardless of which one
actually distinguishes the stations better.

The number of clusters, `k`, is a choice the analyst makes rather than something the
algorithm discovers on its own. Too few clusters lump genuinely different stations
together; too many split one real pattern into arbitrary sub-groups that do not
correspond to anything meaningful about the business. For six stations, a small `k` such
as two or three is the sensible range to start exploring.

Running k-means with a fixed `random_state` makes the clustering reproducible: k-means
initializes its cluster centers randomly, and different random starting points can
converge to different final clusterings, particularly with a small number of stations
where there is less data to smooth out that randomness.

The value of clustering here is not a prediction but a description: it might reveal that
'Bramford Central' and 'College Quad' behave like one cluster - high volume, short
commute-length trips - while 'Riverside Park' and 'Harbor Point' behave like another -
lower volume, longer leisure trips - a grouping the operations team can use to tailor
rebalancing strategy differently for each type of station rather than applying one rule
to all six.

## 1. Practice Questions

### Q1. (Multiple Choice)
Why must station features be scaled before applying k-means clustering?

- A) K-means cannot process more than three features at once. - **Incorrect.** K-means can handle any number of numeric features; there is no such three-feature limit.
- B) Scaling is required only for supervised learning, never for clustering. - **Incorrect.** Scaling matters for clustering precisely because it is a distance-based method, as much as for many supervised methods.
- C) Unscaled features always produce exactly one cluster. - **Incorrect.** Unscaled features can still produce multiple clusters; the issue is which feature dominates the distance calculation.
- D) K-means groups points by distance, so a feature measured on a much larger scale than the others would dominate that distance regardless of its true importance. - **Correct answer.** This scale-domination of the distance calculation is exactly why scaling matters here.

- **Answer:** D

### Q2. (True/False)
Setting a fixed `random_state` when running k-means makes the resulting clustering reproducible across runs.

- **Answer:** True
- **Hint:** Recall how k-means initializes its cluster centers.

### Q3. (Open-Ended)
Explain why choosing the number of clusters `k` is a judgement call rather than something k-means determines automatically.

- **Answer:** K-means requires `k` as an input and will always produce exactly that many clusters regardless of whether that number genuinely matches the structure in the data, so the analyst must choose `k` using judgement, domain knowledge, or a separate diagnostic, rather than the algorithm discovering the right number on its own.
