# Detecting and Removing Duplicate Trips

A duplicate row in the Wheel & Way trip log usually means one physical bike trip got
counted twice: a retry in the ticketing system after a network hiccup, an export job
that ran twice and appended both times, or a rider whose card was tapped in and briefly
registered as two separate short trips. Every one of these inflates ridership counts and
distorts anything computed per trip, such as average duration or revenue per ride.

The most obvious check, `duplicated()` on every column, only catches exact duplicates:
rows that agree on every single field, including the trip id. That is a useful first
pass, but it misses the more common case at Wheel & Way, where a retried request gets a
new trip id but the same rider, the same station, and start and end times only a few
seconds apart.

Catching those near-duplicates means choosing a smaller set of columns that should
uniquely identify one real trip - rider id, start station, and start time rounded to the
nearest minute is a reasonable choice for this dataset - and looking for repeats on that
subset instead of on every column. `duplicated(subset=[...], keep=False)` returns every
row involved in such a group, which lets you inspect the candidates before deciding what
to do with them, rather than deleting blindly.

Deciding what to keep from a duplicate group is a judgement call that should be made
explicitly rather than left to whatever order the rows happen to be in. A reasonable
default for retried requests is to keep the first occurrence and drop the rest, on the
assumption that the first successful write reflects the real trip; but if two
'duplicate' rows actually differ in a field like duration, that is a signal they might
be two genuine trips rather than one trip recorded twice, and deserve a second look
rather than an automatic drop.

It helps to keep count of how many rows a deduplication step removes and to sanity-check
that number against what you would expect. If 40 out of 50,000 trips get removed as
duplicates, that is plausible for a system with occasional retries; if 12,000 get
removed, your identifying subset of columns is probably too loose and is matching
genuinely different trips against each other.

Deduplication should happen early in the cleaning pipeline, before outlier detection or
imputation, because a duplicated row with an unusual duration would otherwise be counted
twice when you look at the distribution of durations, exaggerating how common that
unusual value actually is.

## 1. Practice Questions

### Q1. (Multiple Choice)
Why might checking for duplicates only on rows that match on every single column miss real duplicate trips in the Wheel & Way log?

- A) Because `duplicated()` cannot be applied to a whole dataframe. - **Incorrect.** `duplicated()` works perfectly well across an entire dataframe.
- B) Because exact duplicates never occur in operational data. - **Incorrect.** Exact duplicates can and do occur, for instance from a repeated export job.
- C) Because trip ids are always identical for the same physical trip. - **Incorrect.** A retried request commonly receives a brand-new trip id, which is exactly the problem.
- D) Because a retried request often gets a new trip id, so two rows for the same physical trip may differ on that one column while agreeing on everything else that matters. - **Correct answer.** This mismatched trip id is precisely why an all-columns check misses this case.

- **Answer:** D

### Q2. (True/False)
Using `duplicated(subset=[...], keep=False)` returns every row that participates in a duplicate group, not just the extra copies.

- **Answer:** True
- **Hint:** `keep=False` marks all occurrences, not just the ones after the first.

### Q3. (Open-Ended)
Suggest one sanity check you could run after deduplicating the trip log to confirm the chosen subset of identifying columns was neither too strict nor too loose.

- **Answer:** Compare the number of rows removed against a plausible expectation, such as a small single-digit percentage consistent with occasional retries; a much larger drop suggests the identifying columns are matching genuinely distinct trips together.
