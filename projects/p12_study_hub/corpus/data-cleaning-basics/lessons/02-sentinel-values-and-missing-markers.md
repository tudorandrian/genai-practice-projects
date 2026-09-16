# Sentinel Values and Missing-Data Markers

Not every gap in a dataset shows up as a proper empty cell. Wheel & Way's trip log is
exported from an older ticketing system that was never designed with analysis in mind,
and that system has its own way of saying 'we don't know': a duration of -1 minutes, a
station name of the literal text 'NA', an end time equal to '1970-01-01 00:00:00', and
occasionally a rider age of 999. Every one of these is a sentinel value: a normal-
looking entry that was chosen, by whoever built the export, to stand in for missing
information.

The danger of sentinels is that they pass right through the checks that catch a true
empty cell. A pandas `isna()` call will not flag a duration of -1, because -1 is a
perfectly good integer as far as the dataframe is concerned. If you compute an average
duration without first replacing sentinels with proper missing markers, that -1 pulls
the mean down and nobody sees a warning about it.

The fix starts with discovery, not correction. For every column, look at its extreme
values and its most frequent values side by side: `value_counts()` for text,
`describe()` and a histogram for numbers. A duration column where -1 appears 340 times
and every other value is positive is a strong signal that -1 is not a real duration; it
is a marker somebody chose arbitrarily, most likely because a negative number could
never occur naturally in that column and stood out to whoever designed the export.

Once you have identified the candidate sentinels, replace them explicitly with a real
missing marker (`numpy.nan` in pandas) rather than leaving them as ordinary numbers or
strings. This single step is what lets every later missing-value tool - `isna()`,
`dropna()`, `fillna()` - actually see the gap. Skipping this step is the single most
common reason a supposedly 'clean' dataset still contains distorted averages.

Sentinel conventions are rarely documented anywhere, so you should expect to find them
by pattern rather than by reading a specification. A repeated, suspiciously round or
suspiciously extreme value concentrated in one column - -1, 0 where zero is impossible,
9999, or an empty string masquerading as text - is worth a closer look before you accept
it as genuine data.

It is also worth recording what you found. A short note next to your cleaning code -
'duration == -1 means missing, 340 rows affected' - turns a silent judgement call into
something a reviewer, or you in six months, can verify and reproduce.

## 1. Practice Questions

### Q1. (Multiple Choice)
Why does a sentinel value such as duration = -1 often slip past a naive missing-data check?

- A) Because pandas automatically converts -1 to `NaN` on load. — **Incorrect.** Pandas does no such automatic conversion; -1 stays exactly as loaded.
- B) Because it is stored as an ordinary, valid-looking number, so functions like `isna()` do not recognize it as missing. — **Correct answer.** Sentinels look structurally valid, which is precisely what lets them pass unnoticed.
- C) Because negative numbers cannot be stored in a dataframe column. — **Incorrect.** Dataframe numeric columns store negative numbers without any restriction.
- D) Because sentinel values are always documented in a data dictionary before use. — **Incorrect.** Sentinel conventions are typically undocumented, which is why you must detect them by pattern.

- **Answer:** B

### Q2. (True/False)
Replacing a discovered sentinel value with `numpy.nan` is what allows standard missing-value tools such as `isna()` and `fillna()` to treat it as missing.

- **Answer:** True
- **Hint:** Think about what those functions actually check for under the hood.

### Q3. (Open-Ended)
Describe a concrete way you could decide whether a repeated value of 9999 in a rider-age column is a genuine age or a sentinel for missing data.

- **Answer:** Compare its frequency and plausibility against the rest of the distribution: if 9999 appears far more often than any realistic age and every other value falls in a sensible human range, it is very likely a sentinel rather than a true measurement.
