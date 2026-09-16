# Parsing Timestamps and Extracting Time Features

The Wheel & Way trip log stores `start_time` and `end_time` as plain text strings, and
not always in the same format: most rows use 'YYYY-MM-DD HH:MM:SS', but a batch exported
during a system migration uses 'MM/DD/YYYY HH:MM' instead, with no seconds field at all.
Treating both formats as one text column and comparing them directly would produce
nonsense, since string comparison has nothing to do with chronological order once the
formats differ.

`pandas.to_datetime` can often infer a format automatically, but relying on automatic
inference across a column with two different formats is exactly the situation where it
is safest to be explicit. Splitting the column by which format it matches, parsing each
batch with its own explicit `format` string, and then concatenating the results back
into one proper timestamp column avoids the ambiguity entirely and fails loudly, via
`errors='raise'`, if a row does not match either format.

With both columns parsed as real timestamps, `end_time - start_time` gives an actual
`Timedelta`, which is what should be used to recompute `duration_minutes` rather than
trusting a duration value that arrived as raw text and might already contain one of the
unit-mixing errors from an earlier lesson. Recomputing duration from parsed timestamps
and comparing it against the original column is a strong, independent check on how much
of that original column can be trusted.

Once timestamps are proper datetime objects, useful features fall out almost for free:
the hour of day a trip started, the day of the week, and whether it fell on a weekend.
These features matter operationally at Wheel & Way, since rebalancing bikes between
stations depends heavily on commute-hour demand patterns that only become visible once
you can group trips by hour and weekday rather than treating every trip as an
undifferentiated row.

Time zones deserve explicit attention too. If some historical exports recorded local
Bramford time and a newer system started recording UTC after a migration, an hour-of-day
feature computed carelessly across both eras will be systematically wrong for one of
them. The safe habit is to convert every timestamp to one consistent time zone as an
explicit step, rather than assuming the whole column already shares one.

As with every other cleaning step in this course, timestamp parsing should report how
many rows failed to parse under the expected formats, so that a genuinely malformed row
gets flagged and investigated rather than silently dropped or, worse, coerced into a
timestamp that happens to parse but is chronologically meaningless.

## 1. Practice Questions

### Q1. (Multiple Choice)
Why is it risky to rely purely on automatic format inference when a timestamp column mixes two different date formats?

- A) Automatic inference is always slower than explicit parsing. - **Incorrect.** Speed is not the central risk being described here.
- B) Pandas cannot parse dates automatically under any circumstance. - **Incorrect.** Pandas can and often does infer formats automatically; the risk is ambiguity, not incapability.
- C) Mixed formats always cause an immediate program crash. - **Incorrect.** Mixed formats more often cause silent misparsing than an immediate crash.
- D) Automatic inference can misparse rows silently when the format is ambiguous across the mixed batches, rather than failing loudly. - **Correct answer.** This silent misparsing under ambiguity is exactly why explicit, per-batch formats are safer.

- **Answer:** D

### Q2. (True/False)
Recomputing duration as `end_time - start_time` from properly parsed timestamps provides an independent check on a duration column that arrived as raw text.

- **Answer:** True
- **Hint:** Comparing an independently derived value against the original is a classic validation technique.

### Q3. (Open-Ended)
Explain why mixing local time and UTC timestamps in the same column without converting them could distort an hour-of-day demand analysis.

- **Answer:** Trips recorded in different time zones but treated as if they shared one clock will be shifted relative to each other by the zone offset, so grouping by hour of day would mix genuinely different local hours together and misrepresent when demand actually peaks.
