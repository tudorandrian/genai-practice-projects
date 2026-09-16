# Data Types and Type Coercion

Every column in the Wheel & Way trip log has an intended type: `trip_id` should be an
integer, `start_time` a timestamp, `duration_minutes` a float, `rider_type` a small set
of category strings. When the export tool writes a CSV file, all of that structure is
flattened into plain text, and it is up to whoever loads the file to put the types back.

Left to its own defaults, a CSV loader often guesses wrong in quiet ways. A
`duration_minutes` column that contains one stray value like 'unknown' will force the
entire column to load as text (an 'object' column in pandas terms) instead of a float,
because a column can only have one dtype and text is the only type that can hold every
value. Once that happens, every arithmetic operation on that column either raises an
error or, worse, silently does the wrong thing depending on how it is later coerced.

The practical workflow is to check `dtypes` immediately after loading, before doing
anything else, and compare it against what you expect for each column. A `trip_id`
loaded as text instead of an integer, or a `start_time` loaded as plain text instead of
a timestamp, both mean something upstream does not match your assumptions and needs
attention before you proceed.

Converting a column's type deliberately - with `pandas.to_numeric`,
`pandas.to_datetime`, or `astype`, each given an explicit `errors` argument - is far
safer than letting a loader guess. `errors='coerce'` turns anything it cannot parse into
a missing value instead of raising or, worse, quietly truncating; counting how many
values became missing after a coercion tells you exactly how much of the column did not
match your expected format.

Categorical columns deserve their own type as well. `rider_type` only ever holds
'member' or 'casual' in principle, so storing it as pandas' `category` dtype instead of
generic text saves memory on a large table and, more importantly, makes it easy to spot
an unexpected third value - a typo or a new category nobody told you about - the moment
you list the categories.

Getting types right early pays off throughout the rest of the pipeline: outlier
detection, imputation, and time-based features all assume the columns they touch are
already numeric or already timestamps, and every one of those steps becomes both more
fragile and harder to debug once you start doing type conversions inside them instead of
once, up front.

## 1. Practice Questions

### Q1. (Multiple Choice)
A `duration_minutes` column loads as an 'object' (text) dtype instead of a float. What is the most likely explanation?

- A) At least one value in the column cannot be parsed as a number, forcing pandas to store the whole column as text. — **Correct answer.** A single non-numeric value is enough to make pandas fall back to a text dtype for the whole column.
- B) The column has too many rows for pandas to store as a float. — **Incorrect.** Row count has no bearing on which dtype a column receives.
- C) Float columns are always loaded as text by default in pandas. — **Incorrect.** Pandas infers numeric dtypes by default when every value parses cleanly.
- D) The CSV file must be corrupted beyond use. — **Incorrect.** One unparsable value does not mean the file is corrupted; it usually just needs cleaning.

- **Answer:** A

### Q2. (True/False)
Using `errors='coerce'` when converting a column's type turns unparsable values into missing values instead of raising an exception.

- **Answer:** True
- **Hint:** Consider what alternative behaviours `errors` could control besides raising.

### Q3. (Open-Ended)
Explain why storing `rider_type` as pandas' `category` dtype rather than plain text can help you catch a data quality problem you might otherwise miss.

- **Answer:** Listing a category column's categories immediately surfaces every distinct value it actually contains, so an unexpected third value such as a typo or a new rider type shows up directly instead of being buried among thousands of repeated text values.
