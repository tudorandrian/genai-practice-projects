# String Cleaning and Categorical Normalization

Free-text fields in the Wheel & Way trip log are where inconsistency accumulates
fastest. The same physical station, 'Riverside Park', shows up in the raw export as
'Riverside Park', 'riverside park', 'Riverside  Park' with a double space, and even
'Riverside Prk' from an old firmware version with a typo baked into its station list. To
a computer, each of those is a different string, and therefore a different station,
unless you normalize them first.

A basic normalization pass handles a large share of the problem: strip leading and
trailing whitespace, collapse repeated internal whitespace to a single space, and
standardize letter case, typically to title case for display or lower case for matching.
These operations are cheap, safe, and should be applied to essentially every free-text
column before you do anything else with it.

Whitespace and case differences are the easy cases; genuine misspellings like 'Riverside
Prk' need a different tool. A small, hand-built mapping from known bad values to their
correct form is often the most reliable approach for a fixed, known set of stations: you
look at the distinct values in the column, spot the ones that clearly refer to a real
station under a typo, and add them to a lookup dictionary the cleaning code applies
explicitly.

For larger or less predictable vocabularies, fuzzy string matching - comparing how
similar two strings are, for example with Levenshtein distance - can suggest candidate
corrections automatically, such as noticing that 'Riverside Prk' is very close to the
known station 'Riverside Park'. Fuzzy matching is a good way to generate suggestions,
but it should not be trusted to apply corrections unreviewed: two genuinely different
station names can also be close to each other, and an automatic fix could merge them by
mistake.

Once station names are normalized, converting the column to a `category` dtype with an
explicit, known list of valid categories turns normalization into an ongoing safeguard
rather than a one-time fix: any future value that does not match one of the known
categories will stand out immediately instead of quietly adding a seventh, misspelled
station to a report that should only ever show six.

The same principles apply beyond station names - rider notes, bike model codes, and
payment method labels all benefit from the same sequence: normalize whitespace and case,
correct known misspellings explicitly, and lock the column down to a known set of valid
values once you are confident you have found them all.

## 1. Practice Questions

### Q1. (Multiple Choice)
Why is applying fuzzy string matching to auto-correct station names risky if done without any human review?

- A) Fuzzy matching cannot run on text data at all. - **Incorrect.** Fuzzy matching is specifically designed to compare text strings.
- B) It always runs slower than exact string matching. - **Incorrect.** Speed is a real consideration but not the risk the lesson highlights.
- C) Two genuinely different station names can be close enough to each other that an automatic correction could incorrectly merge them. - **Correct answer.** This risk of merging distinct real values is exactly why review matters.
- D) It can only compare strings of exactly equal length. - **Incorrect.** Levenshtein-style distance measures work on strings of different lengths.

- **Answer:** C

### Q2. (True/False)
Stripping whitespace and standardizing letter case should generally be done before attempting to fix specific misspellings in a text column.

- **Answer:** True
- **Hint:** Cheap, broad fixes first reduce the number of distinct values you still have to inspect.

### Q3. (Open-Ended)
Explain how converting a normalized station column to a `category` dtype with an explicit list of valid values helps catch future data quality problems.

- **Answer:** Any new value that does not match the known category list becomes immediately visible rather than blending in as a new, unnoticed variant, turning normalization from a one-time cleanup into an ongoing check on every new batch of data.
