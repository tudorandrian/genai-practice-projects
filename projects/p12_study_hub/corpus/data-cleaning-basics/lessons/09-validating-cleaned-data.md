# Validating Cleaned Data with Schema Checks

After sentinels are handled, types are fixed, duplicates are removed, outliers are
flagged, gaps are imputed, text is normalized, and timestamps are parsed, it is tempting
to call the Wheel & Way trip log clean and move on. The last step this course covers is
the one that turns that feeling into something checkable: writing down, explicitly, what
a valid row looks like, and testing every row against that description.

A schema for this table is a short, explicit list of expectations: `trip_id` is a
positive integer and unique; `start_time` and `end_time` are valid timestamps with
`end_time` no earlier than `start_time`; `duration_minutes` is a non-negative float,
typically under a generous ceiling like 1,500 minutes unless flagged as an outlier;
`rider_type` is one of exactly 'member' or 'casual'; and `start_station`/`end_station`
each belong to the known set of six stations. None of these rules is complicated on its
own, but together they cover most of the problems the earlier lessons addressed one at a
time.

Writing these expectations as plain Python assertions - `assert (df['duration_minutes']
>= 0).all()`, for instance - is a lightweight way to start, and it fails immediately and
visibly if a rule is broken, which is exactly the behaviour you want from a validation
step: a silent pass on bad data is far more dangerous than a loud failure that forces
someone to look.

For anything beyond a quick script, a schema library that checks a whole dataframe at
once and reports every violation - not just the first one it encounters - is more useful
in practice, since a single validation run can then tell you 'duration has 3 negative
values, station has 2 unknown values' instead of stopping at the very first problem and
leaving the rest undiscovered until the next run.

Validation belongs at the boundary between cleaning and everything downstream: run it
once, right before the cleaned table is written out or handed to another team, so that
every consumer of the file can trust the same set of guarantees without re-deriving them
independently. Running validation earlier, in the middle of cleaning, is also useful as
a sanity check, but the final pre-export check is the one that matters most for trust.

A validation step that never fails is a little suspicious rather than fully reassuring:
either the rules are too loose to catch anything, or the earlier cleaning steps were
unusually thorough. Periodically loosening a rule on purpose, on a scratch copy of the
data, to confirm it does actually catch a planted problem is a good way to check that
your checks are doing real work rather than passing by default.

## 1. Practice Questions

### Q1. (Multiple Choice)
Why is a validation step that raises a loud, visible failure on bad data preferable to one that silently lets bad rows through?

- A) A loud failure forces someone to investigate immediately, whereas a silent pass lets a data quality problem reach downstream reports unnoticed. — **Correct answer.** Forcing visible investigation is precisely the value of a strict, loud validation step.
- B) Loud failures always run faster than silent checks. — **Incorrect.** Execution speed is unrelated to whether a check fails loudly or silently.
- C) Silent validation is not technically possible in Python. — **Incorrect.** Silent validation is entirely possible; it simply fails to alert anyone to a problem.
- D) A loud failure guarantees the underlying data problem is fixed automatically. — **Incorrect.** A loud failure only signals a problem; fixing it still requires a deliberate follow-up step.

- **Answer:** A

### Q2. (True/False)
Running a schema validation step that checks every rule and reports every violation is generally more useful than one that stops at the first violation it finds.

- **Answer:** True
- **Hint:** Consider how much a single validation run can tell you about the true scope of a problem.

### Q3. (Open-Ended)
Explain why deliberately loosening a validation rule on a scratch copy of the data, to see whether it catches a planted problem, is a useful exercise.

- **Answer:** It confirms the check is actually doing meaningful work rather than passing only because the data happens to be clean, giving you evidence that the rule would catch a real problem if one occurred instead of quietly letting it through.
