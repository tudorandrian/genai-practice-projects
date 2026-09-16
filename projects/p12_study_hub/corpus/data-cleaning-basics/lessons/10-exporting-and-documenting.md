# Exporting Clean Datasets and Writing a Data Dictionary

A cleaned Wheel & Way trip log is only useful once other people, and other programs, can
load it without repeating all the work this course covered. The final step is choosing
an export format and writing down, in plain language, exactly what every column means
and what guarantees the cleaning process gives about it.

CSV is the most portable export format - readable by almost any tool - but it loses type
information on the way out: a timestamp column becomes plain text again the moment it is
written to CSV, and whoever loads the file has to know to re-parse it as a date rather
than treating it as a string by accident. Parquet, by contrast, stores each column's
type alongside the data itself, so a downstream reader gets back exactly the timestamp,
integer, or category dtype you cleaned it into, with no re-parsing step and no room for
a mismatched guess.

Parquet is also a columnar format, which means it compresses well and lets a downstream
tool read only the columns it actually needs instead of the whole file - a meaningful
advantage once the trip log grows into the millions of rows. The trade-off is that
Parquet is a binary format: you cannot open it in a plain text editor to eyeball a few
rows the way you can with CSV, so many teams export both, a Parquet file for pipelines
and a small CSV sample for quick human inspection.

Whichever format you choose, a data dictionary should travel with the file: a short
table listing every column's name, its type, a one-sentence description, and any
constraint that validation enforces, such as 'duration_minutes: float, non-negative,
values above 1500 are flagged as outliers rather than removed.' This is the artifact
that lets a colleague, or a future version of you, use the table correctly without re-
reading every line of cleaning code.

The dictionary should also record what the cleaning process changed: how sentinels were
identified and replaced, how duplicates were detected and how many were removed, which
imputation strategy was used for which column, and the schema rules the final table
satisfies. This is not extra bureaucracy - it is the difference between a dataset
someone can build a model or a report on with confidence, and one they have to re-
investigate from scratch before trusting it.

This closes the loop this course opened in its first lesson: data quality is not a
property a dataset either has or lacks in the abstract, but something you establish
through specific, checkable steps, and then communicate clearly enough that the next
person does not have to take your word for it.

## 1. Practice Questions

### Q1. (Multiple Choice)
What is the main advantage of exporting a cleaned dataset to Parquet instead of CSV?

- A) Parquet files can always be opened directly in a plain text editor. - **Incorrect.** Parquet is a binary format and is not directly readable as plain text.
- B) Parquet preserves each column's data type, so a downstream reader does not need to re-parse timestamps or numbers from text. - **Correct answer.** Preserving type information without re-parsing is exactly Parquet's key advantage here.
- C) CSV files cannot be read by any tool other than the one that wrote them. - **Incorrect.** CSV is in fact one of the most broadly portable formats available.
- D) Parquet eliminates the need for any data dictionary. - **Incorrect.** A data dictionary remains valuable regardless of file format.

- **Answer:** B

### Q2. (True/False)
A data dictionary should record not only what each column means, but also what the cleaning process changed about it.

- **Answer:** True
- **Hint:** Consider what a colleague would need to trust the table without redoing your work.

### Q3. (Open-Ended)
Explain one trade-off between exporting a cleaned dataset as CSV versus as Parquet.

- **Answer:** CSV is broadly portable and human-readable in a text editor but loses type information, forcing every reader to re-parse dates and numbers, while Parquet preserves types and compresses well but cannot be opened as plain text for a quick manual look.
