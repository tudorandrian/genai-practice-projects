# P02 - Question Bank

## What it does

A dataset-agnostic question-bank builder: it scans a folder of Markdown
lesson notes, finds every file that contains a "Practice Questions" section,
parses each Multiple Choice / Multi-Select / True-False / Open-Ended block,
and assembles the result into one deterministic JSON question bank plus a
plain-text validation summary.

```
discover_lessons -> parse_questions -> assemble -> write_bank
      (I/O)             (pure)          (pure)       (I/O)
```

Discovery is layout-agnostic: any `*.md` file under the given folder that
has a `## N. Practice Questions` header or at least one `### Qn. (Type)`
question header is treated as a question file, regardless of its file name.
Files with neither (project overviews, module summaries) are skipped
silently. Unknown or out-of-scope question types (`Matching`,
`Fill-in-the-Blank`, ...) are reported with a precise
`Qn: unsupported type '<label>'` error instead of silently vanishing or
corrupting the neighbouring question.

`qbank.build_bank(course_dir, course_id=None) -> (bank, errors)` is the pure
half of the pipeline: it does no I/O beyond reading the lesson files and
never writes anything. It is the entry point another project (a quiz
front-end) is expected to call directly - `run()` is a thin CLI wrapper
around it that adds validation gating and the two output files.

## Run

```bash
uv run p02-question-bank --demo                        # shipped fixtures
uv run p02-question-bank path/to/lessons
uv run p02-question-bank path/to/lessons --check-only   # validate, write nothing
uv run p02-question-bank path/to/lessons --course-id my-course --output out/bank.json
```

`--verbose` logs each lesson's parsed counts at `INFO`; by default only the
summary prints, and any validation errors are logged at `ERROR` level
(visible on stderr regardless of `--verbose`). On success the exit code is
`0`; if any lesson fails validation, or no question files are found, nothing
is written and the exit code is `1`.

### CLI flags

| Flag | Meaning | Default |
|------|---------|---------|
| `course_dir` | Folder to scan for Markdown lessons | required unless `--demo` |
| `--output PATH` | Output JSON path | `output/question-bank.json` |
| `--check-only` | Validate only; write nothing | off |
| `--course-id ID` | Course id used to build question ids | the folder name |
| `--demo` | Run the offline demo on the shipped fixtures | - |
| `--verbose` | Log per-lesson counts at INFO level | off |

## Example output

`uv run p02-question-bank --demo`:

```
p02-question-bank: ok
  lessons: 2
  questions: 4
  wrote: output/
  seconds: 0.01
```

(the `seconds:` line varies run to run; it is not part of any committed
proof.)

`output/summary.txt` (the committed proof, deterministic, no absolute
paths or timestamps):

```
QBANK VALIDATION SUMMARY
========================================
  - e00_valid_baseline.md: 1 MC, 1 T/F, 1 OE
  - e15_multi_select_valid.md: 1 MC, 0 T/F, 0 OE
```

`output/metrics.txt` (figures only, also committed):

```
lessons=2
questions=4
```

## Design notes

- **Parse-partial, refuse-total.** A broken question is still *recovered*
  into the parsed list (so per-file counts are honest about what was seen),
  but the presence of any error anywhere blocks the write: either every
  lesson validates, or nothing is written.
- **Generic header matching.** `### Qn. (Type)` is matched on any label, so
  an unrecognized type is still a clean block boundary - it cannot pollute
  the questions around it, and it is reported by name instead of surfacing
  as a generic parse failure or a misleading "numbering is not consecutive."
- **Multi-Select as a Multiple-Choice variant.** A `(Multi-Select)` header
  reuses the `multiple_choice` output type but allows more than one option
  marked `**Correct answer.**`, as long as the trailing `**Answer:**` line
  lists the same letters (comma-separated). A `(Multiple Choice)` header
  keeps the single-answer constraint.
- **Layout-agnostic identity.** `course_id_from_dir` is just the folder's own
  name, and `unit_id_from_filename` is always `f"{course_id}/{path.stem}"` -
  never `None` - so the tool works on any folder structure or file naming
  convention, not one baked-in scheme.
- **Deterministic output.** No timestamps or run-order dependence anywhere in
  the bank; two runs over the same input are byte-identical.
- **Output contract.** Library functions never `print`; they log through
  `logging.getLogger(__name__)`. `main()` prints at most six summary lines
  and sets the logging level (`WARNING`, or `INFO` with `--verbose`);
  validation errors are logged at `ERROR` so they still reach stderr by
  default.

## Limits

- **Single-line bullets only.** A rationale or hint that wraps onto a second
  line is not recognized as part of the same option (`e12` fixture); the
  continuation line is reported as an unrecognized bullet.
- **A strict option syntax.** Options need a hyphen with a space on each side
  between the option text and its `**Correct answer.**\|**Incorrect.**` marker
  (an en dash or em dash is accepted too); without it the whole option line is
  unrecognized (`e11` fixture).
- **The schema fixes the three question types.** `question_counts` always has
  exactly `multiple_choice`, `true_false`, `open_ended` keys; Multi-Select
  questions are counted under `multiple_choice`.

## Datasets and licences

No external dataset is used. `fixtures/e00…e16_*.md` are seventeen small,
synthetic Markdown files written for this repository to exercise one control
case and sixteen different or missing structures (unknown types, malformed
options, non-consecutive numbering, a valid Multi-Select block, and so on).
None of them reproduce real course content.

## Courses drawn on

- 4 Python for Data Science, AI & Development
