"""qbank.py — build a queryable JSON question bank from lesson Markdown notes.

A didactic question-bank builder covering three question types written in a
"Practice Questions" section of a lesson: Multiple Choice, True/False and
Open-Ended, plus a Multi-Select variant of Multiple Choice that allows more
than one correct option.

PIPELINE  (pure functions between two I/O ends)
    discover_lessons ─▶ parse_questions ─▶ assemble ─▶ write_bank
        (I/O)               (pure)          (pure)        (I/O)

Discovery is layout-agnostic: any Markdown file under the given course
directory that contains a Practice Questions section (or at least one
question header) is treated as a question file, regardless of its file
name. Unknown or out-of-scope question types (``Matching``,
``Fill-in-the-Blank``, ...) are reported with a clear
``Qn: unsupported type '<label>'`` error instead of silently vanishing or
polluting the neighbouring question.

HOW TO RUN
    python qbank.py <course_dir> [--output PATH] [--check-only] [--course-id ID]
    python qbank.py <course_dir> --check-only

DEPENDENCIES  none — standard library only (re, json, pathlib, argparse).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

from shared.demo import DemoResult

log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
OUT_DIR = HERE / "output"
SCHEMA_PATH = HERE / "schema" / "question-bank.schema.json"

Question = dict[str, Any]
Bank = dict[str, Any]

# ---------------------------------------------------------------------------
# Regex contract — the "Practice Questions" format this parser understands.
# ---------------------------------------------------------------------------

# `## 3. Practice Questions`
RE_PQ_SECTION = re.compile(r"^## (\d+)\.\s+Practice Questions\s*$", re.MULTILINE)

# `### Q12. (Any Type Label)` — matched GENERICALLY on the label, so an
# unsupported type is still a block boundary (it cannot pollute its neighbour)
# and can be reported with a precise "unsupported type" error.
RE_Q_HEADER = re.compile(r"^###\s+Q(\d+)\.\s+\(([^)]+)\)\s*$", re.MULTILINE)

# `- A) Option text — **Correct answer.** Rationale.`  (note the em-dash `—`)
RE_MC_OPTION = re.compile(
    r"^- ([A-Z])\)\s+(.+?)\s+—\s+\*\*(Correct answer|Incorrect)[^*]*\*\*\s*(.*)$"
)
# `- **Answer:** B` or `- **Answer:** A, C` (Multi-Select: comma-separated letters)
RE_MC_ANSWER = re.compile(r"^- \*\*Answer:\*\*\s+([A-Z](?:\s*,\s*[A-Z])*)\s*$")
RE_TF_ANSWER = re.compile(r"^- \*\*Answer:\*\*\s+(True|False)\s*$")
RE_HINT = re.compile(r"^- \*\*Hint:\*\*\s+(.+?)\s*$")
RE_OE_ANSWER = re.compile(r"^- \*\*Answer:\*\*\s*(.*)$")

# Base scope: Multiple Choice and Multi-Select both produce a "multiple_choice"
# question (Multi-Select simply allows more than one correct option); anything
# else is reported, not parsed.
KNOWN_TYPES = {
    "Multiple Choice": "multiple_choice",
    "Multi-Select": "multiple_choice",
    "True/False": "true_false",
    "Open-Ended": "open_ended",
}
MULTI_ANSWER_LABELS = {"Multi-Select"}
COUNT_KEYS = ("multiple_choice", "true_false", "open_ended")


# ---------------------------------------------------------------------------
# Parse  (pure: text in, data out, no I/O, never raises on bad content)
# ---------------------------------------------------------------------------


def is_question_file(md_text: str) -> bool:
    """A file is a *question file* if it has a Practice Questions section header
    or at least one `### Qn.` header. Files with neither are ordinary lessons
    (overviews, summaries) and are skipped without error."""
    return bool(RE_PQ_SECTION.search(md_text) or RE_Q_HEADER.search(md_text))


def parse_questions(md_text: str) -> tuple[list[Question], list[str], int | None]:
    """Parse one lesson's Markdown into (questions, errors, section_number).

    Errors are *collected*, never raised — one run reports every problem. A
    broken question is still recovered into `questions` (so counts are honest
    about what was seen) but the presence of any error must block output
    upstream (see `run`).
    """
    errors: list[str] = []

    pq_match = RE_PQ_SECTION.search(md_text)
    section = int(pq_match.group(1)) if pq_match else None
    if section is None:
        errors.append("missing `## N. Practice Questions` section header")

    headers = list(RE_Q_HEADER.finditer(md_text))
    if not headers:
        errors.append("no `### Qn. (Type)` question headers found")
        return [], errors, section

    questions: list[Question] = []
    header_numbers: list[int] = []

    for idx, header in enumerate(headers):
        qnum = int(header.group(1))
        label = header.group(2).strip()
        header_numbers.append(qnum)

        block_start = header.end()
        block_end = headers[idx + 1].start() if idx + 1 < len(headers) else len(md_text)
        # End the block at the next `## ` section or `---` rule, whichever first.
        boundary = re.search(r"^(?:## |---\s*$)", md_text[block_start:block_end], re.MULTILINE)
        if boundary:
            block_end = block_start + boundary.start()
        block = md_text[block_start:block_end]

        if label not in KNOWN_TYPES:
            errors.append(
                f"Q{qnum}: unsupported type '{label}' "
                f"(expected Multiple Choice | Multi-Select | True/False | Open-Ended)"
            )
            continue

        q, q_errors = _parse_question_body(qnum, label, block)
        errors.extend(f"Q{qnum}: {e}" for e in q_errors)
        if q is not None:
            questions.append(q)

    # Consecutive-numbering check runs over ALL headers (known + unsupported),
    # so an unsupported type does not also trigger a spurious "not consecutive".
    for expected, found in enumerate(header_numbers, start=1):
        if found != expected:
            errors.append(
                f"question numbering is not consecutive: expected Q{expected}, found Q{found}"
            )
            break

    return questions, errors, section


def _split_prompt_and_bullets(block: str) -> tuple[str, list[str]]:
    """Everything before the first `- ` bullet is the prompt; the rest are bullets."""
    lines = block.strip("\n").splitlines()
    prompt_lines: list[str] = []
    i = 0
    while i < len(lines) and not lines[i].lstrip().startswith("- "):
        if lines[i].strip():
            prompt_lines.append(lines[i].strip())
        i += 1
    bullet_lines = [ln.rstrip() for ln in lines[i:] if ln.strip()]
    return " ".join(prompt_lines).strip(), bullet_lines


def _parse_question_body(qnum: int, label: str, block: str) -> tuple[Question | None, list[str]]:
    """Dispatch on the (known) type. Returns (question_dict | None, errors)."""
    errors: list[str] = []
    prompt, bullet_lines = _split_prompt_and_bullets(block)
    if not prompt:
        errors.append("empty prompt (no text between the header and first bullet)")

    q: Question = {"number": qnum, "type": KNOWN_TYPES[label], "prompt": prompt}

    if label in ("Multiple Choice", "Multi-Select"):
        options, correct, sub = _parse_mc_bullets(
            bullet_lines, allow_multiple=label in MULTI_ANSWER_LABELS
        )
        errors.extend(sub)
        q["options"] = options
        q["correct"] = correct
    elif label == "True/False":
        answer, hint, sub = _parse_tf_bullets(bullet_lines)
        errors.extend(sub)
        q["options"] = [{"id": "true", "text": "True"}, {"id": "false", "text": "False"}]
        q["correct"] = [answer.lower()] if answer else []
        if hint:
            q["explanation"] = hint
    elif label == "Open-Ended":
        sample_answer, sub = _parse_oe_bullets(bullet_lines)
        errors.extend(sub)
        if sample_answer:
            q["sample_answer"] = sample_answer

    return q, errors


def _parse_mc_bullets(
    bullet_lines: list[str], allow_multiple: bool = False
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Parse a Multiple-Choice/Multi-Select body. Returns (options, correct, errors)."""
    options: list[dict[str, Any]] = []
    inline_correct: list[str] = []
    declared: list[str] | None = None
    errors: list[str] = []

    for line in bullet_lines:
        m_opt = RE_MC_OPTION.match(line)
        if m_opt:
            letter, text, marker, rationale = (
                m_opt.group(1),
                m_opt.group(2).strip(),
                m_opt.group(3),
                m_opt.group(4).strip(),
            )
            options.append({"id": letter, "text": text, "rationale": rationale})
            if marker == "Correct answer":
                inline_correct.append(letter)
            continue
        m_ans = RE_MC_ANSWER.match(line)
        if m_ans:
            declared = [letter.strip() for letter in m_ans.group(1).split(",")]
            continue
        errors.append(f"unrecognized bullet line: {line!r}")

    if not options:
        errors.append(
            "no MC options found "
            "(expected `- A) ... — **Correct answer/Incorrect.** ...`; needs an em-dash `—`)"
        )
    elif len(options) < 2:
        errors.append(f"only {len(options)} option(s) found; expected at least 2")

    if not inline_correct:
        errors.append("no option marked **Correct answer.**")
    elif len(inline_correct) > 1 and not allow_multiple:
        errors.append(
            f"more than one option marked **Correct answer.** in a Multiple Choice "
            f"question: {inline_correct} (use the Multi-Select type for more than "
            f"one correct answer)"
        )

    if declared is None:
        errors.append("missing trailing `- **Answer:** letter` line")
    elif inline_correct and sorted(declared) != sorted(inline_correct):
        errors.append(
            f"trailing `**Answer:** {', '.join(declared)}` does not match the inline correct "
            f"option(s) `{', '.join(inline_correct)}`"
        )

    return options, sorted(inline_correct), errors


def _parse_tf_bullets(bullet_lines: list[str]) -> tuple[str | None, str | None, list[str]]:
    answer: str | None = None
    hint: str | None = None
    errors: list[str] = []
    for line in bullet_lines:
        m_ans = RE_TF_ANSWER.match(line)
        if m_ans:
            answer = m_ans.group(1)
            continue
        m_hint = RE_HINT.match(line)
        if m_hint:
            hint = m_hint.group(1).strip()
            continue
        errors.append(f"unrecognized bullet line: {line!r}")
    if answer is None:
        errors.append("missing `- **Answer:** True|False` line")
    return answer, hint, errors


def _parse_oe_bullets(bullet_lines: list[str]) -> tuple[str | None, list[str]]:
    sample_answer: str | None = None
    has_answer_line = False
    errors: list[str] = []
    for line in bullet_lines:
        m_ans = RE_OE_ANSWER.match(line)
        if m_ans:
            has_answer_line = True
            text = m_ans.group(1).strip()
            if text:
                sample_answer = text
            continue
        errors.append(f"unrecognized bullet line: {line!r}")
    if not has_answer_line:
        errors.append("missing `- **Answer:**` line (may be blank)")
    return sample_answer, errors


# ---------------------------------------------------------------------------
# Discover + identity  (I/O and filename -> id derivation)
# ---------------------------------------------------------------------------


def discover_lessons(course_dir: Path) -> list[Path]:
    """Return the sorted list of Markdown files under `course_dir` that contain
    questions, regardless of file name (layout-agnostic discovery)."""
    candidates = sorted(course_dir.glob("**/*.md"))
    return [p for p in candidates if is_question_file(p.read_text(encoding="utf-8"))]


def course_id_from_dir(course_dir: Path) -> str:
    """The course id is just the course folder's own name (a slug)."""
    return course_dir.name


def _safe_source_dir(course_dir: Path) -> str:
    """A non-leaking representation of the course directory for the bank's
    `source_dir` field: relative to the repository root when `course_dir` is
    inside the repo, otherwise just the directory's own name — never an
    absolute local filesystem path."""
    try:
        return str(course_dir.resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return course_dir.name


def unit_id_from_filename(course_id: str, path: Path) -> str:
    """`<course_id>/<file stem>`, independent of any file naming scheme."""
    return f"{course_id}/{path.stem}"


# ---------------------------------------------------------------------------
# Assemble + write  (pure assembly, then a single I/O write)
# ---------------------------------------------------------------------------


def _finalize_question(q: Question, unit_id: str, file_name: str, section: int | None) -> Question:
    out: Question = {
        "qid": f"{unit_id}-Q{q['number']}",
        "number": q["number"],
        "type": q["type"],
        "prompt": q["prompt"],
    }
    if q["type"] != "open_ended":
        out["options"] = q["options"]
        out["correct"] = q.get("correct", [])
    if "sample_answer" in q:
        out["sample_answer"] = q["sample_answer"]
    if "explanation" in q:
        out["explanation"] = q["explanation"]
    out["source_ref"] = {"file": file_name, "section": section, "anchor": f"Q{q['number']}"}
    return out


def _counts(questions: list[Question]) -> dict[str, int]:
    counts = {k: 0 for k in COUNT_KEYS}
    for q in questions:
        counts[q["type"]] = counts.get(q["type"], 0) + 1
    return counts


def assemble(
    course_id: str,
    source_dir: str,
    parsed: list[tuple[Path, str, list[Question], int | None]],
) -> Bank:
    """Build the final deterministic JSON structure (no timestamps)."""
    units: list[dict[str, Any]] = []
    all_questions: list[Question] = []

    for path, unit_id, questions, section in parsed:
        finalized = [_finalize_question(q, unit_id, path.name, section) for q in questions]
        all_questions.extend(finalized)
        units.append(
            {
                "unit_id": unit_id,
                "source_file": path.name,
                "section": section,
                "question_counts": _counts(questions),
            }
        )

    return {
        "schema_version": "1.0",
        "course": course_id,
        "source_dir": source_dir,
        "question_counts": _counts(all_questions),
        "units": units,
        "questions": all_questions,
    }


def write_bank(bank: Bank, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bank, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_summary(bank: Bank, path: Path) -> None:
    """Deterministic validation transcript: unit file names and counts only,
    no absolute paths, no timestamps."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["QBANK VALIDATION SUMMARY", "=" * 40]
    for unit in bank["units"]:
        c = unit["question_counts"]
        lines.append(
            f"  - {unit['source_file']}: {c['multiple_choice']} MC, "
            f"{c['true_false']} T/F, {c['open_ended']} OE"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Orchestration + CLI
# ---------------------------------------------------------------------------


def build_bank(course_dir: Path, course_id: str | None = None) -> tuple[Bank, list[str]]:
    """Discover, parse and assemble the bank for `course_dir`. Pure: the only
    I/O is reading the lesson files themselves — no output is written.

    Returns (bank, errors). `errors` collects every validation problem found
    across every lesson; the caller decides whether to write the bank.
    """
    cid = course_id or course_id_from_dir(course_dir)
    lessons = discover_lessons(course_dir)

    parsed: list[tuple[Path, str, list[Question], int | None]] = []
    errors: list[str] = []
    for path in lessons:
        text = path.read_text(encoding="utf-8")
        unit_id = unit_id_from_filename(cid, path)
        questions, q_errors, section = parse_questions(text)
        errors.extend(f"{path.name}: {e}" for e in q_errors)
        parsed.append((path, unit_id, questions, section))
        c = _counts(questions)
        log.info(
            "%s: %d MC, %d T/F, %d OE",
            path.name,
            c["multiple_choice"],
            c["true_false"],
            c["open_ended"],
        )

    bank = assemble(cid, _safe_source_dir(course_dir), parsed)
    return bank, errors


def run(course_dir: Path, output: Path, check_only: bool, course_id: str | None) -> int:
    """Build the bank for `course_dir` and, unless `check_only` or a validation
    error is found, write it to `output` and write `output`'s sibling
    `summary.txt`."""
    if not course_dir.is_dir():
        log.error("not a directory: %s", course_dir)
        return 1

    bank, errors = build_bank(course_dir, course_id)
    if not bank["units"]:
        log.error("no question files found under %s", course_dir)
        return 1
    if errors:
        for e in errors:
            log.error("%s", e)
        return 1

    if check_only:
        return 0

    write_bank(bank, output)
    _write_summary(bank, output.parent / "summary.txt")
    return 0


def demo() -> DemoResult:
    """Build a small bank from the two valid shipped fixtures; offline and
    deterministic. Writes output/question-bank.json, output/summary.txt (via
    `run`) and output/metrics.txt."""
    start = time.perf_counter()
    fixtures = HERE / "fixtures"
    output_dir = OUT_DIR
    demo_dir = output_dir / "demo-lessons"
    demo_dir.mkdir(parents=True, exist_ok=True)

    for stem in ("e00_valid_baseline", "e15_multi_select_valid"):
        text = (fixtures / f"{stem}.md").read_text(encoding="utf-8")
        (demo_dir / f"{stem}.md").write_text(text, encoding="utf-8")

    out_path = output_dir / "question-bank.json"
    rc = run(demo_dir, out_path, check_only=False, course_id="p02-demo")

    total = 0
    if out_path.exists():
        bank = json.loads(out_path.read_text(encoding="utf-8"))
        total = sum(bank["question_counts"].values())

    _write_metrics({"lessons": 2, "questions": total}, output_dir / "metrics.txt")

    return DemoResult(
        "p02-question-bank",
        "ok" if rc == 0 else "failed",
        {"lessons": "2", "questions": str(total)},
        seconds=round(time.perf_counter() - start, 2),
    )


def _write_metrics(figures: dict[str, int], path: Path) -> None:
    """Deterministic figures-only proof file: no timestamps, no absolute paths."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{key}={value}" for key, value in figures.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a JSON question bank from Markdown lesson notes."
    )
    p.add_argument(
        "course_dir",
        nargs="?",
        help="Folder to scan for Markdown lesson files with a Practice Questions section.",
    )
    p.add_argument(
        "--output",
        default="output/question-bank.json",
        help="Output JSON path (default: output/question-bank.json).",
    )
    p.add_argument(
        "--check-only", action="store_true", help="Validate only; do not write any output."
    )
    p.add_argument(
        "--course-id",
        help="Course id used to build question ids (default: the course folder name).",
    )
    p.add_argument(
        "--demo", action="store_true", help="Run the offline demo on the shipped fixtures."
    )
    p.add_argument("--verbose", action="store_true", help="Log pipeline steps at INFO level.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s"
    )

    if args.demo:
        result = demo()
        print(f"p02-question-bank: {result.status}")
        for key, value in result.figures.items():
            print(f"  {key}: {value}")
        print("  wrote: output/")
        print(f"  seconds: {result.seconds:.2f}")
        return 0 if result.status == "ok" else 1

    if not args.course_dir:
        print("error: course_dir is required unless --demo is given", file=sys.stderr)
        return 1

    course_dir = Path(args.course_dir)
    output = Path(args.output)
    rc = run(course_dir, output, args.check_only, args.course_id)

    if rc != 0:
        print("error: validation failed; see stderr for details", file=sys.stderr)
        return rc

    if args.check_only:
        print(f"p02-question-bank: {course_dir.name} check-only OK")
        return 0

    bank = json.loads(output.read_text(encoding="utf-8"))
    counts = bank["question_counts"]
    total = sum(counts.values())
    print(f"p02-question-bank: {course_dir.name}")
    print(
        f"  questions={total} (MC={counts['multiple_choice']} "
        f"TF={counts['true_false']} OE={counts['open_ended']})"
    )
    print(f"  wrote: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
