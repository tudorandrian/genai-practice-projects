"""quiz_engine.py - a quiz simulator over the P12 synthetic corpus's question bank.

Builds a merged question bank from ``corpus/`` (three courses, via
``projects.p02_question_bank.qbank.build_bank``), runs a session of random
closed questions (multiple choice + true/false) with shuffled options, scores
answers (handling multi-answer ``correct`` lists), and persists each session to
``output/history.json`` so ``progress.py`` can compute success rates.

Pure functions: ``load_bank`` / ``build_session`` / ``grade`` / ``save_session`` -
all testable without a UI.

CLI: ``python quiz_engine.py --n 5 --seed 42 [--module data-cleaning-basics] [--type true_false]``
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import random
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
CORPUS_DIR = HERE / "corpus"
OUT_DIR = HERE / "output"
HISTORY_PATH = OUT_DIR / "history.json"
BANK_PATH = OUT_DIR / "question-bank.json"

CLOSED_TYPES = ("multiple_choice", "true_false")

Question = dict[str, Any]


def build_bank_from_corpus(
    corpus_dir: str | Path = CORPUS_DIR, out_path: str | Path = BANK_PATH
) -> Path:
    """Build and write the merged question bank for every course under ``corpus_dir``.

    Calls ``qbank.build_bank(course_dir)`` for each course subdirectory,
    concatenates ``questions``/``units`` and sums ``question_counts``, then
    writes the combined JSON to ``out_path``. Raises ``ValueError`` if any
    course fails to validate (mirrors ``qbank.run``'s all-or-nothing behaviour).
    """
    from projects.p02_question_bank import qbank

    corpus_dir = Path(corpus_dir)
    courses = sorted(p for p in corpus_dir.iterdir() if p.is_dir())

    all_questions: list[Question] = []
    all_units: list[dict[str, Any]] = []
    counts = {"multiple_choice": 0, "true_false": 0, "open_ended": 0}
    for course_dir in courses:
        bank, errors = qbank.build_bank(course_dir)
        if errors:
            raise ValueError(f"{course_dir.name}: {errors}")
        all_questions.extend(bank["questions"])
        all_units.extend(bank["units"])
        for key, value in bank["question_counts"].items():
            counts[key] = counts.get(key, 0) + value
        log.info("%s: %s", course_dir.name, bank["question_counts"])

    merged = {
        "schema_version": "1.0",
        "courses": [c.name for c in courses],
        "question_counts": counts,
        "units": all_units,
        "questions": all_questions,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_path


def load_bank(path: str | Path | None = None) -> list[Question]:
    """Load the merged bank and flatten to closed questions only (multiple_choice +
    true_false), tagging each with ``module = qid.split("/")[0]`` (the course slug)."""
    bank_path = Path(path) if path is not None else BANK_PATH
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    questions = [
        {**q, "module": q["qid"].split("/")[0]}
        for q in bank["questions"]
        if q["type"] in CLOSED_TYPES
    ]
    assert len(questions) > 0, "no closed questions found in the bank - is the corpus empty?"
    return questions


def build_session(
    questions: list[Question],
    n: int = 5,
    seed: int | None = None,
    module: str | None = None,
    qtype: str | None = None,
) -> list[Question]:
    """Pick ``n`` random questions (optionally filtered by module/type) and shuffle
    each one's options. Option ``id``s are stable, so shuffling only changes display
    order, not correctness. ``seed`` makes the session repeatable."""
    pool = [
        q
        for q in questions
        if (module is None or q["module"] == module) and (qtype is None or q["type"] == qtype)
    ]
    rng = random.Random(seed)
    chosen = rng.sample(pool, min(n, len(pool)))
    session = []
    for q in chosen:
        options = list(q["options"])
        rng.shuffle(options)
        session.append({**q, "shuffled_options": options})
    return session


def grade(question: Question, answer: Any) -> bool:
    """True if ``answer`` (an id or list of ids) matches the ``correct`` set exactly.
    Handles questions with several correct options."""
    correct = {c.strip().lower() for c in question["correct"]}
    if isinstance(answer, (list, tuple, set)):
        given = {str(a).strip().lower() for a in answer}
    else:
        given = {p.strip().lower() for p in str(answer).replace(",", " ").split()}
    return bool(given) and given == correct


def load_history() -> list[dict[str, Any]]:
    """Return the saved quiz sessions (empty list if none yet)."""
    if HISTORY_PATH.exists():
        return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    return []


def save_session(summary: dict[str, Any]) -> None:
    """Append one session summary to ``output/history.json``."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    history = load_history()
    history.append(summary)
    HISTORY_PATH.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def _session_summary(session: list[Question], answers: list[Any]) -> dict[str, Any]:
    """Build the persisted record for a completed session."""
    details, score = [], 0
    for q, given in zip(session, answers, strict=True):
        correct = grade(q, given)
        score += int(correct)
        details.append(
            {
                "qid": q["qid"],
                "module": q["module"],
                "type": q["type"],
                "given": given,
                "correct": q["correct"],
                "is_correct": correct,
            }
        )
    return {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "score": score,
        "total": len(session),
        "questions": details,
    }


def run_quiz_cli(
    n: int = 5, seed: int | None = None, module: str | None = None, qtype: str | None = None
) -> dict[str, Any]:
    """Interactive console quiz; saves the session and returns its summary."""
    if not BANK_PATH.exists():
        build_bank_from_corpus(CORPUS_DIR, BANK_PATH)
    questions = load_bank(BANK_PATH)
    print(f"{len(questions)} closed questions loaded.")
    session = build_session(questions, n=n, seed=seed, module=module, qtype=qtype)
    answers: list[str] = []
    for i, q in enumerate(session, 1):
        print(f"\n[{i}/{len(session)}] ({q['module']} - {q['type']}) {q['prompt']}")
        for opt in q["shuffled_options"]:
            print(f"   {opt['id']}) {opt['text']}")
        try:
            answers.append(input("Your answer (id): ").strip())
        except (EOFError, KeyboardInterrupt):
            answers.append("")
    summary = _session_summary(session, answers)
    for q, det in zip(session, summary["questions"], strict=True):
        mark = "OK   " if det["is_correct"] else "WRONG"
        print(f"\n{mark} {q['prompt'][:70]}...")
        if not det["is_correct"]:
            print(f"   correct: {', '.join(q['correct'])} | {q.get('explanation', '')}")
    print(f"\nFinal score: {summary['score']}/{summary['total']}")
    save_session(summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entry: run one console quiz session with the given options."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", maxsplit=1)[0])
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--module", default=None, help="e.g. data-cleaning-basics")
    parser.add_argument("--type", dest="qtype", default=None, choices=[None, *CLOSED_TYPES])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s"
    )
    run_quiz_cli(n=args.n, seed=args.seed, module=args.module, qtype=args.qtype)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
