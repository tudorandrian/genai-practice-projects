"""Tests for qbank: parsing, discovery, identity, assembly, and the run()/build_bank()
pipeline, driven by the fixtures in fixtures/."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from projects.p02_question_bank import qbank

pytestmark = pytest.mark.core
HERE = Path(__file__).resolve().parents[1]
FIXTURES = HERE / "fixtures"
EDGE_CASES = sorted(FIXTURES.glob("e*.md"))


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def parse(name: str) -> tuple[list[dict[str, Any]], list[str], int | None]:
    return qbank.parse_questions(fixture(name))


def errors_of(name: str) -> list[str]:
    return parse(name)[1]


def joined(name: str) -> str:
    return " || ".join(errors_of(name))


# ---------------------------------------------------------------------------
# New tests required by this port (brief Step 2): generic discovery, schema
# conformance, and a parametrized sweep over every fixture.
# ---------------------------------------------------------------------------


def test_discovery_is_not_tied_to_a_file_naming_scheme(tmp_path: Path) -> None:
    (tmp_path / "any-name.md").write_text(
        (FIXTURES / "e00_valid_baseline.md").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "notes.md").write_text("# no questions here\n", encoding="utf-8")
    found = qbank.discover_lessons(tmp_path)
    assert [p.name for p in found] == ["any-name.md"]
    assert qbank.unit_id_from_filename("course-x", found[0]) == "course-x/any-name"


@pytest.mark.parametrize("fixture_name", ["e00_valid_baseline", "e15_multi_select_valid"])
def test_valid_bank_matches_the_json_schema(fixture_name: str, tmp_path: Path) -> None:
    (tmp_path / "lesson.md").write_text(
        (FIXTURES / f"{fixture_name}.md").read_text(encoding="utf-8"), encoding="utf-8"
    )
    bank, errors = qbank.build_bank(tmp_path)
    assert errors == []
    schema = json.loads(qbank.SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(bank, schema)  # raises on mismatch, including a Multi-Select
    # question's `correct` list with more than one entry (e15).


def test_valid_baseline_bank_has_one_of_each_type(tmp_path: Path) -> None:
    (tmp_path / "lesson.md").write_text(
        (FIXTURES / "e00_valid_baseline.md").read_text(encoding="utf-8"), encoding="utf-8"
    )
    bank, errors = qbank.build_bank(tmp_path)
    assert errors == []
    assert bank["question_counts"] == {"multiple_choice": 1, "true_false": 1, "open_ended": 1}


# Expected (errors == [] ?, exact recovered-question count) for every fixture,
# derived from what each fixture is designed to show (its heading/content) and
# from the current parser's actual behaviour (computed by running the parser,
# then cross-checked against each fixture's stated intent):
#   - e00, e15 (fully valid): errors == [] and the exact question count.
#   - Strictly invalid (no recoverable question block): errors != [] and 0
#     questions.
#   - Mixed (a valid part plus a broken part, e.g. e14/e16, or a lone broken
#     question still recovered alongside its error, e.g. e04-e13): errors !=
#     [] and the exact number of questions the parser recovers.
# A fixture added under fixtures/e*.md without a matching entry here fails the
# guard test below, so the sweep can never silently start ignoring it.
EDGE_CASE_EXPECTATIONS: dict[str, tuple[bool, int]] = {
    "e00_valid_baseline": (True, 3),
    "e01_missing_pq_section": (False, 1),
    "e02_section_no_questions": (False, 0),
    "e03_unknown_type": (False, 0),
    "e04_mc_no_correct_marker": (False, 1),
    "e05_mc_two_correct_markers": (False, 1),
    "e06_answer_letter_mismatch": (False, 1),
    "e07_mc_missing_answer_line": (False, 1),
    "e08_tf_missing_answer": (False, 1),
    "e09_oe_missing_answer_line": (False, 1),
    "e10_non_consecutive": (False, 2),
    "e11_missing_separator": (False, 1),
    "e12_multiline_rationale": (False, 1),
    "e13_empty_prompt": (False, 1),
    "e14_mixed_valid_and_broken": (False, 3),
    "e15_multi_select_valid": (True, 1),
    "e16_unknown_type_between_valid": (False, 2),
}


def test_edge_case_expectations_cover_every_fixture() -> None:
    # A new fixture file cannot be added without a matching expectation entry.
    assert {p.stem for p in EDGE_CASES} == set(EDGE_CASE_EXPECTATIONS)


@pytest.mark.parametrize("fixture_path", EDGE_CASES, ids=[p.stem for p in EDGE_CASES])
def test_every_edge_case_parses_or_reports(fixture_path: Path) -> None:
    expect_no_errors, expected_count = EDGE_CASE_EXPECTATIONS[fixture_path.stem]
    questions, errors, section = qbank.parse_questions(fixture_path.read_text(encoding="utf-8"))
    if expect_no_errors:
        assert errors == []
    else:
        assert errors != []
    assert len(questions) == expected_count


# ---------------------------------------------------------------------------
# Parsing / validation, one test per edge-case fixture (ported from the
# original 27-test suite; converted from unittest to pytest).
# ---------------------------------------------------------------------------


def test_clean_three_types() -> None:
    questions, errors, section = parse("e00_valid_baseline.md")
    assert errors == []
    assert section == 3
    assert [q["type"] for q in questions] == ["multiple_choice", "true_false", "open_ended"]


def test_open_ended_has_no_options_or_correct() -> None:
    questions, _, _ = parse("e00_valid_baseline.md")
    oe = questions[2]
    assert "options" not in oe
    assert "correct" not in oe


def test_mc_correct_is_a_list() -> None:
    questions, _, _ = parse("e00_valid_baseline.md")
    assert questions[0]["correct"] == ["B"]


def test_missing_pq_section_still_recovers_question() -> None:
    questions, errors, section = parse("e01_missing_pq_section.md")
    assert section is None
    assert len(questions) == 1  # recovered despite the missing header
    assert "missing `## N. Practice Questions` section header" in errors


def test_section_but_no_question_blocks() -> None:
    questions, errors, _ = parse("e02_section_no_questions.md")
    assert questions == []
    assert "no `### Qn. (Type)` question headers found" in errors


def test_empty_course_dir_is_caught_by_run() -> None:
    # discover_lessons on an empty dir -> run() exits 1 with no output.
    with tempfile.TemporaryDirectory() as d:
        rc = qbank.run(Path(d), Path(d) / "out.json", False, "demo-course")
        assert rc == 1
        assert not (Path(d) / "out.json").exists()


def test_unknown_type_alone_reports_unsupported_not_generic() -> None:
    errs = joined("e03_unknown_type.md")
    assert "unsupported type 'Fill-in-the-Blank'" in errs
    assert "unsupported type 'Matching'" in errs


def test_multi_select_with_matching_answer_line_is_valid() -> None:
    # The parser now supports a dedicated Multi-Select label (mapped onto the
    # multiple_choice type): several inline **Correct answer.** markers are
    # allowed as long as the trailing Answer line lists the same letters.
    questions, errors, _ = parse("e15_multi_select_valid.md")
    assert errors == []
    assert questions[0]["type"] == "multiple_choice"
    assert questions[0]["correct"] == ["A", "C"]


def test_unknown_type_between_valid_does_not_pollute_neighbour() -> None:
    questions, errors, _ = parse("e16_unknown_type_between_valid.md")
    # Q1 (MC) and Q3 (TF) recovered cleanly; Q2 (Matching) rejected.
    assert [q["number"] for q in questions] == [1, 3]
    assert questions[0]["prompt"] == "Which artifact is derived?"
    joined_err = " || ".join(errors)
    assert "Q2: unsupported type 'Matching'" in joined_err
    # No pollution of Q1 and no spurious non-consecutive error.
    assert "unrecognized bullet" not in joined_err
    assert "not consecutive" not in joined_err


def test_mc_no_correct_marker() -> None:
    assert "Q1: no option marked **Correct answer.**" in errors_of("e04_mc_no_correct_marker.md")


def test_mc_two_correct_markers() -> None:
    assert "more than one option marked" in joined("e05_mc_two_correct_markers.md")


def test_answer_letter_mismatch() -> None:
    assert "does not match the inline correct" in joined("e06_answer_letter_mismatch.md")


def test_mc_missing_answer_line() -> None:
    assert "Q1: missing trailing `- **Answer:** letter` line" in errors_of(
        "e07_mc_missing_answer_line.md"
    )


def test_tf_missing_answer() -> None:
    assert "Q1: missing `- **Answer:** True|False` line" in errors_of("e08_tf_missing_answer.md")


def test_oe_missing_answer_line() -> None:
    assert "Q1: missing `- **Answer:**` line (may be blank)" in errors_of(
        "e09_oe_missing_answer_line.md"
    )


def test_non_consecutive_numbering() -> None:
    assert "not consecutive" in joined("e10_non_consecutive.md")


def test_missing_separator_breaks_options() -> None:
    errs = joined("e11_missing_separator.md")
    assert "unrecognized bullet line" in errs
    assert "no MC options found" in errs


@pytest.mark.parametrize("dash", ["-", "\u2013", "\u2014"], ids=["hyphen", "en-dash", "em-dash"])
def test_option_separator_accepts_hyphen_en_dash_and_em_dash(dash: str) -> None:
    text = fixture("e00_valid_baseline.md").replace(" - **", f" {dash} **")
    assert qbank.parse_questions(text)[:2] == parse("e00_valid_baseline.md")[:2]


def test_empty_prompt() -> None:
    assert "Q1: empty prompt (no text between the header and first bullet)" in errors_of(
        "e13_empty_prompt.md"
    )


def test_mixed_recovers_all_but_reports_broken() -> None:
    questions, errors, _ = parse("e14_mixed_valid_and_broken.md")
    assert len(questions) == 3  # partial parse: all recovered
    assert len(errors) == 1
    assert "Q2: missing trailing" in errors[0]


# ---------------------------------------------------------------------------
# Identity + assembly (layout-agnostic: no leading-number course ids, no
# lesson-code file names, unit_id_from_filename never returns None).
# ---------------------------------------------------------------------------


def test_course_id_from_dir() -> None:
    assert qbank.course_id_from_dir(Path("intro-to-generative-ai")) == "intro-to-generative-ai"
    assert qbank.course_id_from_dir(Path("nope")) == "nope"


def test_unit_id_from_filename() -> None:
    assert qbank.unit_id_from_filename("demo-course", Path("lesson-1.md")) == "demo-course/lesson-1"
    assert (
        qbank.unit_id_from_filename("demo-course", Path("module/topic-3.md"))
        == "demo-course/topic-3"
    )
    # A name that would not have matched the old M<m>-<c>-L<l> scheme still
    # gets a unit id (the point of the port: never None).
    assert qbank.unit_id_from_filename("demo-course", Path("quiz-1.md")) == "demo-course/quiz-1"


def test_source_dir_is_repo_relative_for_a_dir_inside_the_repo() -> None:
    course_dir = qbank.HERE / "fixtures"
    bank, _errors = qbank.build_bank(course_dir)
    assert bank["source_dir"] == "projects/p02_question_bank/fixtures"
    assert ":" not in bank["source_dir"]  # never a Windows drive letter / absolute path


def test_source_dir_falls_back_to_dir_name_outside_the_repo(tmp_path: Path) -> None:
    course = tmp_path / "some-course"
    course.mkdir()
    (course / "lesson.md").write_text(fixture("e00_valid_baseline.md"), encoding="utf-8")
    bank, _errors = qbank.build_bank(course)
    assert bank["source_dir"] == "some-course"


def test_qid_starts_with_course_id_slash() -> None:
    # Pinned for projects/p12: qid always starts with "<course_id>/", so the
    # module can be recovered with qid.split("/")[0].
    bank = _assemble_baseline()
    assert bank["questions"][0]["qid"].startswith("demo-course/")


def _assemble_baseline() -> dict[str, Any]:
    questions, _, section = parse("e00_valid_baseline.md")
    return qbank.assemble(
        "demo-course", "fixtures", [(Path("lesson.md"), "demo-course/lesson", questions, section)]
    )


def test_qid_and_source_ref() -> None:
    bank = _assemble_baseline()
    first = bank["questions"][0]
    assert first["qid"] == "demo-course/lesson-Q1"
    assert first["source_ref"] == {"file": "lesson.md", "section": 3, "anchor": "Q1"}


def test_counts() -> None:
    bank = _assemble_baseline()
    assert bank["question_counts"] == {"multiple_choice": 1, "true_false": 1, "open_ended": 1}


# ---------------------------------------------------------------------------
# End-to-end run(): exit codes, output, determinism, check-only.
# ---------------------------------------------------------------------------


def _make_course(d: Path, body: str, name: str = "lesson-1.md") -> None:
    sub = d / "unit-1" / "topic"
    sub.mkdir(parents=True)
    (sub / name).write_text(body, encoding="utf-8")


def test_valid_run_writes_json_and_exits_zero() -> None:
    with tempfile.TemporaryDirectory() as d:
        d_path = Path(d)
        _make_course(d_path, fixture("e00_valid_baseline.md"))
        out = d_path / "out" / "bank.json"
        rc = qbank.run(d_path, out, False, "demo-course")
        assert rc == 0
        bank = json.loads(out.read_text(encoding="utf-8"))
        assert bank["question_counts"]["multiple_choice"] == 1
        assert bank["questions"][0]["qid"] == "demo-course/lesson-1-Q1"


def test_deterministic_byte_identical() -> None:
    with tempfile.TemporaryDirectory() as d:
        d_path = Path(d)
        _make_course(d_path, fixture("e00_valid_baseline.md"))
        out = d_path / "out" / "bank.json"
        qbank.run(d_path, out, False, "demo-course")
        first = out.read_bytes()
        qbank.run(d_path, out, False, "demo-course")
        assert first == out.read_bytes()


def test_check_only_writes_nothing() -> None:
    with tempfile.TemporaryDirectory() as d:
        d_path = Path(d)
        _make_course(d_path, fixture("e00_valid_baseline.md"))
        out = d_path / "out" / "bank.json"
        rc = qbank.run(d_path, out, True, "demo-course")
        assert rc == 0
        assert not out.exists()


def test_broken_file_blocks_output() -> None:
    with tempfile.TemporaryDirectory() as d:
        d_path = Path(d)
        _make_course(d_path, fixture("e06_answer_letter_mismatch.md"))
        out = d_path / "out" / "bank.json"
        rc = qbank.run(d_path, out, False, "demo-course")
        assert rc == 1
        assert not out.exists()


# ---------------------------------------------------------------------------
# demo()
# ---------------------------------------------------------------------------


def test_demo_is_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qbank, "OUT_DIR", tmp_path / "output")
    first = qbank.demo()
    second = qbank.demo()
    assert first.figures == second.figures
    assert first.status == "ok"
    assert first.figures["lessons"] == "2"
