"""Tests for the P12 study hub capstone (quiz_engine / progress / tutor / app).

Heavy paths (embeddings, Chroma, the LLM) are only touched by ``rag``-marked
tests, per the ported source's own convention — ``tutor.py`` imports
LangChain/Chroma even in its simplest paths, so every test that touches it,
pure helpers included, stays ``rag`` rather than ``core``. ``quiz_engine`` and
``progress`` run on the real shipped corpus and stay ``core`` (fast, offline).

Run
    uv run pytest projects/p12_study_hub -q            # core only
    uv run pytest projects/p12_study_hub -m rag -q     # needs `uv sync --group rag`
"""

from __future__ import annotations

from pathlib import Path

import pytest

from projects.p02_question_bank import qbank
from projects.p12_study_hub import progress, quiz_engine, tutor
from shared import blocklist

# No blanket module-level ``pytestmark`` here: every test is marked
# individually, since this file mixes ``core`` tests (quiz_engine, progress —
# pure Python and pandas) with ``rag`` tests (tutor.py imports LangChain even
# in its simplest paths) — see p10_meeting_assistant's test file for the same
# convention and the reason it matters: a blanket module-level `core` marker
# would leave every `rag` test *also* matching `-m "core and not network"`,
# which is exactly the CI filter that must not import langchain/chromadb.

HERE = Path(__file__).resolve().parents[1]
CORPUS = HERE / "corpus"
COURSES = sorted(p for p in CORPUS.iterdir() if p.is_dir())


# =============================================================================
# The corpus itself
# =============================================================================


@pytest.mark.core
def test_corpus_has_three_courses_of_ten_lessons_with_three_questions_each() -> None:
    assert [c.name for c in COURSES] == [
        "classic-ml-in-practice",
        "data-cleaning-basics",
        "llm-apps-from-scratch",
    ]
    for course in COURSES:
        bank, errors = qbank.build_bank(course)
        assert errors == [], errors
        assert len(bank["units"]) == 10
        assert bank["question_counts"] == {
            "multiple_choice": 10,
            "true_false": 10,
            "open_ended": 10,
        }


@pytest.mark.core
def test_corpus_contains_no_blocklisted_phrase() -> None:
    assert blocklist.scan(sorted(CORPUS.rglob("*.md"))) == []


# =============================================================================
# quiz_engine
# =============================================================================


@pytest.mark.core
def test_quiz_session_from_the_corpus_bank(
    tmp_output: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(quiz_engine, "OUT_DIR", tmp_output)
    monkeypatch.setattr(quiz_engine, "HISTORY_PATH", tmp_output / "history.json")
    bank_path = quiz_engine.build_bank_from_corpus(CORPUS, tmp_output / "question-bank.json")
    questions = quiz_engine.load_bank(bank_path)
    assert len(questions) == 60
    session = quiz_engine.build_session(questions, n=5, seed=42, module="data-cleaning-basics")
    assert len(session) == 5 and all(q["module"] == "data-cleaning-basics" for q in session)
    answers = [q["correct"] for q in session]
    quiz_engine.save_session(quiz_engine._session_summary(session, answers))
    history = quiz_engine.load_history()
    assert history[-1]["score"] == 5 and history[-1]["total"] == 5


@pytest.mark.core
def test_build_bank_from_corpus_sums_every_course(tmp_output: Path) -> None:
    bank_path = quiz_engine.build_bank_from_corpus(CORPUS, tmp_output / "bank.json")
    questions = quiz_engine.load_bank(bank_path)
    assert len(questions) == 60  # 30 MC + 30 T/F across the three courses; 30 open-ended excluded


@pytest.mark.core
def test_session_is_seed_repeatable_and_filtered(tmp_output: Path) -> None:
    bank_path = quiz_engine.build_bank_from_corpus(CORPUS, tmp_output / "bank.json")
    questions = quiz_engine.load_bank(bank_path)
    s1 = quiz_engine.build_session(questions, n=6, seed=7, qtype="true_false")
    s2 = quiz_engine.build_session(questions, n=6, seed=7, qtype="true_false")
    assert [q["qid"] for q in s1] == [q["qid"] for q in s2]
    assert all(q["type"] == "true_false" for q in s1)


@pytest.mark.core
def test_scoring_single_and_multi_answer() -> None:
    assert quiz_engine.grade({"correct": ["C"]}, "c")
    assert not quiz_engine.grade({"correct": ["C"]}, "a")
    assert quiz_engine.grade({"correct": ["A", "C"]}, "A, C")
    assert not quiz_engine.grade({"correct": ["A", "C"]}, "A")


@pytest.mark.core
def test_history_round_trip(tmp_output: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(quiz_engine, "OUT_DIR", tmp_output)
    monkeypatch.setattr(quiz_engine, "HISTORY_PATH", tmp_output / "history.json")
    assert quiz_engine.load_history() == []
    quiz_engine.save_session({"score": 3, "total": 5, "questions": []})
    assert len(quiz_engine.load_history()) == 1


# =============================================================================
# progress
# =============================================================================


@pytest.mark.core
def test_normalize_status() -> None:
    assert progress._normalize("COMPLETED - 6 lessons") == "COMPLETED"
    assert progress._normalize("in progress now") == "IN PROGRESS"
    assert progress._normalize("—") is None


@pytest.mark.core
def test_table_status_column_extraction() -> None:
    text = (
        "| File | Status |\n|---|---|\n| a.md | COMPLETED |\n| b.md | NOT STARTED |\n| c.md | — |\n"
    )
    assert progress._statuses_from_table(text) == ["COMPLETED", "NOT STARTED"]


@pytest.mark.core
def test_progress_scan_reads_status_tables() -> None:
    df = progress.scan_statuses(CORPUS)
    assert len(df) == 30 and set(df["status"]) <= set(progress.CANONICAL_STATUSES)
    assert set(df["course"]) == {c.name for c in COURSES}


@pytest.mark.core
def test_aggregate_pivots_by_course_and_status() -> None:
    df = progress.scan_statuses(CORPUS)
    pivot = progress.aggregate(df)
    assert set(pivot.index) == {c.name for c in COURSES}
    assert (pivot["TOTAL"] == 10).all()


@pytest.mark.core
def test_success_rate_from_history(tmp_path: Path) -> None:
    history_path = tmp_path / "h.json"
    history_path.write_text(
        '[{"questions": ['
        '{"module": "data-cleaning-basics", "type": "true_false", "is_correct": true},'
        '{"module": "data-cleaning-basics", "type": "true_false", "is_correct": false}]}]',
        encoding="utf-8",
    )
    result = progress.success_rate(history_path)
    assert result["sessions"] == 1
    assert result["by_module"]["data-cleaning-basics"] == pytest.approx(0.5)


@pytest.mark.core
def test_success_rate_with_no_history_file(tmp_path: Path) -> None:
    result = progress.success_rate(tmp_path / "missing.json")
    assert result == {"sessions": 0, "by_module": {}, "by_type": {}}


# =============================================================================
# tutor — imports LangChain even in its simplest paths, so every test here
# stays `rag` (per the rename map's note that module C keeps @pytest.mark.rag).
# =============================================================================


@pytest.mark.rag
def test_grounding_prompt_demands_context_and_refusal() -> None:
    assert "ONLY" in tutor.PROMPT_TEMPLATE
    assert tutor.REFUSAL in tutor.PROMPT_TEMPLATE
    assert "{context}" in tutor.PROMPT_TEMPLATE
    assert "{question}" in tutor.PROMPT_TEMPLATE


@pytest.mark.rag
def test_sources_dedupe() -> None:
    class FakeDoc:
        def __init__(self, source: str) -> None:
            self.metadata = {"source": source}

    docs = [
        FakeDoc("data-cleaning-basics/lessons/01-why-data-quality-matters.md"),
        FakeDoc("data-cleaning-basics/lessons/01-why-data-quality-matters.md"),
        FakeDoc("classic-ml-in-practice/lessons/07-clustering-stations-with-kmeans.md"),
    ]
    assert tutor.format_sources(docs) == [
        "data-cleaning-basics/lessons/01-why-data-quality-matters.md",
        "classic-ml-in-practice/lessons/07-clustering-stations-with-kmeans.md",
    ]


@pytest.mark.rag
def test_stub_llm_provider_needs_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """The stub is deterministic and derived from its own input (T14-b) — never
    a fixed string regardless of the prompt, and never the exact ``REFUSAL``
    string (that's reserved for ``ask()``'s relevance gate and a real
    provider's grounding behaviour)."""
    monkeypatch.setenv("TUTOR_LLM_PROVIDER", "stub")
    prompt = (
        "Context:\nSentinel values look like ordinary numbers but mean missing data."
        "\n\nQuestion: What?\n\nAnswer:"
    )
    first = tutor.ask_llm(prompt)
    second = tutor.ask_llm(prompt)
    assert first == second  # deterministic given the same input
    assert "Sentinel values look like ordinary numbers" in first  # echoes the retrieved context
    assert first != tutor.REFUSAL
    assert tutor.ask_llm("Context:\nSomething else entirely.\n\nQuestion: X?\n\nAnswer:") != first


@pytest.mark.rag
def test_stub_provider_argument_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TUTOR_LLM_PROVIDER", "ollama")  # would need a running service
    prompt = "Context:\nFoo bar baz qux.\n\nQuestion: Q?\n\nAnswer:"
    result = tutor.ask_llm(prompt, provider="stub")  # explicit argument wins
    assert "Foo bar baz qux" in result
    assert result != tutor.REFUSAL


@pytest.mark.rag
def test_finds_lesson_notes() -> None:
    assert len(tutor.find_lessons(CORPUS)) == 30


@pytest.mark.rag
def test_index_and_ask_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Real embeddings + a real Chroma index over the shipped corpus, with the
    stub LLM provider — proves retrieval genuinely finds a relevant lesson, and
    (T14-b/T14-c) that the stub's answer is derived from that retrieved
    *lesson prose*, not merely from the bracketed source label ask() used to
    assemble into the context: a distinctive phrase from the sentinel-values
    lesson must actually appear in the answer, deterministic for a repeated
    question, different for a different (still on-topic) question, and never
    the bare ``REFUSAL`` string."""
    monkeypatch.setattr(tutor, "INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr(tutor, "MANIFEST_PATH", tmp_path / "index" / "manifest.json")
    report = tutor.index_lessons(reindex=True, corpus_dir=CORPUS)
    assert report["total_files"] == 30

    question = "What is a sentinel value in a messy dataset?"
    result = tutor.ask(question, provider="stub")
    assert result["sources"], "a relevant lesson should have been retrieved"
    assert any("sentinel" in s for s in result["sources"])
    assert result["answer"].startswith("STUB:")
    assert result["answer"] != tutor.REFUSAL
    # A real sentence from the retrieved lesson, not the bracketed source path
    # (a 61-character label alone would consume the stub's whole 60-char
    # slice budget — see tutor.py's ask() docstring/comment for why the
    # context fed to the LLM carries no "[source]" label at all).
    assert "Sentinel conventions are rarely documented" in result["answer"]

    repeat = tutor.ask(question, provider="stub")
    assert repeat["answer"] == result["answer"]  # deterministic given the same retrieved context

    other = tutor.ask("How does k-means clustering decide the number of clusters?", provider="stub")
    assert (
        other["answer"] != result["answer"]
    )  # genuinely derived from (different) retrieved context


@pytest.mark.rag
# langchain_chroma's default relevance function assumes cosine similarity maps
# into [0, 1] and warns whenever a score falls outside that range. A genuinely
# off-topic question — the exact case this test exercises — legitimately
# scores *negative* (see RELEVANCE_MIN's docstring in tutor.py), so the
# warning is expected here, not a bug to chase; silencing it keeps this
# specific, understood case out of the test-run warning summary.
@pytest.mark.filterwarnings("ignore:Relevance scores must be between 0 and 1:UserWarning")
def test_trap_question_refuses_via_relevance_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tutor, "INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr(tutor, "MANIFEST_PATH", tmp_path / "index" / "manifest.json")
    tutor.index_lessons(reindex=True, corpus_dir=CORPUS)

    result = tutor.ask("What is the capital city of France?", provider="stub")
    assert result == {"answer": tutor.REFUSAL, "sources": []}
