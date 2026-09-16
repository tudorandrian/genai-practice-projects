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

import sys
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

# Adjudicated 2026-09-16 (heavy.yml run
# https://github.com/tudorandrian/genai-practice-projects/actions/runs/35076304269):
# the relevance gate's margin heuristic assumes an on-topic question's best-retrieved
# chunk stands out from the rest; that assumption is empirically false on macOS, where
# a genuinely on-topic question was measured to produce a flat score field and a
# genuinely off-topic one a peaked field — the opposite of what the heuristic needs,
# and not something a threshold, a ratio, or computing the similarity a different way
# can fix (`_diagnostic_snapshot` on that run showed the store's own score and this
# module's own cosine similarity agreeing with each other; the geometry itself is what
# differs there). See projects/p12_study_hub/README.md "Limits" for the measured
# numbers and how to re-measure if this ever needs revisiting. Not `skip`: skipping
# would hide that the assertions below are known to fail on this one platform, not that
# they don't apply. Not `strict`: if a future runner or model revision makes them pass,
# that must not break CI.
_MACOS_RELEVANCE_GATE_XFAIL_REASON = (
    "relevance gate margin heuristic doesn't hold on macOS - see "
    "projects/p12_study_hub/README.md 'Limits' and heavy.yml run "
    "https://github.com/tudorandrian/genai-practice-projects/actions/runs/35076304269"
)


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
@pytest.mark.xfail(
    sys.platform == "darwin", reason=_MACOS_RELEVANCE_GATE_XFAIL_REASON, strict=False
)
def test_index_and_ask_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Real embeddings + a real Chroma index over the shipped corpus, with the
    stub LLM provider — proves retrieval genuinely finds a relevant lesson, and
    (T14-b/T14-c) that the stub's answer is derived from that retrieved
    *lesson prose*, not merely from the bracketed source label ask() used to
    assemble into the context.

    Fix round 2 (A2): this used to assert one exact, hard-coded phrase from
    the sentinel-values lesson appeared in the answer. That pins the
    embedding model's ranking to one specific platform's output, which is not
    guaranteed bit-identical everywhere (it failed on macOS, where a
    different — still on-topic — chunk ranked first). The invariant that
    actually matters, and is checked below instead, is platform-independent
    by construction: whatever chunk `ask()` itself reports as the top
    retrieved source, the stub's answer must be a genuine excerpt of *that*
    lesson's real text — not a fixed phrase, not the bracketed source label,
    not a hallucination.

    Adjudicated xfail on macOS only (see `_MACOS_RELEVANCE_GATE_XFAIL_REASON`
    above): on that platform this exact question was measured to produce a
    flat score field with the correct lesson ranking second, not first, so
    the relevance gate refuses it — a real limitation of the margin
    heuristic, not a bug in how the similarity is computed (own-cosine and
    the store's own score agreed with each other on that run). The assertion
    below is unchanged and still correct; it is expected to keep failing on
    macOS specifically until the underlying limitation is addressed.
    """
    monkeypatch.setattr(tutor, "INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr(tutor, "MANIFEST_PATH", tmp_path / "index" / "manifest.json")
    report = tutor.index_lessons(reindex=True, corpus_dir=CORPUS)
    assert report["total_files"] == 30

    question = "What is a sentinel value in a messy dataset?"
    result = tutor.ask(question, provider="stub")
    # Fix round 5 (T15-g point 2): on assertion failure, dump collection metadata,
    # the query embedding's norm, and each candidate's own-cosine + store-relevance
    # score side by side — `_diagnostic_snapshot` is only ever called here (an assert
    # message expression is evaluated by Python only when the condition is false), so
    # this costs nothing when the assertion passes.
    assert result["sources"], (
        f"a relevant lesson should have been retrieved\n{tutor._diagnostic_snapshot(question)}"
    )
    assert any("sentinel" in s for s in result["sources"])
    assert result["answer"].startswith("STUB:")
    assert result["answer"] != tutor.REFUSAL

    # The stub echoes the start of the top-ranked chunk's own text (see
    # ask_llm's stub branch): confirm that snippet is a real excerpt of the
    # lesson ask() itself names as the top source (result["sources"][0]),
    # not a hard-coded phrase and not the bracketed source label (a
    # 61-character label alone would consume the stub's whole 60-char slice
    # budget — see tutor.py's ask() docstring/comment for why the context
    # fed to the LLM carries no "[source]" label at all).
    echoed = result["answer"].removeprefix("STUB: ")
    top_lesson_text = (CORPUS / result["sources"][0]).read_text(encoding="utf-8")
    assert echoed and echoed in " ".join(top_lesson_text.split())

    repeat = tutor.ask(question, provider="stub")
    assert repeat["answer"] == result["answer"]  # deterministic given the same retrieved context

    other = tutor.ask("How does k-means clustering decide the number of clusters?", provider="stub")
    assert (
        other["answer"] != result["answer"]
    )  # genuinely derived from (different) retrieved context


@pytest.mark.rag
@pytest.mark.xfail(
    sys.platform == "darwin", reason=_MACOS_RELEVANCE_GATE_XFAIL_REASON, strict=False
)
def test_trap_question_refuses_via_relevance_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A genuinely off-topic question must refuse via the relevance gate, never
    reach the LLM.

    Fix round 2 (A2) traced an earlier macOS failure here to Chroma having no
    explicit distance metric (fixed: COLLECTION_METADATA, EMBED_NORMALIZE) and
    re-measured RELEVANCE_MIN as an absolute floor under the corrected
    configuration. heavy.yml still failed on macOS after that — the actual
    retrieval pulled chunks from three unrelated courses and answered from
    one, meaning nothing stood out on that platform even though the absolute
    floor was cleared. Fix round 3's diagnosis (tune the absolute number
    again) would only move the problem to the next platform; fix round 4
    (ruling T15-e) replaced the absolute floor as the primary discriminator
    with a margin: refuse unless the best chunk clearly stands out from a
    wider "background" sample (`RELEVANCE_MARGIN`, see its comment in
    tutor.py for the measured evidence this formulation was chosen over a
    narrower one that didn't separate reliably). That is scale-free by
    construction, not by calibration — it does not depend on what absolute
    numbers a given platform's embedding/HNSW build happens to produce.
    Verified the corrected configuration (cosine space, normalized
    embeddings) is genuinely in effect, not just requested, in
    `test_embeddings_are_actually_normalized_and_cosine_configured` below —
    yet `heavy.yml` still disagreed on macOS after fix round 4, with *opposite*
    verdicts (a genuinely grounded question refused, this trap answered) that
    a mis-scaled margin cannot produce; the retrieved documents for this exact
    question were also identical, same order, to a pre-cosine-config run —
    evidence the configuration was silently not taking effect on that
    platform's Chroma/HNSW build specifically, not merely mis-scaled. Fix
    round 5 (ruling T15-g) stopped trusting the store's own relevance score
    for the gate's decision at all: `ask()` now computes cosine similarity
    itself, as a plain dot product of embedding vectors read back directly
    from the store (`_retrieve_with_own_similarities`) — the same arithmetic
    on every platform, up to float noise, regardless of what distance space
    the store itself is configured (or silently defaults) to use. Not
    verified on macOS directly (no macOS environment available here); a
    residual platform-dependence risk is recorded in this project's README
    "Limits".
    """
    monkeypatch.setattr(tutor, "INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr(tutor, "MANIFEST_PATH", tmp_path / "index" / "manifest.json")
    tutor.index_lessons(reindex=True, corpus_dir=CORPUS)

    question = "What is the capital city of France?"
    result = tutor.ask(question, provider="stub")
    # See test_index_and_ask_end_to_end's comment on _diagnostic_snapshot: only
    # evaluated (and so only costs anything) when this assertion actually fails.
    assert result == {"answer": tutor.REFUSAL, "sources": []}, tutor._diagnostic_snapshot(question)


# _clears_relevance_gate is pure Python (no LangChain/Chroma import) — see tutor.py's
# module docstring: "Every LangChain/Chroma import stays inside the function that needs
# it, so this module ... import[s] cleanly without the rag group installed." Testing it
# directly with synthetic scores needs no embeddings, no store, and no platform's
# particular numbers — exactly what fix round 4 (ruling T15-e, point 3) asked for: a test
# that pins "refuse when nothing stands out" without needing a platform to reproduce it.
@pytest.mark.core
def test_relevance_gate_refuses_when_no_chunk_stands_out() -> None:
    # A flat score profile — every candidate similarly (here, similarly *high*)
    # relevant — must refuse: nothing distinguishes a genuine top match from generic
    # background, which is exactly the shape heavy.yml's macOS failure showed for a
    # real off-topic question (three unrelated courses, nothing standing out). This
    # also proves the gate is not merely re-implementing the old absolute floor: every
    # score here clears the old RELEVANCE_MIN=0.35 by a wide margin, and it still
    # refuses, because "high" was never itself the point once the margin criterion
    # replaced it — see RELEVANCE_MARGIN's comment in tutor.py.
    flat_high = [(f"doc{i}", 0.55) for i in range(tutor.RELEVANCE_POOL_K)]
    assert tutor._clears_relevance_gate(flat_high) is False

    flat_low = [(f"doc{i}", 0.12) for i in range(tutor.RELEVANCE_POOL_K)]
    assert tutor._clears_relevance_gate(flat_low) is False


@pytest.mark.core
def test_relevance_gate_accepts_when_the_top_chunk_stands_out() -> None:
    # One clearly-relevant chunk well above a low, flat background: this is the
    # on-topic shape and must be accepted.
    peaked = [("top", 0.9)] + [(f"bg{i}", 0.1) for i in range(tutor.RELEVANCE_POOL_K - 1)]
    assert tutor._clears_relevance_gate(peaked) is True


@pytest.mark.core
def test_relevance_gate_still_enforces_the_absolute_sanity_floor() -> None:
    # A huge margin cannot rescue a top score that is not even a real match — the
    # absolute floor (RELEVANCE_MIN) is still checked first, as a sanity bound.
    below_floor = [("top", 0.01)] + [(f"bg{i}", -0.5) for i in range(tutor.RELEVANCE_POOL_K - 1)]
    assert tutor._clears_relevance_gate(below_floor) is False


@pytest.mark.core
def test_relevance_gate_passes_with_too_few_candidates_to_estimate_a_background() -> None:
    # A corpus smaller than TOP_K + 1 chunks can't supply a background sample; the
    # gate falls back to the absolute floor alone rather than refusing everything.
    tiny = [("only", 0.9), ("second", 0.8)]
    assert tutor._clears_relevance_gate(tiny) is True


@pytest.mark.rag
def test_embeddings_are_actually_normalized_and_cosine_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fix round 4 (ruling T15-e, point 1): verify the corrected configuration is
    genuinely in effect at runtime, not merely requested — the whole premise of fix
    round 2's fix depends on it actually reaching the collection and the encoder on
    every platform, and this is the one thing this repository's test suite can check
    for itself (macOS's own configuration is not something this machine can inspect).
    """
    monkeypatch.setattr(tutor, "INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr(tutor, "MANIFEST_PATH", tmp_path / "index" / "manifest.json")
    tutor.index_lessons(reindex=True, corpus_dir=CORPUS)

    store = tutor._store()
    assert store._collection.metadata == {"hnsw:space": "cosine"}

    embeddings = tutor._embeddings()
    assert embeddings.encode_kwargs.get("normalize_embeddings") is True

    vector = embeddings.embed_query("what is a sentinel value?")
    norm = sum(x * x for x in vector) ** 0.5
    assert abs(norm - 1.0) < 1e-3, f"embedding vector is not unit-norm (norm={norm})"
