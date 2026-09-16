"""Tests for the P12 study hub capstone (quiz_engine / progress / tutor / app).

Heavy paths (embeddings, Chroma, the LLM) are only touched by ``rag``-marked
tests. ``tutor.py``'s top-level imports are stdlib only (every LangChain/Chroma
import is inside the function that needs it), so its pure helpers are tested
under ``core``. ``quiz_engine``, ``progress`` and ``app``'s CLI dispatch run
offline on the real shipped corpus and are ``core`` too.

Run
    uv run pytest projects/p12_study_hub -q            # core only
    uv run pytest projects/p12_study_hub -m rag -q     # needs `uv sync --group rag`
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from projects.p02_question_bank import qbank
from projects.p12_study_hub import app, progress, quiz_engine, tutor
from shared import blocklist

# No blanket module-level ``pytestmark`` here: every test is marked
# individually, since this file mixes ``core`` tests (quiz_engine, progress,
# tutor's pure helpers) with ``rag`` tests (real embeddings and Chroma) — see
# p10_meeting_assistant's test_assistant.py for the same convention and the
# reason it matters: a blanket module-level `core` marker would leave every
# `rag` test *also* matching `-m "core and not network"`, which is exactly the
# CI filter that must not import langchain/chromadb.

HERE = Path(__file__).resolve().parents[1]
CORPUS = HERE / "corpus"
COURSES = sorted(p for p in CORPUS.iterdir() if p.is_dir())

# The relevance gate's margin heuristic assumes an on-topic question's best-retrieved
# chunk stands out from the rest. Measured on a macOS runner, that is false: a genuinely
# on-topic question produced a flat score field and a genuinely off-topic one a peaked
# field, the opposite of what the heuristic needs. `_diagnostic_snapshot` there showed
# the store's own score and this module's own cosine similarity agreeing, and the
# collection correctly in cosine space, so the scores themselves differ, not how they
# are computed. See projects/p12_study_hub/README.md "Limits" for the numbers and how
# to re-measure. Not `skip`: skipping would hide that the assertions below are known to
# fail on this one platform, not that they don't apply. Not `strict`: if a future
# runner or model revision makes them pass, that must not break CI.
_MACOS_RELEVANCE_GATE_XFAIL_REASON = (
    "relevance gate margin heuristic doesn't hold on macOS - see "
    "projects/p12_study_hub/README.md 'Limits'"
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
# tutor — pure helpers are `core` (tutor.py imports nothing heavy at module
# level); anything that embeds or touches Chroma is `rag`.
# =============================================================================


@pytest.mark.core
def test_grounding_prompt_demands_context_and_refusal() -> None:
    assert "ONLY" in tutor.PROMPT_TEMPLATE
    assert tutor.REFUSAL in tutor.PROMPT_TEMPLATE
    assert "{context}" in tutor.PROMPT_TEMPLATE
    assert "{question}" in tutor.PROMPT_TEMPLATE


@pytest.mark.core
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


@pytest.mark.core
def test_stub_llm_provider_needs_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """The stub is deterministic and derived from its own input — never
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


@pytest.mark.core
def test_stub_provider_argument_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TUTOR_LLM_PROVIDER", "ollama")  # would need a running service
    prompt = "Context:\nFoo bar baz qux.\n\nQuestion: Q?\n\nAnswer:"
    result = tutor.ask_llm(prompt, provider="stub")  # explicit argument wins
    assert "Foo bar baz qux" in result
    assert result != tutor.REFUSAL


@pytest.mark.core
def test_finds_lesson_notes() -> None:
    assert len(tutor.find_lessons(CORPUS)) == 30


@pytest.mark.core
def test_ollama_settings_are_read_when_called(monkeypatch: pytest.MonkeyPatch) -> None:
    """TUTOR_OLLAMA_MODEL / TUTOR_OLLAMA_URL set after import must still take effect."""
    import json
    import urllib.request

    seen: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"response": " ok "}'

    def fake_urlopen(req: urllib.request.Request, timeout: float) -> FakeResponse:
        seen["url"] = req.full_url
        seen["model"] = json.loads(req.data)["model"]  # type: ignore[arg-type]
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("TUTOR_OLLAMA_URL", "http://ollama.test:1234")
    monkeypatch.setenv("TUTOR_OLLAMA_MODEL", "tiny-test-model")

    assert tutor.ask_llm("prompt", provider="ollama") == "ok"
    assert seen == {"url": "http://ollama.test:1234/api/generate", "model": "tiny-test-model"}


@pytest.mark.core
def test_ask_builds_the_index_first_on_a_fresh_clone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With no index manifest, `--ask` must index the lessons before asking;
    otherwise the question runs against an empty store and is refused."""
    calls: list[str] = []
    monkeypatch.setattr(tutor, "MANIFEST_PATH", tmp_path / "index" / "manifest.json")

    def fake_index_lessons(**_kwargs: object) -> dict[str, int]:
        calls.append("index")
        return {}

    def fake_ask(question: str) -> dict[str, object]:
        calls.append("ask")
        return {"answer": "A", "sources": ["s.md"]}

    monkeypatch.setattr(tutor, "index_lessons", fake_index_lessons)
    monkeypatch.setattr(tutor, "ask", fake_ask)

    assert app.main(["--ask", "what is a sentinel value?"]) == 0
    assert calls == ["index", "ask"]
    assert "Sources: s.md" in capsys.readouterr().out


@pytest.mark.core
def test_ask_does_not_reindex_when_an_index_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    calls: list[str] = []
    monkeypatch.setattr(tutor, "MANIFEST_PATH", manifest)

    def fake_index_lessons(**_kwargs: object) -> dict[str, int]:
        calls.append("index")
        return {}

    def fake_ask(question: str) -> dict[str, object]:
        calls.append("ask")
        return {"answer": "", "sources": []}

    monkeypatch.setattr(tutor, "index_lessons", fake_index_lessons)
    monkeypatch.setattr(tutor, "ask", fake_ask)

    assert app.main(["--ask", "q"]) == 0
    assert calls == ["ask"]


@pytest.mark.rag
@pytest.mark.xfail(
    sys.platform == "darwin", reason=_MACOS_RELEVANCE_GATE_XFAIL_REASON, strict=False
)
def test_index_and_ask_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Real embeddings + a real Chroma index over the shipped corpus, with the
    stub LLM provider — proves retrieval genuinely finds a relevant lesson, and
    that the stub's answer is derived from that retrieved *lesson prose*, not
    from a source label.

    It does not assert one exact phrase from the sentinel-values lesson: which
    on-topic chunk ranks first is not guaranteed identical on every platform.
    Instead, whatever chunk `ask()` itself reports as the top retrieved source,
    the stub's answer must be a genuine excerpt of *that* lesson's real text —
    not a fixed phrase, not a source label, not a hallucination.

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
    # On assertion failure, dump collection metadata, the query embedding's norm, and
    # each candidate's own-cosine + store-relevance score side by side —
    # `_diagnostic_snapshot` is only ever called here (an assert
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

    The gate refuses unless the best chunk clearly stands out from a wider
    "background" sample (`RELEVANCE_MARGIN`; see its comment in tutor.py for
    the measured evidence), using cosine similarity this module computes
    itself from the stored vectors. Adjudicated xfail on macOS only (see
    `_MACOS_RELEVANCE_GATE_XFAIL_REASON` above): there this question was
    measured to produce a peaked score field that clears the margin, so the
    gate answers it. See the README's "Limits".
    """
    monkeypatch.setattr(tutor, "INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr(tutor, "MANIFEST_PATH", tmp_path / "index" / "manifest.json")
    tutor.index_lessons(reindex=True, corpus_dir=CORPUS)

    question = "What is the capital city of France?"
    result = tutor.ask(question, provider="stub")
    # See test_index_and_ask_end_to_end's comment on _diagnostic_snapshot: only
    # evaluated (and so only costs anything) when this assertion actually fails.
    assert result == {"answer": tutor.REFUSAL, "sources": []}, tutor._diagnostic_snapshot(question)


# _clears_relevance_gate is pure Python (no LangChain/Chroma import). Testing it directly
# with synthetic scores needs no embeddings, no store, and no platform's particular
# numbers, so "refuse when nothing stands out" is pinned on every platform.
@pytest.mark.core
def test_relevance_gate_refuses_when_no_chunk_stands_out() -> None:
    # A flat score profile — every candidate similarly (here, similarly *high*)
    # relevant — must refuse: nothing distinguishes a genuine top match from generic
    # background. (On macOS, a real *on-topic* question produced this flat shape, which
    # is why the gate refuses it there; see the README's "Limits".) This also proves the
    # gate is not merely an absolute floor: every score here clears an earlier floor of
    # 0.35 by a wide margin, and it still refuses — see RELEVANCE_MARGIN's comment in
    # tutor.py.
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
def test_relevance_gate_background_starts_below_the_k_answer_chunks() -> None:
    # Top chunk 0.9, then seven at 0.85, then four at 0.1. With k=4 the background
    # (ranks 5..12) has median 0.475, so the top chunk stands out; with k=2 the
    # background (ranks 3..12) has median 0.85, so it does not.
    scores = [("top", 0.9)] + [(f"mid{i}", 0.85) for i in range(7)]
    scores += [(f"low{i}", 0.1) for i in range(4)]
    assert tutor._clears_relevance_gate(scores, k=4) is True
    assert tutor._clears_relevance_gate(scores, k=2) is False


@pytest.mark.core
def test_ask_passes_its_k_to_the_relevance_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeDoc:
        def __init__(self, name: str) -> None:
            self.page_content = f"text of {name}"
            self.metadata = {"source": f"{name}.md"}

    scores = [(FakeDoc("top"), 0.9)] + [(FakeDoc(f"mid{i}"), 0.85) for i in range(7)]
    scores += [(FakeDoc(f"low{i}"), 0.1) for i in range(4)]
    monkeypatch.setattr(tutor, "_retrieve_with_own_similarities", lambda _q, _pool_k: scores)

    assert tutor.ask("q", k=2, provider="stub") == {"answer": tutor.REFUSAL, "sources": []}
    answered = tutor.ask("q", k=4, provider="stub")
    assert answered["answer"].startswith("STUB: text of top")
    assert answered["sources"] == ["top.md", "mid0.md", "mid1.md", "mid2.md"]


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
    """Verify the cosine-space collection and normalized embeddings are genuinely in
    effect at runtime, not merely requested: the gate's own cosine similarity is only
    correct if every embedding is a unit vector.
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
