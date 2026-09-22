"""Tests for rag_chatbot.py (Project P11).

Ingestion (``load_documents``/``split_documents``) and the LLM provider seam
(``create_llm``) both import LangChain packages even in their simplest paths, so
those tests are marked ``rag`` even though they need no network or downloaded
embeddings - no ``core`` test may import langchain, chromadb or
sentence-transformers. The grounding prompt, the source formatter and ``ask()`` are
pure Python exercised against fakes and stay ``core``. The full index + real
embeddings + retrieval path (the evaluation set, its negative control, and the
trap question) needs the ``rag`` group installed, and asserts on the TOP-ranked
source (or, for the trap question, on the retrieved chunks directly) rather than
mere presence in the result - see ``test_evaluation_questions_retrieve_the_expected_source``'s
docstring for why that distinction matters.

Run
    uv run pytest projects/p11_rag_chatbot -q            # core only
    uv run pytest projects/p11_rag_chatbot -m rag -q     # needs `uv sync --group rag`
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from projects.p11_rag_chatbot import rag_chatbot, synthetic_docs


class FakeDoc:
    def __init__(self, source: str, page: int | None = None) -> None:
        self.metadata: dict[str, Any] = {"source": source}
        if page is not None:
            self.metadata["page"] = page


class FakeQA:
    def __init__(self, result: str, docs: list[Any]) -> None:
        self._result, self._docs = result, docs

    def invoke(self, _inputs: dict[str, str]) -> dict[str, Any]:
        return {"result": self._result, "source_documents": self._docs}


@pytest.fixture
def synthetic_data_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "data"
    synthetic_docs.write_all(folder)
    return folder


# =============================================================================
# synthetic_docs.write_all
# =============================================================================


@pytest.mark.rag
def test_write_all_creates_three_documents(tmp_path: Path) -> None:
    paths = synthetic_docs.write_all(tmp_path / "data")
    assert set(paths) == {"acme_handbook.pdf", "engineering_notes.md", "support_faq.txt"}
    assert all(p.exists() for p in paths.values())


@pytest.mark.rag
def test_write_all_refuses_to_overwrite_an_existing_file(tmp_path: Path) -> None:
    """data/ is where a user's own documents live (and Git ignores it), so a
    colliding name must never be silently replaced."""
    folder = tmp_path / "data"
    folder.mkdir()
    mine = folder / "support_faq.txt"
    mine.write_text("my own notes", encoding="utf-8")
    with pytest.raises(FileExistsError, match="support_faq.txt"):
        synthetic_docs.write_all(folder)
    assert mine.read_text(encoding="utf-8") == "my own notes"
    assert not (folder / "acme_handbook.pdf").exists()  # nothing written at all


@pytest.mark.rag
def test_write_all_overwrites_only_when_asked(tmp_path: Path) -> None:
    folder = tmp_path / "data"
    folder.mkdir()
    (folder / "support_faq.txt").write_text("old", encoding="utf-8")
    paths = synthetic_docs.write_all(folder, overwrite=True)
    assert paths["support_faq.txt"].read_text(encoding="utf-8") == synthetic_docs.FAQ


# =============================================================================
# Ingestion - real LangChain loaders/splitters (rag)
# =============================================================================


@pytest.mark.rag
def test_loads_pdf_md_and_txt(synthetic_data_dir: Path) -> None:
    docs = rag_chatbot.load_documents(synthetic_data_dir)
    sources = {os.path.basename(d.metadata.get("source", "")) for d in docs}
    assert "acme_handbook.pdf" in sources
    assert "engineering_notes.md" in sources
    assert "support_faq.txt" in sources


@pytest.mark.rag
def test_split_preserves_source_metadata(synthetic_data_dir: Path) -> None:
    docs = rag_chatbot.load_documents(synthetic_data_dir)
    chunks = rag_chatbot.split_documents(docs)
    assert len(chunks) > 0
    assert all("source" in c.metadata for c in chunks)


# =============================================================================
# Grounding prompt (pure string; core)
# =============================================================================


@pytest.mark.core
def test_grounding_prompt_demands_context_only_and_refusal() -> None:
    assert "ONLY" in rag_chatbot.PROMPT_TEMPLATE
    assert rag_chatbot.REFUSAL in rag_chatbot.PROMPT_TEMPLATE
    assert "{context}" in rag_chatbot.PROMPT_TEMPLATE
    assert "{question}" in rag_chatbot.PROMPT_TEMPLATE


# =============================================================================
# Source formatting (pure Python; core)
# =============================================================================


@pytest.mark.core
def test_formats_file_and_page_and_dedupes() -> None:
    docs = [
        FakeDoc("/x/acme_handbook.pdf", page=0),
        FakeDoc("/x/acme_handbook.pdf", page=0),  # duplicate -> collapsed
        FakeDoc("/x/notes.md"),
    ]
    sources = rag_chatbot.format_sources(docs)
    assert sources == ["acme_handbook.pdf (p.1)", "notes.md"]


# =============================================================================
# LLM provider seam - imports langchain_core (rag)
# =============================================================================


@pytest.mark.rag
def test_stub_provider_needs_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """The stub is deterministic and derived from its own input - never a
    single fixed string regardless of the prompt, and never the exact ``REFUSAL``
    string (that's reserved for a real provider's grounding behaviour)."""
    monkeypatch.setenv("RAG_LLM_PROVIDER", "stub")
    llm = rag_chatbot.create_llm()
    prompt = "Context:\nACME was founded in 2014.\n\nQuestion: When?\n\nAnswer:"
    first = llm.invoke(prompt)
    second = llm.invoke(prompt)
    assert first == second  # deterministic given the same input
    assert "ACME was founded in 2014" in first  # echoes the retrieved context
    assert first != rag_chatbot.REFUSAL
    assert llm.invoke("a different prompt") != first  # genuinely input-derived


@pytest.mark.rag
def test_create_llm_provider_argument_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_LLM_PROVIDER", "ollama")  # would need a running service
    llm = rag_chatbot.create_llm(provider="stub")  # explicit argument wins
    assert llm.invoke("anything") == "STUB: anything"


# =============================================================================
# ask() - pure logic against a fake QA object (core)
# =============================================================================


@pytest.mark.core
def test_a_refusal_cites_no_sources() -> None:
    # Retrieval always returns chunks, but citing them under "the documents do not say"
    # would present unrelated files as the evidence for a refusal (seen in the UI).
    docs = [FakeDoc("/d/acme_handbook.pdf", page=0)]
    for answer in (rag_chatbot.REFUSAL, f' "{rag_chatbot.REFUSAL}" '):
        result = rag_chatbot.ask(FakeQA(answer, docs), "What is the annual revenue?")
        assert result == {"answer": rag_chatbot.REFUSAL, "sources": []}


@pytest.mark.core
def test_ask_returns_answer_and_sources() -> None:
    qa = FakeQA("  25 days.  ", [FakeDoc("/d/acme_handbook.pdf", page=0)])
    result = rag_chatbot.ask(qa, "How many vacation days?")
    assert result["answer"] == "25 days."
    assert result["sources"] == ["acme_handbook.pdf (p.1)"]


@pytest.mark.core
def test_rebuild_index_explains_an_empty_corpus(tmp_path: Path) -> None:
    """A fresh clone's data/ holds only .gitkeep; indexing nothing must say how to
    create the corpus instead of failing inside Chroma."""
    with pytest.raises(rag_chatbot.EmptyCorpusError, match="--demo"):
        rag_chatbot._rebuild_index([], tmp_path / "index")


@pytest.mark.core
def test_cli_reports_an_empty_corpus_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def empty_index(reindex: bool = False) -> Any:
        raise rag_chatbot.EmptyCorpusError("no documents to index in data/ - run --demo")

    monkeypatch.setattr(rag_chatbot, "build_index", empty_index)
    assert rag_chatbot.main(["-q", "How many vacation days?"]) == 1
    assert "no documents to index" in capsys.readouterr().out


# =============================================================================
# load_eval_questions() - pure file parsing (core)
# =============================================================================


@pytest.mark.core
def test_load_eval_questions_excludes_trap_row_and_header() -> None:
    pairs = rag_chatbot.load_eval_questions()
    assert len(pairs) == 12
    assert ("How many paid vacation days do employees get?", "acme_handbook.pdf") in pairs
    questions = [question for question, _source in pairs]
    assert not any("annual revenue" in question for question in questions)


# =============================================================================
# Evaluation set as a retrieval test (rag) - proves grounding end to end
# =============================================================================


@pytest.fixture
def eval_qa(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """A real build_index()/build_chain() over the synthetic corpus, with the
    stub LLM - shared by the retrieval tests below."""
    monkeypatch.setattr(rag_chatbot, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(rag_chatbot, "PERSIST_DIR", tmp_path / "index")
    synthetic_docs.write_all(tmp_path / "data")
    store = rag_chatbot.build_index(reindex=True)
    return rag_chatbot.build_chain(store, llm=rag_chatbot.create_llm(provider="stub"))


@pytest.mark.rag
def test_evaluation_questions_retrieve_the_expected_source(eval_qa: Any) -> None:
    """Each grounded question's expected source must be the TOP-ranked source, not
    merely present somewhere in the returned list. With ``CHUNK_SIZE=200`` the
    corpus splits into 9 chunks for ``TOP_K=3`` (see ``rag_chatbot.py``'s comment
    next to ``CHUNK_SIZE``), so retrieval only ever returns a third of the corpus
    and must genuinely discriminate - a presence-only assertion would still pass
    if every query returned the whole corpus."""
    for question, expected_source in rag_chatbot.load_eval_questions():
        result = rag_chatbot.ask(eval_qa, question)
        top_source = result["sources"][0]
        assert expected_source in top_source, (question, result["sources"])


@pytest.mark.rag
def test_wrong_document_is_not_ranked_first(eval_qa: Any) -> None:
    """Negative control for the rank assertion above: a question grounded in the
    FAQ must not surface the handbook or the engineering notes as its top-ranked
    source. This is what makes the assertion above capable of failing."""
    result = rag_chatbot.ask(eval_qa, "Where is parking available?")
    top_source = result["sources"][0]
    assert "support_faq.txt" in top_source
    assert "acme_handbook.pdf" not in top_source
    assert "engineering_notes.md" not in top_source


@pytest.mark.rag
def test_trap_question_retrieves_no_revenue_related_chunk(eval_qa: Any) -> None:
    """The evaluation set's trap question has no answer anywhere in the corpus -
    asserted at the retrieval layer directly. This deliberately does NOT go
    through the stub's answer text: the stub now echoes whatever context WAS
    retrieved rather than judging whether an answer exists, so an
    answer-text assertion here would be true by construction regardless of the
    question (the corpus contains no chunk about revenue, so this is checkable
    offline without a real LLM; the end-to-end refusal from a real provider is
    ``test_real_llm_refuses_the_trap_question``, ``llm``-marked, at the end of this file)."""
    retrieved = eval_qa.retriever.invoke("What is ACME Robotics' annual revenue?")
    assert not any("revenue" in doc.page_content.lower() for doc in retrieved)


# =============================================================================
# demo() - deterministic across repeated runs (rag)
# =============================================================================


@pytest.mark.rag
def test_demo_writes_deterministic_session_and_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Each call gets its own PERSIST_DIR: reusing one directory for two in-process
    # rebuilds hits a Windows-only chromadb/hnswlib quirk (rmtree fails because the
    # first client's memory-mapped index file is still open) that never arises
    # across two separate `--demo` *processes* - the real proof command - since a
    # process exit releases the OS-level file lock. Separate directories test the
    # actual property in question (repeated builds of the same corpus are
    # byte-identical) without depending on that unrelated OS behaviour.
    monkeypatch.setattr(rag_chatbot, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(rag_chatbot, "OUT_DIR", tmp_path / "output")

    monkeypatch.setattr(rag_chatbot, "PERSIST_DIR", tmp_path / "index1")
    first = rag_chatbot.demo()
    session_1 = (tmp_path / "output" / "session.txt").read_text(encoding="utf-8")
    metrics_1 = (tmp_path / "output" / "metrics.txt").read_text(encoding="utf-8")

    monkeypatch.setattr(rag_chatbot, "PERSIST_DIR", tmp_path / "index2")
    second = rag_chatbot.demo()
    session_2 = (tmp_path / "output" / "session.txt").read_text(encoding="utf-8")
    metrics_2 = (tmp_path / "output" / "metrics.txt").read_text(encoding="utf-8")

    assert first.status == "ok"
    assert second.status == "ok"
    assert session_1 == session_2
    assert metrics_1 == metrics_2
    assert "provider=stub" in metrics_1
    assert "questions=3" in metrics_1


# =============================================================================
# Real LLM (llm) - Qwen2.5 1.5B served by Ollama; skipped when Ollama is not running
# =============================================================================


@pytest.fixture
def real_llm_qa(ollama: tuple[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """``eval_qa`` with the real ``ollama`` provider instead of the stub."""
    url, model = ollama
    monkeypatch.setenv("RAG_OLLAMA_URL", url)
    monkeypatch.setenv("RAG_OLLAMA_MODEL", model)
    monkeypatch.setattr(rag_chatbot, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(rag_chatbot, "PERSIST_DIR", tmp_path / "index")
    synthetic_docs.write_all(tmp_path / "data")
    store = rag_chatbot.build_index(reindex=True)
    return rag_chatbot.build_chain(store, llm=rag_chatbot.create_llm(provider="ollama"))


@pytest.mark.llm
def test_real_llm_answers_a_grounded_question_from_the_documents(real_llm_qa: Any) -> None:
    result = rag_chatbot.ask(real_llm_qa, "How many paid vacation days do employees get?")
    assert "25" in result["answer"]
    assert result["answer"] != rag_chatbot.REFUSAL
    assert "acme_handbook.pdf" in result["sources"][0]


@pytest.mark.llm
def test_real_llm_refuses_the_trap_question(real_llm_qa: Any) -> None:
    """End to end, the refusal the retrieval-layer trap test above cannot check offline:
    the corpus has no revenue figure, and the grounding prompt makes the model say so
    instead of inventing one."""
    result = rag_chatbot.ask(real_llm_qa, "What is ACME Robotics' annual revenue?")
    assert result["answer"].strip().strip('"') == rag_chatbot.REFUSAL
