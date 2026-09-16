"""rag_chatbot.py - a Retrieval-Augmented Generation chatbot over your own documents.

Project P11 - the repository's first ``rag``-tier project. Answers questions about
*your* private documents (PDF, Markdown, text) rather than from the model's general
knowledge. The pipeline:

    load -> split (200/20, keep source+page) -> embed (MiniLM) -> Chroma (on disk)
    question -> embed -> top-k similar chunks -> grounded prompt -> LLM -> answer + sources

The answer is generated ONLY from the retrieved context, and every answer cites the
files/pages it came from - the standard technique for reducing hallucinations and
connecting an LLM to private data.

Key seams
  * ``create_llm()`` isolates the LLM provider: ``ollama`` (default, a local model
    served by Ollama, no API key), ``openai`` (key from env), ``stub`` (offline, used
    by the tests and by ``demo()``). No API key ever lives in the code - every
    key/URL/model name is read from the environment at call time, not cached at
    import time, so tests and operators can override any of them without reloading
    the module.
  * The Chroma index persists on disk under ``chroma_index/``: a second run loads it
    without re-embedding; ``--reindex`` rebuilds it from scratch.

Run
    uv run p11-rag-chatbot --reindex                     # (re)build the index from data/
    uv run p11-rag-chatbot -q "How many vacation days?"  # single question
    uv run p11-rag-chatbot                                # interactive Q&A loop
    uv run p11-rag-chatbot --ui                            # Gradio web UI
    uv run p11-rag-chatbot --demo                          # offline-ish demo, writes output/
    uv run pytest projects/p11_rag_chatbot -q               # core tests, no rag group needed
    uv run pytest projects/p11_rag_chatbot -m rag -q        # needs the rag group installed

Dependencies  langchain-core, langchain-text-splitters, langchain-huggingface,
langchain-chroma, langchain-ollama, langchain-openai, chromadb, sentence-transformers,
pypdf, fpdf2 - the ``rag`` dependency group. Every one of those imports stays inside a
function so this module (and its ``core``-marked tests) stay importable without the group installed.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from shared.demo import DemoResult

log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
PERSIST_DIR = HERE / "chroma_index"
OUT_DIR = HERE / "output"
EVAL_QUESTIONS_PATH = HERE / "eval_questions.md"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Small enough that the synthetic corpus splits into several chunks per document
# (9 total, vs. TOP_K=3) so retrieval must genuinely discriminate between
# candidates instead of trivially returning the entire corpus for every query -
# see test_evaluation_questions_retrieve_the_expected_source's rank assertion.
# 250 chars also yields 9 chunks but occasionally merges two adjacent, unrelated
# handbook facts (e.g. "vacation days" + "remote work") into one chunk that then
# out-competes the actually-relevant chunk for a *different* nearby question (e.g.
# "WiFi password") - verified empirically across all twelve real eval questions
# before picking 200/20 over 250/30.
CHUNK_SIZE = 200
CHUNK_OVERLAP = 20
TOP_K = 3

DEFAULT_OLLAMA_MODEL = "qwen2.5:1.5b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.0  # deterministic generation

# Grounding prompt: role + "use ONLY the context" + explicit refusal behaviour +
# cite the source. This is what keeps answers anchored to the documents.
REFUSAL = "I cannot find that information in the documents."
PROMPT_TEMPLATE = (
    "You are a helpful assistant answering questions about a set of private "
    "documents. Use ONLY the context below to answer. If the answer is not "
    f'contained in the context, reply exactly: "{REFUSAL}" and nothing else. '
    "Do not use outside knowledge.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

DEMO_QUESTION_COUNT = 3


class EmptyCorpusError(ValueError):
    """Raised when there is nothing in ``data/`` to index."""


# =============================================================================
# Ingestion
# =============================================================================


def load_documents(folder: str | Path | None = None) -> list[Any]:
    """Load every supported document in ``folder`` (default: ``DATA_DIR``).

    .pdf via pypdf (one Document per page, with a 0-based ``page`` in metadata),
    .md/.txt as one Document per file. Every Document carries its ``source`` path.
    Returns a list of LangChain Documents. Read directly rather than through
    ``langchain-community``'s loaders, which produced the same text and chunks for
    this corpus and would add that package (and its dependency tree) for two loops.
    """
    from langchain_core.documents import Document  # lazy import, see module docstring
    from pypdf import PdfReader  # lazy import, see module docstring

    folder = Path(folder) if folder is not None else DATA_DIR
    documents: list[Any] = []
    for path in sorted(folder.iterdir()):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            for page_number, page in enumerate(PdfReader(str(path)).pages):
                documents.append(
                    Document(
                        page_content=page.extract_text().strip(),
                        metadata={"source": str(path), "page": page_number},
                    )
                )
        elif suffix in (".md", ".txt"):
            documents.append(
                Document(
                    page_content=path.read_text(encoding="utf-8"), metadata={"source": str(path)}
                )
            )
    return documents


def split_documents(documents: list[Any]) -> list[Any]:
    """Split documents into overlapping chunks, preserving provenance metadata
    (source file + page) on every chunk."""
    from langchain_text_splitters import (  # lazy import, see module docstring
        RecursiveCharacterTextSplitter,
    )

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(documents)


# =============================================================================
# Index (persistent Chroma)
# =============================================================================


def _embeddings() -> Any:
    from langchain_huggingface import (  # lazy import, see module docstring
        HuggingFaceEmbeddings,
    )

    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def _load_and_split(folder: str | Path | None = None) -> list[Any]:
    """``load_documents`` + ``split_documents`` in one step - shared by
    ``build_index`` and ``demo()`` so the latter doesn't load and split the corpus
    twice just to report the ``chunks`` figure."""
    return split_documents(load_documents(folder))


def _rebuild_index(chunks: list[Any], persist_dir: str | Path | None = None) -> Any:
    """(Re)embed ``chunks`` into a fresh persistent Chroma index, replacing
    whatever was at ``persist_dir`` (default: ``PERSIST_DIR``, read at call time).
    Raises ``EmptyCorpusError`` if ``chunks`` is empty."""
    if not chunks:
        # Chroma would otherwise fail with an unexplained "Expected Embeddings to be
        # non-empty" error. data/ is empty on a fresh clone until the corpus is written.
        raise EmptyCorpusError(
            "no documents to index in data/ - run `uv run p11-rag-chatbot --demo` or "
            "`uv run python -m projects.p11_rag_chatbot.synthetic_docs` first"
        )
    from langchain_chroma import Chroma  # lazy import, see module docstring

    persist_dir = Path(persist_dir) if persist_dir is not None else PERSIST_DIR
    log.info("build_index: embedding %d chunks...", len(chunks))
    if persist_dir.exists():
        import shutil  # lazy import, see module docstring

        shutil.rmtree(persist_dir)
    return Chroma.from_documents(chunks, _embeddings(), persist_directory=str(persist_dir))


def build_index(
    reindex: bool = False,
    folder: str | Path | None = None,
    persist_dir: str | Path | None = None,
) -> Any:
    """Build or load the persistent Chroma index.

    If the index already exists and ``reindex`` is False, it is loaded from disk
    with no re-embedding. Otherwise the documents are (re)loaded, split and embedded.
    Returns the Chroma vector store. ``folder``/``persist_dir`` default to the
    module-level ``DATA_DIR``/``PERSIST_DIR`` *read at call time*, so tests can
    monkeypatch those module attributes instead of passing arguments.
    """
    from langchain_chroma import Chroma  # lazy import, see module docstring

    folder = Path(folder) if folder is not None else DATA_DIR
    persist_dir = Path(persist_dir) if persist_dir is not None else PERSIST_DIR

    if persist_dir.exists() and any(persist_dir.iterdir()) and not reindex:
        log.info("build_index: loading existing index from %s (no re-embedding)", persist_dir.name)
        return Chroma(persist_directory=str(persist_dir), embedding_function=_embeddings())

    return _rebuild_index(_load_and_split(folder), persist_dir)


# =============================================================================
# LLM provider seam + RAG chain
# =============================================================================


def create_llm(provider: str | None = None) -> Any:
    """Return a LangChain LLM. Provider chosen by ``provider``, falling back to
    ``RAG_LLM_PROVIDER`` (default ``ollama``):

        ollama (default) - a local model served by Ollama, no API key
        openai           - ChatOpenAI (reads OPENAI_API_KEY from env)
        stub             - a deterministic offline LLM, no model at all (used by
                           the tests and demo())

    Only this function's body changes to swap providers; no API key ever lives in
    the code. Model/URL names are read from the environment at call time, not
    cached, so tests can override them without reloading the module.
    """
    provider = (provider or os.environ.get("RAG_LLM_PROVIDER", "ollama")).lower()

    if provider == "stub":
        from langchain_core.language_models.llms import (  # lazy import, see module docstring
            LLM,
        )

        def _call(
            self: Any,
            prompt: str,
            stop: list[str] | None = None,
            run_manager: Any = None,
            **kwargs: Any,
        ) -> str:
            """A deterministic offline answer *derived from its own prompt*
            rather than a single fixed string: it echoes a snippet of the
            retrieved context (``"STUB: " + context[:60]``). A grounded question
            therefore visibly gets a grounded-looking answer instead of every
            question - grounded or not - reading as a refusal; ``REFUSAL`` is reserved for a real
            provider's grounding behaviour (see the retrieval-layer trap test,
            which checks this offline without one)."""
            context = prompt.split("Context:\n", 1)[-1].split("\n\nQuestion:", 1)[0]
            # Collapse whitespace (the retrieved context can span several lines)
            # so the answer is always a single line - readable in
            # output/session.txt and consistent with _show()'s one-line-per-field
            # console output.
            return f"STUB: {' '.join(context.split())[:60]}"

        # Built with type(name, bases, namespace) rather than a `class _StubLLM
        # (LLM):` statement: LLM is an optional dependency resolved at call time
        # (only present with the rag group installed), so without the rag group
        # `ignore_missing_imports` makes it resolve to `Any`, and `strict`
        # refuses to let a *class statement* subclass `Any` - even through an
        # explicit `Any`-typed alias, which mypy still treats as subclassing
        # `Any` (verified empirically; a `# type: ignore` here would be a real
        # suppression in one supported environment and an unused one in the
        # other). `type()` is a function call, not a class statement, so that
        # check doesn't apply to it, and - because `type.__new__` resolves the
        # most-derived metaclass among its bases before constructing - it
        # correctly invokes LLM's Pydantic metaclass at runtime exactly as a
        # `class` statement would, so the resulting object behaves identically.
        namespace = {"_llm_type": property(lambda self: "stub"), "_call": _call}
        stub_llm_cls = type("_StubLLM", (LLM,), namespace)
        return stub_llm_cls()

    if provider == "openai":
        from langchain_openai import ChatOpenAI  # lazy import, see module docstring

        model = os.environ.get("RAG_OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        return ChatOpenAI(model=model, temperature=LLM_TEMPERATURE)

    from langchain_ollama import ChatOllama  # lazy import, see module docstring

    model = os.environ.get("RAG_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    url = os.environ.get("RAG_OLLAMA_URL", DEFAULT_OLLAMA_URL)
    return ChatOllama(model=model, base_url=url, temperature=LLM_TEMPERATURE)


class GroundedQA:
    """Retrieve the top chunks, stuff them into the grounding prompt, make one LLM call.

    This is what LangChain's legacy ``RetrievalQA`` "stuff" chain did, rebuilt on
    ``langchain-core`` alone: ``RetrievalQA`` lived in the ``langchain`` 0.3 line,
    which has published advisories fixed only in 1.x, where the class moved to
    ``langchain-classic``. The contract is unchanged:
    ``invoke({"query": q}) -> {"result": str, "source_documents": list}``, with chunks
    joined by a blank line, as the "stuff" chain joined them.
    """

    def __init__(self, retriever: Any, answer_chain: Any) -> None:
        self.retriever = retriever
        self._answer_chain = answer_chain

    def invoke(self, inputs: dict[str, str]) -> dict[str, Any]:
        question = inputs["query"]
        docs = self.retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)
        answer = self._answer_chain.invoke({"context": context, "question": question})
        return {"result": str(answer), "source_documents": docs}


def build_chain(vector_store: Any, llm: Any = None) -> GroundedQA:
    """Wire retrieval, the grounding prompt and the LLM, returning answer + sources."""
    from langchain_core.output_parsers import (  # lazy import, see module docstring
        StrOutputParser,
    )
    from langchain_core.prompts import (  # lazy import, see module docstring
        PromptTemplate,
    )

    llm = llm if llm is not None else create_llm()
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
    # StrOutputParser accepts both a plain LLM's str and a chat model's message.
    return GroundedQA(
        retriever=vector_store.as_retriever(search_kwargs={"k": TOP_K}),
        answer_chain=prompt | llm | StrOutputParser(),
    )


def format_sources(source_documents: list[Any]) -> list[str]:
    """Turn retrieved documents into unique 'file (p.N)' citation strings."""
    sources: list[str] = []
    for doc in source_documents:
        name = Path(doc.metadata.get("source", "?")).name
        page = doc.metadata.get("page")
        label = f"{name} (p.{page + 1})" if isinstance(page, int) else name
        if label not in sources:
            sources.append(label)
    return sources


def ask(qa: Any, question: str) -> dict[str, Any]:
    """Ask one question; return ``{"answer": str, "sources": list[str]}``.

    A refusal cites no sources: retrieval always returns the nearest chunks, and listing
    them under "I cannot find that information" would present them as its evidence.
    """
    result = qa.invoke({"query": question})
    answer = result["result"].strip()
    if answer.strip('"') == REFUSAL:
        return {"answer": REFUSAL, "sources": []}
    return {"answer": answer, "sources": format_sources(result.get("source_documents", []))}


# =============================================================================
# Evaluation set
# =============================================================================


def load_eval_questions(path: str | Path | None = None) -> list[tuple[str, str]]:
    """Parse ``eval_questions.md``'s ``| # | Question | Expected answer | Source |``
    table into ``(question, expected_source)`` pairs.

    Skips the header and separator rows. The table's final row is a **trap**
    question with no expected source (``Source`` column reads ``(absent)``) - it
    is deliberately excluded here, since a row with no expected source cannot
    assert a retrieval hit; see ``test_trap_question_retrieves_no_revenue_related_chunk``
    for the assertion the trap row exists to prove.
    """
    path = Path(path) if path is not None else EVAL_QUESTIONS_PATH
    pairs: list[tuple[str, str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 4:
            continue
        number, question, _expected_answer, source = cells
        if number.startswith("#") or set(number) <= {"-"}:  # header / separator row
            continue
        if source == "(absent)":  # the trap row - no source to assert a retrieval hit
            continue
        pairs.append((question.strip("*"), source.strip("*")))
    return pairs


# =============================================================================
# Session log
# =============================================================================


def _write_session(question: str, result: dict[str, Any], session_path: Path | None = None) -> None:
    """Append one Q/A entry. Interactive runs log to ``output/runs/session.txt``
    (gitignored), so questions about private documents never reach the committed
    ``output/session.txt``, which only ``demo()`` writes."""
    if session_path is None:
        session_path = OUT_DIR / "runs" / "session.txt"
    session_path.parent.mkdir(parents=True, exist_ok=True)
    entry = (
        f"Q: {question}\nA: {result['answer']}\n"
        f"Sources: {', '.join(result['sources']) or '(none)'}\n"
    )
    # A blank line separates entries, but the file itself must end with exactly one
    # trailing newline (not a trailing blank line) to satisfy the repo's
    # end-of-file-fixer pre-commit hook, so the separator is a *prefix* on every
    # entry but the first rather than a suffix on every entry.
    prefix = "\n" if session_path.exists() and session_path.stat().st_size > 0 else ""
    with session_path.open("a", encoding="utf-8") as handle:
        handle.write(prefix + entry)


def _show(question: str, result: dict[str, Any]) -> None:
    print(f"\nQ: {question}\nA: {result['answer']}")
    print(f"Sources: {', '.join(result['sources']) or '(none)'}")
    _write_session(question, result)


def build_ui(qa: Any) -> Any:
    """Gradio chat UI over the RAG chain."""
    import gradio as gr  # lazy import, see module docstring

    def _respond(message: str, _history: Any) -> str:
        result = ask(qa, message)
        sources = result["sources"]
        suffix = ("\n\n_Sources: " + ", ".join(sources) + "_") if sources else ""
        _write_session(message, result)
        return str(result["answer"]) + suffix

    return gr.ChatInterface(
        fn=_respond,
        title="RAG chatbot - ask your documents",
        description="Answers ONLY from the indexed documents (Chroma + your chosen LLM "
        "provider), with cited sources.",
        analytics_enabled=False,
    )


# =============================================================================
# demo()
# =============================================================================


def demo() -> DemoResult:
    """Write the synthetic corpus, build the index, ask three questions through the
    ``stub`` provider, and write ``output/session.txt`` and ``output/metrics.txt``.

    Tier ``rag``: needs LangChain + Chroma + the MiniLM embedding model, so it never
    runs in CI and is skipped by ``uv run demo``/``uv run demo --models`` - only
    ``uv run demo --all`` runs it.

    The LLM provider is force-pinned to ``stub`` (not "whichever provider is
    configured") so the committed proof does not depend on whether Ollama happens
    to be running on this machine - see the README for running the chain against a
    real provider. Retrieval itself is real (the MiniLM embeddings and the Chroma
    index), and the stub's answer echoes the retrieved context rather than a fixed
    string, so both ``sources`` and ``answer`` in ``output/session.txt``
    demonstrate genuine grounding.

    The index is always rebuilt (``reindex=True``) so the reported ``chunks`` figure
    and the retrieved sources never depend on a stale ``chroma_index/`` directory
    left over from a previous manual run. Any existing ``output/session.txt`` is
    removed first, so the three demo questions are the file's only content - two
    consecutive ``--demo`` runs are byte-identical.
    """
    start = time.perf_counter()

    from projects.p11_rag_chatbot import synthetic_docs

    synthetic_docs.write_all(DATA_DIR)
    chunks = _load_and_split(DATA_DIR)
    store = _rebuild_index(chunks, PERSIST_DIR)
    qa = build_chain(store, llm=create_llm(provider="stub"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session_path = OUT_DIR / "session.txt"
    session_path.unlink(missing_ok=True)

    questions = [question for question, _source in load_eval_questions()[:DEMO_QUESTION_COUNT]]
    for question in questions:
        _write_session(question, ask(qa, question), session_path)

    metrics_lines = [f"chunks={len(chunks)}", "provider=stub", f"questions={len(questions)}"]
    (OUT_DIR / "metrics.txt").write_text("\n".join(metrics_lines) + "\n", encoding="utf-8")

    seconds = time.perf_counter() - start
    figures = {"chunks": str(len(chunks)), "provider": "stub", "questions": str(len(questions))}
    log.info("demo: indexed %d chunks, asked %d questions", len(chunks), len(questions))
    return DemoResult("p11-rag-chatbot", "ok", figures, seconds=round(seconds, 2))


# =============================================================================
# CLI
# =============================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument("--reindex", action="store_true", help="Rebuild the index from data/.")
    p.add_argument("-q", "--question", help="Ask a single question and exit.")
    p.add_argument("--ui", action="store_true", help="Launch the Gradio web UI.")
    p.add_argument("--port", type=int, default=7860, help="Port for the Gradio UI.")
    p.add_argument(
        "--host", default="127.0.0.1", help="Bind address for the Gradio UI (default: 127.0.0.1)."
    )
    p.add_argument("--demo", action="store_true", help="Run the demo (needs the rag group).")
    p.add_argument("--verbose", action="store_true", help="Log at INFO level.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI: run the demo, index the documents and answer one question / a loop /
    serve the UI."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s"
    )

    if args.demo:
        result = demo()
        print(f"p11-rag-chatbot: {result.status}")
        for key, value in result.figures.items():
            print(f"  {key}: {value}")
        print("  wrote: output/")
        print(f"  seconds: {result.seconds:.2f}")
        return 0 if result.status == "ok" else 1

    try:
        vector_store = build_index(reindex=args.reindex)
    except EmptyCorpusError as exc:
        print(f"p11-rag-chatbot: {exc}")
        return 1
    qa = build_chain(vector_store)

    if args.ui:
        build_ui(qa).launch(server_name=args.host, server_port=args.port)
        return 0
    if args.question:
        _show(args.question, ask(qa, args.question))
        return 0

    print("Ask your documents (empty line to exit):")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        _show(question, ask(qa, question))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
