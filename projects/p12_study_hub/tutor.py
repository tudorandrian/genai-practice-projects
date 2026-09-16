"""tutor.py - a RAG tutor over the P12 synthetic corpus's lessons.

Indexes every lesson note (``corpus/**/lessons/*.md``) into a persistent Chroma
store and answers questions using ONLY the retrieved passages, citing the
source lesson for each answer. Reindexing is incremental: a manifest of file
hashes detects new / changed / removed lessons and updates only those.

The LLM lives behind one seam, ``ask_llm(prompt) -> str``, dispatched by
``TUTOR_LLM_PROVIDER`` (or an explicit ``provider`` argument, which wins):
``ollama`` (default, local via Ollama, no key), ``openai`` (key from env),
``stub`` (offline, used by the tests and by ``demo()``). Embeddings are local
(all-MiniLM-L6-v2), so lesson content never leaves the machine.

Every LangChain/Chroma import stays inside the function that needs it, so this
module (and its ``core``-marked tests) import cleanly without the ``rag`` group
installed.

CLI: ``python tutor.py --reindex`` | ``python tutor.py --ask "..."``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import statistics
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
CORPUS_DIR = HERE / "corpus"
INDEX_DIR = HERE / "index"
MANIFEST_PATH = INDEX_DIR / "manifest.json"
OUT_DIR = HERE / "output"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Explicit cosine space + normalized embeddings. EMBED_NORMALIZE keeps every embedding a
# unit vector (checked at runtime by
# test_embeddings_are_actually_normalized_and_cosine_configured), so a dot product of two
# embeddings *is* their cosine similarity. COLLECTION_METADATA asks Chroma for cosine
# space, which shapes which candidates its nearest-neighbour search returns. The
# relevance gate then scores those candidates itself, from the raw vectors
# (_retrieve_with_own_similarities), so its numbers do not depend on a vector store's
# relevance-score formula. Where the gate does not work (macOS), see the README's
# "Limits".
EMBED_NORMALIZE = True
COLLECTION_METADATA = {"hnsw:space": "cosine"}
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4
# How many candidates to pull just to estimate the relevance gate's "background" level
# below - deliberately wider than TOP_K; see RELEVANCE_MARGIN's comment for why.
RELEVANCE_POOL_K = 12
# Relevance gate, absolute sanity floor only: the best retrieved chunk must clear this,
# but it is set far below any realistic on-topic score. The discriminator is
# RELEVANCE_MARGIN below. An absolute floor alone did not transfer between platforms: an
# earlier floor of 0.35, tuned on Windows, let an off-topic question through on macOS.
RELEVANCE_MIN = 0.05
# Relevance gate: refuse unless the best-retrieved chunk stands out from the
# "background" (the typical similarity among lower-ranked candidates) by at least this
# margin. The premise is that an on-topic question's best chunk is clearly more similar
# to the question than generic corpus content is, and an off-topic question's is not.
# The margin is invariant to shifting all scores by a constant, but not to rescaling
# them, so it is still a measured number. The premise holds on Windows and Linux and
# fails on macOS; see the README's "Limits".
#
# Why the background is ranks TOP_K+1..RELEVANCE_POOL_K, not a narrow top-TOP_K margin
# (e.g. top1 vs the median of just the top 4): measured on this machine, a narrow margin
# does not separate reliably - a question that strongly matches one lesson often pulls
# several near-duplicate chunks *from that same lesson* into the top 4 (chunking with
# overlap does this on purpose), so top1 doesn't stand out much above rank 2-4 even
# though the question is genuinely on-topic. Measured top1-vs-top4-median margins:
# on-topic 0.0234-0.2693, off-topic 0.0058-0.0549 - these overlap, so that formulation
# was tried and rejected, not assumed to work. Comparing top1 against the median of
# ranks 5..12 instead (a wider, and so more representative, sample of "generic corpus
# content" that excludes the near-duplicate on-topic chunks) separates cleanly: measured
# on-topic margin 0.2018-0.3783, off-topic margin 0.0160-0.0748 - no overlap, with clear
# air on both sides of the value below. 6 on-topic and 10 clearly off-topic questions,
# on Windows, real embeddings over the real corpus, cosine space + normalized embeddings,
# scored with this module's own cosine similarity (the store's relevance score gave the
# same numbers to four decimal places).
RELEVANCE_MARGIN = 0.13

# Defaults only: TUTOR_OLLAMA_MODEL / TUTOR_OLLAMA_URL are read when ask_llm runs, not at
# import, so tests and operators can override them without reloading the module.
DEFAULT_OLLAMA_MODEL = "qwen2.5:1.5b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
LLM_MAX_TOKENS = 400

REFUSAL = "The lessons do not cover this topic."
PROMPT_TEMPLATE = (
    "You are a study tutor. Answer the question using ONLY the lesson context "
    "below. If the answer is not in the context, reply exactly: "
    f'"{REFUSAL}" Do not use outside knowledge. Keep the answer concise.\n\n'
    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)


# =============================================================================
# Lessons -> chunks
# =============================================================================


def find_lessons(corpus_dir: str | Path = CORPUS_DIR) -> list[Path]:
    """Every lesson-note file under ``corpus_dir/<course>/lessons/*.md``."""
    return sorted(Path(corpus_dir).rglob("lessons/*.md"))


def _hash(text: str) -> str:
    return hashlib.sha1(
        text.encode("utf-8")
    ).hexdigest()  # content fingerprint only, not security-sensitive


def _chunks_for(path: Path, corpus_dir: Path) -> tuple[list[Any], str]:
    """Split one lesson into chunks carrying its corpus-relative path as 'source'."""
    from langchain_text_splitters import (  # lazy import, see module docstring
        RecursiveCharacterTextSplitter,
    )

    rel = str(path.relative_to(corpus_dir)).replace("\\", "/")
    text = path.read_text(encoding="utf-8", errors="replace")
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.create_documents([text], metadatas=[{"source": rel}]), rel


def _embeddings() -> Any:
    from langchain_huggingface import (  # lazy import, see module docstring
        HuggingFaceEmbeddings,
    )

    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL, encode_kwargs={"normalize_embeddings": EMBED_NORMALIZE}
    )


def _store(embeddings: Any = None) -> Any:
    from langchain_chroma import Chroma  # lazy import, see module docstring

    return Chroma(
        persist_directory=str(INDEX_DIR),
        embedding_function=embeddings or _embeddings(),
        collection_metadata=COLLECTION_METADATA,
    )


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity as a plain dot product - exact (up to float noise) for two
    unit-normalized vectors, which EMBED_NORMALIZE guarantees both `a` and `b` are
    here. Deliberately does not go through any vector store's own relevance-score
    formula; see COLLECTION_METADATA's comment above.
    """
    return sum(x * y for x, y in zip(a, b, strict=True))


def _retrieve_with_own_similarities(question: str, pool_k: int) -> list[tuple[Any, float]]:
    """Retrieve `pool_k` candidates from the persisted store and score each one's
    relevance to `question` with cosine similarity computed directly from raw
    embedding vectors read back from the collection - never from
    `similarity_search_with_relevance_scores`, whose output depends on the collection's
    distance-space configuration and the store's score formula. Returns
    `(Document, similarity)` pairs sorted by that similarity, descending - the ranking
    used for both the relevance gate and, if it passes, the answer's context.

    Chroma's own nearest-neighbour *selection* (which `pool_k` candidates come back at
    all) still goes through the collection's configured distance space; only the
    scoring and ranking from here on is this function's own arithmetic.
    """
    from langchain_core.documents import Document  # lazy import, see module docstring

    embeddings = _embeddings()
    query_vector = embeddings.embed_query(question)
    raw = _store(embeddings)._collection.query(
        query_embeddings=[query_vector],
        n_results=pool_k,
        include=["documents", "metadatas", "embeddings"],
    )
    texts = raw.get("documents") or [[]]
    metas = raw.get("metadatas") or [[]]
    vectors = raw.get("embeddings") or [[]]
    scored = [
        (
            Document(page_content=text, metadata=dict(meta or {})),
            _cosine(query_vector, list(vector)),
        )
        for text, meta, vector in zip(texts[0], metas[0], vectors[0], strict=True)
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored


def _diagnostic_snapshot(question: str, k: int = RELEVANCE_POOL_K) -> str:
    """A human-readable dump of exactly what the relevance gate sees for `question`:
    the collection's own metadata as read back from the live store, the measured norm
    of the query embedding, and for both this module's own cosine similarity (what the
    gate is based on) and the store's own relevance score (what it is not based on) -
    printed side by side so a disagreement between them is visible directly, not
    inferred.

    Built to be called from an assert's message expression (`assert cond,
    _diagnostic_snapshot(question)`), which Python only evaluates when `cond` is
    false - so this costs nothing when a test passes, and pytest's failure report
    shows the returned string in full, so a failure on a platform nobody can inspect
    directly states what that platform computed instead of leaving it to be inferred.
    Returns a string rather than printing directly: this module's own convention is
    that library functions never print, only the caller (here, a test's assert)
    decides what becomes visible.

    Deliberately defensive: this function runs inside a failing assert's message
    expression, on the one platform this repository cannot otherwise inspect. If any
    individual piece (store setup, the store's own relevance score, ...) raises there,
    letting that exception propagate would replace the original assertion failure with
    an unrelated one and destroy exactly the evidence this function exists to capture.
    Every external call is caught separately and reported inline as an "ERROR ..." line
    instead - this always returns a snapshot, even a partial one, never raises itself.
    """
    lines = [f"--- relevance diagnostic: {question!r} ---"]

    try:
        embeddings = _embeddings()
        store = _store(embeddings)
    except Exception as exc:
        lines.append(f"ERROR setting up embeddings/store: {exc.__class__.__name__}: {exc}")
        return "\n".join(lines)

    try:
        lines.append(
            f"collection metadata (read back from the live store): {store._collection.metadata}"
        )
    except Exception as exc:
        lines.append(f"ERROR reading collection metadata: {exc.__class__.__name__}: {exc}")

    try:
        norm = sum(x * x for x in embeddings.embed_query(question)) ** 0.5
        lines.append(f"query embedding norm: {norm:.6f}")
    except Exception as exc:
        lines.append(f"ERROR embedding the query: {exc.__class__.__name__}: {exc}")

    own: list[tuple[Any, float]] = []
    try:
        own = _retrieve_with_own_similarities(question, k)
    except Exception as exc:
        lines.append(f"ERROR computing own cosine similarities: {exc.__class__.__name__}: {exc}")

    store_scored: list[tuple[Any, float]] = []
    try:
        store_scored = store.similarity_search_with_relevance_scores(question, k=k)
    except Exception as exc:
        lines.append(
            f"ERROR getting the store's own relevance score: {exc.__class__.__name__}: {exc}"
        )

    if not own and not store_scored:
        return "\n".join(lines)

    try:
        lines.append(f"{'source':<70} {'own_cosine':>10} {'store_relevance':>16}")
        seen: set[str] = set()
        for doc, sim in own:
            match = next((s for d, s in store_scored if d.page_content == doc.page_content), None)
            seen.add(doc.page_content)
            store_col = "-" if match is None else f"{match:.4f}"
            lines.append(f"{doc.metadata.get('source', '?'):<70} {sim:>10.4f} {store_col:>16}")
        for doc, score in store_scored:
            if doc.page_content in seen:
                continue
            lines.append(
                f"{doc.metadata.get('source', '?'):<70} {'(not in own top-k)':>10} {score:>16.4f}"
            )
    except Exception as exc:
        lines.append(f"ERROR formatting the comparison table: {exc.__class__.__name__}: {exc}")

    return "\n".join(lines)


def index_lessons(reindex: bool = False, corpus_dir: str | Path = CORPUS_DIR) -> dict[str, int]:
    """(Re)index the lessons into persistent Chroma; incremental unless ``reindex``.

    Returns a report dict ``{indexed, changed, removed, total_files}``.
    """
    from langchain_chroma import Chroma  # lazy import, see module docstring

    corpus_dir = Path(corpus_dir)
    lessons = find_lessons(corpus_dir)
    current = {
        str(p.relative_to(corpus_dir)).replace("\\", "/"): _hash(
            p.read_text(encoding="utf-8", errors="replace")
        )
        for p in lessons
    }
    manifest = (
        {}
        if reindex or not MANIFEST_PATH.exists()
        else json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    )
    embeddings = _embeddings()

    if reindex or not manifest:
        if INDEX_DIR.exists():
            import shutil  # lazy import, see module docstring

            shutil.rmtree(INDEX_DIR)
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        all_chunks = []
        for path in lessons:
            chunks, _ = _chunks_for(path, corpus_dir)
            all_chunks.extend(chunks)
        Chroma.from_documents(
            all_chunks,
            embeddings,
            persist_directory=str(INDEX_DIR),
            collection_metadata=COLLECTION_METADATA,
        )
        report = {"indexed": len(lessons), "changed": 0, "removed": 0, "total_files": len(lessons)}
    else:  # incremental
        new = [
            p for p in lessons if str(p.relative_to(corpus_dir)).replace("\\", "/") not in manifest
        ]
        changed = [
            p
            for p in lessons
            if (rel := str(p.relative_to(corpus_dir)).replace("\\", "/")) in manifest
            and manifest[rel] != current[rel]
        ]
        removed = [rel for rel in manifest if rel not in current]
        store = _store(embeddings)
        for rel in removed + [str(p.relative_to(corpus_dir)).replace("\\", "/") for p in changed]:
            # Public API only: unlike the ported source, this does not reach into
            # the private `_collection` attribute (a LangChain/Chroma internal
            # that could break on a dependency bump). `.get(where=...)` and
            # `.delete(ids=...)` are both part of langchain_chroma's public
            # Chroma vectorstore surface.
            matches = store.get(where={"source": rel})
            ids = matches.get("ids", [])
            if ids:
                store.delete(ids=ids)
        added = []
        for path in new + changed:
            chunks, _ = _chunks_for(path, corpus_dir)
            added.extend(chunks)
        if added:
            store.add_documents(added)
        report = {
            "indexed": len(new),
            "changed": len(changed),
            "removed": len(removed),
            "total_files": len(lessons),
        }

    MANIFEST_PATH.write_text(json.dumps(current, ensure_ascii=False), encoding="utf-8")
    log.info("index_lessons: %s", report)
    return report


# =============================================================================
# LLM seam + ask
# =============================================================================


def ask_llm(prompt: str, provider: str | None = None) -> str:
    """The LLM provider seam. ``provider`` wins over ``TUTOR_LLM_PROVIDER``
    (default ``ollama``): ``ollama`` | ``openai`` | ``stub``.

    The stub echoes a snippet of its own prompt's retrieved context (the same
    shape as P11's ``rag_chatbot.py`` stub) rather than a fixed string - this
    keeps a grounded stub answer visibly distinct from a refusal. ``REFUSAL``
    stays reserved for ``ask()``'s relevance gate (fired *before* this function
    is ever called) and for a real provider's own grounding behaviour.
    """
    provider = (provider or os.environ.get("TUTOR_LLM_PROVIDER", "ollama")).lower()

    if provider == "stub":
        context = prompt.split("Context:\n", 1)[-1].split("\n\nQuestion:", 1)[0]
        return f"STUB: {' '.join(context.split())[:60]}"

    if provider == "openai":
        from openai import OpenAI  # lazy import, see module docstring

        client = OpenAI()  # OPENAI_API_KEY from env
        model = os.environ.get("TUTOR_OPENAI_MODEL", "gpt-4o-mini")
        resp = client.chat.completions.create(
            model=model, temperature=0, messages=[{"role": "user", "content": prompt}]
        )
        return (resp.choices[0].message.content or "").strip()

    import urllib.request  # lazy import, see module docstring

    model = os.environ.get("TUTOR_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    url = os.environ.get("TUTOR_OLLAMA_URL", DEFAULT_OLLAMA_URL)
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": LLM_MAX_TOKENS},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{url}/api/generate", data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=180) as resp:  # local Ollama call only
        return json.loads(resp.read().decode("utf-8")).get("response", "").strip()


def format_sources(docs: list[Any]) -> list[str]:
    """Unique source lesson paths from retrieved documents."""
    sources: list[str] = []
    for doc in docs:
        src = doc.metadata.get("source", "?")
        if src not in sources:
            sources.append(src)
    return sources


def _clears_relevance_gate(scored: list[tuple[Any, float]], k: int = TOP_K) -> bool:
    """True if the best-retrieved chunk stands out from the rest enough to trust an
    answer grounded in it - see RELEVANCE_MARGIN's comment above for the measured
    reasoning behind this specific formulation (an absolute floor alone was tried and
    did not transfer between platforms; a narrow top-k margin was also tried and
    measured not to separate reliably). The background is every candidate ranked
    below the ``k`` an answer would use.
    """
    if not scored:
        return False
    scores = sorted((s for _, s in scored), reverse=True)
    top1 = scores[0]
    if top1 < RELEVANCE_MIN:
        return False
    background = scores[k:]
    if not background:
        # Too few candidates to estimate a background (a corpus smaller than
        # k + 1 chunks) - the absolute floor above is all there is to check.
        return True
    return top1 - statistics.median(background) >= RELEVANCE_MARGIN


def ask(question: str, k: int = TOP_K, provider: str | None = None) -> dict[str, Any]:
    """Retrieve top-k lesson chunks and answer grounded in them, citing sources.

    If the best retrieved chunk doesn't clear the relevance gate (`_clears_relevance_gate`),
    refuse immediately (the lessons don't cover it) - this is what makes an off-topic
    'trap' question refuse instead of the small model answering from general
    knowledge, except on macOS; see the README's "Limits". Returns
    ``{"answer": str, "sources": list[str]}``.
    """
    # Retrieve a wider pool than `k` purely to give the relevance gate a representative
    # "background" to compare the top chunk against (RELEVANCE_POOL_K); the answer
    # itself still only ever uses the first `k` of those. Own-computed cosine
    # similarity, not the store's relevance score - see
    # _retrieve_with_own_similarities's docstring.
    scored = _retrieve_with_own_similarities(question, max(k, RELEVANCE_POOL_K))
    if not _clears_relevance_gate(scored, k):
        return {"answer": REFUSAL, "sources": []}
    docs = [d for d, _ in scored[:k]]
    # Lesson text only - no "[source]\n" label in the context fed to the LLM.
    # PROMPT_TEMPLATE never asks the model to cite sources inline from such a
    # label; citations are attached out-of-band via format_sources(docs)
    # below, which reads doc.metadata directly and does not depend on this
    # string at all (mirroring P11's rag_chatbot.py: its GroundedQA
    # feeds the model page_content only, and sources come from the retrieved
    # documents' metadata, never from in-context labels).
    context = "\n\n".join(d.page_content for d in docs)
    answer = ask_llm(PROMPT_TEMPLATE.format(context=context, question=question), provider=provider)
    return {"answer": answer.strip(), "sources": format_sources(docs)}


def main(argv: list[str] | None = None) -> int:
    """CLI: ``--reindex`` to (re)build the lesson index, ``--ask`` to query the tutor."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", maxsplit=1)[0])
    parser.add_argument("--reindex", action="store_true", help="rebuild the lesson index")
    parser.add_argument("--ask", help="ask the tutor a question")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s"
    )

    if args.reindex or not MANIFEST_PATH.exists():
        report = index_lessons(reindex=args.reindex)
        print(f"[index] {report}")
    if args.ask:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        result = ask(args.ask)
        print(f"\nQ: {args.ask}\nA: {result['answer']}\nSources: {', '.join(result['sources'])}")
        with (OUT_DIR / "tutor_session.txt").open("a", encoding="utf-8") as handle:
            handle.write(
                f"Q: {args.ask}\nA: {result['answer']}\nSources: {', '.join(result['sources'])}\n\n"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
