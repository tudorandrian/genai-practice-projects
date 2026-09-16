"""tutor.py — a RAG tutor over the P12 synthetic corpus's lessons.

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
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
CORPUS_DIR = HERE / "corpus"
INDEX_DIR = HERE / "index"
MANIFEST_PATH = INDEX_DIR / "manifest.json"
OUT_DIR = HERE / "output"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4
# Relevance gate: if the best retrieved chunk scores below this, the lessons
# don't cover the question, so we refuse WITHOUT calling the LLM — a robust
# guard a small model would otherwise ignore for well-known facts.
RELEVANCE_MIN = 0.15

OLLAMA_MODEL = os.environ.get("TUTOR_OLLAMA_MODEL", "qwen2.5:1.5b")
OLLAMA_URL = os.environ.get("TUTOR_OLLAMA_URL", "http://localhost:11434")
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
    from langchain_text_splitters import (  # noqa: PLC0415 — lazy import, see module docstring
        RecursiveCharacterTextSplitter,
    )

    rel = str(path.relative_to(corpus_dir)).replace("\\", "/")
    text = path.read_text(encoding="utf-8", errors="replace")
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.create_documents([text], metadatas=[{"source": rel}]), rel


def _embeddings() -> Any:
    from langchain_huggingface import (  # noqa: PLC0415 — lazy import, see module docstring
        HuggingFaceEmbeddings,
    )

    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def _store(embeddings: Any = None) -> Any:
    from langchain_chroma import Chroma  # noqa: PLC0415 — lazy import, see module docstring

    return Chroma(persist_directory=str(INDEX_DIR), embedding_function=embeddings or _embeddings())


def index_lessons(reindex: bool = False, corpus_dir: str | Path = CORPUS_DIR) -> dict[str, int]:
    """(Re)index the lessons into persistent Chroma; incremental unless ``reindex``.

    Returns a report dict ``{indexed, changed, removed, total_files}``.
    """
    from langchain_chroma import Chroma  # noqa: PLC0415 — lazy import, see module docstring

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
            import shutil  # noqa: PLC0415 — lazy import, see module docstring

            shutil.rmtree(INDEX_DIR)
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        all_chunks = []
        for path in lessons:
            chunks, _ = _chunks_for(path, corpus_dir)
            all_chunks.extend(chunks)
        Chroma.from_documents(all_chunks, embeddings, persist_directory=str(INDEX_DIR))
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

    The stub echoes a snippet of its own prompt's retrieved context (in the
    spirit of ``shared.testing.stub_llm`` and P11's ``rag_chatbot.py``; T14-b)
    rather than a fixed string — this keeps a grounded stub answer visibly
    distinct from a refusal. ``REFUSAL`` stays reserved for ``ask()``'s
    relevance gate (fired *before* this function is ever called) and for a
    real provider's own grounding behaviour.
    """
    provider = (provider or os.environ.get("TUTOR_LLM_PROVIDER", "ollama")).lower()

    if provider == "stub":
        context = prompt.split("Context:\n", 1)[-1].split("\n\nQuestion:", 1)[0]
        return f"STUB: {' '.join(context.split())[:60]}"

    if provider == "openai":
        from openai import OpenAI  # noqa: PLC0415 — lazy import, see module docstring

        client = OpenAI()  # OPENAI_API_KEY from env
        model = os.environ.get("TUTOR_OPENAI_MODEL", "gpt-4o-mini")
        resp = client.chat.completions.create(
            model=model, temperature=0, messages=[{"role": "user", "content": prompt}]
        )
        return (resp.choices[0].message.content or "").strip()

    import urllib.request  # noqa: PLC0415 — lazy import, see module docstring

    payload = json.dumps(
        {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": LLM_MAX_TOKENS},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate", data=payload, headers={"Content-Type": "application/json"}
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


def ask(question: str, k: int = TOP_K, provider: str | None = None) -> dict[str, Any]:
    """Retrieve top-k lesson chunks and answer grounded in them, citing sources.

    If the best retrieved chunk is below the relevance gate, refuse immediately
    (the lessons don't cover it) — this is what makes an off-topic 'trap'
    question reliably refuse instead of the small model answering from general
    knowledge. Returns ``{"answer": str, "sources": list[str]}``.
    """
    scored = _store().similarity_search_with_relevance_scores(question, k=k)
    if not scored or max(s for _, s in scored) < RELEVANCE_MIN:
        return {"answer": REFUSAL, "sources": []}
    docs = [d for d, _ in scored]
    # Lesson text only — no "[source]\n" label in the context fed to the LLM.
    # PROMPT_TEMPLATE never asks the model to cite sources inline from such a
    # label; citations are attached out-of-band via format_sources(docs)
    # below, which reads doc.metadata directly and does not depend on this
    # string at all (mirroring P11's rag_chatbot.py: RetrievalQA's "stuff"
    # chain feeds the model page_content only, and sources come from
    # return_source_documents=True, never from in-context labels).
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
