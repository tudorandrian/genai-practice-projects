# P12 — Study Hub Assistant (capstone)

## What it does

The capstone project: three earlier techniques from this repository, wired
together over one small, original corpus. `quiz_engine.py` builds a question
bank from Markdown lessons (reusing P02's `qbank.build_bank`, unmodified),
runs a scored quiz session; `progress.py` scans each course's status table
and reports quiz success rates; `tutor.py` is a RAG chatbot (P11's pattern)
that answers questions grounded only in the lessons, with citations.
`app.py` serves all three through one Gradio interface, or headless in a CLI.

The three modules are coupled only through files — `output/history.json`
(quiz_engine writes it, progress.py reads it), `corpus/` (all three read it),
`index/` (tutor.py's persistent Chroma store) — never by calling into each
other's internals:

```
qbank.build_bank(course_dir)         # P02, unmodified, per course
  -> quiz_engine.build_bank_from_corpus(corpus_dir) -> question-bank.json
  -> quiz_engine.load_bank/build_session/grade/save_session -> history.json

progress.scan_statuses(corpus_dir)   # course README "| # | Lesson | Status |"
  -> progress.aggregate -> pivot table
  -> progress.success_rate(history.json)  # consumes quiz_engine's output

tutor.find_lessons(corpus_dir) -> tutor.index_lessons -> Chroma (index/)
  -> tutor.ask(question) -> {"answer", "sources"}
```

**The corpus is original, not ported.** `projects/p12_study_hub/corpus/`
holds three invented courses — `data-cleaning-basics`, `classic-ml-in-
practice`, `llm-apps-from-scratch` — ten lessons each, three practice
questions per lesson (Multiple Choice / True-False / Open-Ended), 90
questions total. See "Datasets and licences" and
[ADR 0003](../../docs/decisions/0003-p12-synthetic-corpus.md) for why.

## Run

```bash
uv sync --group rag                                  # langchain, chromadb, sentence-transformers, ...
uv run p12-study-hub --demo                          # quiz + progress + tutor, writes output/
uv run p12-study-hub --quiz --n 5 --seed 42           # quiz in the console
uv run p12-study-hub --dashboard                      # progress scan + chart
uv run p12-study-hub --ask "what is a sentinel value?" # RAG tutor
uv run p12-study-hub --reindex                         # rebuild the lesson index
uv run p12-study-hub                                   # Gradio UI (3 tabs), http://127.0.0.1:7860
uv run p12-study-hub --host 0.0.0.0                     # UI: listen on every interface
uv run pytest projects/p12_study_hub -q                  # core tests, no rag group needed
uv run pytest projects/p12_study_hub -m rag -q            # needs the rag group installed
```

`--verbose` logs at `INFO`; by default only the summary lines print. `--demo`
builds the merged question bank from `corpus/`, runs a seeded 5-question
session answered correctly, scans progress and saves `output/progress.png`,
(re)indexes the lessons and asks one question with the `stub` provider
(force-pinned — see "Design notes"), and writes `output/tutor_session.txt`
and `output/metrics.txt`.

To run the tutor against a real LLM instead of the stub (`--demo` always
pins `stub`):

```bash
# Ollama running locally, e.g. `ollama pull qwen2.5:1.5b`
export TUTOR_LLM_PROVIDER=ollama        # TUTOR_OLLAMA_MODEL/_URL override the defaults
uv run p12-study-hub --ask "How does chunk overlap help retrieval?"
```

`TUTOR_LLM_PROVIDER=openai` (with `OPENAI_API_KEY`, `TUTOR_OPENAI_MODEL`)
works the same way. `.env.example` lists every key P12 reads; no key is
required for the default `ollama` provider or for `stub` itself.

## Example output

`uv run p12-study-hub --demo`:

```
p12-study-hub: ok
  questions: 5
  session_score: 5/5
  lessons_indexed: 30
  provider: stub
  wrote: output/
  seconds: 40.03
```

`seconds` varies run to run and is reported only on the console and in the
returned `DemoResult`, never in a committed file — running `--demo` twice in
a row leaves `git status` clean for the tracked files. `output/metrics.txt`
(committed, deterministic): `questions=5`, `session_score=5/5`,
`lessons_indexed=30`, `provider=stub`.

`output/tutor_session.txt` (committed, deterministic — real retrieval
against the real corpus, `stub` LLM):

```
Q: What does the missing-value lesson recommend for a skewed numeric column?
A: STUB: For numeric columns you intend to keep, filling gaps with th
Sources: data-cleaning-basics/lessons/06-missing-value-imputation.md
```

The `stub` answer is a truncated echo of the top-ranked retrieved chunk's
own lesson text (`"STUB: " + context[:60]`, in the spirit of
`shared.testing.stub_llm` and P11's `rag_chatbot.py`) — deterministic and
visibly derived from the actual retrieved prose, not a real generated
answer; see "Design notes" for why `demo()` pins `stub` and for the context
shape that keeps this slice on real lesson content, not a citation label.
`output/progress.png`/`output/history.json` regenerate every run, never committed.

## Design notes

- **`app.py` composes, it does not reimplement.** Every quiz/progress/tutor
  behaviour lives in its own module; `app.py`'s Gradio tabs and CLI dispatch
  are a thin layer calling `quiz_engine`/`progress`/`tutor` by their real
  names (an earlier draft of the source project imported them as bare
  `A`/`B`/`C` — this port names them properly throughout).
- **`quiz_engine.build_bank_from_corpus` is the one addition over P02.** It
  calls `qbank.build_bank(course_dir)` once per course — P02 itself is
  untouched — and concatenates `questions`/`units`, summing
  `question_counts`, into one merged bank. `load_bank` then keeps only
  `multiple_choice`/`true_false` (`CLOSED_TYPES`) and tags each question's
  `module` from `qid.split("/")[0]`, i.e. the course slug.
- **No "lab README" branch.** The source `progress_scan.py` recognised two
  status formats: a `## Status:` line (private-repo lab READMEs) and a
  Markdown table (course READMEs). This corpus only ever uses the table
  format, so `progress.scan_statuses` implements table scanning only.
- **`demo()` force-pins `stub`, not "whichever provider is configured".**
  Same reasoning as P10/P11 — the answer must not depend on whether Ollama
  happens to be running. Retrieval is real (real MiniLM embeddings, a real
  Chroma index over the real corpus), and the stub's answer echoes the
  retrieved context rather than a fixed string (T14-b, matching P11's
  `rag_chatbot.py` stub), so both `Sources` and the answer text demonstrate
  genuine grounding — an earlier version always returned the fixed
  `REFUSAL` string, a literal port of the source's `ask_llm` that made a
  correctly-grounded answer indistinguishable from a refusal.
- **The context fed to the LLM carries no `"[source]"` label (T14-c).** An
  earlier `ask()` prefixed every chunk with `f"[{source}]\n"` before joining
  it into `context`; for any 60+ character corpus-relative path that alone
  consumed the stub's whole `context[:60]` slice, never reaching the
  lesson's own prose (P11's chain never has this problem: `page_content`
  only, citations out-of-band). `context` is `page_content` only now;
  `format_sources(docs)` still reads `doc.metadata` directly, unaffected.
  Correct for a real provider too, since `PROMPT_TEMPLATE` never asked it to
  cite from such a label.
- **The relevance gate refuses before ever calling the LLM.** `tutor.ask`
  checks the best retrieved chunk's similarity score against
  `RELEVANCE_MIN`; an off-topic question refuses immediately with empty
  `sources`, which is what makes a trap question ("What is the capital of
  France?") reliably refuse regardless of which LLM provider is behind it.
- **No private Chroma internals.** The source reached into
  `store._collection.delete(where=...)`, a LangChain/Chroma implementation
  detail not covered by its public API contract. `index_lessons`'s
  incremental path instead uses the public `store.get(where=...)` to find
  matching ids and `store.delete(ids=...)` to remove them — same effect,
  no reliance on an attribute that could disappear on a dependency bump.
- **Output contract.** Library functions never `print`; they log through
  `logging.getLogger(__name__)`. `main()` prints at most six summary lines
  for `--demo`; the interactive `--quiz`/`--dashboard`/`--ask` subcommands
  print their own richer console output, matching P10/P11's convention for
  direct (non-demo) CLI modes.

## Limits

- **The `stub` provider never actually generates an answer** — it echoes a
  snippet of the retrieved context (matching P11's `rag_chatbot.py` stub), to
  prove the pipeline deterministically without a real model, not to
  demonstrate generation quality; run against `ollama`/`openai` (see "Run")
  for an actual generated answer.
- **`ollama` and `openai` each need their own setup.** CI never exercises
  either — the tests force `stub` throughout, and `demo()` force-pins it too.
- **The corpus is intentionally small** (30 lessons, 90 questions) — enough
  to exercise every code path (multi-course filtering, incremental
  reindexing, a genuine relevance gate) without a large, slow index.
- **No authentication or rate limiting on the Gradio server.** `build_ui()`
  is a local demo UI; the CLI binds `127.0.0.1` by default — pass `--host
  0.0.0.0` to listen on every interface.

## Datasets and licences

There is no external dataset. Every file under `projects/p12_study_hub/
corpus/` (30 lessons, 3 course READMEs) is original prose and original
practice questions, written specifically for this repository (see
[ADR 0003](../../docs/decisions/0003-p12-synthetic-corpus.md) for why the
source capstone's corpus could not be reused). Two fictional scenarios
recur across the lessons — "Wheel & Way" (bike-share) and "Fenwick Labs" /
"Pebble" (documentation assistant) — neither refers to anything real.

The embedding model is
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
(Apache-2.0), downloaded on first use and cached locally; this project only
performs inference against it. `index/` (the persisted Chroma store) and
`output/history.json`/`output/progress.png`/`output/question-bank.json` are
all regenerated artefacts, never tracked by git.

## Courses drawn on

- 4 Python for Data Science, AI & Development
- 8 Machine Learning with Python
- 15 Fundamentals of AI Agents Using RAG and LangChain
- 16 Project: Generative AI Applications with RAG and LangChain
