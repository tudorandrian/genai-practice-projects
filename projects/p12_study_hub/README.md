# P12 - Study Hub Assistant (capstone)

## What it does

The capstone project: three earlier techniques from this repository, wired
together over one small, original corpus. `quiz_engine.py` builds a question
bank from Markdown lessons (reusing P02's `qbank.build_bank`, unmodified),
runs a scored quiz session; `progress.py` scans each course's status table
and reports quiz success rates; `tutor.py` is a RAG chatbot (P11's pattern)
that answers questions grounded only in the lessons, with citations.
`app.py` serves all three through one Gradio interface, or headless in a CLI.

The three modules are coupled only through files - `output/history.json`
(quiz_engine writes it, progress.py reads it), `corpus/` (all three read it),
`index/` (tutor.py's persistent Chroma store) - never by calling into each
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
holds three invented courses - `data-cleaning-basics`, `classic-ml-in-
practice`, `llm-apps-from-scratch` - ten lessons each, three practice
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
(force-pinned - see "Design notes"), and writes `output/tutor_session.txt`
and `output/metrics.txt`. `--ask` and the UI's Tutor tab build the lesson
index first if it does not exist yet (the first question on a fresh clone
takes longer).

To run the tutor against a real LLM instead of the stub (`--demo` always
pins `stub`):

```bash
docker compose --profile llm up -d        # Qwen2.5 1.5B served by Ollama on localhost
export TUTOR_LLM_PROVIDER=ollama        # TUTOR_OLLAMA_MODEL/_URL override the defaults
uv run p12-study-hub --ask "How does chunk overlap help retrieval?"
```

`TUTOR_LLM_PROVIDER=openai` (with `OPENAI_API_KEY`, `TUTOR_OPENAI_MODEL`)
works the same way. `.env.example` lists every key P12 reads. Nothing loads `.env` automatically:
export the variables, or copy the file to `.env` (gitignored) and run with
`uv run --env-file .env p12-study-hub …`. No key is required for the default `ollama` provider or for `stub` itself.

## Example output

`uv run p12-study-hub --demo`:

```
p12-study-hub: ok
  questions: 5
  session_score: 5/5
  lessons_indexed: 30
  provider: stub
  wrote: output/
```

`seconds` varies run to run and is reported only on the console and in the
returned `DemoResult`, never in a committed file - running `--demo` twice in
a row leaves `git status` clean for the tracked files. `output/metrics.txt`
(committed, deterministic): `questions=5`, `session_score=5/5`,
`lessons_indexed=30`, `provider=stub`.

`output/tutor_session.txt` (committed, deterministic - real retrieval
against the real corpus, `stub` LLM):

```
Q: What does the missing-value lesson recommend for a skewed numeric column?
A: STUB: For numeric columns you intend to keep, filling gaps with th
Sources: data-cleaning-basics/lessons/06-missing-value-imputation.md
```

The `stub` answer is a truncated echo of the top-ranked retrieved chunk's
own lesson text (`"STUB: " + context[:60]`, the same shape as P11's
`rag_chatbot.py` stub) - deterministic and
visibly derived from the actual retrieved prose, not a real generated
answer; see "Design notes" for why `demo()` pins `stub` and for the context
shape that keeps this slice on real lesson content, not a citation label.
`output/progress.png`/`output/history.json` regenerate every run, never committed.

## Design notes

- **`app.py` composes, it does not reimplement.** Every quiz/progress/tutor
  behaviour lives in its own module; `app.py`'s Gradio tabs and CLI dispatch
  are a thin layer calling `quiz_engine`/`progress`/`tutor` by their real
  names (an earlier draft of the source project imported them as bare
  `A`/`B`/`C` - this port names them properly throughout).
- **`quiz_engine.build_bank_from_corpus` is the one addition over P02.** It
  calls `qbank.build_bank(course_dir)` once per course - P02 itself is
  untouched - and concatenates `questions`/`units`, summing
  `question_counts`, into one merged bank. `load_bank` then keeps only
  `multiple_choice`/`true_false` (`CLOSED_TYPES`) and tags each question's
  `module` from `qid.split("/")[0]`, i.e. the course slug.
- **No "lab README" branch.** The source `progress_scan.py` recognised two
  status formats: a `## Status:` line (private-repo lab READMEs) and a
  Markdown table (course READMEs). This corpus only ever uses the table
  format, so `progress.scan_statuses` implements table scanning only.
- **`demo()` force-pins `stub`, not "whichever provider is configured".**
  Same reasoning as P10/P11 - the answer must not depend on whether Ollama
  happens to be running. Retrieval is real (real MiniLM embeddings, a real
  Chroma index over the real corpus), and the stub's answer echoes the
  retrieved context rather than a fixed string (matching P11's
  `rag_chatbot.py` stub), so both `Sources` and the answer text demonstrate
  genuine grounding - an earlier version always returned the fixed
  `REFUSAL` string, a literal port of the source's `ask_llm` that made a
  correctly-grounded answer indistinguishable from a refusal.
- **The context fed to the LLM carries no `"[source]"` label.** An
  earlier `ask()` prefixed every chunk with `f"[{source}]\n"` before joining
  it into `context`; for any 60+ character corpus-relative path that alone
  consumed the stub's whole `context[:60]` slice, never reaching the
  lesson's own prose (P11's chain never has this problem: `page_content`
  only, citations out-of-band). `context` is `page_content` only now;
  `format_sources(docs)` still reads `doc.metadata` directly, unaffected.
  Correct for a real provider too, since `PROMPT_TEMPLATE` never asked it to
  cite from such a label.
- **The relevance gate refuses before ever calling the LLM.** `tutor.ask`
  refuses immediately, with empty `sources`, unless the best retrieved
  chunk's similarity clearly stands out from the rest (see "Limits" for the
  gate's exact basis) - which is what makes a trap question ("What is the
  capital of France?") refuse regardless of which LLM provider is behind it,
  except on macOS; see "Limits".
- **Private Chroma internals: avoided where possible, disclosed where not.**
  The source reached into `store._collection.delete(where=...)`, a
  LangChain/Chroma implementation detail not covered by its public API
  contract. `index_lessons`'s incremental path avoids that specific one: it
  uses the public `store.get(where=...)` to find matching ids and
  `store.delete(ids=...)` to remove them - same effect, no reliance on an
  attribute that could disappear on a dependency bump. **This is no longer
  true of the module as a whole.** The relevance gate ("Limits" below) calls `store._collection.query(..., include=["embeddings"])` to
  retrieve raw embedding vectors and reads `store._collection.metadata` in
  the diagnostic - both `tutor.py`, both on the private `_collection`
  attribute. This was not avoidable with LangChain's public `Chroma`
  wrapper: `similarity_search_with_relevance_scores` and its siblings return
  scores and documents, but no documented public method returns a
  candidate's own embedding vector alongside it, which is exactly what
  computing cosine similarity independently of the store requires (see
  "Limits"). Consequence: a future `langchain_chroma`/`chromadb` release
  could rename or restructure `_collection` and break the mechanism the
  relevance gate scores with - the same
  fragility this bullet originally promised was absent, now knowingly
  accepted for one specific, load-bearing reason rather than silently
  reintroduced.
- **Output contract.** Library functions never `print`; they log through
  `logging.getLogger(__name__)`. `main()` prints at most six summary lines
  for `--demo`; the interactive `--quiz`/`--dashboard`/`--ask` subcommands
  print their own richer console output, matching P10/P11's convention for
  direct (non-demo) CLI modes.

## Limits

- **The `stub` provider never actually generates an answer** - it echoes a
  snippet of the retrieved context (matching P11's `rag_chatbot.py` stub), to
  prove the pipeline deterministically without a real model, not to
  demonstrate generation quality; run against `ollama`/`openai` (see "Run")
  for an actual generated answer.
- **`ollama` and `openai` each need their own setup.** The `rag` tests and
  `demo()` force-pin `stub`; one `llm`-marked test asks the tutor an on-topic and an
  off-topic question with Qwen2.5 1.5B served by Ollama (`heavy.yml`'s `llm` job, or
  locally after `docker compose --profile llm up -d`). `openai` is never exercised.
- **The corpus is intentionally small** (30 lessons, 90 questions) - enough
  to exercise every code path (multi-course filtering, incremental
  reindexing, a genuine relevance gate) without a large, slow index.
- **No authentication or rate limiting on the Gradio server.** `build_ui()`
  is a local demo UI; the CLI binds `127.0.0.1` by default - pass `--host
  0.0.0.0` to listen on every interface.
- **How the relevance gate works, and what was measured.** `tutor.ask`
  retrieves `RELEVANCE_POOL_K=12` candidates, scores each with cosine
  similarity computed from the stored, unit-normalized vectors
  (`_retrieve_with_own_similarities`, not the store's relevance score), and
  refuses unless the top score beats the median of ranks
  `TOP_K+1..RELEVANCE_POOL_K` by `RELEVANCE_MARGIN=0.13`; `RELEVANCE_MIN=0.05`
  is only a sanity floor. The margin was measured on Windows with 6 on-topic
  and 10 clearly off-topic questions: on-topic margins 0.20–0.38, off-topic
  0.02–0.07. It is unaffected by shifting every score by a constant but not by
  rescaling, so it is a calibrated number, not a platform-free one. Two
  earlier forms were tried and dropped: an absolute floor (0.35, tuned on
  Windows), which let an off-topic question through on macOS, and a margin
  against only the top 4, whose on-topic and off-topic ranges overlapped. To
  check the gate on another platform, call `tutor._diagnostic_snapshot` with
  an on-topic and an off-topic question: it prints the collection metadata,
  the query-embedding norm, and each candidate's own cosine next to the
  store's relevance score.
- **The relevance gate does not work on macOS; this is a limit of the
  technique, not an unfixed bug.** The gate assumes a genuinely on-topic
  question's best chunk stands out from the rest. On a macOS CI runner,
  `_diagnostic_snapshot` showed that assumption failing, while `own_cosine`
  matched `store_relevance` to four decimal places on every row and the
  collection metadata read back as `{'hnsw:space': 'cosine'}` (so neither the
  arithmetic nor the configuration was at fault). For the on-topic question
  "What is a sentinel value in a messy dataset?", the top scores were
  `0.4138, 0.3990, 0.3928, 0.3870, …, 0.3643` - nearly flat, with the correct
  lesson (`02-sentinel-values-and-missing-markers.md`) ranked **second**;
  margin ≈ 0.04, so the gate refuses a question the lessons cover. For the
  trap question "What is the capital city of France?", the scores were
  peaked - `0.4006`, then a `0.2230`–`0.3263` tail - margin ≈ 0.16, so the gate
  answers a question the lessons do not cover. The same model over the same
  corpus gives a **flat** field for an on-topic question and a **peaked** one
  for an off-topic question there, the opposite of what a standout-match
  heuristic needs. Recalibrating a threshold and computing the similarity
  independently of the store were both tried and did not fix it. A ratio or
  z-score criterion was not tried, and is not expected to help, because it
  relies on the same standout premise. The two `rag` tests that exercise
  this (`test_index_and_ask_end_to_end`,
  `test_trap_question_refuses_via_relevance_gate`) are marked
  `xfail(sys.platform == "darwin", strict=False)`: not skipped, so the known
  failure stays visible, and not strict, so a future runner, model revision
  or embedding library update that makes them pass does not break CI.

## Datasets and licences

There is no external dataset. Every file under `projects/p12_study_hub/
corpus/` (30 lessons, 3 course READMEs) is original prose and original
practice questions, written specifically for this repository (see
[ADR 0003](../../docs/decisions/0003-p12-synthetic-corpus.md) for why the
source capstone's corpus could not be reused). Two fictional scenarios
recur across the lessons - "Wheel & Way" (bike-share) and "Fenwick Labs" /
"Pebble" (documentation assistant) - neither refers to anything real.

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
