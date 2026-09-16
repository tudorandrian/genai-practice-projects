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
  refuses immediately, with empty `sources`, unless the best retrieved
  chunk's similarity clearly stands out from the rest (see "Limits" for the
  gate's exact basis and history) — which is what makes a trap question
  ("What is the capital of France?") reliably refuse regardless of which LLM
  provider is behind it.
- **Private Chroma internals: avoided where possible, disclosed where not.**
  The source reached into `store._collection.delete(where=...)`, a
  LangChain/Chroma implementation detail not covered by its public API
  contract. `index_lessons`'s incremental path avoids that specific one: it
  uses the public `store.get(where=...)` to find matching ids and
  `store.delete(ids=...)` to remove them — same effect, no reliance on an
  attribute that could disappear on a dependency bump. **This is no longer
  true of the module as a whole.** The relevance gate (fix round 5, "Limits"
  below) calls `store._collection.query(..., include=["embeddings"])` to
  retrieve raw embedding vectors and reads `store._collection.metadata` in
  the diagnostic — both `tutor.py`, both on the private `_collection`
  attribute. This was not avoidable with LangChain's public `Chroma`
  wrapper: `similarity_search_with_relevance_scores` and its siblings return
  scores and documents, but no documented public method returns a
  candidate's own embedding vector alongside it, which is exactly what
  computing cosine similarity independently of the store requires (see
  "Limits"). Consequence: a future `langchain_chroma`/`chromadb` release
  could rename or restructure `_collection` and break the mechanism that
  makes the relevance gate's scoring platform-independent — the same
  fragility this bullet originally promised was absent, now knowingly
  accepted for one specific, load-bearing reason rather than silently
  reintroduced.
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
- **The relevance gate computes its own similarity and its own margin — the
  only thing about it that is not platform-independent by construction is
  one measured number.** History, across three attempted fixes: (1) the
  tutor's Chroma collection had no explicit distance metric, so "relevance
  score" came from applying a cosine-shaped formula to whatever Chroma
  defaulted to — confirmed to make an off-topic "trap" question that
  reliably refused on Windows/Linux score high enough to *not* refuse on
  macOS. Fixing that (explicit `COLLECTION_METADATA`/`EMBED_NORMALIZE`,
  verified actually in effect at runtime on this machine by
  `test_embeddings_are_actually_normalized_and_cosine_configured`) was
  necessary but not sufficient. (2) Replacing the absolute floor with a
  margin against a wider "background" pool (`RELEVANCE_MARGIN`,
  `RELEVANCE_POOL_K`) — still using the *store's* relevance score — fixed
  Windows/Linux but macOS then failed with **opposite verdicts**: a
  genuinely grounded question refused, and the trap question answered,
  retrieving the same documents in the same order a pre-cosine-config run
  had. Opposite verdicts rule out a mis-scaled number (that would push both
  classes the same direction); identical retrieval across two different
  configurations is direct evidence the configuration was not actually
  reaching that platform's Chroma/HNSW build, even though it verifiably was
  reaching this machine's. (3) **`tutor.ask` no longer asks the store for a
  relevance score at all.** `_retrieve_with_own_similarities` reads raw
  embedding vectors back from the collection and computes cosine similarity
  itself — a plain dot product of vectors already confirmed unit-normalized
  — so the gate's numbers are this module's own arithmetic on every
  platform, up to float noise, regardless of what distance space the store
  is configured (or silently defaults) to use internally. The margin logic
  from (2) is otherwise unchanged and still scale-free by construction:
  refuse unless the top chunk clears the median of a wider background by
  `RELEVANCE_MARGIN`; `RELEVANCE_MIN=0.05` remains only a conservative
  absolute sanity floor, not the discriminator.
  **What is still per-platform-measured, and so still a residual risk:**
  `RELEVANCE_MARGIN=0.13` and `RELEVANCE_POOL_K=12` were measured on this
  machine only (6 on-topic / 10 off-topic questions; re-measured again in
  fix round 5 using this module's own cosine similarity — identical to four
  decimal places, since this machine's store relevance score and its own
  cosine similarity already agreed; see the comment above `RELEVANCE_MARGIN`
  in `tutor.py`). That agreement was never reproducible here in the first
  place, so this machine cannot confirm the fix resolves the disagreement —
  only that it removes the one channel (the store's own relevance-score
  formula) through which a platform could silently diverge from what this
  module actually computed and verified. If the gate ever misfires on a
  specific platform again, first check whether `_diagnostic_snapshot`'s two
  columns (`own_cosine` vs `store_relevance`) themselves disagree there — if
  they do, that platform's Chroma/HNSW build is doing something unrelated to
  this module's own arithmetic, which is a different bug from a
  mis-calibrated margin. If they agree with each other but still misfire,
  re-measure `RELEVANCE_MARGIN` on that platform (a mix of on-topic and
  clearly off-topic questions, comparing top1 against the median of ranks
  `TOP_K+1..RELEVANCE_POOL_K`, `_diagnostic_snapshot` prints exactly the
  numbers needed) rather than adjusting it blind.
- **Adjudicated limitation (2026-09-16): the relevance gate does not work
  correctly on macOS, and this is a real limit of the technique, not an
  unfixed bug.** The gate is a margin heuristic: it assumes a genuinely
  on-topic question's best-matching chunk stands out from the rest, and
  refuses when nothing does. That assumption holds on Windows and Linux (see
  the measured numbers above) but does not hold on macOS, confirmed by
  `_diagnostic_snapshot` on [heavy.yml run
  35076304269](https://github.com/tudorandrian/genai-practice-projects/actions/runs/35076304269):
  `own_cosine` matched `store_relevance` to four decimal places on every row
  (the store's arithmetic was never the problem, and computing cosine
  independently changed nothing there), and the collection metadata read
  back from the live store on that run was correctly `{'hnsw:space':
  'cosine'}` (the configuration *was* in effect — the identical retrieval
  order noticed in an earlier round was a coincidence of a flat score field,
  not a sign of a stale configuration). The scores themselves show why: for
  the genuinely on-topic question "What is a sentinel value in a messy
  dataset?", the top candidates on that platform were `0.4138, 0.3990,
  0.3928, 0.3870, …, 0.3643` — nearly flat, and the correct lesson
  (`02-sentinel-values-and-missing-markers.md`) ranked **second**, not
  first; margin ≈ 0.04, well under `RELEVANCE_MARGIN=0.13`, so the gate
  refuses a question the lessons do cover. For the trap question "What is
  the capital city of France?", the same platform produced a genuine peak —
  `0.4006`, then a `0.2230`–`0.3263` tail — margin ≈ 0.16, clearing
  `RELEVANCE_MARGIN`, so the gate answers a question the lessons do not
  cover. On that platform, the same model over the same corpus produces a
  **flat** field for an on-topic question and a **peaked** one for an
  off-topic question — the opposite of what a standout-match heuristic
  needs. This is not fixable by recalibrating a threshold, switching to a
  ratio or a z-score, or changing who computes the similarity (all three
  were tried, in that order, across fix rounds 3-5) — the geometry of the
  scores on that platform contradicts the heuristic's premise directly. The
  two `rag` tests that exercise this (`test_index_and_ask_end_to_end`,
  `test_trap_question_refuses_via_relevance_gate`) are marked
  `xfail(sys.platform == "darwin", strict=False)` for exactly this reason —
  not skipped, so the known failure stays visible, and not strict, so a
  future runner, model revision, or embedding library update that happens
  to fix this is free to surprise everyone by passing. To re-measure whether
  this is still true on a given macOS runner, call `tutor._diagnostic_snapshot`
  with an on-topic and an off-topic question and compare the shape of the
  `own_cosine` column, the same way this finding was produced.

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
