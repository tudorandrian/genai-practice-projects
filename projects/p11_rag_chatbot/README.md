# P11 - RAG Chatbot (LangChain + Chroma)

## What it does

Answers questions about *your own* documents - PDF, Markdown, plain text - instead
of from the model's general knowledge, using Retrieval-Augmented Generation. This
is the repository's first `rag`-tier project: it needs LangChain, a persistent
Chroma store and a sentence-transformers embedding model, so `uv run demo` and
`uv run demo --models` skip it - only `uv run demo --all` runs it.

The pipeline, in `rag_chatbot.py`:

```
load_documents(folder)    -> [Document]         # pypdf pages / text files, keeps source+page
split_documents(docs)     -> [Document]          # RecursiveCharacterTextSplitter (200/20)
build_index(reindex, ...) -> Chroma              # embed with MiniLM, persist to chroma_index/
create_llm(provider)      -> LLM                 # THE provider seam
build_chain(store, llm)   -> GroundedQA          # grounding prompt + source return
ask(qa, question)         -> {"answer","sources"}  # one question in, one grounded answer out
```

The answer is generated **only** from the retrieved context: the grounding prompt
instructs the model to answer exclusively from the supplied context and to reply
with a fixed refusal string, `REFUSAL`, when the answer is not in it. Every answer
also carries the file (and PDF page) it was retrieved from - the standard technique
for reducing hallucinations and connecting an LLM to private data.

`synthetic_docs.py` generates a small, invented company corpus (`acme_handbook.pdf`,
`engineering_notes.md`, `support_faq.txt`) with specific, checkable facts, and
`eval_questions.md` pairs twelve questions with the source file that answers each
one, plus one **trap** question with no answer in the corpus - see "Design notes".

## Run

```bash
uv sync --group rag                                    # langchain, chromadb, sentence-transformers, ...
uv run p11-rag-chatbot --demo                           # builds the index, asks 3 questions, writes output/
uv run p11-rag-chatbot --reindex                        # rebuild the index from data/
uv run p11-rag-chatbot -q "How many vacation days?"      # single question, real provider
uv run p11-rag-chatbot                                    # interactive Q&A loop
uv run p11-rag-chatbot --ui                                # Gradio web UI, http://127.0.0.1:7860
uv run pytest projects/p11_rag_chatbot -q                  # core tests, no rag group needed
uv run pytest projects/p11_rag_chatbot -m rag -q           # full pipeline, needs the rag group
```

`--verbose` logs at `INFO`; by default only the summary lines print. On a fresh
clone `data/` is empty: run `--demo` (or `uv run python -m
projects.p11_rag_chatbot.synthetic_docs`) once before `--reindex`, `-q`, the loop
or `--ui`, which otherwise stop with a message saying so. `--demo`
generates the synthetic corpus into `data/` (never tracked - see "Datasets and
licences"), rebuilds the index, asks three questions with the `stub` provider
(force-pinned - see "Design notes"), and writes `output/session.txt` and
`output/metrics.txt`.

To run the CLI against a real LLM instead of the stub (`--demo` always pins
`stub`):

```bash
docker compose --profile llm up -d        # Qwen2.5 1.5B served by Ollama on localhost
export RAG_LLM_PROVIDER=ollama          # RAG_OLLAMA_MODEL/_URL override the defaults
uv run p11-rag-chatbot -q "Who is the CEO of ACME Robotics?"
```

`RAG_LLM_PROVIDER=openai` (with `OPENAI_API_KEY`, `RAG_OPENAI_MODEL`) works the
same way. `.env.example` lists every key P11 reads. Nothing loads `.env` automatically:
export the variables, or copy the file to `.env` (gitignored) and run with
`uv run --env-file .env p11-rag-chatbot …`. No key is required for the default `ollama` provider or for `stub` itself.

## Example output

`uv run p11-rag-chatbot --demo`:

```
p11-rag-chatbot: ok
  chunks: 9
  provider: stub
  questions: 3
  wrote: output/
```

`seconds` varies run to run and is reported only on the console and in the
returned `DemoResult`, never in a committed file - running `--demo` twice in a row
leaves `git status` clean. `output/metrics.txt` (committed, deterministic):
`chunks=9`, `provider=stub`, `questions=3`, one `key=value` per line.

`output/session.txt` (committed, deterministic - retrieval is real, and the
`stub` provider's answer echoes the retrieved context rather than a fixed string,
so both `Sources` and the answer text demonstrate genuine grounding):

```
Q: How many paid vacation days do employees get?
A: STUB: Every full-time employee receives 25 days of paid vacation p
Sources: acme_handbook.pdf (p.1), support_faq.txt

Q: Who is the CEO of ACME Robotics?
A: STUB: ACME Robotics - Employee Handbook ACME Robotics was founded
Sources: acme_handbook.pdf (p.1), support_faq.txt, engineering_notes.md
```

The `stub` answer is a truncated echo of the top-ranked retrieved chunk
(`"STUB: " + context[:60]`) -
deterministic and visibly derived from what was retrieved, not a real generated
answer; see "Design notes" for why `demo()` pins `stub`.

## Design notes

- **One provider seam.** `create_llm(provider=None)` is the only place an LLM is
  constructed; `provider` falls back to `RAG_LLM_PROVIDER` (default `ollama`).
  `stub` returns a small `LLM` subclass that echoes the retrieved context, used
  by the tests and `demo()`. An earlier stub returned a canned response
  whatever the input, which made the refusal assertion true by construction;
  deriving the answer from the prompt closes that gap and reserves the exact
  `REFUSAL` string for a real provider's own grounding behaviour. No API key
  ever lives in code, and every model/URL name is read
  from the environment at call time, not cached, so tests can override them
  without reloading the module.
- **`demo()` force-pins `stub`, not "whichever provider is configured".** The
  default provider is `ollama`; most machines running the test suite have none
  listening, and a machine that does would produce a different answer. A
  committed artefact must not depend on which services happen to be running -
  see P10's [ADR 0007](../../docs/decisions/0007-p10-provider-seam-and-fallback.md)
  for the same reasoning. Retrieval is still real, so `sources` in
  `output/session.txt` demonstrate actual grounding.
- **The evaluation set doubles as a retrieval test - asserting rank, not mere
  presence.** `eval_questions.md`'s table is parsed by `load_eval_questions()`
  into `(question, expected_source)` pairs, excluding the header, the separator
  row and the trap row (`Source` reads `(absent)` - no expected source, no
  retrieval hit to assert). `test_evaluation_questions_retrieve_the_expected_source`
  asserts the expected file is the *top-ranked* source (`result["sources"][0]`),
  with `test_wrong_document_is_not_ranked_first` as a negative control. An
  earlier presence-only assertion could not fail, because `TOP_K` covered the
  whole corpus (see `CHUNK_SIZE` below). The trap
  question is asserted at the retrieval layer directly
  (`test_trap_question_retrieves_no_revenue_related_chunk`: no retrieved
  chunk mentions revenue), not through the stub's answer text, which is true
  by construction regardless of the question and so could never have caught
  anything on its own.
- **Persistence, and why `CHUNK_SIZE` is small.** `build_index(reindex=False)`
  loads an existing `chroma_index/` directory without re-embedding;
  `reindex=True` (or `--reindex`) rebuilds it from `data/`; `demo()` always
  passes `reindex=True` so `chunks`/sources never depend on a stale index
  directory. `CHUNK_SIZE=200`/`CHUNK_OVERLAP=20` splits the corpus into 9 chunks
  for `TOP_K=3`, forcing genuine discrimination; 250/30 also yields 9 chunks but
  was rejected - it occasionally merges two unrelated handbook facts into one
  chunk that then out-competes the actually-relevant chunk for a different
  question (verified against all twelve real questions before picking 200/20).
- **Marker discipline.** `load_documents`/`split_documents` import
  `pypdf`/`langchain_core`/`langchain_text_splitters`, and `create_llm` imports
  `langchain_core`, even for their simplest (`stub`, local-file) paths - so those
  tests are marked `rag`, not `core`, even though none needs a network
  connection. Only the pure-Python pieces (the grounding prompt, `format_sources`,
  `ask()` against a fake QA object, `load_eval_questions()`'s file parsing) stay
  `core`. Every LangChain/Chroma/embedding import stays inside the function that
  needs it, so this module (and its `core` tests) import cleanly without the
  `rag` group installed.
- **Output contract.** Library functions never `print`; they log through
  `logging.getLogger(__name__)`. `main()` prints at most six summary lines and
  sets the logging level (`WARNING`, or `INFO` with `--verbose`).

## Limits

- **`ollama` and `openai` each need their own setup** (`ollama serve` + a pulled
  model, or an `OPENAI_API_KEY`). The `rag` tests and `demo()` force-pin `stub`;
  the `llm`-marked tests run the grounded answer and the trap-question refusal
  against Qwen2.5 1.5B served by Ollama (`heavy.yml`'s `llm` job, or locally after
  `docker compose --profile llm up -d`). `openai` is never exercised.
- **The `stub` provider never actually generates an answer** - it echoes a
  snippet of the retrieved context, to prove the pipeline deterministically
  without a real model, not to demonstrate generation quality; see "Run" for a
  real provider.
- **`all-MiniLM-L6-v2` is small, and the corpus is tiny and stylistically
  uniform** (short, similarly-phrased sentences) - enough to exercise every
  loader, chunking and citation path, but together they make chunk-level
  retrieval more sensitive to exact chunk boundaries than a larger, more varied
  corpus or a stronger embedding model would be; see "Design notes" for the
  chunk-size trade-off this surfaced.
- **No authentication or rate limiting on the Gradio server.** `build_ui()` is a
  local demo UI, not hardened for public exposure; the CLI binds `127.0.0.1` by
  default - pass `--host 0.0.0.0` to listen on every interface.

## Datasets and licences

There is no external dataset. `synthetic_docs.py`'s `HANDBOOK`/`ENGINEERING`/`FAQ`
are hand-written, fixed text, written specifically for this repository, with
facts invented for the exercise (a fictional company, "ACME Robotics").
`write_all()` renders them into `data/acme_handbook.pdf` (via `fpdf2`, pure
Python), `data/engineering_notes.md` and `data/support_faq.txt`; none of the
three is ever tracked by git (`.gitignore`: `projects/p11_rag_chatbot/data/*`,
with an exception for `data/.gitkeep`) - `demo()` and the `rag`-marked tests
regenerate the corpus on demand instead of committing it.

The embedding model is
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
from the Hugging Face Hub (Apache-2.0, by the Sentence-Transformers project -
Reimers & Gurevych, *Sentence-BERT*, 2019), downloaded on first use and cached
locally; this project only performs inference against it. `chroma_index/` (the
persisted vector store) is likewise never tracked (`.gitignore`:
`projects/*/chroma_index/`).

## Courses drawn on

- 15 Fundamentals of AI Agents Using RAG and LangChain
- 16 Project: Generative AI Applications with RAG and LangChain
