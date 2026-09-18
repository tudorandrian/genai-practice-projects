# Changelog

## Unreleased

- Ollama 0.34.2 (pinned by digest) in `compose.yaml` and `heavy.yml`.
- The release gate records the publication: every item ticked, evidence from this repository.

## 1.1.0 - 2026-09-16

Hardening and preparation for publication. No project changes its command-line interface or
the results committed in `output/`, except where noted.

### Security and privacy

- Runs on the user's own data write to the gitignored `output/runs/` (P08, P10, P11, P12), so
  only `--demo` writes the committed outputs.
- Gradio analytics are off; the Gradio apps limit upload size and keep exception details in the
  server log.
- P09 no longer sends CORS headers (`flask-cors` removed). P07 and P09 never start the
  Werkzeug debugger, and on loopback they reject a foreign Host header (DNS rebinding).
- The default Hugging Face models load a pinned Hub commit (P08-P12). Container images are
  pinned by digest in the Dockerfile, `compose.yaml` and `heavy.yml`; Ollama moves to 0.34.1.
- CI pins every action to a commit SHA, keeps no credentials after checkout, and audits the
  locked dependencies with `pip-audit`. The test image runs as an unprivileged user, and
  `.dockerignore` excludes local data.
- The blocklist detects home-directory paths by their shape instead of naming an account.

### Changed

- P11 and P12 run on LangChain 1.x, because the `langchain` 0.3 packages had advisories fixed
  only in 1.x. P11's `RetrievalQA` chain is replaced by `GroundedQA` on `langchain-core`, and
  its loaders by `pypdf`; prompts, answers and chunks are byte-identical. `langchain` and
  `langchain-community` are no longer dependencies.
- The em dash is no longer used anywhere in the repository. P02's option separator is a spaced
  hyphen (`- A) text - **Correct answer.**`); en and em dashes are still accepted as input.
- CI runs the pre-commit hooks on every file; `heavy.yml` also runs on pull requests that touch
  P08-P12, `shared/` or the dependencies; the ruff hook uses the `uv.lock` version.

### Fixed

- P07 folds Romanian diacritics before the lexicon lookup (`excelentă și plăcută` scored
  neutral before).
- P11 no longer cites sources under a refusal.

### Added

- Real-LLM tests (`llm` marker) for P10-P12 against Qwen2.5 1.5B served by Ollama, a compose
  `llm` profile that serves it locally, and a `heavy.yml` job that runs them.
- Publication tooling: `docs/publication.md` (the runbook for the owner's steps),
  `scripts/prepare_public_history.py` (builds and verifies a publishable copy of the history,
  pushing nothing) and `.github/branch-protection.json` (branch protection as code).
- `CITATION.cff`, `SECURITY.md` (reporting, scope, data handling, dependency advisories) and
  `THIRD_PARTY_NOTICES.md` (what the MIT licence covers, and what keeps its own terms).

### Documentation

- The root README is rewritten as the project's presentation: scope, requirements, running a
  single project, tests, layout and a documentation index.
- The release gate gains item A8 (history, commit identities and pull-request descriptions) and
  a dated publication record; the pull request template is generic.

## 1.0.0 - 2026-09-16

Fifteen pull requests, squash-merged in order:

- chore: scaffold
- feat(shared): dataset cache, demo runner, registry, fixtures, blocklist
- feat(p01): mini-ETL - twelve public datasets, config schema tests, demo entry point
- feat(p02): question bank - layout-agnostic discovery, JSON Schema, fixtures as test parameters
- feat(p03): linear regression across seven public datasets - checksummed loaders, coefficient ranking test
- feat(p04): decision-tree classifier - synthetic primary data, rules export, course-hosted source removed
- feat(p05): K-Means + PCA segmentation - offline synthetic primary, network-marked public sets, saved plots
- feat(p06): pipeline + grid search - model card per run, course-hosted mirror removed
- feat(p07): sentiment API - English contract, waitress production flag, JSON-error tests
- feat(p08): BLIP captioning - device auto-select, single model load in batch mode
- feat(p09): Flask chatbot - English contract, history-window test, dated-model note
- feat(p10): meeting assistant - English sections, stub fallback when no LLM, end-to-end stub test
- feat(p11): RAG chatbot - English contract, evaluation set as a retrieval test
- feat(p12): study hub on a synthetic corpus - P02 bank integration, progress tables, RAG tutor
- chore(release): 1.0.0 - README, certificate map, release gate. Also the largest code change
  of the release, from getting `heavy.yml` green on macOS:
  - P10's synthetic audio converts the AIFF/AIFF-C files macOS's speech engine writes to WAV and
    waits for its asynchronous writes to finish; Whisper is pinned to CPU, because the MPS
    backend produced noise transcripts.
  - P12's relevance gate compares the top chunk against the background of a wider pool, with
    cosine similarity computed from the stored vectors. Limitation: on macOS it cannot reliably
    separate on-topic from off-topic questions, so two P12 tests are `xfail` there (see P12's
    README, "Limits").
