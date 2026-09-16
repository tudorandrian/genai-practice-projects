# Changelog

## 1.0.0 — 2026-09-16

Fifteen pull requests, squash-merged in order:

- chore: scaffold
- feat(shared): dataset cache, demo runner, registry, fixtures, blocklist
- feat(p01): mini-ETL — twelve public datasets, config schema tests, demo entry point
- feat(p02): question bank — layout-agnostic discovery, JSON Schema, fixtures as test parameters
- feat(p03): linear regression across seven public datasets — checksummed loaders, coefficient ranking test
- feat(p04): decision-tree classifier — synthetic primary data, rules export, course-hosted source removed
- feat(p05): K-Means + PCA segmentation — offline synthetic primary, network-marked public sets, saved plots
- feat(p06): pipeline + grid search — model card per run, course-hosted mirror removed
- feat(p07): sentiment API — English contract, waitress production flag, JSON-error tests
- feat(p08): BLIP captioning — device auto-select, single model load in batch mode
- feat(p09): Flask chatbot — English contract, history-window test, dated-model note
- feat(p10): meeting assistant — English sections, stub fallback when no LLM, end-to-end stub test
- feat(p11): RAG chatbot — English contract, evaluation set as a retrieval test
- feat(p12): study hub on a synthetic corpus — P02 bank integration, progress tables, RAG tutor
- chore(release): 1.0.0 — README, certificate map, release gate. Also the largest code change
  of the release, from getting `heavy.yml` green on macOS:
  - P10's synthetic audio converts the AIFF/AIFF-C files macOS's speech engine writes to WAV and
    waits for its asynchronous writes to finish; Whisper is pinned to CPU, because the MPS
    backend produced noise transcripts.
  - P12's relevance gate compares the top chunk against the background of a wider pool, with
    cosine similarity computed from the stored vectors. Limitation: on macOS it cannot reliably
    separate on-topic from off-topic questions, so two P12 tests are `xfail` there (see P12's
    README, "Limits").
