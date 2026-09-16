# 0003: P12's corpus is written from scratch, not ported

## Context

The source capstone reads `courses/` in the private certificate repository -
a real question bank, real `M*-L*.md` lesson notes, and README status
tables - all belonging to a third-party provider and unpublishable; the
blocklist's `M[0-9]-[0-9]-L(AB)?[0-9]` pattern flags that naming directly.

## Decision

Write three original, fictional courses (`data-cleaning-basics`,
`classic-ml-in-practice`, `llm-apps-from-scratch`; 10 lessons each, 90
questions) under `projects/p12_study_hub/corpus/`, in the format the
already-merged P02 `qbank` parser expects. Every sentence of prose, every
question, and every README status table is original, written for this
repository - checked by `test_corpus_contains_no_blocklisted_phrase`.

## Consequences

- Quiz, progress and tutor questions reference invented material (a fictional
  bike-share company; a fictional documentation assistant), not real facts.
- The corpus is small (30 lessons) by design; `demo()`'s figures reflect that
  scale, not the source capstone's original counts.
- Writing the corpus was the majority of this port's effort.
