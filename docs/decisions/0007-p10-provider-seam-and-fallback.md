# 0007: P10's LLM provider seam degrades to the stub, not to a raised error

## Context

`summarize_with_llm(prompt)` dispatches across four providers by
`MEETING_LLM_PROVIDER`: `ollama` (default, local, over HTTP), `openai`,
`local` (offline transformers), `stub`. Ollama is an external process this
repo does not manage; a working summary should not start raising just
because it stopped.

## Decision

Only `ollama` degrades: on any `OSError` (`urllib.error.URLError` is one),
log a warning and return `"[ollama unavailable — stub used] " +
stub(prompt)` — visible in the text, never silent. `openai`/`local` still
raise (a missing key or weights need attention). `demo()` separately
force-pins `MEETING_LLM_PROVIDER=stub`, so `output/summary.txt` never
depends on whether Ollama is running.

## Consequences

- The default provider stays usable offline, no traceback.
- Callers needing a real model ran check the `[ollama unavailable ...]`
  prefix, not a separate exception type.
- `openai`/`local` failures still raise; only `ollama` degrades.
