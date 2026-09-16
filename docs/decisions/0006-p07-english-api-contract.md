# 0006: P07's JSON API contract is English, not the source's Romanian

## Context

The source `server.py` exposed a Flask API with a Romanian JSON contract:
response keys `scor`/`eroare`, labels `pozitiv`/`negativ`/`neutru`, Romanian
error messages. ADR 0002 requires English identifiers repository-wide, and
this is the first web API in the port, so its wire contract is external.

## Decision

`POST /sentiment` now returns `{"text", "score", "sentiment"}` with labels
`positive`/`negative`/`neutral`; error bodies use `{"error": "..."}` in
English. `LEXICON`'s words stay Romanian — they are scoring *data* for a
Romanian-text domain, not repository identifiers, so ADR 0002 does not apply
to them (documented in the project README).

## Consequences

- Any consumer written against the old contract (`scor`, `eroare`,
  `pozitiv`/`negativ`/`neutru`) must update field names and label values;
  there is no compatibility shim.
- The contract is now consistent with every other project's English-only
  identifier rule; the demo domain (Romanian input text) is unaffected.
