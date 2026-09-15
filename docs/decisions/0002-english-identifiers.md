# 0002: Full English — specs, identifiers, and API contracts

## Context

The projects were originally written with a mix of languages across specs, variable
names, JSON keys, and enum values. A bilingual repository reads as unfinished to a
human reviewer and as noise to an automated screener.

## Decision

Every project is fully translated to English as it is ported: specs are translated
and condensed, identifiers and API keys are renamed to their English equivalents,
and the tests are updated in the same commit so each port stays green throughout.

## Consequences

- The repository reads as a single, coherent body of work regardless of which
  project a visitor opens first.
- Each rename is direct, visible evidence of the port being read, understood, and
  improved, not just copied.
- Every project PR must update its tests alongside any renamed identifier so no
  commit is red.
