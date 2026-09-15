# 0004: CI matrix — Ubuntu and Windows on every PR, macOS on demand

## Context

CI needs to catch real cross-platform issues (path separators, line endings) without
burning the free Actions minutes budget of a private repository. None of the code
has OS-specific logic, so a third platform on every push would mostly repeat what
Ubuntu already covers.

## Decision

`ci.yml` runs the required checks on a matrix of `ubuntu-latest` and
`windows-latest` for every push and pull request. macOS, plus the `models` and
`rag` dependency groups, run only on manual dispatch (`heavy.yml`), exercised once
before the private-to-public flip and after any dependency bump.

## Consequences

- Every PR pays for two OS runs, not three; macOS minutes (billed at 10x) are spent
  only when there is a specific reason to.
- Windows in the required matrix catches path and line-ending mistakes that a
  Linux-only CI would miss.
- Branch protection requires both `checks (ubuntu-latest)` and
  `checks (windows-latest)`, plus `gitleaks`.
