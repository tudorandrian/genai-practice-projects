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

## Later

The "no OS-specific logic" premise stopped holding with P10 and P12. P10's synthetic
audio depends on each OS's own speech engine and handles their differences
(`synthetic_audio.py` converts the AIFF/AIFF-C files macOS writes to WAV, waits for
engines that write asynchronously, and gives a Linux-specific install hint); `heavy.yml`
installs `espeak-ng` on Linux only. Two P12 retrieval tests are `xfail` on macOS (see
P12's README, "Limits"). Both projects are exercised on Ubuntu and macOS by `heavy.yml`,
not by the per-PR matrix.
