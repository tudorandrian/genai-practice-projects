## Project
<!-- P01 … P12, or scaffold / release-prep -->

## Ported
## Renamed
## Added (new tests, improvements)
## Cut (and why)
## Proof
Command that regenerated `output/metrics.txt` and the transcript:

## Automated checks (green before review starts)
- [ ] `ci` green: pre-commit hooks, ruff, mypy, core tests on Ubuntu and Windows, blocklist, release check, gitleaks
- [ ] `heavy` green when it ran (P08-P12, `shared/` or dependency changes): models, RAG, demos, real-LLM tests

## Manual review
- [ ] no file > 100 KB, no course material, no local paths
- [ ] README has all eight headings; datasets have source and licence
- [ ] reviewed against spec §5

🤖 Generated with [Claude Code](https://claude.com/claude-code)
