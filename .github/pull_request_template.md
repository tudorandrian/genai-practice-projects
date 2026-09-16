## Summary
<!-- What changes and why. Link the issue or decision record if there is one. -->

## Changes
-

## Proof
<!-- The commands you ran and their result, for example `uv run demo --all` followed by
`git status --porcelain -- projects/*/output`. -->

## Automated checks (green before review starts)
- [ ] `ci` green: pre-commit hooks, ruff, mypy, core tests on Ubuntu and Windows, blocklist, dependency audit, release check, gitleaks
- [ ] `heavy` green when it ran (P08-P12, `shared/` or dependency changes): models, RAG, demos, real-LLM tests

## Manual review
- [ ] Nothing unpublishable: no secrets, course material, personal data, local paths, or files over 100 KB
- [ ] A changed project README keeps its seven sections, and every dataset has a source and a licence
- [ ] Documentation and `CHANGELOG.md` are updated where behaviour changed
