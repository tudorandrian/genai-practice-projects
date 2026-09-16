# Security policy

## Reporting a vulnerability

Please report a vulnerability privately through GitHub's **Report a vulnerability** button
on this repository's Security tab, not in a public issue. This is a personal practice
repository maintained on a best-effort basis; expect an acknowledgement within a week.

## Scope

The projects are local demos, not hosted services:

- The Flask and Gradio servers (P07-P12) bind `127.0.0.1` by default and have no
  authentication or rate limiting. Do not expose them with `--host 0.0.0.0` on a
  network you do not trust.
- Ollama (compose profile `llm`) has no authentication either; `compose.yaml` publishes its
  port on `127.0.0.1` only.
- API keys are read from environment variables only. `.env` is gitignored, and every commit
  is scanned by gitleaks (pre-commit locally, full history in CI).

## Known advisories in dependencies

Dependencies are audited against the PyPA advisory database with `pip-audit`, and
Dependabot opens monthly update pull requests. Advisories that remain open, and why they do
not apply here:

| Package | Advisories | Why it does not apply |
|---|---|---|
| `chromadb` 1.5.9 | CVE-2026-45829, CVE-2026-45830, CVE-2026-45831, CVE-2026-45833 | All four are in Chroma's HTTP server and its role-based authorization. P11 and P12 use Chroma embedded in the process through `langchain-chroma`, with no server. No fixed release exists yet. |
