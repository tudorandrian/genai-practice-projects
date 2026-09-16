# Security policy

## Reporting a vulnerability

Report a vulnerability privately with GitHub's **Report a vulnerability** button on this
repository's Security tab, not in a public issue. This is a personal practice repository
maintained on a best-effort basis; expect an acknowledgement within a week.

## Scope

The projects are local demos, not hosted services.

- **Network exposure.** The Flask and Gradio servers (P07-P12) bind `127.0.0.1` by default
  and have no authentication or rate limiting. Do not start them with `--host 0.0.0.0` on a
  network you do not trust. The Flask apps never enable the Werkzeug debugger, send no CORS
  headers and reject request bodies over 64 KiB. The Gradio apps limit uploads (P08 25 MB,
  P10 200 MB) and show exception types, not details, in the browser.
- **Ollama.** The compose `llm` profile publishes Ollama, which has no authentication, on
  `127.0.0.1` only.
- **Secrets.** API keys are read from environment variables only. `.env` is gitignored, and
  gitleaks scans every commit (pre-commit locally, the full history in CI).

## Data handling

- **Your files stay local.** Runs on your own data (P08 images, P10 recordings, P11
  documents, P12 questions) write to `projects/<project>/output/runs/`, which Git ignores.
  Only `--demo` writes the committed files in `output/`.
- **Third-party APIs are opt-in.** P10-P12 send transcripts, document excerpts and questions
  to OpenAI only when `MEETING_LLM_PROVIDER`, `RAG_LLM_PROVIDER` or `TUTOR_LLM_PROVIDER` is set
  to `openai`. The default is a local Ollama model, with a local stub as fallback.
- **No telemetry from the UIs.** Gradio's usage analytics are turned off in every app.
  Model weights and public datasets are downloaded from their publishers on first use.

## Dependencies

CI audits every locked dependency with `pip-audit` on each pull request, Dependabot alerts
and security updates are enabled, and Dependabot opens monthly update pull requests.
Advisories that remain open, and why they do not apply here:

| Package | Advisories | Why it does not apply |
|---|---|---|
| `chromadb` 1.5.9 | CVE-2026-45829, CVE-2026-45830, CVE-2026-45831, CVE-2026-45833 (PYSEC-2026-311, -3813, -3814, -3815) | All four are in Chroma's HTTP server and its multi-tenant authorization. P11 and P12 use Chroma embedded in the process through `langchain-chroma`, with no server. No fixed release exists yet; `ci.yml` ignores exactly these four IDs. |
