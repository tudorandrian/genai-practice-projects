# Contributing

This is a personal portfolio repository; the rules below are the ones I hold myself to.

## Two rules that never bend

1. **Every commit is publishable.** No secrets, no file that originates from a course
   provider, no local machine paths, no file over 100 KB, no notebooks or model files.
   The pre-commit hooks enforce this; `--no-verify` is not used.
2. **Public only through the gate.** The repository stays private until every item in
   [`docs/release-gate.md`](docs/release-gate.md) is ticked.

## Tooling

    uv sync                         # core + dev groups
    uv sync --group models          # + torch, transformers, gradio
    uv sync --group rag             # + LangChain, Chroma, sentence-transformers
    uv run pre-commit install
    uv run pytest -m "core and not network"
    uv run ruff check . && uv run ruff format --check . && uv run mypy
    docker compose run --rm tests   # the same core suite on Linux

## Pull requests

Branch per project (`p01-mini-etl` … `p12-study-hub`), one PR each, squash-merged with the
PR title as the commit subject. The PR template asks what was ported, renamed, added and cut,
and for the command that regenerated the run proofs.

## Conventions

- Python 3.13, `ruff` rules `E F I UP B SIM N`, `mypy --strict`.
- Console output is a summary of at most six lines; details go to `output/`.
- Library code logs, it never prints.
- Every dataset has a source and a licence in the project README.
