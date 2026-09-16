# Contributing

This is a personal portfolio repository; the rules below are the ones I hold myself to.

## Two rules that never bend

1. **Every commit is publishable.** No secrets, no file that originates from a course
   provider, no local machine paths, no file over 100 KB, no notebooks or model files.
   Pre-commit checks size (`check-added-large-files`), secrets (`gitleaks`) and blocked
   words and local paths (`shared.blocklist`). File types are checked by the core test
   `test_no_tracked_binary_artifact` and by the offline `release_check` step in CI;
   `.gitignore` alone does not stop `git add -f`. `--no-verify` is not used.
2. **Publication goes through the gate.** The repository is published only when every item
   in [`docs/release-gate.md`](docs/release-gate.md) is ticked, and each publication step is
   recorded there with its date and evidence.

## Tooling

    uv sync                         # core + dev groups
    uv sync --group models          # + torch, transformers, gradio
    uv sync --group rag             # + LangChain, Chroma, sentence-transformers
    uv run pre-commit install
    uv run pytest -m "core and not network"
    uv run ruff check . && uv run ruff format --check . && uv run mypy
    docker compose run --rm tests   # the same core suite on Linux
    docker compose --profile llm up -d && uv run pytest -m llm   # P10-P12 against Qwen2.5 1.5B
    docker compose --profile llm down                            # stop Ollama (the model volume stays)

## Automated checks come before review

Nobody reads a pull request whose automated checks are not green. In order:

1. **On commit:** pre-commit (`uv run pre-commit install` once) checks file size, end of
   file and whitespace, merge markers, secrets (gitleaks), ruff (the `uv.lock` version) and
   the blocklist.
2. **On every pull request:** `ci.yml` runs the same pre-commit hooks on every file, then
   ruff, mypy, the `core` tests on Ubuntu and Windows, the blocklist, a dependency audit
   (`pip-audit` against `uv.lock`), the offline release check, and gitleaks over the full
   history. Branch protection on `main` requires
   `checks (ubuntu-latest)`, `checks (windows-latest)` and `gitleaks` to pass, on a branch
   that is up to date with `main`.
3. **On pull requests that touch P08-P12, `shared/` or the dependencies:** `heavy.yml` runs
   the model and RAG tests and every demo on Ubuntu and macOS, and the `llm` tests against
   Qwen2.5 1.5B. It is not a required check (it does not run on every pull request), so
   review waits for it too when it runs. It can also be started by hand from the Actions tab.
4. **Then manual review**, against the pull request template's checklist.

## Pull requests

One branch and one pull request per change, squash-merged with the pull request title as the
commit subject. Titles follow Conventional Commits (`feat(p07): ...`, `fix: ...`, `docs: ...`,
`chore(deps): ...`). The pull request template asks for a summary, the changes, and the
commands that prove them, such as the demo run that regenerated the committed outputs.

## Conventions

- Python 3.13, `ruff` rules `E F I UP B SIM N`, `mypy --strict`.
- Console output is a summary of at most six lines; details go to `output/`.
- Library code logs, it never prints.
- Every dataset has a source and a licence in the project README.
- Each project README keeps the same seven sections: What it does, Run, Example output,
  Design notes, Limits, Datasets and licences, Courses drawn on.
- `--demo` is the only writer of the committed files in `output/`; runs on the user's own
  data write to `output/runs/`, which Git ignores.
- Prose uses the ASCII hyphen (`-`); the em dash is not used anywhere, which a core test
  checks.
