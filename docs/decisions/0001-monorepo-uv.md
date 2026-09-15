# 0001: One monorepo, managed by uv

## Context

The twelve practice projects previously lived as separate trees, each with its own
`requirements.txt` pinning the same core packages (pandas, scikit-learn, Flask, …)
twelve times over. A flat, single-environment install would make even the lightest
project (an ETL script) pull in torch and transformers.

## Decision

One repository, one `pyproject.toml`, managed by `uv` with a committed `uv.lock`.
Dependencies are split into `dependency-groups`: `core` (offline, deterministic
projects), `models` (Hugging Face weights), `rag` (LangChain + Chroma), and `dev`
(test and lint tooling). Projects declare only the group they need.

## Consequences

- One lockfile, one CI workflow, one `uv sync` to reproduce the whole repository.
- Cheap projects stay cheap: `uv sync` alone does not install torch.
- Heavier groups (`models`, `rag`) are opt-in via `uv sync --group …` and exercised
  in the manual `heavy.yml` workflow rather than on every push.
