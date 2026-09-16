# 0005: Underscore project directories, hyphenated console scripts

## Context

Each project is both a Python package (imported by its tests and by `shared/`) and
a console entry point declared in `[project.scripts]`. Python module paths must be
importable identifiers; hyphens are not legal in a module path.

## Decision

Project directories use underscores, e.g. `projects/p01_mini_etl/`, so they are
directly importable. The console-script name registered in `[project.scripts]`
keeps the hyphenated, conventional CLI form, e.g. `p01-mini-etl`.

## Consequences

- `import projects.p01_mini_etl` works without special-casing.
- The CLI name a user types (`p01-mini-etl`) stays conventional and readable,
  decoupled from the import path.
- Every project PR must register its script under the hyphenated name pointing at
  the underscored module path. The core test
  `shared/tests/test_scaffold.py::test_every_registry_entry_matches_its_console_script`
  checks that every `shared/registry.py` entry has a `[project.scripts]` entry under its
  slug, pointing at the same module, and that the module exists.
