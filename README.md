# Generative AI Engineering: Twelve Practice Projects

[![CI](https://github.com/tudorandrian/genai-practice-projects/actions/workflows/ci.yml/badge.svg)](https://github.com/tudorandrian/genai-practice-projects/actions/workflows/ci.yml)
[![Licence: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](.python-version)

Twelve independent Python projects that progress from data preparation and classical machine
learning to web APIs, pretrained transformer models and retrieval-augmented generation (RAG).
Every project can be installed, run and tested on its own, and each has its own README with
complete instructions.

The projects are practice work, written while completing the IBM Generative AI Engineering
Professional Certificate (Coursera, 16 courses, completed August 2026,
[credential](https://coursera.org/verify/professional-cert/L9H5FUFBJVTS)). Each one applies
course techniques to a problem and data of its own. No course material (notebooks, lesson
text, quizzes or lab datasets) is included; [docs/certificate.md](docs/certificate.md) maps
each project to the courses it draws on.

**About the name.** The repository is called `genai-practice-projects`: *genai* stands for
generative AI, and *practice* states the scope. These are learning projects built to an
engineering standard, not products.

## Contents

- [Scope and standards](#scope-and-standards)
- [The projects](#the-projects)
- [Requirements](#requirements)
- [Getting started](#getting-started)
- [Running a single project](#running-a-single-project)
- [Running every demo](#running-every-demo)
- [Tests and quality checks](#tests-and-quality-checks)
- [Optional: a real local LLM](#optional-a-real-local-llm)
- [Repository layout](#repository-layout)
- [Documentation](#documentation)
- [Security, citation and licence](#security-citation-and-licence)

## Scope and standards

- **Independent projects.** Each project is a package under `projects/` with its own command,
  tests, `--demo` mode and README. No project depends on another, except P12, which reuses
  P02's question parser.
- **Reproducible.** Dependencies are locked in `uv.lock`. After the first model download,
  every demo runs offline and deterministically. The files committed in each project's
  `output/` folder are exactly what its `--demo` produces.
- **Tested on three operating systems.** CI runs the core tests on Ubuntu and Windows for
  every pull request. The model, RAG and real-LLM tests run on Ubuntu and macOS
  (`.github/workflows/heavy.yml`). One known platform limit: P12's tutor cannot tell
  on-topic from off-topic questions on macOS (two tests are expected failures there; the
  P12 README's *Limits* gives the measurements).
- **Limits stated.** Every project README has a *Limits* section and lists each dataset's
  source and licence.

## The projects

| Project | What it does | Group | Demo |
|---|---|---|---|
| [P01 mini-ETL](projects/p01_mini_etl/README.md) | Command-line ETL that audits, imputes and exports a messy table in six input formats. | core | `uv run p01-mini-etl --demo` |
| [P02 question bank](projects/p02_question_bank/README.md) | Parses Markdown lesson notes into a question bank, validated against a JSON Schema. | core | `uv run p02-question-bank --demo` |
| [P03 regression](projects/p03_regression/README.md) | Linear regression on seven public datasets, with a disciplined evaluation workflow. | core | `uv run p03-regression --demo` |
| [P04 decision tree](projects/p04_decision_tree/README.md) | Explainable decision-tree classifier whose exported rules are the model. | core | `uv run p04-decision-tree --demo` |
| [P05 segmentation](projects/p05_segmentation/README.md) | K-Means and PCA customer segmentation, with the number of clusters chosen by a metric. | core | `uv run p05-segmentation --demo` |
| [P06 ML pipeline](projects/p06_ml_pipeline/README.md) | scikit-learn `Pipeline` with `GridSearchCV`, writing a model card per run. | core | `uv run p06-ml-pipeline --demo` |
| [P07 sentiment API](projects/p07_sentiment_api/README.md) | Flask JSON API that scores the sentiment of short Romanian texts. | core | `uv run p07-sentiment-api --demo` |
| [P08 image captioning](projects/p08_image_captioning/README.md) | Captions images with BLIP, in a Gradio UI and in batch. | models | `uv run p08-image-captioning --demo` |
| [P09 chatbot](projects/p09_chatbot/README.md) | Local Flask chatbot on BlenderBot with a bounded conversation history. | models | `uv run p09-chatbot --demo` |
| [P10 meeting assistant](projects/p10_meeting_assistant/README.md) | Transcribes a recording with Whisper, then summarises it with an LLM. | models | `uv run p10-meeting-assistant --demo` |
| [P11 RAG chatbot](projects/p11_rag_chatbot/README.md) | Answers questions from your own PDF, Markdown or text files with LangChain and Chroma, citing sources. | rag | `uv run p11-rag-chatbot --demo` |
| [P12 study hub](projects/p12_study_hub/README.md) | Capstone: a quiz, a progress tracker and a RAG tutor over an original synthetic course corpus. Reuses P02's question parser. The tutor's relevance gate does not hold on macOS (see its Limits). | rag | `uv run p12-study-hub --demo` |

## Requirements

- **[uv](https://docs.astral.sh/uv/getting-started/installation/) 0.12.15 or a later 0.12
  release**, and Git. uv installs Python 3.13 itself when it is missing.
- **Disk and network.** The `core` group is small. The `models` and `rag` groups install
  PyTorch and download model weights on first use: BLIP (P08) about 950 MB, BlenderBot (P09)
  about 700 MB, Whisper tiny.en (P10) about 150 MB, the embedding model (P11, P12) about 90 MB.
- **Optional.** Docker, for the Linux test container and the local Ollama LLM. On Linux,
  `espeak-ng` for P10's synthetic recording.

The projects are tested on Windows, Ubuntu and macOS.

## Getting started

```bash
git clone https://github.com/tudorandrian/genai-practice-projects.git
cd genai-practice-projects
uv sync                                  # the core and dev groups: enough for P01-P07
uv run pytest -m "core and not network"  # the fast test suite
uv run demo                              # the P01-P07 demos
```

Each project belongs to one dependency group, shown in the table above:

| Group | Install | Projects |
|---|---|---|
| `core` | `uv sync` | P01-P07 |
| `models` | `uv sync --group models` | P08-P10 |
| `rag` | `uv sync --group rag` (includes `models`) | P11-P12 |

## Running a single project

Every project runs independently. Its README has the complete instructions: inputs, options,
example output, design notes and limits. The steps are the same for all twelve, shown here
for P07:

```bash
uv sync                                          # 1. install the project's group (P07: core)
uv run p07-sentiment-api --help                  # 2. list its options
uv run p07-sentiment-api --demo                  # 3. run the offline demo
uv run pytest projects/p07_sentiment_api -q      # 4. run its tests only
```

Most projects also run on your own data or as a service. For example, P07 serves its API on
`http://127.0.0.1:5000`:

```bash
uv run p07-sentiment-api
curl -s -X POST http://127.0.0.1:5000/sentiment -H "Content-Type: application/json" -d "{\"text\": \"Un produs excelent\"}"
```

The servers and web UIs bind to `127.0.0.1` by default. `--demo` regenerates the committed
files in `projects/<project>/output/`. Runs on your own files write to `output/runs/`, which
Git ignores, so your data never overwrites the committed files.

## Running every demo

```bash
uv run demo              # P01-P07: offline and deterministic
uv run demo --models     # adds P08-P10 (needs the models group)
uv run demo --all        # adds P11-P12 (needs the rag group)
uv run demo --all --strict  # also fail if a selected demo was skipped (what CI uses on macOS)
```

The runner writes a summary table to `output/demo-summary.md`. LLM steps in the demos always
use a deterministic stub, so the results do not depend on a running model. A demo that cannot
run on the machine (for example P10 without a speech engine) reports `skipped`; `--strict`
turns that into a non-zero exit.

## Tests and quality checks

Tests are selected with pytest markers:

| Marker | What it covers | Needs |
|---|---|---|
| `core` | Logic, parsers, APIs and demos without model weights | `uv sync` |
| `network` | Downloads of public datasets | internet access |
| `models` | Real inference with Hugging Face models | `models` group |
| `rag` | Indexing and retrieval with Chroma | `rag` group |
| `llm` | P10-P12 against a real LLM | `rag` group and Ollama |

```bash
uv run pytest -m "core and not network"              # what CI requires on every pull request
uv run pytest -m "models or rag"                     # the heavier suites
uv run ruff check . && uv run ruff format --check . && uv run mypy
uv run pre-commit install                            # run the checks on every commit
docker compose run --rm tests                        # the core suite in a Linux container
```

[CONTRIBUTING.md](CONTRIBUTING.md) describes the checks that run before any manual review.

## Optional: a real local LLM

P10-P12 can use Qwen2.5 1.5B, served by Ollama in Docker, instead of the stub:

```bash
docker compose --profile llm up -d   # Ollama on 127.0.0.1:11434; pulls the model (~1 GB)
uv sync --group rag
uv run pytest -m llm                 # skipped when Ollama is not running
docker compose --profile llm down    # stop Ollama; the downloaded model is kept
```

Each project README explains how to point the project itself at Ollama or at the OpenAI API.

## Repository layout

```text
projects/pNN_name/   one project: code, tests, README, output/ with the committed demo results
shared/              demo runner, project registry, dataset cache, test fixtures, blocklist scanner
scripts/             release_check.py: the automated pre-publication checks
docs/                certificate mapping, architecture decision records, release gate
.github/             CI workflows, Dependabot configuration, pull request template
compose.yaml         the Linux test container and the optional Ollama service
```

## Documentation

- [docs/certificate.md](docs/certificate.md): the courses each project draws on and the
  technique it demonstrates.
- [docs/decisions/](docs/decisions/): architecture decision records (monorepo with uv, English
  identifiers, P12's original corpus, CI matrix, package naming, P07's API contract, P10's
  provider fallback).
- [docs/publication.md](docs/publication.md): the publication runbook, step by step.
- [docs/release-gate.md](docs/release-gate.md): the publication checklist, with the evidence
  for each item and a dated record of the publication steps.
- [CHANGELOG.md](CHANGELOG.md): changes by release. The
  [commit history](https://github.com/tudorandrian/genai-practice-projects/commits/main) has
  one squash-merged pull request per change.
- [CONTRIBUTING.md](CONTRIBUTING.md): the repository's rules, tooling and review process.

## Security, citation and licence

- **Security.** [SECURITY.md](SECURITY.md) explains how to report a vulnerability and which
  dependency advisories do not apply here.
- **Citation.** [CITATION.cff](CITATION.cff) provides citation metadata; GitHub shows it as
  *Cite this repository*.
- **Licence.** The code and the original material are [MIT-licensed](LICENSE). Committed
  third-party datasets keep their own licences, and model weights are downloaded from their
  publishers at run time; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). IBM and
  Coursera are trademarks of their respective owners; this repository is not affiliated
  with or endorsed by either.
