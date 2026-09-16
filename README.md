# genai-practice-projects

[![CI](https://github.com/tudorandrian/genai-practice-projects/actions/workflows/ci.yml/badge.svg)](https://github.com/tudorandrian/genai-practice-projects/actions/workflows/ci.yml)

Twelve self-directed practice projects I built while completing the IBM Generative AI Engineering Professional Certificate (16 courses, August 2026, [verify](https://coursera.org/verify/professional-cert/L9H5FUFBJVTS)). Each project applies techniques from one or more courses to a problem and dataset of my own; the course material itself — notebooks, lesson text, quizzes, lab datasets — is IBM's and is not in this repository.

## Practice work, professional standard

These are not products, and this repository does not suggest they are. What is senior-level is
the engineering around them: every project is reproducible from a clean clone (P10's
synthetic audio needs a working text-to-speech engine; on the Ubuntu runner it reports
`skipped`), tested in CI on two operating systems ([workflow runs](https://github.com/tudorandrian/genai-practice-projects/actions/workflows/ci.yml)),
documented in English with its limits stated, and free of any material that belongs to IBM or
Coursera. The judgement on display is in the boundaries drawn, the tests written, and the
delivery discipline — one pull request per project, squash-merged, history readable end to end
([commit history](https://github.com/tudorandrian/genai-practice-projects/commits/main),
[pull requests](https://github.com/tudorandrian/genai-practice-projects/pulls?q=is%3Apr)).

## Quick start

    git clone https://github.com/tudorandrian/genai-practice-projects
    cd genai-practice-projects
    uv sync
    uv run pytest -m "core and not network"
    uv run demo

`uv sync` installs the `core` group — enough for P01-P07 and the command above. P08-P10 need
`uv sync --group models`; P11-P12 need `uv sync --group rag` (a superset of `models`). The
table below marks each project's tier; `uv run demo --all` needs the `rag` group installed first.

## The twelve projects

| Project | What it does | Tier | Run it |
|---|---|---|---|
| [P01 mini-ETL](projects/p01_mini_etl/README.md) | Dataset-agnostic CLI ETL: audits, imputes, and exports a messy tabular file across six input formats. | core | `uv run p01-mini-etl --demo` |
| [P02 question bank](projects/p02_question_bank/README.md) | Dataset-agnostic question-bank builder that parses Markdown lesson notes into a validated JSON bank. | core | `uv run p02-question-bank --demo` |
| [P03 regression](projects/p03_regression/README.md) | Linear regression run across seven public datasets to show the disciplined workflow, not just the model call. | core | `uv run p03-regression --demo` |
| [P04 decision tree](projects/p04_decision_tree/README.md) | Explainable decision-tree classifier where the exported rule tree *is* the model. | core | `uv run p04-decision-tree --demo` |
| [P05 segmentation](projects/p05_segmentation/README.md) | Unsupervised K-Means + PCA customer segmentation with an objectively chosen k. | core | `uv run p05-segmentation --demo` |
| [P06 ML pipeline](projects/p06_ml_pipeline/README.md) | Production-shaped `Pipeline` + `GridSearchCV` classifier with a model card written per run. | core | `uv run p06-ml-pipeline --demo` |
| [P07 sentiment API](projects/p07_sentiment_api/README.md) | Small Flask JSON API that scores short texts for sentiment; the first script-to-service step. | core | `uv run p07-sentiment-api --demo` |
| [P08 image captioning](projects/p08_image_captioning/README.md) | Turns any image into a caption with BLIP, served via Gradio and batch inference. | models | `uv run p08-image-captioning --demo` |
| [P09 chatbot](projects/p09_chatbot/README.md) | Local Flask + Hugging Face chatbot (BlenderBot) with a windowed conversation history. | models | `uv run p09-chatbot --demo` |
| [P10 meeting assistant](projects/p10_meeting_assistant/README.md) | Chains Whisper transcription into an LLM summarization step, with a documented provider seam. | models | `uv run p10-meeting-assistant --demo` |
| [P11 RAG chatbot](projects/p11_rag_chatbot/README.md) | Retrieval-augmented Q&A over your own PDF/Markdown/text documents via LangChain + Chroma. | rag | `uv run p11-rag-chatbot --demo` |
| [P12 study hub](projects/p12_study_hub/README.md) | Capstone: P02's quiz engine, a progress tracker, and a RAG tutor over one original synthetic corpus. | rag | `uv run p12-study-hub --demo` |

Run every project's demo in one command:

    uv run demo              # core group, P01-P07, offline, deterministic
    uv run demo --models     # + models group, P08-P10 — downloads model weights on first
                              # run, then cached locally: BLIP (P08) ~950 MB, BlenderBot
                              # (P09) ~700 MB, Whisper-tiny.en (P10) ~150 MB
    uv run demo --all        # + rag group, P11-P12 — + the embedding model (~90 MB);
                              # LLM steps always use the deterministic stub provider in the demo

P10-P12 also run against a real local model, Qwen2.5 1.5B served by Ollama in Docker:

    docker compose --profile llm up -d   # starts Ollama on 127.0.0.1:11434, pulls the model (~1 GB)
    uv sync --group rag
    uv run pytest -m llm                 # the real-model tests; they skip when Ollama is not running

## More

- [docs/certificate.md](docs/certificate.md) — which courses each project draws on, and the technique it demonstrates.
- [CONTRIBUTING.md](CONTRIBUTING.md) — the two rules this repository holds itself to, and the tooling.
- [docs/release-gate.md](docs/release-gate.md) — the checklist this repository had to clear before it went public.
- [SECURITY.md](SECURITY.md) — how to report a vulnerability, and the dependency advisories that do not apply here.

## Licence

The code and the original material are [MIT-licensed](LICENSE). Committed third-party datasets keep
their own licences, and model weights are downloaded from their publishers at run time; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). IBM and Coursera are trademarks of their
owners; this repository is not affiliated with or endorsed by either.
