# genai-practice-projects

Twelve self-directed practice projects I built while completing the IBM Generative AI Engineering Professional Certificate (16 courses, August 2026, [verify](https://coursera.org/verify/professional-cert/L9H5FUFBJVTS)). Each project applies techniques from one or more courses to a problem and dataset of my own; the course material itself — notebooks, lesson text, quizzes, lab datasets — is IBM's and is not in this repository.

**Status:** work in progress — projects are being ported one pull request at a time. See [CHANGELOG.md](CHANGELOG.md).

## Quick start

    git clone https://github.com/tudorandrian/genai-practice-projects
    cd genai-practice-projects
    uv sync
    uv run pytest -m "core and not network"
    uv run demo
