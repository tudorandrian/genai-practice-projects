"""Shared pytest fixtures (`tmp_output`, `ollama`); loaded through the root conftest.py.

The LLM-backed projects (P10-P12) each define their own stub provider next to the code
it stands in for. What is shared is `ollama`: the check, for the `llm`-marked tests, that
a real model is being served.
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import pytest

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:1.5b"


@pytest.fixture
def tmp_output(tmp_path: Path) -> Path:
    out = tmp_path / "output"
    out.mkdir()
    return out


@pytest.fixture(scope="session")
def ollama() -> tuple[str, str]:
    """``(url, model)`` of a running Ollama that serves the model, from ``OLLAMA_URL`` and
    ``OLLAMA_MODEL`` (defaults: localhost, ``qwen2.5:1.5b``; see compose.yaml's ``llm``
    profile).

    Skips when Ollama is not reachable or lacks the model, so a plain ``pytest`` run on a
    machine without Ollama stays green. With ``REQUIRE_OLLAMA=1`` (set by the workflow
    job that exists to run these tests) the same condition fails instead: a missing
    server must never turn that job into a green run of skipped tests.
    """
    url = os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL).rstrip("/")
    model = os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=5) as resp:
            served = {m["name"] for m in json.loads(resp.read().decode("utf-8"))["models"]}
        problem = None if model in served else f"Ollama at {url} does not serve {model}"
    except OSError as exc:  # connection refused, timeout (URLError is an OSError)
        problem = f"Ollama is not reachable at {url} ({exc})"
    if problem is not None:
        hint = "start it with `docker compose --profile llm up -d`"
        if os.environ.get("REQUIRE_OLLAMA") == "1":
            pytest.fail(f"{problem}; REQUIRE_OLLAMA=1 - {hint}")
        pytest.skip(f"{problem} - {hint}")
    return url, model
