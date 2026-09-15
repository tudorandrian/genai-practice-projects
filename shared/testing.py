"""Shared pytest fixtures; loaded through the root conftest.py."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest


@pytest.fixture
def tmp_output(tmp_path: Path) -> Path:
    out = tmp_path / "output"
    out.mkdir()
    return out


@pytest.fixture
def stub_llm() -> Callable[[str], str]:
    def reply(prompt: str) -> str:
        return f"STUB: {prompt[:40].strip()}"

    return reply
