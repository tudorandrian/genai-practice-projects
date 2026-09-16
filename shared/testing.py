"""Shared pytest fixture (`tmp_output`); loaded through the root conftest.py.

The LLM-backed projects (P10-P12) each define their own stub provider next to the code
it stands in for, so there is no shared LLM fixture here.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def tmp_output(tmp_path: Path) -> Path:
    out = tmp_path / "output"
    out.mkdir()
    return out
