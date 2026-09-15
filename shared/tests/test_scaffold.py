import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.core
ROOT = Path(__file__).resolve().parents[2]


def test_pyproject_declares_the_four_groups_and_five_markers() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert set(data["dependency-groups"]) == {"core", "models", "rag", "dev"}
    markers = {m.split(":")[0] for m in data["tool"]["pytest"]["ini_options"]["markers"]}
    assert markers == {"core", "network", "models", "rag", "llm"}


def test_no_tracked_file_over_100kb() -> None:
    import subprocess

    files = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True
    ).stdout
    # uv.lock is a mechanically generated lockfile, not course material, a secret, or a
    # model file; the 100 KB rule targets those, so the lockfile is exempt by convention
    # (mirrored in the check-added-large-files pre-commit hook).
    names = [f for f in files.decode().split("\0") if f and f != "uv.lock"]
    big = [f for f in names if (ROOT / f).stat().st_size > 100 * 1024]
    assert big == []
