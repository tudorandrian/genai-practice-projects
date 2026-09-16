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


def test_no_tracked_binary_artifact() -> None:
    import subprocess

    from scripts.release_check import BINARY_SUFFIXES

    files = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True
    ).stdout
    # .gitignore alone does not stop `git add -f model.pkl`; this test runs on every PR.
    bad = [f for f in files.decode().split("\0") if f and Path(f).suffix in BINARY_SUFFIXES]
    assert bad == []


def test_every_registry_entry_matches_its_console_script() -> None:
    import importlib.util

    from shared.registry import ENTRIES

    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "scripts"
    ]
    assert set(scripts) - {"demo"} == {entry.slug for entry in ENTRIES}
    for entry in ENTRIES:
        target_module = entry.target.split(":")[0]
        script_module = scripts[entry.slug].split(":")[0]
        assert script_module == target_module, entry.slug
        # find_spec locates the module without importing it, so models/rag projects stay
        # cheap to check in the core environment.
        assert importlib.util.find_spec(target_module) is not None, entry.slug


def test_no_tracked_text_file_contains_an_em_dash() -> None:
    from shared.blocklist import tracked_text_files

    em_dash = chr(0x2014)  # spelled as a code point so this file passes its own test
    offenders = [
        str(path.relative_to(ROOT))
        for path in tracked_text_files()
        if em_dash in path.read_text(encoding="utf-8", errors="replace")
    ]
    assert offenders == [], "use the ASCII hyphen '-' instead of the em dash"


def test_default_hugging_face_models_are_pinned_to_a_hub_commit() -> None:
    import re

    from projects.p08_image_captioning import captioner
    from projects.p09_chatbot import engine
    from projects.p10_meeting_assistant import assistant
    from projects.p11_rag_chatbot import rag_chatbot
    from projects.p12_study_hub import tutor

    revisions = [
        captioner.MODEL_REVISION,
        engine.MODEL_REVISION,
        assistant.WHISPER_REVISION,
        assistant.DEFAULT_LOCAL_LLM_REVISION,
        rag_chatbot.EMBED_REVISION,
        tutor.EMBED_REVISION,
    ]
    assert all(re.fullmatch(r"[0-9a-f]{40}", rev) for rev in revisions)
    assert rag_chatbot.EMBED_REVISION == tutor.EMBED_REVISION


def test_container_images_are_pinned_by_digest_and_ollama_matches_ci() -> None:
    import re

    pinned = re.compile(r"(?:FROM|--from=|image:)\s*(\S+)")
    found = {
        name: pinned.findall((ROOT / name).read_text(encoding="utf-8"))
        for name in ("Dockerfile", "compose.yaml", ".github/workflows/heavy.yml")
    }
    images = [image for names in found.values() for image in names]
    assert images and all("@sha256:" in image for image in images)
    assert set(found["compose.yaml"]) == set(found[".github/workflows/heavy.yml"])


def test_branch_protection_as_code_requires_the_ci_jobs() -> None:
    import json

    protection = json.loads((ROOT / ".github" / "branch-protection.json").read_text("utf-8"))
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "matrix: { os: [ubuntu-latest, windows-latest] }" in ci and "\n  gitleaks:\n" in ci
    assert protection["required_status_checks"]["contexts"] == [
        "checks (ubuntu-latest)",
        "checks (windows-latest)",
        "gitleaks",
    ]
    assert protection["enforce_admins"] and protection["required_linear_history"]
