"""scripts/prepare_public_history.py on a small throwaway repository."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts import prepare_public_history as pph

pytestmark = pytest.mark.core

FIXTURE = pph.FIXTURES[0]
OLD_EMAIL = "someone@university.example"
NEW_EMAIL = "1+someone@users.noreply.github.com"
REMOVED_ENTRY = '    "acc" + "ount9",\n'  # joins to a value no current pattern has
KEPT_ENTRY = '    "lara" + "gon",\n'  # joins to a current pattern


def _commit(repo: Path, files: dict[str, str], message: str) -> str:
    for name, text in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))
    identity = ["-c", "user.name=Some One", "-c", f"user.email={OLD_EMAIL}"]
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), *identity, "commit", "-q", "-m", message], check=True)
    return pph.git(repo, "rev-parse", "HEAD").decode().strip()


@pytest.fixture
def source(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "source"
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    blocklist_v1 = "PATTERNS = [\n" + KEPT_ENTRY + REMOVED_ENTRY + "]\n"
    _commit(
        repo,
        {FIXTURE: "original question\n", pph.BLOCKLIST_FILE: blocklist_v1},
        "feat: first " + chr(0x2014) + " with a dash",
    )
    blocklist_v2 = "PATTERNS = [\n" + KEPT_ENTRY + "]\n"
    rewritten = _commit(
        repo,
        {FIXTURE: "invented question\n", pph.BLOCKLIST_FILE: blocklist_v2},
        "fix: rewrite the fixture",
    )
    _commit(repo, {"README.md": "done\n"}, "docs: readme")
    return repo, rewritten


def test_scrub_blocklist_drops_only_entries_the_current_blocklist_lacks() -> None:
    text = ("PATTERNS = [\n" + KEPT_ENTRY + REMOVED_ENTRY + "]\n").encode()
    assert (
        pph.scrub_blocklist(text, ["lara" + "gon"])
        == ("PATTERNS = [\n" + KEPT_ENTRY + "]\n").encode()
    )


def test_rewrite_produces_a_verified_publishable_clone(
    source: tuple[Path, str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo, rewritten = source
    dest = tmp_path / "public"
    code = pph.main(
        [str(repo), str(dest), "--old-email", OLD_EMAIL, "--new-email", NEW_EMAIL,
         "--fixtures-rewritten-in", rewritten]
    )  # fmt: skip
    assert code == 0, capsys.readouterr().out
    out = capsys.readouterr().out
    assert "3 commits rewritten" in out
    assert "verified: publishable" in out

    first = pph.git(dest, "rev-list", "--max-parents=0", "HEAD").decode().strip()
    assert pph.git(dest, "show", f"{first}:{FIXTURE}") == b"invented question\n"
    assert REMOVED_ENTRY.encode() not in pph.git(dest, "show", f"{first}:{pph.BLOCKLIST_FILE}")
    assert KEPT_ENTRY.encode() in pph.git(dest, "show", f"{first}:{pph.BLOCKLIST_FILE}")
    assert set(pph.git(dest, "log", "--format=%ae%n%ce").decode().split()) == {NEW_EMAIL}
    assert pph.git(dest, "log", "-1", "--format=%s", first).decode().strip() == (
        "feat: first - with a dash"
    )
    assert pph.git(dest, "rev-parse", "HEAD^{tree}") == pph.git(repo, "rev-parse", "HEAD^{tree}")


def test_refuses_an_existing_destination(source: tuple[Path, str], tmp_path: Path) -> None:
    repo, _ = source
    assert pph.main([str(repo), str(tmp_path), "--old-email", "a", "--new-email", "b"]) == 1
