from pathlib import Path

import pytest

from shared import blocklist

pytestmark = pytest.mark.core


def test_scan_flags_each_pattern_with_line_numbers(tmp_path: Path) -> None:
    bad = tmp_path / "bad.md"
    # Written as "M2-1-" + "LAB3" so this fixture's own text doesn't trip the blocklist
    # scan of this very file.
    lab_code = "M2-1-" + "LAB3"
    # "lara" + "gon" for the same reason: the plain word is itself a blocked pattern
    # (the variable is named without the literal spelling for the same reason).
    dir_name = "lara" + "gon"
    bad.write_text(
        f"fine line\nsee {lab_code} for details\npath C:\\{dir_name}\\bin\n",
        encoding="utf-8",
    )
    hits = blocklist.scan([bad])
    assert [(h[1], h[2]) for h in hits] == [
        (2, "M[0-9]-[0-9]-L(AB)?[0-9]"),
        (3, dir_name),
    ]


def test_allowed_files_are_skipped(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    # "I" + "BM" so this sentence doesn't trip a scan of this test file itself.
    ibm = "I" + "BM"
    text = f"{ibm} Generative AI Engineering Professional Certificate\n"
    readme.write_text(text, encoding="utf-8")
    assert blocklist.scan([readme], allowed={"README.md"}) == []


def test_tracked_repository_is_clean() -> None:
    assert blocklist.scan(blocklist.tracked_text_files(), allowed=blocklist.ALLOWED) == []


def test_only_the_root_readme_is_allowed_not_a_nested_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The allow-list holds repository-relative paths: a project's README.md must still
    # be scanned even though its file name matches the allowed root README.md.
    monkeypatch.setattr(blocklist, "ROOT", tmp_path)
    blocked = "Cour" + "sera"
    root_readme = tmp_path / "README.md"
    nested_readme = tmp_path / "projects" / "x" / "README.md"
    nested_readme.parent.mkdir(parents=True)
    for path in (root_readme, nested_readme):
        path.write_text(f"{blocked} certificate\n", encoding="utf-8")
    hits = blocklist.scan([root_readme, nested_readme], allowed=blocklist.ALLOWED)
    assert [h[0] for h in hits] == [nested_readme]


def test_env_example_is_a_scanned_text_file() -> None:
    assert blocklist.ROOT / ".env.example" in blocklist.tracked_text_files()
