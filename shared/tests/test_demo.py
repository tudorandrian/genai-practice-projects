from pathlib import Path

import pytest

from shared import demo, registry

pytestmark = pytest.mark.core


def ok_demo() -> demo.DemoResult:
    return demo.DemoResult(name="p00-ok", status="ok", figures={"rows": "5"})


def boom_demo() -> demo.DemoResult:
    raise RuntimeError("boom")


def test_runner_times_each_entry_and_catches_failures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    entries = [
        registry.Entry("p00-ok", "core", "shared.tests.test_demo:ok_demo"),
        registry.Entry("p00-boom", "core", "shared.tests.test_demo:boom_demo"),
        registry.Entry("p00-heavy", "models", "shared.tests.test_demo:ok_demo"),
    ]
    results = demo.run_demo(entries, tiers={"core"}, out_path=tmp_path / "demo-summary.md")
    assert [r.status for r in results] == ["ok", "failed", "skipped"]
    assert results[1].note.startswith("RuntimeError: boom")
    text = (tmp_path / "demo-summary.md").read_text(encoding="utf-8")
    assert "| p00-ok | ok |" in text and "rows=5" in text

    # Every entry - run or tier-skipped - must print its own console line, not
    # just land in the summary file; a tier-skipped entry that never runs is
    # exactly the case that silently regressed before (see shared/demo.py).
    lines = capsys.readouterr().out.splitlines()
    assert any(line.startswith("p00-ok") and "ok" in line for line in lines)
    assert any(line.startswith("p00-boom") and "failed" in line for line in lines)
    assert any(line.startswith("p00-heavy") and "skipped" in line for line in lines)


def test_main_tier_flags() -> None:
    assert demo.tiers_from_args([]) == {"core"}
    assert demo.tiers_from_args(["--models"]) == {"core", "models"}
    assert demo.tiers_from_args(["--all"]) == {"core", "models", "rag"}


def multiline_demo() -> demo.DemoResult:
    raise ValueError("first line | with a pipe\nsecond line")


def test_summary_keeps_one_table_row_per_project_when_a_note_has_pipes_and_newlines(
    tmp_path: Path,
) -> None:
    entries = [registry.Entry("p00-messy", "core", "shared.tests.test_demo:multiline_demo")]
    out = tmp_path / "demo-summary.md"
    demo.run_demo(entries, tiers={"core"}, out_path=out)
    rows = [line for line in out.read_text(encoding="utf-8").splitlines() if "p00-messy" in line]
    assert len(rows) == 1
    assert r"first line \| with a pipe second line" in rows[0]
    # Five cells: the escaped pipe must not count as a column separator.
    assert rows[0].replace(r"\|", "").count("|") == 6


def test_main_writes_the_summary_under_the_repository_root_whatever_the_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Path] = []

    def fake_run_demo(
        entries: list[registry.Entry], tiers: set[str], out_path: Path
    ) -> list[demo.DemoResult]:
        seen.append(out_path)
        return []

    monkeypatch.setattr(demo, "run_demo", fake_run_demo)
    monkeypatch.chdir(tmp_path)
    assert demo.main([]) == 0
    assert seen == [Path(demo.__file__).resolve().parents[1] / "output" / "demo-summary.md"]


def skipping_demo() -> demo.DemoResult:
    return demo.DemoResult(name="p00-skip", status="skipped", note="no speech engine")


def test_strict_flag_is_parsed() -> None:
    assert demo.parse_args([]).strict is False
    assert demo.parse_args(["--all", "--strict"]).strict is True
    assert demo.parse_args(["--all", "--strict"]).tiers == {"core", "models", "rag"}


def test_a_selected_project_that_skips_fails_only_in_strict_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F5 of the 2026-09-22 audit: a green demo step must mean every selected demo
    ran. A tier that was not selected is not a skip of that kind."""
    entries = [
        registry.Entry("p00-ok", "core", "shared.tests.test_demo:ok_demo"),
        registry.Entry("p00-skip", "core", "shared.tests.test_demo:skipping_demo"),
        registry.Entry("p00-heavy", "models", "shared.tests.test_demo:ok_demo"),
    ]
    monkeypatch.setattr(demo, "ENTRIES", entries)
    monkeypatch.setattr(demo, "REPO_ROOT", tmp_path)
    assert demo.main([]) == 0  # permissive: skipped is not failed
    assert demo.main(["--strict"]) == 1  # p00-skip was selected and did not run
    monkeypatch.setattr(demo, "ENTRIES", entries[:1] + entries[2:])
    assert demo.main(["--strict"]) == 0  # p00-heavy is tier-skipped, which is fine


def skipping_demo_with_a_different_name() -> demo.DemoResult:
    """Reports `skipped`, like `skipping_demo`, but under a name that does not
    match its own registry slug - strict mode must still catch this."""
    return demo.DemoResult(name="not-the-slug", status="skipped", note="no speech engine")


def test_strict_mode_catches_a_skip_even_when_the_result_name_differs_from_the_slug(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F4 of the 2026-09-22 whole-branch review: strict mode must pair each entry
    with its result by position (the order run_demo appends them in), not by
    matching DemoResult.name against the registry slug."""
    entries = [
        registry.Entry("p00-ok", "core", "shared.tests.test_demo:ok_demo"),
        registry.Entry(
            "p00-renamed-skip", "core", "shared.tests.test_demo:skipping_demo_with_a_different_name"
        ),
    ]
    monkeypatch.setattr(demo, "ENTRIES", entries)
    monkeypatch.setattr(demo, "REPO_ROOT", tmp_path)
    assert demo.main([]) == 0  # permissive: skipped is not failed
    assert demo.main(["--strict"]) == 1  # selected, skipped, name != slug - still fails


def test_a_selected_tier_without_its_group_is_skipped_with_the_install_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Fresh-clone test 2026-09-23: `uv run demo --models` before `uv sync --group models`
    # printed ModuleNotFoundError / KeyError instead of saying what to install.
    monkeypatch.setattr(demo, "_module_available", lambda name: name != "transformers")
    entries = [
        registry.Entry("p00-ok", "core", "shared.tests.test_demo:ok_demo"),
        registry.Entry("p00-heavy", "models", "shared.tests.test_demo:boom_demo"),
    ]
    results = demo.run_demo(entries, tiers={"core", "models"}, out_path=tmp_path / "s.md")
    assert [r.status for r in results] == ["ok", "skipped"]
    assert "uv sync --group models" in results[1].note
    assert "transformers" in results[1].note


def test_rag_tier_also_needs_the_models_modules() -> None:
    assert set(demo.TIER_MODULES["models"]) <= set(demo.TIER_MODULES["rag"])
    assert demo.TIER_MODULES.get("core", ()) == ()


def test_missing_group_is_empty_when_everything_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(demo, "_module_available", lambda _name: True)
    assert demo.missing_group("rag") == []
    assert demo.missing_group("core") == []
