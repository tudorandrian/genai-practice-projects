from pathlib import Path

import pytest

from shared import demo, registry

pytestmark = pytest.mark.core


def ok_demo() -> demo.DemoResult:
    return demo.DemoResult(name="p00-ok", status="ok", figures={"rows": "5"})


def boom_demo() -> demo.DemoResult:
    raise RuntimeError("boom")


def test_runner_times_each_entry_and_catches_failures(tmp_path: Path) -> None:
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


def test_main_tier_flags() -> None:
    assert demo.tiers_from_args([]) == {"core"}
    assert demo.tiers_from_args(["--models"]) == {"core", "models"}
    assert demo.tiers_from_args(["--all"]) == {"core", "models", "rag"}
