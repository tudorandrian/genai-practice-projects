from pathlib import Path

import pytest
import requests

from scripts import release_check

pytestmark = pytest.mark.core


def test_every_check_has_a_name_and_runs() -> None:
    # network=False: this is a "core" test, and CI runs "core and not network", so it
    # must not make real HTTP requests — links-resolve reports "skip" instead of
    # actually checking URLs (see release_check.check_links).
    results = release_check.run_all(root=release_check.ROOT, network=False)
    assert {r.name for r in results} >= {
        "no-large-files",
        "no-binary-artifacts",
        "blocklist",
        "datasets-have-licences",
        "metrics-fresh",
        "links-resolve",
    }
    for r in results:
        assert r.status in {"pass", "fail", "skip"}, r


def test_links_check_is_skipped_without_network() -> None:
    result = release_check.check_links(release_check.ROOT, network=False)
    assert result.status == "skip"


def test_large_file_check_exempts_uv_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The root uv.lock is a generated lockfile that must be committed; it is exempt from
    # the 100 KB rule, mirroring shared/tests/test_scaffold.py and the
    # check-added-large-files pre-commit hook.
    big = tmp_path / "uv.lock"
    big.write_text("x" * (200 * 1024), encoding="utf-8")
    small = tmp_path / "keep.txt"
    small.write_text("fine", encoding="utf-8")

    monkeypatch.setattr(release_check, "tracked", lambda _root: [big, small])
    result = release_check.check_large_files(tmp_path)
    assert result.status == "pass"


def test_large_file_check_exempts_only_the_root_uv_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The pre-commit hook and the scaffold test exempt only the root lockfile; a large
    # uv.lock in a subdirectory is an ordinary large file.
    nested = tmp_path / "projects" / "p99_x" / "uv.lock"
    nested.parent.mkdir(parents=True)
    nested.write_text("x" * (200 * 1024), encoding="utf-8")

    monkeypatch.setattr(release_check, "tracked", lambda _root: [nested])
    result = release_check.check_large_files(tmp_path)
    assert result.status == "fail"
    assert "p99_x" in result.detail


def test_metrics_fresh_compares_commit_history_not_mtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A project whose output/metrics.txt was committed in the same commit as its
    # code (or later) must never be named as missing or stale, however file mtimes
    # on disk read right now (checkout order, not commit history). Synthetic, not
    # a real project's git history: a later commit that legitimately touches a
    # project's code without changing its byte-identical output — exactly what a
    # fix wave does — would otherwise make this test flaky against the live repo.
    project = tmp_path / "projects" / "p99_fresh"
    (project / "output").mkdir(parents=True)
    (project / "app.py").write_text("x = 1\n", encoding="utf-8")
    (project / "output" / "metrics.txt").write_text("ok\n", encoding="utf-8")

    def fake_last_commit_epoch(_root: Path, paths: list[Path]) -> int:
        # Proof committed in the same commit as the code: never stale.
        return 100

    monkeypatch.setattr(release_check, "_last_commit_epoch", fake_last_commit_epoch)
    result = release_check.check_metrics_fresh(tmp_path)
    assert "p99_fresh" not in result.detail


def test_metrics_fresh_fails_only_when_the_proof_is_missing(tmp_path: Path) -> None:
    # A missing output/metrics.txt is the one unambiguous defect this check can
    # call a fail — no honest regeneration can be blocked by anything else.
    project = tmp_path / "projects" / "p99_missing"
    project.mkdir(parents=True)
    (project / "app.py").write_text("x = 1\n", encoding="utf-8")
    result = release_check.check_metrics_fresh(tmp_path)
    assert result.status == "fail"
    assert "p99_missing" in result.detail
    assert "missing" in result.detail.lower()


def test_metrics_fresh_reports_a_stale_proof_as_skip_with_the_action_to_take(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A proof whose last commit predates the project's last code commit is real,
    # useful signal — but a code change can legitimately leave output byte-identical, so
    # git records no new commit for it and the proof's timestamp can never honestly
    # catch up. That must not be a fail (unsatisfiable by any honest action); it is a
    # skip naming what to do: re-run `uv run demo --all` and confirm `git status` is
    # clean.
    project = tmp_path / "projects" / "p99_stale"
    (project / "output").mkdir(parents=True)
    (project / "app.py").write_text("x = 1\n", encoding="utf-8")
    (project / "output" / "metrics.txt").write_text("ok\n", encoding="utf-8")

    def fake_last_commit_epoch(_root: Path, paths: list[Path]) -> int:
        # Code newer than the proof: exactly the "stale, not missing" case.
        return 200 if any(p.suffix == ".py" for p in paths) else 100

    monkeypatch.setattr(release_check, "_last_commit_epoch", fake_last_commit_epoch)
    result = release_check.check_metrics_fresh(tmp_path)
    assert result.status == "skip"
    assert "p99_stale" in result.detail
    assert "uv run demo --all" in result.detail
    assert "git status" in result.detail


def _project_tree(tmp_path: Path, readmes: dict[str, str]) -> Path:
    (tmp_path / "README.md").write_text("", encoding="utf-8")
    for slug, text in readmes.items():
        project = tmp_path / "projects" / slug
        project.mkdir(parents=True)
        (project / "README.md").write_text(text, encoding="utf-8")
    return tmp_path


def test_dataset_licence_check_accepts_prose_without_a_table(tmp_path: Path) -> None:
    # A project whose data needs no third-party licence because nothing was
    # sourced externally states that in prose (P07's actual wording, reused here), not
    # a `|...|...licence` table row — this must be accepted, not just the table form.
    root = _project_tree(
        tmp_path,
        {
            "p07_sentiment_api": (
                "## Datasets and licences\n\n"
                "There is no external dataset. `LEXICON` is a small hand-written "
                "Romanian sentiment word list, written for this repository.\n"
            )
        },
    )
    result = release_check.check_dataset_licences(root)
    assert result.status == "pass"


def test_dataset_licence_check_still_fails_a_silent_section(tmp_path: Path) -> None:
    root = _project_tree(
        tmp_path,
        {"p99_nothing_said": "## Datasets and licences\n\nTBD.\n"},
    )
    result = release_check.check_dataset_licences(root)
    assert result.status == "fail"
    assert "p99_nothing_said" in result.detail


def test_dataset_licence_check_does_not_match_mit_inside_an_ordinary_word(
    tmp_path: Path,
) -> None:
    # Without a \b word boundary, "MIT" matches inside "committed"
    # ("com-MIT-ted"), so a section with no licence stated at all would pass.
    root = _project_tree(
        tmp_path,
        {"p99_no_licence": "## Datasets and licences\n\nThis project uses a committed CSV file.\n"},
    )
    result = release_check.check_dataset_licences(root)
    assert result.status == "fail"
    assert "p99_no_licence" in result.detail


def test_links_check_skips_localhost_and_example_hosts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Loopback and RFC 2606 example domains are documentation illustrations, not
    # links to check — assert no HTTP request is even attempted for them.
    def _unexpected_call(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("check_links should not make a request for an example host")

    monkeypatch.setattr(requests, "head", _unexpected_call)
    root = _project_tree(
        tmp_path,
        {
            "p07_sentiment_api": (
                "curl http://127.0.0.1:5000/sentiment\n"
                "See https://example.com/not-a-real-link and http://localhost:7860.\n"
            )
        },
    )
    result = release_check.check_links(root)
    assert result.status == "pass"


def test_links_check_reports_private_repo_404_as_skip_not_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Resp:
        status_code = 404

    monkeypatch.setattr(requests, "head", lambda *a, **k: _Resp())
    root = _project_tree(tmp_path, {"p01_mini_etl": f"    git clone {release_check.REPO_URL}\n"})
    result = release_check.check_links(root)
    assert result.status == "skip"
    assert "private" in result.detail


def test_links_check_treats_any_repo_subpath_404_the_same_way(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The README also links to Actions/commits/pulls under the repo's own path, not
    # just the bare repo URL — all of those 404 while private too.
    class _Resp:
        status_code = 404

    monkeypatch.setattr(requests, "head", lambda *a, **k: _Resp())
    root = _project_tree(
        tmp_path,
        {
            "p01_mini_etl": (
                f"[CI]({release_check.REPO_URL}/actions/workflows/ci.yml/badge.svg)\n"
                f"[history]({release_check.REPO_URL}/commits/main)\n"
            )
        },
    )
    result = release_check.check_links(root)
    assert result.status == "skip"
    assert "private" in result.detail
    assert "actions/workflows/ci.yml" in result.detail
    assert "commits/main" in result.detail


def test_links_check_skips_a_backtick_wrapped_loopback_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression: Markdown inline code around a loopback URL (`` `http://127.0.0.1` ``,
    # P09's actual wording) must not let the closing backtick leak into the extracted
    # URL — that would corrupt the hostname and defeat the loopback skip below.
    def _unexpected_call(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("check_links should not make a request for a loopback host")

    monkeypatch.setattr(requests, "head", _unexpected_call)
    root = _project_tree(
        tmp_path,
        {"p09_chatbot": "demo reached only from `http://127.0.0.1`, but it would not be an\n"},
    )
    result = release_check.check_links(root)
    assert result.status == "pass"


def test_links_check_still_fails_a_genuinely_broken_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _raise(*_args: object, **_kwargs: object) -> None:
        raise requests.ConnectionError("no such host")

    monkeypatch.setattr(requests, "head", _raise)
    root = _project_tree(
        tmp_path, {"p01_mini_etl": "See https://this-host-does-not-exist.invalid-test.\n"}
    )
    result = release_check.check_links(root)
    assert result.status == "fail"
    assert "this-host-does-not-exist.invalid-test" in result.detail


def test_main_offline_makes_no_request_and_exits_1_on_a_failing_check(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # ci.yml runs `python -m scripts.release_check --offline`: it must stay off the
    # network and fail the job when any check fails.
    seen: list[bool] = []

    def fake_run_all(root: Path = release_check.ROOT, *, network: bool = True) -> list[object]:
        seen.append(network)
        return [
            release_check.Check("ok-check", "pass"),
            release_check.Check("bad-check", "fail", "boom"),
        ]

    monkeypatch.setattr(release_check, "run_all", fake_run_all)
    assert release_check.main(["--offline"]) == 1
    assert seen == [False]
    assert "FAIL  bad-check" in capsys.readouterr().out
