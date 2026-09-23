"""Mechanical half of docs/release-gate.md. Exit 1 on any failing check.

Covers gate items A3-A5 and A7 (A1/A2/A6 are recorded from CI/heavy.yml runs, not run
here - see docs/release-gate.md). `ci.yml` runs `python -m scripts.release_check --offline`
on every push and pull request (Ubuntu job); the link check needs a run without
`--offline`.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import requests

from shared import blocklist

ROOT = Path(__file__).resolve().parents[1]
BINARY_SUFFIXES = {".ipynb", ".pkl", ".joblib", ".sqlite3", ".bin", ".pt", ".pth", ".h5", ".keras"}
# The root uv.lock is a generated lockfile that must be committed for reproducible
# installs; it is not course material, a secret or a model file, so it is exempt from
# the 100 KB rule. The exemption is by path relative to the repository root, mirroring
# shared/tests/test_scaffold.py and the check-added-large-files pre-commit hook
# (.pre-commit-config.yaml): a uv.lock in any other directory is not exempt.
LARGE_FILE_EXEMPT = {"uv.lock"}


@dataclass
class Check:
    name: str
    status: str  # pass | fail | skip
    detail: str = ""


def tracked(root: Path) -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True
    ).stdout.decode()
    return [root / f for f in out.split("\0") if f]


def check_large_files(root: Path) -> Check:
    big = [
        p
        for p in tracked(root)
        if p.relative_to(root).as_posix() not in LARGE_FILE_EXEMPT and p.stat().st_size > 100 * 1024
    ]
    return Check("no-large-files", "fail" if big else "pass", ", ".join(str(p) for p in big))


def check_binary_artifacts(root: Path) -> Check:
    bad = [p for p in tracked(root) if p.suffix in BINARY_SUFFIXES]
    return Check("no-binary-artifacts", "fail" if bad else "pass", ", ".join(str(p) for p in bad))


def check_blocklist(root: Path) -> Check:
    hits = blocklist.scan(
        blocklist.tracked_text_files(root=root), allowed=blocklist.ALLOWED, root=root
    )
    return Check(
        "blocklist", "fail" if hits else "pass", "; ".join(f"{p.name}:{n}" for p, n, _ in hits[:10])
    )


# Licence identifiers, and the prose a project uses instead of a table row when its data
# needs no third-party licence because nothing was sourced from anywhere else ("no
# external dataset", "hand-written", ...). Either form is acceptable evidence - gate item
# A5 requires a source and a licence be stated, not any particular Markdown formatting.
#
# The short identifiers (MIT, BSD, ODC, Apache, CC/CC0/CC-BY) are wrapped in \b word
# boundaries: without them, "MIT" matches inside an ordinary word
# like "committed" ("com-MIT-ted"), which would make a section pass with no licence
# stated at all. "licen[cs]e" and the multi-word prose phrases below don't need this -
# they aren't short enough to collide with unrelated English words.
DATASET_LICENCE_EVIDENCE = re.compile(
    r"licen[cs]e|public domain|\b(?:CC0|CC-BY|CC|BSD|ODC|MIT|Apache)\b|"
    r"no external dataset|hand-written|synthetic|invented|not derived from",
    re.I,
)


def check_dataset_licences(root: Path) -> Check:
    """Every project README must name its data's source and licence (gate item A5).

    This is a smoke test, not an audit. It proves that each project's
    "## Datasets and licences" section exists and says *something* recognisable as
    licence or provenance evidence, in a table row or in prose - it does not prove that
    a project with several datasets documents every one of them, nor that the licence
    named is the correct one for what actually ships. Per-dataset completeness is what
    the blind review (gate item B) and the owner's own read (docs/release-gate.md C8)
    are for; this check only catches a section that is missing or silent altogether.
    """
    missing = []
    for readme in sorted((root / "projects").glob("p*/README.md")):
        text = readme.read_text(encoding="utf-8")
        section = (
            text.split("## Datasets and licences")[-1] if "## Datasets and licences" in text else ""
        )
        if "## Datasets and licences" not in text or not DATASET_LICENCE_EVIDENCE.search(section):
            missing.append(readme.parent.name)
    return Check("datasets-have-licences", "fail" if missing else "pass", ", ".join(missing))


def _last_commit_epoch(root: Path, paths: list[Path]) -> int:
    """Unix timestamp of the last commit touching any of `paths` (0 if none tracked)."""
    existing = [p for p in paths if p.exists()]
    if not existing:
        return 0
    rel = [str(p.relative_to(root)) for p in existing]
    out = subprocess.run(
        ["git", "log", "-1", "--format=%ct", "--", *rel],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return int(out) if out else 0


DEMO_SUMMARY = Path("output") / "demo-summary.md"
_SUMMARY_HEADER = re.compile(r"^# Demo summary - (\d{4}-\d{2}-\d{2} \d{2}:\d{2}) UTC$")
_SUMMARY_ROW = re.compile(r"^\| (p\d\d-[a-z0-9-]+) \| (ok|skipped|failed) \|")


def _demo_run_evidence(root: Path) -> tuple[int, dict[str, str]]:
    """When `uv run demo` last wrote its summary (UTC epoch, minute precision) and each
    project's status in it; `(0, {})` if there is no readable summary."""
    path = root / DEMO_SUMMARY
    if not path.exists():
        return 0, {}
    lines = path.read_text(encoding="utf-8").splitlines()
    header = _SUMMARY_HEADER.match(lines[0]) if lines else None
    if header is None:
        return 0, {}
    when = datetime.strptime(header.group(1), "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
    statuses = {m.group(1): m.group(2) for line in lines if (m := _SUMMARY_ROW.match(line))}
    return int(when.timestamp()), statuses


def _output_is_clean(root: Path, project: Path) -> bool:
    """True when no tracked file under `project/output` differs from the last commit."""
    out = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--",
            str((project / "output").relative_to(root)),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return out == ""


def check_metrics_fresh(root: Path) -> Check:
    """Every project must have a committed `output/metrics.txt`: fail if it is missing,
    or if a `uv run demo` run newer than the project's last code commit changed a stale
    proof. A stale proof (its last commit predates the project's last code commit)
    passes once a later demo run reproduces it; otherwise it is reported as a `skip`
    naming the required action.

    Git commit history, not `Path.stat().st_mtime`, is what "predates" means here: a
    fresh clone (or CI checkout) writes every tracked file to disk in whatever order
    the checkout happens to use, so on-disk mtimes carry no information about which
    file was *authored* more recently - the last commit touching each path does.

    That recency comparison is a useful signal but cannot be a pass/fail gate: a code
    change can leave a project's output byte-identical, and git records no new commit
    for an unchanged file. The proof's commit timestamp then stays behind the code's
    permanently, with no honest action able to advance it - "touching" the file without
    a real content change would only game the check. So a stale timestamp alone is real
    evidence worth surfacing (as `skip`, with the action to take), not by itself an
    unambiguous defect; it becomes one only once a demo run reproduces it and still
    leaves the committed proof different from what the code now produces.

    A stale timestamp is cleared by evidence instead: a `uv run demo` summary written
    after the project's last code commit that lists the project as `ok`, with no tracked
    change under its `output/`. That is exactly the action the skip message asks for. If
    the demo ran and changed a committed proof, the proof is wrong, and that is a fail.
    """
    missing: list[str] = []
    stale: list[str] = []
    changed: list[str] = []
    demo_time, demo_status = _demo_run_evidence(root)
    for project in sorted((root / "projects").glob("p*")):
        metrics = project / "output" / "metrics.txt"
        if not metrics.exists():
            missing.append(project.name)
            continue
        code_time = _last_commit_epoch(root, sorted(project.glob("*.py")))
        if _last_commit_epoch(root, [metrics]) >= code_time:
            continue
        reproduced = (
            demo_time >= code_time and demo_status.get(project.name.replace("_", "-")) == "ok"
        )
        if not reproduced:
            stale.append(project.name)
        elif not _output_is_clean(root, project):
            changed.append(project.name)
    if missing:
        return Check("metrics-fresh", "fail", f"missing output/metrics.txt: {', '.join(missing)}")
    if changed:
        return Check(
            "metrics-fresh",
            "fail",
            f"proof differs from what the code produces now: {', '.join(changed)} - "
            "review `git diff -- projects/*/output` and commit the regenerated proof",
        )
    if stale:
        return Check(
            "metrics-fresh",
            "skip",
            f"code changed since these proofs were last committed: {', '.join(stale)} - "
            "re-run `uv run demo --all` and confirm `git status` stays clean",
        )
    return Check("metrics-fresh", "pass")


# RFC 2606 reserved example domains, plus loopback: these appear in READMEs only as
# "run this locally" illustrations, never as references a reader would follow, so they
# are not links in the sense gate item A7 means and are skipped rather than checked.
EXAMPLE_HOSTS = {"example.com", "example.net", "example.org", "example.invalid"}
LOOPBACK_HOSTS = {"localhost", "127.0.0.1"}
# The repository's own URL - and anything under it (Actions runs, the CI badge, the
# commit history, the pull-request list) - cannot resolve for a logged-out client while
# the repository is private: gate item A7's own wording ("resolves for a logged-out
# client") makes that a post-publication check by definition, not a broken link.
REPO_URL = "https://github.com/tudorandrian/genai-practice-projects"


def _is_example_host(url: str) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    return host in LOOPBACK_HOSTS or host in EXAMPLE_HOSTS


def check_links(root: Path, *, network: bool = True) -> Check:
    """Every link in the root and per-project READMEs must resolve for a logged-out
    client (gate item A7).

    Two categories of URL are deliberately not treated as failures: documentation
    example hosts (`_is_example_host`) are skipped outright, and any URL under the
    repository's own GitHub path (the bare repo URL, an Actions run, the CI badge, the
    commit history, the pull-request list, ...), which all 404 for a logged-out client
    until the repository is made public, are reported as a distinct "skip"
    rather than lumped in with genuine breakage. Any other URL that fails to resolve
    still fails this check.
    """
    # Real HTTP requests: excluded from the "core" marker (CI runs "core and not
    # network" and must stay offline), so callers that need an offline run pass
    # network=False and get a "skip" instead of a real check.
    if not network:
        return Check("links-resolve", "skip", "network disabled")
    urls: set[str] = set()
    for md in [root / "README.md", *sorted((root / "projects").glob("p*/README.md"))]:
        # Backtick excluded too: a URL wrapped in Markdown inline code (`` `http://…` ``)
        # would otherwise capture the closing backtick as part of the "URL", which
        # defeats _is_example_host's exact hostname match below.
        found = re.findall(r"https?://[^\s)>\]`]+", md.read_text(encoding="utf-8"))
        # A URL at the end of a sentence also picks up trailing punctuation that was
        # never part of the URL (e.g. "See <repo-url>." - the check needs the bare URL
        # to recognise REPO_URL below, not "<repo-url>.").
        urls |= {u.rstrip(".,;:") for u in found}
    urls = {u for u in urls if not _is_example_host(u)}

    broken = []
    pending_private = []
    for url in sorted(urls):
        try:
            code = requests.head(url, allow_redirects=True, timeout=15).status_code
        except requests.RequestException as exc:
            broken.append(f"ERR {url} ({exc.__class__.__name__})")
            continue
        if code == 404 and (url == REPO_URL or url.startswith(REPO_URL + "/")):
            pending_private.append(url)
        elif code >= 400 and code != 999:  # 999 = LinkedIn's bot answer
            broken.append(f"{code} {url}")

    if broken:
        return Check("links-resolve", "fail", "; ".join(broken))
    if pending_private:
        return Check(
            "links-resolve",
            "skip",
            f"repository is private; re-run after publication: {', '.join(pending_private)}",
        )
    return Check("links-resolve", "pass", "")


def run_all(root: Path = ROOT, *, network: bool = True) -> list[Check]:
    return [
        check_large_files(root),
        check_binary_artifacts(root),
        check_blocklist(root),
        check_dataset_licences(root),
        check_metrics_fresh(root),
        check_links(root, network=network),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="release_check")
    parser.add_argument(
        "--offline", action="store_true", help="skip the link check (no network requests)"
    )
    args = parser.parse_args(argv)
    results = run_all(network=not args.offline)
    for r in results:
        print(f"{r.status.upper():5} {r.name:24} {r.detail}")
    return 0 if all(r.status != "fail" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
