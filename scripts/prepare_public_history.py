"""Build a publishable copy of this repository's history in a new local clone.

Release-gate item A8 requires that the history of ``main`` holds nothing unpublishable.
This script clones ``main`` into a new bare repository and rewrites that clone only. Commit
order, dates, authors' names and the final tree stay as they are; four things change:

1. Every version of the P02 fixtures in ``FIXTURES`` written before
   ``--fixtures-rewritten-in`` is replaced by the version from that commit. The first
   versions paraphrased course quiz questions.
2. Older versions of ``shared/blocklist.py`` lose the split-string entries that the current
   blocklist no longer has (they spelled out a local account name).
3. The author and committer e-mail ``--old-email`` becomes ``--new-email``.
4. The em dash in commit messages becomes an ASCII hyphen.

The rewritten clone is then verified. Nothing is pushed: publishing it is the owner's step,
described in docs/publication.md.

Usage:
    uv run python -m scripts.prepare_public_history SOURCE DEST \\
        --old-email OLD --new-email NEW
"""

from __future__ import annotations

import argparse
import contextlib
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from shared import blocklist

FIXTURES = tuple(
    f"projects/p02_question_bank/fixtures/{name}.md"
    for name in (
        "e06_answer_letter_mismatch",
        "e08_tf_missing_answer",
        "e09_oe_missing_answer_line",
        "e15_multi_select_valid",
    )
)
FIXTURES_REWRITTEN_IN = "599ee40"  # fix: final whole-branch review findings (#18)
BLOCKLIST_FILE = "shared/blocklist.py"
EM_DASH = chr(0x2014).encode("utf-8")
_SPLIT_LITERAL = re.compile(rb'^\s*"([^"]*)"\s*\+\s*"([^"]*)",\s*$')


def git(repo: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True).stdout


def scrub_blocklist(contents: bytes, current_patterns: Iterable[str]) -> bytes:
    """Drop ``"a" + "b",`` entries whose joined value is not a current blocklist pattern."""
    keep = set(current_patterns)
    lines = contents.splitlines(keepends=True)
    kept = []
    for line in lines:
        match = _SPLIT_LITERAL.match(line)
        if match and (match.group(1) + match.group(2)).decode("utf-8") not in keep:
            continue
        kept.append(line)
    return b"".join(kept)


def _blob_ids(repo: Path, rev: str, path: str) -> set[str]:
    """Every blob id ``path`` has had in the history reachable from ``rev``."""
    ids = set()
    for line in git(repo, "rev-list", "--objects", rev, "--", path).decode().splitlines():
        parts = line.split(" ", 1)
        if len(parts) == 2 and parts[1] == path:
            ids.add(parts[0])
    return ids


def rewrite(repo: Path, old_email: str, new_email: str, fixtures_rewritten_in: str) -> None:
    """Rewrite ``repo`` (a fresh clone) in place with git-filter-repo."""
    import git_filter_repo as fr  # dev dependency, needed only here

    replacements: dict[bytes, tuple[set[bytes], bytes]] = {}
    for path in FIXTURES:
        try:
            new = git(repo, "show", f"{fixtures_rewritten_in}:{path}")
        except subprocess.CalledProcessError:
            continue  # not in this history
        old = _blob_ids(repo, f"{fixtures_rewritten_in}^", path)
        replacements[path.encode()] = ({i.encode() for i in old}, new)

    def message_callback(message: bytes) -> bytes:
        return message.replace(EM_DASH, b"-")

    def file_info_callback(
        filename: bytes, mode: bytes, blob_id: bytes, value: Any
    ) -> tuple[bytes, bytes, bytes]:
        if filename in replacements:
            old_ids, new = replacements[filename]
            if blob_id in old_ids:
                return filename, mode, value.insert_file_with_contents(new)
        if filename == BLOCKLIST_FILE.encode():
            contents = value.get_contents_by_identifier(blob_id)
            cleaned = scrub_blocklist(contents, blocklist.PATTERNS)
            if cleaned != contents:
                return filename, mode, value.insert_file_with_contents(cleaned)
        return filename, mode, blob_id

    with tempfile.TemporaryDirectory() as tmp:
        mailmap = Path(tmp) / "mailmap"
        mailmap.write_text(f"<{new_email}> <{old_email}>\n", encoding="utf-8")
        with contextlib.chdir(repo):
            args = fr.FilteringOptions.parse_args(
                ["--mailmap", str(mailmap), "--prune-empty", "never", "--quiet"]
            )
            fr.RepoFilter(
                args, message_callback=message_callback, file_info_callback=file_info_callback
            ).run()


def verify(repo: Path, source_tree: str, old_email: str, old_fixture_ids: set[str]) -> list[str]:
    """Return every way the rewritten clone falls short; empty means publishable."""
    problems = []
    if git(repo, "rev-parse", "HEAD^{tree}").decode().strip() != source_tree:
        problems.append("the final tree differs from the source")
    identities = git(repo, "log", "--all", "--format=%ae%n%ce").decode().split()
    if old_email in identities:
        problems.append(f"{old_email} is still an author or committer e-mail")
    if EM_DASH in git(repo, "log", "--all", "--format=%B"):
        problems.append("a commit message still contains an em dash")
    objects = git(repo, "rev-list", "--objects", "--all").decode().splitlines()
    paths: dict[str, set[str]] = {}
    for line in objects:
        parts = line.split(" ", 1)
        if len(parts) == 2:
            paths.setdefault(parts[0], set()).add(parts[1])
    if leftover := old_fixture_ids & set(paths):
        problems.append(f"{len(leftover)} original fixture version(s) remain")
    compiled = [(p, re.compile(p)) for p in blocklist.PATTERNS]
    for oid, names in paths.items():
        if git(repo, "cat-file", "-t", oid).strip() != b"blob":
            continue
        data = git(repo, "cat-file", "-p", oid)
        if BLOCKLIST_FILE in names and scrub_blocklist(data, blocklist.PATTERNS) != data:
            problems.append(f"{BLOCKLIST_FILE} blob {oid[:9]} still has a removed entry")
        text = data.decode("utf-8", errors="replace")
        for name in names - blocklist.ALLOWED:
            problems.extend(
                f"{name} blob {oid[:9]} matches {pattern!r}"
                for pattern, rx in compiled
                if rx.search(text)
            )
    return problems


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build and verify a publishable history clone.")
    p.add_argument("source", help="URL or path of the repository to copy (its main branch)")
    p.add_argument("dest", type=Path, help="new directory for the rewritten clone")
    p.add_argument("--old-email", required=True, help="e-mail to replace in every commit")
    p.add_argument("--new-email", required=True, help="replacement, e.g. the GitHub no-reply")
    p.add_argument("--fixtures-rewritten-in", default=FIXTURES_REWRITTEN_IN)
    args = p.parse_args(argv)
    if args.dest.exists():
        print(f"prepare_public_history: {args.dest} already exists", file=sys.stderr)
        return 1

    subprocess.run(
        ["git", "clone", "--bare", "--quiet", "--no-local", "--single-branch", "--branch", "main",
         "--no-tags", args.source, str(args.dest)],
        check=True,
    )  # fmt: skip
    source_tree = git(args.dest, "rev-parse", "HEAD^{tree}").decode().strip()
    old_fixture_ids: set[str] = set()
    for path in FIXTURES:
        with contextlib.suppress(subprocess.CalledProcessError):
            old_fixture_ids |= _blob_ids(args.dest, f"{args.fixtures_rewritten_in}^", path)

    rewrite(args.dest, args.old_email, args.new_email, args.fixtures_rewritten_in)
    problems = verify(args.dest, source_tree, args.old_email, old_fixture_ids)
    commits = git(args.dest, "rev-list", "--count", "HEAD").decode().strip()
    print(f"prepare_public_history: {commits} commits rewritten in {args.dest}")
    print(f"  commit map: {args.dest / 'filter-repo' / 'commit-map'}")
    for problem in problems:
        print(f"  FAIL {problem}")
    print("  verified: publishable" if not problems else f"  {len(problems)} problem(s)")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
