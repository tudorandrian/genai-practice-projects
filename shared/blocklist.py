"""No course material, no local paths: scanner used by the test, the hook and the release check."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

# Every literal below is written as an explicit "+" concatenation when it would
# otherwise match its own pattern, so this module doesn't flag itself.
PATTERNS: list[str] = [
    "I" + "BM",
    "Cour" + "sera",
    "Skills " + "Network",
    r"M[0-9]-[0-9]-L(AB)?[0-9]",
    "cf-courses-" + "data",
    "lara" + "gon",
    "watson" + "x",
]
ALLOWED: set[str] = {"README.md", "docs/certificate.md"}
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".csv",
    ".tsv",
    ".html",
    ".cfg",
    ".ini",
    "",
}
_COMPILED = [(p, re.compile(p)) for p in PATTERNS]
ROOT = Path(__file__).resolve().parents[1]


def tracked_text_files(staged_only: bool = False) -> list[Path]:
    cmd = (
        ["git", "diff", "--cached", "--name-only", "-z"]
        if staged_only
        else ["git", "ls-files", "-z"]
    )
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, check=True).stdout.decode()
    return [
        ROOT / f
        for f in out.split("\0")
        if f and Path(f).suffix in TEXT_SUFFIXES and (ROOT / f).is_file()
    ]


def scan(paths: Iterable[Path], allowed: set[str] | None = None) -> list[tuple[Path, int, str]]:
    hits: list[tuple[Path, int, str]] = []
    for path in paths:
        rel = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name
        if allowed and rel in allowed:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), 1):
            for name, rx in _COMPILED:
                if rx.search(line):
                    hits.append((path, lineno, name))
    return hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="blocklist")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--staged", action="store_true")
    args = parser.parse_args(argv)
    files = tracked_text_files(staged_only=args.staged and not args.all)
    hits = scan(files, allowed=ALLOWED)
    for path, lineno, name in hits:
        print(f"{path}:{lineno}: matches {name}")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
