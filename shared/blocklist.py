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
    r"[A-Za-z]:[\\/]{1,2}Users[\\/]{1,2}[^\\/\s]",  # a Windows home directory
    r"/(home|Users)/[A-Za-z]",  # a Linux or macOS home directory
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
    ".cff",
    ".ini",
    ".example",  # .env.example: the one place a real key is most likely pasted by mistake
    "",
}
_COMPILED = [(p, re.compile(p)) for p in PATTERNS]
ROOT = Path(__file__).resolve().parents[1]


def tracked_text_files(staged_only: bool = False, root: Path | None = None) -> list[Path]:
    root = root or ROOT
    cmd = (
        ["git", "diff", "--cached", "--name-only", "-z"]
        if staged_only
        else ["git", "ls-files", "-z"]
    )
    out = subprocess.run(cmd, cwd=root, capture_output=True, check=True).stdout.decode()
    return [
        root / f
        for f in out.split("\0")
        if f and Path(f).suffix in TEXT_SUFFIXES and (root / f).is_file()
    ]


def scan(
    paths: Iterable[Path], allowed: set[str] | None = None, root: Path | None = None
) -> list[tuple[Path, int, str]]:
    """Return (path, line number, pattern) for every blocked match.

    `allowed` holds paths relative to `root` (default: the repository), so only the root
    README.md is exempt, not every file named README.md.
    """
    root = root or ROOT
    hits: list[tuple[Path, int, str]] = []
    for path in paths:
        rel = path.relative_to(root).as_posix() if path.is_relative_to(root) else path.name
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
