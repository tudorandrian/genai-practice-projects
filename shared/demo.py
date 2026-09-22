"""`uv run demo [--models] [--all]` - run every registered project's demo() and summarise."""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from shared.registry import ENTRIES, Entry

Status = Literal["ok", "skipped", "failed"]
# Anchored to the repository, not the working directory, like every project's output/.
REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class DemoResult:
    name: str
    status: Status
    figures: dict[str, str] = field(default_factory=dict)
    seconds: float = 0.0
    note: str = ""


def _resolve(target: str) -> Callable[[], DemoResult]:
    module_name, func_name = target.split(":")
    return getattr(importlib.import_module(module_name), func_name)


def run_demo(entries: list[Entry], tiers: set[str], out_path: Path) -> list[DemoResult]:
    results: list[DemoResult] = []
    for entry in entries:
        if entry.tier not in tiers:
            result = DemoResult(entry.slug, "skipped", note=f"tier {entry.tier} not selected")
        else:
            start = time.perf_counter()
            try:
                result = _resolve(entry.target)()
                result.seconds = round(time.perf_counter() - start, 2)
            except Exception as exc:  # one project failing must not stop the others
                result = DemoResult(
                    entry.slug,
                    "failed",
                    seconds=round(time.perf_counter() - start, 2),
                    note=f"{type(exc).__name__}: {exc}",
                )
        results.append(result)
        # CLI runner output, not a library log (spec 6): one status line per project.
        print(
            f"{result.name:<24} {result.status:<8} {result.seconds:>6.1f}s  "
            f"{' '.join(f'{k}={v}' for k, v in result.figures.items())}{result.note}"
        )
    _write_summary(results, out_path)
    return results


def _cell(text: str) -> str:
    """Make free text safe inside one Markdown table cell: exception messages often
    carry newlines or a literal `|`, either of which would break the row."""
    return " ".join(text.replace("|", r"\|").splitlines())


def _write_summary(results: list[DemoResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Demo summary - {datetime.now(UTC):%Y-%m-%d %H:%M} UTC",
        "",
        "| project | status | seconds | figures | note |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        figures = _cell(" ".join(f"{k}={v}" for k, v in r.figures.items()))
        lines.append(f"| {r.name} | {r.status} | {r.seconds} | {figures} | {_cell(r.note)} |")
    ok = sum(r.status == "ok" for r in results)
    lines += ["", f"{ok}/{len(results)} ok"]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="demo")
    parser.add_argument(
        "--models", action="store_true", help="also run projects that need model weights"
    )
    parser.add_argument("--all", action="store_true", help="run every project (models + rag)")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 if a selected project reports skipped (a tier not selected is fine)",
    )
    return parser


def parse_args(argv: list[str]) -> argparse.Namespace:
    args = _parser().parse_args(argv)
    args.tiers = {"core"}
    if args.models or args.all:
        args.tiers.add("models")
    if args.all:
        args.tiers.add("rag")
    return args


def tiers_from_args(argv: list[str]) -> set[str]:
    tiers: set[str] = parse_args(argv).tiers
    return tiers


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    results = run_demo(ENTRIES, args.tiers, REPO_ROOT / "output" / "demo-summary.md")
    if any(r.status == "failed" for r in results):
        return 1
    if args.strict:
        selected = {e.slug for e in ENTRIES if e.tier in args.tiers}
        # A project that was selected and still reported `skipped` (no speech engine,
        # a timeout, ...) means the run did not cover what it claims to cover.
        if any(r.status == "skipped" and r.name in selected for r in results):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
