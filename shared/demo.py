"""`uv run demo [--models] [--all]` — run every registered project's demo() and summarise."""

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
            except Exception as exc:  # noqa: BLE001 — one project failing must not stop the others
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


def _write_summary(results: list[DemoResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Demo summary — {datetime.now(UTC):%Y-%m-%d %H:%M} UTC",
        "",
        "| project | status | seconds | figures | note |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        figures = " ".join(f"{k}={v}" for k, v in r.figures.items())
        lines.append(f"| {r.name} | {r.status} | {r.seconds} | {figures} | {r.note} |")
    ok = sum(r.status == "ok" for r in results)
    lines += ["", f"{ok}/{len(results)} ok"]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def tiers_from_args(argv: list[str]) -> set[str]:
    parser = argparse.ArgumentParser(prog="demo")
    parser.add_argument(
        "--models", action="store_true", help="also run projects that need model weights"
    )
    parser.add_argument("--all", action="store_true", help="run every project (models + rag)")
    args = parser.parse_args(argv)
    tiers = {"core"}
    if args.models or args.all:
        tiers.add("models")
    if args.all:
        tiers.add("rag")
    return tiers


def main(argv: list[str] | None = None) -> int:
    tiers = tiers_from_args(sys.argv[1:] if argv is None else argv)
    results = run_demo(ENTRIES, tiers, Path("output") / "demo-summary.md")
    return 0 if all(r.status != "failed" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
