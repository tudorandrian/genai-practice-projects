"""progress.py - a progress dashboard over the P12 synthetic corpus's status tables.

Scans ``corpus/<course>/README.md`` files for the ``| # | Lesson | Status |``
table, normalizes statuses to four canonical states (NOT STARTED | IN PROGRESS |
COMPLETED | REVIEW), aggregates with Pandas, and saves a bar chart to
``output/progress.png``. Also reads the quiz ``history.json`` (``quiz_engine``'s
output) to compute per-module success rates - the integration point where this
module consumes ``quiz_engine``'s.

Pure functions: ``scan_statuses`` / ``aggregate`` / ``success_rate``.
CLI: ``python progress.py``
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless backend, must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
CORPUS_DIR = HERE / "corpus"
OUT_DIR = HERE / "output"
HISTORY_PATH = OUT_DIR / "history.json"  # written by quiz_engine (coupling via file)

CANONICAL_STATUSES = ["NOT STARTED", "IN PROGRESS", "COMPLETED", "REVIEW"]


def _normalize(text: str) -> str | None:
    """Return the canonical status found in ``text`` (or None)."""
    upper = text.upper()
    for status in CANONICAL_STATUSES:
        if status in upper:
            return status
    return None


def _statuses_from_table(text: str) -> list[str]:
    """Extract the Status-column cells from every markdown table in ``text``."""
    results = []
    lines = [ln for ln in text.splitlines() if ln.strip().startswith("|")]
    idx = None
    for ln in lines:
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if idx is None:
            if any(c.lower() == "status" for c in cells):
                idx = next(i for i, c in enumerate(cells) if c.lower() == "status")
            continue
        if set("".join(cells)) <= set("-: "):  # separator row |---|
            continue
        if idx < len(cells):
            status = _normalize(cells[idx])
            if status:
                results.append(status)
    return results


def scan_statuses(corpus_dir: str | Path = CORPUS_DIR) -> pd.DataFrame:
    """Scan ``corpus_dir`` for course ``README.md`` status tables.

    Returns a DataFrame ``[course, item, status]``. Every lesson course README
    uses the same table convention, so - unlike the private course tree this
    was ported from - there is no separate "lab README" status-line format to
    handle.
    """
    corpus_dir = Path(corpus_dir)
    rows = []
    unrecognized = []
    for readme in sorted(corpus_dir.rglob("README.md")):
        rel = readme.relative_to(corpus_dir)
        course = rel.parts[0]
        text = readme.read_text(encoding="utf-8", errors="replace")
        statuses = _statuses_from_table(text)
        if not statuses:
            unrecognized.append(str(rel))
        for status in statuses:
            rows.append({"course": course, "item": str(rel), "status": status})
    df = pd.DataFrame(rows, columns=["course", "item", "status"])
    df.attrs["unrecognized"] = unrecognized
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot: rows = course, columns = status, values = counts."""
    if df.empty:
        return pd.DataFrame()
    pivot = df.pivot_table(
        index="course", columns="status", values="item", aggfunc="count", fill_value=0
    ).reindex(columns=CANONICAL_STATUSES, fill_value=0)
    pivot["TOTAL"] = pivot.sum(axis=1)
    return pivot


def success_rate(history_path: str | Path | None = None) -> dict[str, Any]:
    """Per-module / per-type quiz success rate from ``quiz_engine``'s history."""
    path = Path(history_path) if history_path is not None else HISTORY_PATH
    if not path.exists():
        return {"sessions": 0, "by_module": {}, "by_type": {}}
    history = json.loads(path.read_text(encoding="utf-8"))
    questions = [q for s in history for q in s.get("questions", [])]
    if not questions:
        return {"sessions": len(history), "by_module": {}, "by_type": {}}
    df = pd.DataFrame(questions)
    by_module = df.groupby("module")["is_correct"].mean().round(3).to_dict()
    by_type = df.groupby("type")["is_correct"].mean().round(3).to_dict()
    return {
        "sessions": len(history),
        "answered": len(df),
        "by_module": by_module,
        "by_type": by_type,
    }


def plot_progress(pivot: pd.DataFrame, path: str | Path | None = None) -> Path:
    """Grouped bar chart of statuses per course -> ``output/progress.png``."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(path) if path else OUT_DIR / "progress.png"
    statuses = [c for c in CANONICAL_STATUSES if c in pivot.columns]
    fig, ax = plt.subplots(figsize=(11, 6))
    pivot[statuses].plot(
        kind="bar", width=0.8, color=["#BDBDBD", "#FFB74D", "#66BB6A", "#42A5F5"], ax=ax
    )
    ax.set_title("Study progress by course (status counts)")
    ax.set_xlabel("course")
    ax.set_ylabel("lessons")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="status")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def run_dashboard_cli() -> str:
    """Scan, aggregate, plot, and build a text report; returns the report string."""
    df = scan_statuses()
    pivot = aggregate(df)
    plot_progress(pivot)
    rate = success_rate()
    lines = [
        "=== Study progress (from corpus/ READMEs) ===",
        pivot.to_string() if not pivot.empty else "(no statuses found)",
        "",
        f"Files scanned but with no recognizable status: {len(df.attrs.get('unrecognized', []))}",
        "",
        f"=== Quiz history ({rate['sessions']} sessions) ===",
    ]
    if rate["by_module"]:
        lines.append("success rate by module: " + json.dumps(rate["by_module"]))
        lines.append("success rate by type:   " + json.dumps(rate["by_type"]))
    else:
        lines.append("(no quiz sessions yet - run the quiz first)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry: scan statuses, aggregate, save the chart, print the report."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", maxsplit=1)[0])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s"
    )
    print(run_dashboard_cli())
    print(f"\nChart saved to {OUT_DIR / 'progress.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
