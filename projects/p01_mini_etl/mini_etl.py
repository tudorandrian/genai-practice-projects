"""Mini-ETL - clean a messy CSV and export it to CSV / JSON / Excel.

PIPELINE
    load_data  ->  audit (before)  ->  replace_sentinel  ->  coerce_numeric
               ->  impute  ->  audit (after)  ->  export  ->  write_report

The tool is dataset-agnostic: the input path/URL, the missing-value sentinel,
the header list and the per-column imputation strategies are all supplied on
the command line (or through a JSON config), so the same script cleans any
tabular CSV without code changes.

HOW TO RUN
    # Synthetic fixture shipped with the project (offline, deterministic):
    python mini_etl.py --config config.example.json

    # A dataset with per-domain settings baked into a config file:
    python mini_etl.py --config configs/penguins_biology.json

    # Any CSV, fully from the command line:
    python mini_etl.py --input my.csv --sentinel "?" \\
        --numeric-cols age,income --mean-cols age --mode-cols city --drop-cols id

    # Headerless CSV -> attach a header list:
    python mini_etl.py --input data.csv --headers-file headers.csv --drop-cols price

    # Deliberately-broken inputs for a guided tour of failure modes:
    python mini_etl.py --input anti_examples/a1_inconsistent_sentinels.csv --numeric-cols age,income

DEPENDENCIES
    pandas, numpy (required) - openpyxl (only for the .xlsx export)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shared.demo import DemoResult

log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output"


# =============================================================================
# Extract
# =============================================================================


def _detect_format(path: str, override: str | None) -> str:
    """Pick the input format from an explicit override or the file extension."""
    if override:
        return override.lower()
    suffix = Path(path).suffix.lower().lstrip(".")
    return suffix if suffix in {"csv", "tsv", "txt", "json", "xlsx", "xls"} else "csv"


def load_data(
    path_or_url: str,
    headers: list[str] | None = None,
    fmt: str | None = None,
    sep: str | None = None,
) -> pd.DataFrame:
    """Load a tabular file into a DataFrame across several input forms.

    Supported forms (auto-detected from the extension, or forced with ``fmt``):
        csv / txt   -> pd.read_csv  (delimiter ',' or ``sep``)
        tsv         -> pd.read_csv  (delimiter '\\t' unless ``sep`` overrides)
        json        -> pd.read_json (list-of-records)
        xlsx / xls  -> pd.read_excel (needs openpyxl)

    When ``headers`` is given, a delimited file is read as header-less and the
    supplied column names are attached afterwards.
    """
    kind = _detect_format(path_or_url, fmt)

    if kind == "json":
        df = pd.read_json(path_or_url)
    elif kind in {"xlsx", "xls"}:
        df = pd.read_excel(path_or_url)
    else:  # csv / tsv / txt
        delimiter = sep if sep is not None else ("\t" if kind == "tsv" else ",")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", pd.errors.ParserWarning)
            df = pd.read_csv(
                path_or_url,
                header=None if headers else "infer",
                sep=delimiter,
                encoding="utf-8",
                index_col=False,  # never promote an extra delimiter-split field to an index
            )
        if any(issubclass(w.category, pd.errors.ParserWarning) for w in caught):
            raise ValueError(
                f"'{path_or_url}': rows have a different number of fields than the "
                f"header when split on delimiter {delimiter!r}. The delimiter may be "
                f"wrong for this file (e.g. pass --sep ';')."
            )

    if headers:
        if df.shape[1] != len(headers):
            raise ValueError(
                f"Header count mismatch: file has {df.shape[1]} columns but "
                f"{len(headers)} header names were provided."
            )
        df.columns = headers
    return df


# =============================================================================
# Audit
# =============================================================================


def audit(df: pd.DataFrame) -> dict[str, Any]:
    """Snapshot shape, per-column null counts and dtypes."""
    nulls = df.isnull().sum()
    return {
        "shape": df.shape,
        "nulls_per_col": nulls[nulls > 0].sort_values(ascending=False),
        "total_nulls": int(nulls.sum()),
        "dtypes": df.dtypes,
    }


# =============================================================================
# Transform
# =============================================================================


def replace_sentinel(df: pd.DataFrame, sentinel: str = "?") -> pd.DataFrame:
    """Turn every occurrence of ``sentinel`` into NaN.

    Functional form (``df = df.replace(...)``) rather than ``inplace=True`` for
    forward-compatibility with pandas 3.x.
    """
    return df.replace(sentinel, np.nan)


def coerce_numeric(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """Cast the given columns to float.

    A value that cannot be parsed raises a ValueError naming the guilty column,
    instead of surfacing a raw pandas traceback.
    """
    df = df.copy()
    for col in numeric_cols:
        if col not in df.columns:
            raise ValueError(f"coerce_numeric: unknown column '{col}'.")
        try:
            df[col] = pd.to_numeric(df[col], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Column '{col}' contains a value that is not numeric even "
                f"after sentinel removal: {exc}"
            ) from exc
    return df


def impute(df: pd.DataFrame, strategies: dict[str, str]) -> pd.DataFrame:
    """Fill or drop missing values per column.

    strategy in {"mean", "mode", "drop"}:
        mean -> column average   (continuous numerics)
        mode -> most frequent    (discrete / categorical)
        drop -> dropna(subset)   (critical columns, e.g. price)
    """
    df = df.copy()
    for col, strategy in strategies.items():
        if col not in df.columns:
            raise ValueError(f"impute: unknown column '{col}'.")
        if strategy == "mean":
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == "mode":
            df[col] = df[col].fillna(df[col].mode(dropna=True)[0])
        elif strategy == "drop":
            df = df.dropna(subset=[col])
        else:
            raise ValueError(
                f"impute: unknown strategy '{strategy}' for column '{col}' "
                f"(expected mean | mode | drop)."
            )
    return df


# =============================================================================
# Load (write)
# =============================================================================


def export(df: pd.DataFrame, out_dir: str, stem: str = "clean") -> list[str]:
    """Write the cleaned frame as CSV + JSON + Excel. Returns the file paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    csv_path = out / f"{stem}.csv"
    df.to_csv(csv_path, index=False)
    written.append(str(csv_path))

    json_path = out / f"{stem}.json"
    df.to_json(json_path, orient="records", indent=2)
    written.append(str(json_path))

    xlsx_path = out / f"{stem}.xlsx"
    try:
        df.to_excel(xlsx_path, index=False)  # needs openpyxl
        written.append(str(xlsx_path))
    except ImportError:
        log.warning(
            "openpyxl not installed -> skipped .xlsx export (install with: pip install openpyxl)"
        )
    return written


# =============================================================================
# Report
# =============================================================================


def write_report(
    before: dict[str, Any],
    after: dict[str, Any],
    strategies: dict[str, str],
    out_dir: str,
    filename: str = "cleaning_report.txt",
) -> str:
    """Write a plain-text summary of the cleaning run."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / filename

    lines: list[str] = []
    lines.append("MINI-ETL CLEANING REPORT")
    lines.append("=" * 40)
    lines.append(f"Rows x cols  before : {before['shape'][0]} x {before['shape'][1]}")
    lines.append(f"Rows x cols  after  : {after['shape'][0]} x {after['shape'][1]}")
    lines.append(f"Rows dropped        : {before['shape'][0] - after['shape'][0]}")
    lines.append("")
    lines.append("Missing values BEFORE (per column):")
    lines.append(before["nulls_per_col"].to_string() if before["total_nulls"] else "  (none)")
    lines.append(f"  total: {before['total_nulls']}")
    lines.append("")
    lines.append("Missing values AFTER (per column):")
    lines.append(after["nulls_per_col"].to_string() if after["total_nulls"] else "  (none)")
    lines.append(f"  total: {after['total_nulls']}")
    lines.append("")
    lines.append("Imputation strategy per column:")
    if strategies:
        for col, strategy in strategies.items():
            lines.append(f"  {col}: {strategy}")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append("Final dtypes:")
    lines.append(after["dtypes"].to_string())

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(report_path)


def _write_metrics(report: dict[str, Any], path: Path) -> None:
    """Deterministic figures-only proof file: no timestamps, no absolute paths."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"rows={report['rows']}",
        f"missing_before={report['missing_before']}",
        f"missing_after={report['missing_after']}",
        f"files_written={len(report['files'])}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_demo_summary(report: dict[str, Any], path: Path) -> None:
    """Deterministic demo transcript: the same figures printed on success, minus
    the elapsed-seconds line (which is never the same twice) - no timestamps,
    no absolute paths."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "p01-mini-etl: ok",
        f"  rows: {report['rows']}",
        f"  missing_before: {report['missing_before']}",
        f"  missing_after: {report['missing_after']}",
        "  wrote: output/",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =============================================================================
# Config / CLI
# =============================================================================


def _split(value: str | None) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()] if value else []


def _resolve_relative(value: str | None, base: Path | None) -> str | None:
    """Resolve a path *from a config file* against that config file's own
    directory, so `input`/`out_dir` work the same regardless of the caller's
    current directory. Absolute paths and URLs pass through unchanged."""
    if not value or base is None:
        return value
    if "://" in value or Path(value).is_absolute():
        return value
    return str((base / value).resolve())


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    """Merge a JSON config file (if any) with command-line flags.

    CLI flags override config-file keys. A relative `input`/`out_dir` coming
    from the config file is resolved against the config file's own directory
    (not the caller's cwd), so `--config path/to/some.json` works the same
    from any working directory. Returns a normalised dict with keys: input,
    sentinel, headers, numeric_cols, strategies, out_dir, stem.
    """
    cfg: dict[str, Any] = {}
    config_dir: Path | None = None
    if args.config:
        config_path = Path(args.config)
        config_dir = config_path.parent
        cfg = json.loads(config_path.read_text(encoding="utf-8"))

    # Headers: inline list from config, or a one-line CSV file via --headers-file
    headers = cfg.get("headers")
    if args.headers_file:
        headers = _split(Path(args.headers_file).read_text(encoding="utf-8").strip())

    # Strategies: start from config, then layer the convenience flags on top
    strategies: dict[str, str] = dict(cfg.get("strategies", {}))
    for col in _split(args.mean_cols):
        strategies[col] = "mean"
    for col in _split(args.mode_cols):
        strategies[col] = "mode"
    for col in _split(args.drop_cols):
        strategies[col] = "drop"

    numeric = cfg.get("numeric_cols", [])
    if args.numeric_cols:
        numeric = _split(args.numeric_cols)

    return {
        "input": args.input or _resolve_relative(cfg.get("input"), config_dir),
        "format": args.format or cfg.get("format"),
        "sep": args.sep if args.sep is not None else cfg.get("sep"),
        "sentinel": args.sentinel if args.sentinel is not None else cfg.get("sentinel", "?"),
        "headers": headers,
        "numeric_cols": numeric,
        "strategies": strategies,
        "out_dir": args.out_dir or _resolve_relative(cfg.get("out_dir", "output"), config_dir),
        "stem": args.stem or cfg.get("stem", "clean"),
    }


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    """Execute the full pipeline from a normalised (or bare, config-file-shaped) dict.

    Only ``input`` is required; every other key falls back to the same default
    ``build_config`` would use, so a config file can be loaded and run directly
    (``json.loads(path.read_text())``) as well as through the CLI.

    Returns a report dict: rows (final row count), missing_before /
    missing_after (total null counts around imputation) and files (the
    paths written by ``export``).
    """
    if not cfg.get("input"):
        raise ValueError("No input specified (use --input or 'input' in the config).")

    log.info("loading: %s", cfg["input"])
    df = load_data(
        cfg["input"], headers=cfg.get("headers"), fmt=cfg.get("format"), sep=cfg.get("sep")
    )
    log.info("loaded %d rows x %d cols", df.shape[0], df.shape[1])

    df = replace_sentinel(df, cfg.get("sentinel", "?"))
    numeric_cols = cfg.get("numeric_cols") or []
    if numeric_cols:
        df = coerce_numeric(df, numeric_cols)
    before = audit(df)  # audit after sentinel->NaN so counts reflect real missingness
    log.info("missing values after sentinel replacement: %d", before["total_nulls"])

    strategies = cfg.get("strategies") or {}
    df = impute(df, strategies)
    after = audit(df)
    log.info("after imputation: %d rows, %d missing", after["shape"][0], after["total_nulls"])

    written = export(df, cfg.get("out_dir", "output"), cfg.get("stem", "clean"))
    report_path = write_report(before, after, strategies, cfg.get("out_dir", "output"))
    for path in written:
        log.info("wrote %s", path)
    log.info("wrote %s", report_path)

    return {
        "rows": len(df),
        "missing_before": before["total_nulls"],
        "missing_after": after["total_nulls"],
        "files": written,
    }


def demo() -> DemoResult:
    """Clean the shipped synthetic fixture into output/; offline and deterministic."""
    start = time.perf_counter()
    cfg = json.loads((HERE / "config.example.json").read_text(encoding="utf-8"))
    cfg["input"] = str(HERE / cfg["input"])
    cfg["out_dir"] = str(OUT_DIR)
    report = run(cfg)
    _write_metrics(report, OUT_DIR / "metrics.txt")
    _write_demo_summary(report, OUT_DIR / "summary.txt")
    return DemoResult(
        "p01-mini-etl",
        "ok",
        {
            "rows": str(report["rows"]),
            "missing_before": str(report["missing_before"]),
            "missing_after": str(report["missing_after"]),
        },
        seconds=round(time.perf_counter() - start, 2),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Clean a messy CSV and export it to CSV / JSON / Excel."
    )
    p.add_argument("--config", help="JSON config file (CLI flags override its keys).")
    p.add_argument("--input", help="Input file path or URL (csv/tsv/txt/json/xlsx).")
    p.add_argument(
        "--format", help="Force input format (csv|tsv|txt|json|xlsx); default: by extension."
    )
    p.add_argument(
        "--sep", help="Field delimiter for delimited input (e.g. ';'); default ',' or tab for .tsv."
    )
    p.add_argument("--sentinel", help="Missing-value sentinel to replace (default '?').")
    p.add_argument("--headers-file", help="One-line CSV of column names for headerless input.")
    p.add_argument("--numeric-cols", help="Comma-separated columns to cast to float.")
    p.add_argument("--mean-cols", help="Comma-separated columns to impute with the mean.")
    p.add_argument("--mode-cols", help="Comma-separated columns to impute with the mode.")
    p.add_argument("--drop-cols", help="Comma-separated columns whose NaN rows are dropped.")
    p.add_argument("--out-dir", help="Output directory (default 'output').")
    p.add_argument("--stem", help="Base name for output files (default 'clean').")
    p.add_argument(
        "--demo", action="store_true", help="Run the offline demo on the shipped fixture."
    )
    p.add_argument("--verbose", action="store_true", help="Log pipeline steps at INFO level.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s"
    )

    if args.demo:
        result = demo()
        print(f"p01-mini-etl: {result.status}")
        for key, value in result.figures.items():
            print(f"  {key}: {value}")
        print("  wrote: output/")
        print(f"  seconds: {result.seconds:.2f}")
        return 0 if result.status == "ok" else 1

    try:
        cfg = build_config(args)
        report = run(cfg)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"p01-mini-etl: {Path(cfg['input']).name}")
    print(
        f"  rows={report['rows']} missing_before={report['missing_before']} "
        f"missing_after={report['missing_after']}"
    )
    print(f"  wrote: {', '.join(report['files'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
