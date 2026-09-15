"""Tests for mini_etl: load/transform/export unit tests, config-driven end-to-end
coverage over every shipped dataset, and a determinism check for demo()."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from projects.p01_mini_etl import mini_etl

pytestmark = pytest.mark.core

HERE = Path(__file__).resolve().parents[1]
CONFIGS = sorted((HERE / "configs").glob("*.json"))


# =============================================================================
# Load
# =============================================================================


def test_csv_with_headers(tmp_path: Path) -> None:
    (tmp_path / "h.csv").write_text("1,2\n3,4\n", encoding="utf-8")
    df = mini_etl.load_data(str(tmp_path / "h.csv"), headers=["a", "b"])
    assert list(df.columns) == ["a", "b"]
    assert df.shape == (2, 2)


def test_header_count_mismatch_raises(tmp_path: Path) -> None:
    (tmp_path / "h.csv").write_text("1,2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        mini_etl.load_data(str(tmp_path / "h.csv"), headers=["only_one"])


def test_tsv_delimiter(tmp_path: Path) -> None:
    (tmp_path / "t.tsv").write_text("a\tb\n1\t2\n", encoding="utf-8")
    df = mini_etl.load_data(str(tmp_path / "t.tsv"))
    assert list(df.columns) == ["a", "b"]
    assert int(df.iloc[0]["b"]) == 2


def test_json_records(tmp_path: Path) -> None:
    (tmp_path / "d.json").write_text(
        json.dumps([{"x": 1, "y": "a"}, {"x": 2, "y": "b"}]), encoding="utf-8"
    )
    df = mini_etl.load_data(str(tmp_path / "d.json"))
    assert df.shape == (2, 2)


def test_semicolon_sep_override(tmp_path: Path) -> None:
    (tmp_path / "s.csv").write_text("a;b\n1;2\n", encoding="utf-8")
    df = mini_etl.load_data(str(tmp_path / "s.csv"), sep=";")
    assert list(df.columns) == ["a", "b"]


# =============================================================================
# Transform
# =============================================================================


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "num": ["1", "2", "?", "4"],
            "cat": ["x", "?", "x", "y"],
            "crit": ["10", "20", "30", "?"],
        }
    )


def test_replace_sentinel() -> None:
    df = mini_etl.replace_sentinel(_frame(), "?")
    assert int(df.isnull().sum().sum()) == 3


def test_replace_is_not_inplace() -> None:
    original = _frame()
    mini_etl.replace_sentinel(original, "?")
    # original must be untouched (functional form, pandas-3 safe)
    assert "?" in original["num"].tolist()


def test_coerce_numeric_ok() -> None:
    df = mini_etl.replace_sentinel(_frame(), "?")
    df = mini_etl.coerce_numeric(df, ["num", "crit"])
    assert df["num"].dtype.kind == "f"


def test_coerce_numeric_bad_value_raises_named() -> None:
    df = pd.DataFrame({"a": ["1", "HELLO", "3"]})
    with pytest.raises(ValueError, match="'a'"):
        mini_etl.coerce_numeric(df, ["a"])


def test_impute_mean_mode_drop() -> None:
    df = mini_etl.replace_sentinel(_frame(), "?")
    df = mini_etl.coerce_numeric(df, ["num", "crit"])
    df = mini_etl.impute(df, {"num": "mean", "cat": "mode", "crit": "drop"})
    assert int(df.isnull().sum().sum()) == 0
    assert df.shape[0] == 3  # one row dropped for missing 'crit'
    assert df.loc[2, "num"] == pytest.approx((1 + 2 + 4) / 3)


def test_impute_unknown_strategy_raises() -> None:
    with pytest.raises(ValueError):
        mini_etl.impute(pd.DataFrame({"a": [1]}), {"a": "median"})


# =============================================================================
# Export and report
# =============================================================================


def test_export_writes_csv_json(tmp_output: Path) -> None:
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    written = mini_etl.export(df, str(tmp_output), stem="out")
    names = {Path(p).name for p in written}
    assert "out.csv" in names
    assert "out.json" in names
    reloaded = pd.read_csv(tmp_output / "out.csv")
    assert len(reloaded) == 2


def test_report_mentions_shapes(tmp_output: Path) -> None:
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    before = mini_etl.audit(df)
    after = mini_etl.audit(df)
    path = mini_etl.write_report(before, after, {"a": "mean"}, str(tmp_output))
    text = Path(path).read_text(encoding="utf-8")
    assert "CLEANING REPORT" in text
    assert "Final dtypes" in text


# =============================================================================
# End to end
# =============================================================================


def test_pipeline_on_sentinel_csv(tmp_path: Path) -> None:
    (tmp_path / "raw.csv").write_text(
        "id,val,grp\n1,10,a\n2,?,a\n3,30,b\n4,40,?\n", encoding="utf-8"
    )
    cfg = {
        "input": str(tmp_path / "raw.csv"),
        "format": None,
        "sep": None,
        "sentinel": "?",
        "headers": None,
        "numeric_cols": ["val"],
        "strategies": {"val": "mean", "grp": "mode"},
        "out_dir": str(tmp_path / "out"),
        "stem": "clean",
    }
    report = mini_etl.run(cfg)
    out = pd.read_csv(tmp_path / "out" / "clean.csv")
    assert len(out) == 4
    assert int(out["val"].isnull().sum()) == 0
    assert not (out.astype(str) == "?").any().any()
    assert report["rows"] == 4
    assert report["missing_before"] == 2
    assert report["missing_after"] == 0


def test_run_returns_report_dict() -> None:
    df_report_keys = {"rows", "missing_before", "missing_after", "files"}
    cfg = json.loads((HERE / "config.example.json").read_text(encoding="utf-8"))
    cfg["input"] = str(HERE / cfg["input"])
    cfg["out_dir"] = str(HERE / "output")
    report = mini_etl.run(cfg)
    assert df_report_keys <= report.keys()


# =============================================================================
# Every shipped config, and demo() determinism (brief §Step 3)
# =============================================================================


@pytest.mark.parametrize("config_path", CONFIGS, ids=[p.stem for p in CONFIGS])
def test_every_shipped_config_cleans_its_dataset(config_path: Path, tmp_output: Path) -> None:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    cfg["input"] = str(HERE / cfg["input"])
    cfg["out_dir"] = str(tmp_output)
    report = mini_etl.run(cfg)
    assert report["rows"] > 0
    assert report["missing_after"] <= report["missing_before"]
    assert {p.suffix for p in map(Path, report["files"])} == {".csv", ".json", ".xlsx"}


def test_demo_is_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first = mini_etl.demo().figures
    second = mini_etl.demo().figures
    assert first == second and first["missing_after"] == "0"
