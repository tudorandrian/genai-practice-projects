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


def test_a3_default_path_raises_a_controlled_error() -> None:
    # a3 is ';'-delimited with comma-decimal values. Read with the tool's
    # anglophone defaults (no --sep) the header/data field counts disagree
    # (pandas' own ParserWarning), which load_data now turns into a named
    # ValueError instead of silently collapsing the four columns into one.
    path = HERE / "anti_examples" / "a3_european_decimal_comma.csv"
    with pytest.raises(ValueError, match="different number of fields"):
        mini_etl.load_data(str(path))


def test_a3_with_correct_sep_loads_its_four_real_columns() -> None:
    # Naming the actual delimiter avoids the mismatch entirely and loads the
    # four real columns cleanly.
    path = HERE / "anti_examples" / "a3_european_decimal_comma.csv"
    df = mini_etl.load_data(str(path), sep=";")
    assert df.shape == (3, 4)
    assert list(df.columns) == ["id", "screen_inch", "weight_kg", "price"]


def test_cli_reports_a3_delimiter_mismatch_without_writing_clean_files(
    capsys: pytest.CaptureFixture[str], tmp_output: Path
) -> None:
    # End to end: running the CLI on a3 with no --sep must fail loudly (exit
    # 1, `error:` on stderr) instead of exiting 0 with a corrupted clean.*.
    path = HERE / "anti_examples" / "a3_european_decimal_comma.csv"
    exit_code = mini_etl.main(["--input", str(path), "--out-dir", str(tmp_output)])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert "different number of fields" in captured.err
    assert list(tmp_output.glob("clean.*")) == []


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


def test_run_returns_report_dict(tmp_output: Path) -> None:
    df_report_keys = {"rows", "missing_before", "missing_after", "files"}
    cfg = json.loads((HERE / "config.example.json").read_text(encoding="utf-8"))
    cfg["input"] = str(HERE / cfg["input"])
    cfg["out_dir"] = str(tmp_output)
    report = mini_etl.run(cfg)
    assert df_report_keys <= report.keys()


# =============================================================================
# Every shipped config, and demo() determinism (brief §Step 3)
# =============================================================================


@pytest.mark.parametrize("config_path", CONFIGS, ids=[p.stem for p in CONFIGS])
def test_every_shipped_config_cleans_its_dataset(config_path: Path, tmp_output: Path) -> None:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    # Shipped configs store `input` relative to their own directory (configs/),
    # same resolution build_config applies — see test_config_relative_paths_below.
    cfg["input"] = str((config_path.parent / cfg["input"]).resolve())
    cfg["out_dir"] = str(tmp_output)
    report = mini_etl.run(cfg)
    assert report["rows"] > 0
    assert report["missing_after"] <= report["missing_before"]
    assert {p.suffix for p in map(Path, report["files"])} == {".csv", ".json", ".xlsx"}


def test_demo_is_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mini_etl, "OUT_DIR", tmp_path / "output")
    first = mini_etl.demo().figures
    second = mini_etl.demo().figures
    assert first == second and first["missing_after"] == "0"


# =============================================================================
# Config path resolution (README §Run works from the repo root, not just
# from inside the project directory)
# =============================================================================


def test_config_relative_paths_resolve_against_config_dir(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "raw.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    config_path = configs_dir / "c.json"
    config_path.write_text(
        json.dumps({"input": "../data/raw.csv", "out_dir": "../output"}), encoding="utf-8"
    )

    args = mini_etl.parse_args(["--config", str(config_path)])
    cfg = mini_etl.build_config(args)

    assert Path(cfg["input"]) == (tmp_path / "data" / "raw.csv").resolve()
    assert Path(cfg["out_dir"]) == (tmp_path / "output").resolve()


# =============================================================================
# README "Running every shipped config once" table (deferred item 5): pin the
# per-dataset figures so the table can't drift silently from the code.
# =============================================================================

# {stem: (rows before impute, nulls before impute, rows after impute)} — nulls
# after impute is always 0 for every shipped config.
README_TABLE_EXPECTATIONS: dict[str, tuple[int, int, int]] = {
    "airline_passengers": (144, 0, 144),
    "cars_automotive": (406, 14, 406),
    "gdp_economics": (77, 0, 77),
    "health_spending": (274, 0, 274),
    "iris_botany": (150, 0, 150),
    "laptops_sample": (15, 5, 14),
    "movies_entertainment": (250, 606, 250),
    "penguins_biology": (344, 18, 344),
    "population_demographics": (77, 0, 77),
    "restaurant_tips": (244, 0, 244),
    "stocks_finance": (437, 0, 437),
    "titanic_history": (891, 181, 891),
    "weather_climate": (366, 0, 366),
}


def test_readme_table_expectations_cover_every_config() -> None:
    # A new/renamed config cannot be added without a matching entry here.
    assert {p.stem for p in CONFIGS} == set(README_TABLE_EXPECTATIONS)


@pytest.mark.parametrize("config_path", CONFIGS, ids=[p.stem for p in CONFIGS])
def test_readme_table_rows_and_nulls_are_pinned(config_path: Path, tmp_output: Path) -> None:
    rows_before, nulls_before, rows_after = README_TABLE_EXPECTATIONS[config_path.stem]
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    cfg["input"] = str((config_path.parent / cfg["input"]).resolve())

    df = mini_etl.load_data(
        cfg["input"], headers=cfg.get("headers"), fmt=cfg.get("format"), sep=cfg.get("sep")
    )
    df = mini_etl.replace_sentinel(df, cfg.get("sentinel", "?"))
    before = mini_etl.audit(df)
    assert df.shape[0] == rows_before
    assert before["total_nulls"] == nulls_before

    cfg["out_dir"] = str(tmp_output)
    report = mini_etl.run(cfg)
    assert report["rows"] == rows_after
    assert report["missing_after"] == 0


def test_config_cli_input_override_is_not_resolved_against_config_dir(tmp_path: Path) -> None:
    # A CLI --input is relative to the caller's cwd, not to --config's directory.
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    config_path = configs_dir / "c.json"
    config_path.write_text(json.dumps({"input": "ignored.csv"}), encoding="utf-8")

    args = mini_etl.parse_args(["--config", str(config_path), "--input", "cli-given.csv"])
    cfg = mini_etl.build_config(args)

    assert cfg["input"] == "cli-given.csv"
