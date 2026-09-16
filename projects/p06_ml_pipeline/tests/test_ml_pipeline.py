"""Tests for ml_pipeline.py: dataset registry, pipeline structure, no-leakage,
grid search + evaluation, the model card, round-trip persistence, determinism
and the saved plots.

Run:
    uv run pytest projects/p06_ml_pipeline -q                # offline only (weather)
    uv run pytest projects/p06_ml_pipeline -m network -q     # adult/credit_g/churn/
                                                               # online_shoppers/bank_marketing
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from projects.p06_ml_pipeline import ml_pipeline
from projects.p06_ml_pipeline.ml_pipeline import Dataset

pytestmark = pytest.mark.core

NETWORK_DATASETS = ("adult", "credit_g", "churn", "online_shoppers", "bank_marketing")


# =============================================================================
# Registry
# =============================================================================


def test_registry_has_at_least_five_datasets() -> None:
    assert len(ml_pipeline.DATASETS) >= 5


def test_weather_is_mixed_and_binary() -> None:
    ds = ml_pipeline.load_data("weather")
    assert len(ds.num_cols) >= 1  # has numeric cols
    assert len(ds.cat_cols) >= 1  # has categorical cols
    assert ds.y.nunique() == 2  # binary target
    assert ds.pos_label == "Yes"  # minority = rain


@pytest.mark.parametrize("name", NETWORK_DATASETS)
@pytest.mark.network
def test_downloaded_datasets_are_mixed_and_binary(name: str) -> None:
    ds = ml_pipeline.load_data(name)
    assert len(ds.num_cols) >= 1, name
    assert len(ds.cat_cols) >= 1, name
    assert ds.y.nunique() == 2, name


def test_unknown_dataset_raises() -> None:
    with pytest.raises(ValueError):
        ml_pipeline.load_data("nope")


# =============================================================================
# Pipeline structure
# =============================================================================


def test_column_detection_partitions_all_columns() -> None:
    ds = ml_pipeline.load_data("weather")
    assert set(ds.num_cols) | set(ds.cat_cols) == set(ds.X.columns)
    assert set(ds.num_cols) & set(ds.cat_cols) == set()


def test_pipeline_has_preproc_then_clf() -> None:
    ds = ml_pipeline.load_data("weather")
    pipe = ml_pipeline.build_pipeline(ds, LogisticRegression())
    assert isinstance(pipe, Pipeline)
    assert [name for name, _ in pipe.steps] == ["preproc", "clf"]


def test_set_params_swaps_only_the_classifier() -> None:
    # The same preprocessing object is reused; only 'clf' changes.
    ds = ml_pipeline.load_data("weather")
    pipe = ml_pipeline.build_pipeline(ds, RandomForestClassifier())
    preproc_before = pipe.named_steps["preproc"]
    pipe.set_params(clf=LogisticRegression(max_iter=1000))
    assert pipe.named_steps["preproc"] is preproc_before
    assert isinstance(pipe.named_steps["clf"], LogisticRegression)


# =============================================================================
# No leakage
# =============================================================================


def test_target_not_present_among_features() -> None:
    ds = ml_pipeline.load_data("weather")
    assert "rain_today" not in ds.X.columns  # today's rain is target-only
    assert "rain_yesterday" in ds.X.columns  # yesterday's rain is a legal feature


# =============================================================================
# Tuning and evaluation
# =============================================================================


def _fit_weather() -> tuple[Dataset, GridSearchCV, pd.DataFrame, pd.Series]:
    ds = ml_pipeline.load_data("weather")
    x_train, x_test, y_train, y_test = train_test_split(
        ds.X, ds.y, test_size=0.2, random_state=42, stratify=ds.y
    )
    pipe = ml_pipeline.build_pipeline(ds, RandomForestClassifier(random_state=42))
    search = ml_pipeline.tune_model(pipe, ml_pipeline.RF_GRID, x_train, y_train)
    return ds, search, x_test, y_test


def test_grid_search_returns_best_and_reports_metrics() -> None:
    ds, search, x_test, y_test = _fit_weather()
    m = ml_pipeline.evaluate(search, ds, x_test, y_test)
    assert "clf__n_estimators" in m["best_params"]
    for key in ("accuracy", "recall_pos", "precision_pos", "f1_pos", "roc_auc"):
        assert 0.0 <= m[key] <= 1.0
    assert m["accuracy"] > 0.7  # synthetic is learnable


def test_persisted_model_round_trips(tmp_output: Path) -> None:
    import joblib

    ds, search, x_test, y_test = _fit_weather()
    path = tmp_output / "test_roundtrip.joblib"
    joblib.dump(search.best_estimator_, path)
    # Safe: this reloads the file this same test just wrote.
    reloaded = joblib.load(path)
    assert np.array_equal(reloaded.predict(x_test), search.predict(x_test))


# =============================================================================
# The model card
# =============================================================================


def test_model_card_records_grid_winner_and_scores(
    tmp_output: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ml_pipeline, "OUT_DIR", tmp_output)
    result = ml_pipeline.run_one("weather")
    card = (tmp_output / "model_card_weather.md").read_text(encoding="utf-8")
    for needle in (
        "# Model card",
        "## Data",
        "## Candidates and grid",
        "## Winner",
        "## Test-set scores",
        "## Limits",
    ):
        assert needle in card
    assert result["winner"] in card and f"{result['f1']:.3f}" in card


# =============================================================================
# Determinism
# =============================================================================


def test_two_runs_identical() -> None:
    def run() -> tuple[dict[str, Any], float, float]:
        ds = ml_pipeline.load_data("weather")
        x_train, x_test, y_train, y_test = train_test_split(
            ds.X, ds.y, test_size=0.2, random_state=42, stratify=ds.y
        )
        pipe = ml_pipeline.build_pipeline(ds, RandomForestClassifier(random_state=42))
        search = ml_pipeline.tune_model(pipe, ml_pipeline.RF_GRID, x_train, y_train)
        m = ml_pipeline.evaluate(search, ds, x_test, y_test)
        return m["best_params"], round(m["accuracy"], 6), round(m["recall_pos"], 6)

    assert run() == run()


# =============================================================================
# Saved plots (never shown)
# =============================================================================


def test_run_one_writes_plots_and_metrics(
    tmp_output: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import matplotlib

    monkeypatch.setattr(ml_pipeline, "OUT_DIR", tmp_output)
    result = ml_pipeline.run_one("weather")
    for name in (
        "01_confusion_rf_weather.png",
        "02_rf_importances_weather.png",
        "03_confusion_lr_weather.png",
        "04_lr_coefficients_weather.png",
    ):
        assert (tmp_output / name).stat().st_size > 1000
    assert matplotlib.get_backend().lower().startswith("agg")
    text = (tmp_output / "metrics_weather.txt").read_text(encoding="utf-8")
    assert "weather" in text
    assert result["winner"] in ("Random Forest", "Logistic Regression")
    assert 0.0 <= result["f1"] <= 1.0
    assert 0.0 <= result["roc_auc"] <= 1.0
