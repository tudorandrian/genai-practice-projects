"""Tests for regression.py: dataset registry, anti-leakage split, reference
metrics, determinism and coefficient ranking.

Run:
    uv run pytest projects/p03_regression -q                # offline only
    uv run pytest projects/p03_regression -m network -q     # california + diamonds
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from projects.p03_regression import regression

pytestmark = pytest.mark.core

OFFLINE_SHAPES = {
    "diabetes": (442, 10),
    "co2": (1067, 6),
    "mpg": (392, 6),
    "tips": (244, 2),
    "penguins": (342, 3),
}
NETWORK_SHAPES = {
    "california": (20640, 8),
    "diamonds": (53940, 6),
}


# =============================================================================
# Registry
# =============================================================================


def test_offline_datasets_load_with_expected_shapes() -> None:
    for name, (n, p) in OFFLINE_SHAPES.items():
        ds = regression.load_data(name)
        assert ds.X.shape == (n, p), name
        assert len(ds.y) == n, name
        assert len(ds.feature_names) == p, name


@pytest.mark.network
def test_downloaded_datasets_load_with_expected_shapes() -> None:
    for name, (n, p) in NETWORK_SHAPES.items():
        ds = regression.load_data(name)
        assert ds.X.shape == (n, p), name
        assert len(ds.y) == n, name
        assert len(ds.feature_names) == p, name


def test_registry_has_at_least_five_datasets() -> None:
    assert len(regression.DATASETS) >= 5


def test_unknown_dataset_raises() -> None:
    with pytest.raises(ValueError):
        regression.load_data("nope")


@pytest.mark.parametrize("name", ["diabetes", "co2", "mpg", "tips", "penguins"])
def test_offline_datasets_load_clean(name: str) -> None:
    ds = regression.load_data(name)
    assert len(ds.X) == len(ds.y) > 100
    assert not ds.X.isna().any().any()


@pytest.mark.network
@pytest.mark.parametrize("name", ["california", "diamonds"])
def test_downloaded_datasets_load_clean(name: str) -> None:
    ds = regression.load_data(name)
    assert len(ds.X) == len(ds.y) > 1000


# =============================================================================
# Anti-leakage
# =============================================================================


def test_scaler_centres_train_only() -> None:
    ds = regression.load_data("diabetes")
    x_train_s, x_test_s, _y_train, _y_test, _scaler = regression.preprocess(ds.X, ds.y)
    # Train columns are centred (scaler fit on train)...
    assert np.abs(x_train_s.mean(axis=0)).max() < 1e-9
    # ...but the test set is NOT forced to zero mean (no fit on test).
    assert np.abs(x_test_s.mean(axis=0)).max() > 1e-6


def test_split_sizes() -> None:
    ds = regression.load_data("diabetes")
    _x_train_s, _x_test_s, y_train, y_test, _scaler = regression.preprocess(ds.X, ds.y)
    assert len(y_train) == 353  # 442 * 0.8
    assert len(y_test) == 89


# =============================================================================
# Reference values
# =============================================================================


def _metrics(name: str) -> dict[str, float | np.ndarray]:
    ds = regression.load_data(name)
    x_train_s, x_test_s, y_train, y_test, _scaler = regression.preprocess(ds.X, ds.y)
    model = regression.train_model(x_train_s, y_train)
    return regression.evaluate(model, x_test_s, y_test)


def test_diabetes_reference_metrics() -> None:
    m = _metrics("diabetes")
    assert m["R2"] == pytest.approx(0.453, abs=1e-3)
    assert m["MSE"] == pytest.approx(2900.2, abs=1.0)


def test_co2_is_strong_fit() -> None:
    m = _metrics("co2")
    assert m["R2"] > 0.7  # acceptance target for the primary dataset


def test_mpg_reasonable_fit() -> None:
    m = _metrics("mpg")
    assert m["R2"] > 0.7


# =============================================================================
# Determinism
# =============================================================================


def test_two_runs_identical_metrics() -> None:
    def run() -> dict[str, float | np.ndarray]:
        ds = regression.load_data("diabetes")
        x_train_s, x_test_s, y_train, y_test, _scaler = regression.preprocess(ds.X, ds.y)
        model = regression.train_model(x_train_s, y_train)
        m = regression.evaluate(model, x_test_s, y_test)
        return {k: m[k] for k in ("MAE", "MSE", "RMSE", "R2")}

    assert run() == run()


# =============================================================================
# Coefficient ranking
# =============================================================================


def test_ranked_by_absolute_value_desc() -> None:
    ds = regression.load_data("co2")
    x_train_s, _x_test_s, y_train, _y_test, _scaler = regression.preprocess(ds.X, ds.y)
    model = regression.train_model(x_train_s, y_train)
    ranking = regression.rank_coefficients(model, ds.feature_names)
    abs_vals = [abs(c) for _, c in ranking]
    assert abs_vals == sorted(abs_vals, reverse=True)
    assert len(ranking) == len(ds.feature_names)


def test_metrics_file_lists_ranked_coefficients(
    tmp_output: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # run_one(name) with no suffix defaults its tag to "_<name>", so the file is
    # metrics_tips.txt, not the bare "metrics.txt" (regression.run_one docstring).
    monkeypatch.setattr(regression, "OUT_DIR", tmp_output)
    result = regression.run_one("tips")
    text = (tmp_output / "metrics_tips.txt").read_text(encoding="utf-8")
    assert "R2" in text
    assert float(result["R2"]) > 0.4
    assert text.index("total_bill") < text.index("size")  # strongest coefficient first
