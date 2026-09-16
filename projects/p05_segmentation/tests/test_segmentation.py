"""Tests for segmentation.py: dataset registry, standardization, k selection,
ground-truth recovery, PCA reporting, determinism and the saved plots.

Run:
    uv run pytest projects/p05_segmentation -q                # offline only
    uv run pytest projects/p05_segmentation -m network -q     # wholesale/taxis/planets/winequality
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from projects.p05_segmentation import segmentation

pytestmark = pytest.mark.core

OFFLINE_DATASETS = ("customers", "digits")
NETWORK_DATASETS = ("wholesale", "taxis", "planets", "winequality")

# Dataset -> the column name its reference label would carry if load_data
# ever leaked it into the feature frame (None where no such column exists).
LABEL_COLUMNS: dict[str, str | None] = {
    "customers": None,
    "digits": None,
    "wholesale": "Channel",
    "taxis": None,
    "planets": None,
    "winequality": "quality",
}


# =============================================================================
# Registry
# =============================================================================


def test_registry_has_at_least_five_datasets() -> None:
    assert len(segmentation.DATASETS) >= 5


def test_synthetic_customers_has_at_least_three_features_and_true_k() -> None:
    ds = segmentation.load_data("customers")
    assert len(ds.X) >= 300
    assert ds.X.shape[1] >= 3
    assert ds.true_k == 3


@pytest.mark.parametrize("name", OFFLINE_DATASETS)
def test_offline_datasets_load_numeric_and_unlabelled(name: str) -> None:
    ds = segmentation.load_data(name)
    assert all(np.issubdtype(dt, np.number) for dt in ds.X.dtypes), name
    assert ds.X.shape[1] >= 3, name
    label_col = LABEL_COLUMNS[name]
    if label_col is not None:
        assert label_col not in ds.X.columns, name


@pytest.mark.network
@pytest.mark.parametrize("name", NETWORK_DATASETS)
def test_downloaded_datasets_load_numeric_and_unlabelled(name: str) -> None:
    ds = segmentation.load_data(name)
    assert all(np.issubdtype(dt, np.number) for dt in ds.X.dtypes), name
    assert ds.X.shape[1] >= 3, name
    label_col = LABEL_COLUMNS[name]
    if label_col is not None:
        assert label_col not in ds.X.columns, name


def test_unknown_dataset_raises() -> None:
    with pytest.raises(ValueError):
        segmentation.load_data("nope")


# =============================================================================
# Scaling
# =============================================================================


def test_preprocess_standardizes_features() -> None:
    ds = segmentation.load_data("customers")
    x_scaled = segmentation.preprocess(ds)
    assert np.abs(x_scaled.mean(axis=0)).max() < 1e-9
    assert np.abs(x_scaled.std(axis=0) - 1).max() < 1e-6


def test_preprocess_tolerates_constant_columns() -> None:
    # digits has always-zero corner pixels; preprocess must not blow up.
    ds = segmentation.load_data("digits")
    x_scaled = segmentation.preprocess(ds)
    assert x_scaled.shape[0] == len(ds.X)


# =============================================================================
# Choosing k
# =============================================================================


def test_choose_k_recovers_the_three_synthetic_segments() -> None:
    ds = segmentation.load_data("customers")
    x = segmentation.preprocess(ds)
    rows, best_k = segmentation.choose_k(x, k_max=8)
    assert best_k == 3
    assert max(rows, key=lambda r: r["silhouette"])["k"] == 3


def test_inertia_is_monotonically_non_increasing() -> None:
    ds = segmentation.load_data("customers")
    x = segmentation.preprocess(ds)
    rows, _best_k = segmentation.choose_k(x, k_max=8)
    inertias = [r["inertia"] for r in rows]
    assert inertias == sorted(inertias, reverse=True)


# =============================================================================
# Ground-truth recovery
# =============================================================================


def test_synthetic_segments_recovered() -> None:
    ds = segmentation.load_data("customers")
    x = segmentation.preprocess(ds)
    _km, labels = segmentation.fit_final(x, 3)
    ari = segmentation.validate_against_truth(ds, labels)
    assert ari is not None
    assert ari > 0.9  # near-perfect recovery of the 3 recipes


def test_profile_has_one_row_per_segment() -> None:
    ds = segmentation.load_data("customers")
    x = segmentation.preprocess(ds)
    _km, labels = segmentation.fit_final(x, 3)
    profile = segmentation.segment_profile(ds, labels)
    assert len(profile) == 3
    assert int(profile["n"].sum()) == len(ds.X)


# =============================================================================
# PCA
# =============================================================================


def test_two_components_and_variance_reported() -> None:
    ds = segmentation.load_data("customers")
    x = segmentation.preprocess(ds)
    pts, evr = segmentation.pca_2d(x)
    assert pts.shape == (len(ds.X), 2)
    assert len(evr) == 2
    assert 0.0 < evr[:2].sum() <= 1.0 + 1e-9


# =============================================================================
# Determinism
# =============================================================================


def test_two_runs_identical() -> None:
    def run() -> tuple[int, tuple[int, ...], tuple[float, ...]]:
        ds = segmentation.load_data("customers")
        x = segmentation.preprocess(ds)
        rows, best_k = segmentation.choose_k(x, ds.k_max)
        _km, labels = segmentation.fit_final(x, best_k)
        return best_k, tuple(labels.tolist()), tuple(round(r["silhouette"], 6) for r in rows)

    assert run() == run()


# =============================================================================
# Saved plots (never shown)
# =============================================================================


def test_elbow_plot_is_saved_not_shown(tmp_output: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import matplotlib

    monkeypatch.setattr(segmentation, "OUT_DIR", tmp_output)
    segmentation.run_one("customers")
    assert (tmp_output / "elbow_silhouette_customers.png").stat().st_size > 1000
    assert matplotlib.get_backend().lower().startswith("agg")


def test_run_one_writes_pca_plot_and_metrics(
    tmp_output: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(segmentation, "OUT_DIR", tmp_output)
    result = segmentation.run_one("customers")
    assert (tmp_output / "pca_segments_customers.png").stat().st_size > 1000
    text = (tmp_output / "metrics_customers.txt").read_text(encoding="utf-8")
    assert "customers" in text
    assert result["best_k"] == 3
