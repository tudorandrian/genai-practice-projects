"""Tests for decision_tree.py: dataset registry, one-hot preprocessing,
stratified split, synthetic-drug rule recovery, depth comparison, determinism,
importance ranking and the rules export.

Run:
    uv run pytest projects/p04_decision_tree -q                # offline only
    uv run pytest projects/p04_decision_tree -m network -q     # penguins + titanic
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from projects.p04_decision_tree import decision_tree

pytestmark = pytest.mark.core

OFFLINE_DATASETS = ("drug", "iris", "wine", "breast_cancer")
NETWORK_DATASETS = ("penguins", "titanic")


# =============================================================================
# Registry
# =============================================================================


def test_registry_has_at_least_five_datasets() -> None:
    assert len(decision_tree.DATASETS) >= 5


@pytest.mark.parametrize("name", OFFLINE_DATASETS)
def test_offline_datasets_load_and_shapes_line_up(name: str) -> None:
    ds = decision_tree.load_data(name)
    assert len(ds.X_raw) == len(ds.y), name
    assert len(ds.y.unique()) >= 2, name
    assert set(ds.numeric) | set(ds.categorical) == set(ds.X_raw.columns), name


@pytest.mark.network
@pytest.mark.parametrize("name", NETWORK_DATASETS)
def test_downloaded_datasets_load_and_shapes_line_up(name: str) -> None:
    ds = decision_tree.load_data(name)
    assert len(ds.X_raw) == len(ds.y), name
    assert len(ds.y.unique()) >= 2, name
    assert set(ds.numeric) | set(ds.categorical) == set(ds.X_raw.columns), name


def test_unknown_dataset_raises() -> None:
    with pytest.raises(ValueError):
        decision_tree.load_data("nope")


# =============================================================================
# Preprocess
# =============================================================================


def test_get_dummies_expands_categoricals() -> None:
    ds = decision_tree.load_data("drug")
    x = decision_tree.preprocess(ds)
    for cat in ds.categorical:
        assert cat not in x.columns
    assert "BP_HIGH" in x.columns
    assert "Sex_F" in x.columns
    assert "Age" in x.columns
    assert "Na_to_K" in x.columns


def test_numeric_only_dataset_is_unchanged_by_preprocess() -> None:
    ds = decision_tree.load_data("iris")
    x = decision_tree.preprocess(ds)
    assert list(x.columns) == list(ds.X_raw.columns)


# =============================================================================
# Stratification
# =============================================================================


def test_class_proportions_preserved_within_tolerance() -> None:
    ds = decision_tree.load_data("drug")
    x = decision_tree.preprocess(ds)
    _x_tr, _x_te, y_tr, y_te = decision_tree.stratified_split(x, ds.y)
    p_tr = decision_tree.class_proportions(y_tr)
    p_te = decision_tree.class_proportions(y_te)
    for cls in p_tr.index:
        assert p_tr[cls] == pytest.approx(p_te[cls], abs=0.02), cls


# =============================================================================
# Synthetic drug: rules and accuracy
# =============================================================================


def _fit_drug(
    depth: int = 4,
) -> tuple[decision_tree.Dataset, pd.DataFrame, DecisionTreeClassifier, pd.DataFrame, pd.Series]:
    ds = decision_tree.load_data("drug")
    x = decision_tree.preprocess(ds)
    x_tr, x_te, y_tr, y_te = decision_tree.stratified_split(x, ds.y)
    model = decision_tree.train_tree(x_tr, y_tr, max_depth=depth)
    return ds, x, model, x_te, y_te


def test_synthetic_drug_is_perfectly_separable_at_depth4() -> None:
    _ds, _x, model, x_te, y_te = _fit_drug(4)
    acc = decision_tree.evaluate(model, x_te, y_te)["accuracy"]
    assert acc == 1.0  # rules are exactly recoverable by construction


def test_extracted_rules_recover_the_generator() -> None:
    _ds, x, model, _x_te, _y_te = _fit_drug(4)
    rules = decision_tree.extract_rules(model, list(x.columns))
    assert "Na_to_K" in rules  # the dominant split
    assert "DrugY" in rules  # Na_to_K > 15 leaf
    assert "DrugC" in rules  # BP LOW leaf


def test_predict_patient_matches_hand_rules() -> None:
    ds, x, model, _x_te, _y_te = _fit_drug(4)
    cols = list(x.columns)
    cases = [
        ({"Age": 30, "Sex": "F", "BP": "HIGH", "Cholesterol": "HIGH", "Na_to_K": 8.0}, "DrugA"),
        ({"Age": 65, "Sex": "M", "BP": "HIGH", "Cholesterol": "NORMAL", "Na_to_K": 8.0}, "DrugB"),
        ({"Age": 45, "Sex": "F", "BP": "LOW", "Cholesterol": "HIGH", "Na_to_K": 9.0}, "DrugC"),
        ({"Age": 45, "Sex": "M", "BP": "NORMAL", "Cholesterol": "HIGH", "Na_to_K": 9.0}, "DrugX"),
        (
            {"Age": 20, "Sex": "F", "BP": "NORMAL", "Cholesterol": "NORMAL", "Na_to_K": 30.0},
            "DrugY",
        ),
    ]
    for profile, expected in cases:
        assert decision_tree.predict_patient(model, ds, cols, profile) == expected, profile


# =============================================================================
# Depth comparison
# =============================================================================


def test_depth4_at_least_as_accurate_as_depth3_on_drug() -> None:
    ds = decision_tree.load_data("drug")
    x = decision_tree.preprocess(ds)
    x_tr, x_te, y_tr, y_te = decision_tree.stratified_split(x, ds.y)
    acc4 = decision_tree.evaluate(decision_tree.train_tree(x_tr, y_tr, 4), x_te, y_te)["accuracy"]
    acc3 = decision_tree.evaluate(decision_tree.train_tree(x_tr, y_tr, 3), x_te, y_te)["accuracy"]
    assert acc4 >= acc3  # extra depth recovers the last rule


# =============================================================================
# Determinism
# =============================================================================


def test_two_runs_identical_accuracy() -> None:
    def run() -> float:
        ds = decision_tree.load_data("drug")
        x = decision_tree.preprocess(ds)
        x_tr, x_te, y_tr, y_te = decision_tree.stratified_split(x, ds.y)
        model = decision_tree.train_tree(x_tr, y_tr, 4)
        return float(decision_tree.evaluate(model, x_te, y_te)["accuracy"])

    assert run() == run()


def test_synthetic_drug_is_stable_and_balanced() -> None:
    a, b = decision_tree._synthetic_drug(), decision_tree._synthetic_drug()
    assert a.equals(b)
    assert a["Drug"].value_counts().min() >= 60


# =============================================================================
# Importance ranking
# =============================================================================


def test_importances_sorted_desc_and_named() -> None:
    ds = decision_tree.load_data("drug")
    x = decision_tree.preprocess(ds)
    x_tr, _x_te, y_tr, _y_te = decision_tree.stratified_split(x, ds.y)
    model = decision_tree.train_tree(x_tr, y_tr, 4)
    ranking = decision_tree.rank_importances(model, list(x.columns))
    vals = [v for _, v in ranking]
    assert vals == sorted(vals, reverse=True)
    assert ranking[0][0] == "Na_to_K"  # the dominant feature


# =============================================================================
# Rules export
# =============================================================================


def test_rules_export_names_real_features(
    tmp_output: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(decision_tree, "OUT_DIR", tmp_output)
    decision_tree.run_one("drug")
    rules = (tmp_output / "rules_drug.txt").read_text(encoding="utf-8")
    assert "|--- " in rules  # sklearn export_text format
    assert "Na_to_K" in rules and "class: " in rules
