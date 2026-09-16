"""decision_tree.py - an explainable decision-tree classifier for tabular data.

The point is not accuracy for its own sake but **explainability**: a decision
tree whose drawn diagram *is* the model's rule ("if Na/K > 15 -> DrugY"). The
disciplined workflow around it - one-hot encoding of categoricals, a
stratified train/test split, a depth comparison (3 vs 4), and the extracted
text rules - is shown to be dataset-agnostic by running the exact same
pipeline across several public classification datasets, mixing numeric-only
and mixed numeric+categorical data, binary and multi-class targets:

    drug           (synthetic generator)     5 drug classes - PRIMARY, mixed features
    iris           (sklearn)                 3 species      - 4 numeric features
    wine           (sklearn)                 3 cultivars    - 13 numeric features
    breast_cancer  (sklearn)                 2 classes      - 30 numeric features
    penguins       (seaborn palmerpenguins)  3 species      - mixed (island/sex + numeric)
    titanic        (seaborn)                 2 classes      - mixed (sex/embarked + numeric)

PIPELINE  (pure functions)
    load_data -> explore_data -> preprocess -> stratified_split
              -> train_tree -> evaluate -> plot_* / extract_rules -> predict_patient

HOW TO RUN
    python decision_tree.py                    # default dataset: drug (synthetic)
    python decision_tree.py --dataset iris      # a numeric-only 3-class set
    python decision_tree.py --dataset all       # run every dataset + comparison table
    python decision_tree.py --demo              # drug + iris, writes output/metrics.txt

DEPENDENCIES  numpy, pandas, scikit-learn, matplotlib
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless backend - never plt.show()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.datasets import (  # noqa: E402
    load_breast_cancer,
    load_iris,
    load_wine,
)
from sklearn.metrics import accuracy_score, classification_report  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.tree import (  # noqa: E402
    DecisionTreeClassifier,
    export_text,
    plot_tree,
)

from shared.datasets import fetch  # noqa: E402
from shared.demo import DemoResult  # noqa: E402

log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output"
RANDOM_STATE = 42
TEST_SIZE = 0.3  # 70/30 stratified split

_SEABORN = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/"

# Datasets fetched through shared.datasets.fetch (checksum-verified download,
# cached once under GENAI_DATA_DIR). Everything else is either the synthetic
# drug generator or bundled with scikit-learn.
DATASET_SOURCES: dict[str, tuple[str, str]] = {
    "penguins": (
        _SEABORN + "penguins.csv",
        "e07636bd8af74260099ea2f8678e2eabbf35def579940cc76f67061ee16c06c1",
    ),
    "titanic": (
        _SEABORN + "titanic.csv",
        "81787d320d7f7b03df935e91de8bd19e11d45c5bbcab86ef4d4a76dc91b7d4f2",
    ),
}

DATASETS = ("drug", "iris", "wine", "breast_cancer", "penguins", "titanic")


# =============================================================================
# Dataset registry
# =============================================================================


@dataclass
class Dataset:
    """A uniform container: raw features (numeric + categorical) + target.

    ``categorical`` names the columns that get one-hot encoded in
    ``preprocess``. Everything else in ``X_raw`` passes through unchanged.
    """

    name: str
    X_raw: pd.DataFrame
    y: pd.Series
    categorical: list[str]
    target_name: str
    domain: str
    source: str
    numeric: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.numeric = [c for c in self.X_raw.columns if c not in self.categorical]


def _synthetic_drug(n: int = 600) -> pd.DataFrame:
    """A deterministic drug200-style dataset: same columns, clean rules.

    The rules are exactly recoverable by a depth-4 tree, so accuracy is
    ~1.000 - which makes the drawn tree an exact, auditable statement of the
    rule instead of an approximation.
    """
    rng = np.random.default_rng(RANDOM_STATE)  # fixed seed -> deterministic
    # Na_to_K is drawn from two bands around the 15 threshold (25% high, 75%
    # low) rather than one wide uniform range, so all five classes stay
    # well-represented instead of DrugY swamping the rest.
    is_high_na = rng.random(n) < 0.25
    na_to_k = np.where(is_high_na, rng.uniform(15.1, 40, n), rng.uniform(6, 15, n)).round(3)
    df = pd.DataFrame(
        {
            "Age": rng.integers(15, 75, n),
            "Sex": rng.choice(["F", "M"], n),
            "BP": rng.choice(["LOW", "NORMAL", "HIGH"], n, p=[0.3, 0.3, 0.4]),
            "Cholesterol": rng.choice(["NORMAL", "HIGH"], n),
            "Na_to_K": na_to_k,
        }
    )

    def label(r: pd.Series) -> str:
        if r["Na_to_K"] > 15:
            return "DrugY"
        if r["BP"] == "HIGH":
            return "DrugA" if r["Age"] <= 50 else "DrugB"
        if r["BP"] == "LOW":
            return "DrugC"
        return "DrugX"

    df["Drug"] = df.apply(label, axis=1)
    return df


def _load_drug() -> pd.DataFrame:
    """Load the drug dataset - always the deterministic synthetic generator."""
    return _synthetic_drug()


def load_data(name: str) -> Dataset:
    """Load one classification dataset by name into a uniform Dataset container."""
    if name == "drug":
        df = _load_drug()
        feats = ["Age", "Sex", "BP", "Cholesterol", "Na_to_K"]
        return Dataset(
            "drug",
            df[feats].copy(),
            df["Drug"].astype(str),
            categorical=["Sex", "BP", "Cholesterol"],
            target_name="Drug",
            domain="pharmacology",
            source="synthetic drug200-style generator (see Design notes)",
        )

    if name == "iris":
        raw = load_iris(as_frame=True)
        y = raw.target.map(dict(enumerate(raw.target_names))).astype(str)
        return Dataset(
            "iris",
            raw.data,
            y,
            categorical=[],
            target_name="species",
            domain="botany",
            source="sklearn.datasets.load_iris (Fisher 1936)",
        )

    if name == "wine":
        raw = load_wine(as_frame=True)
        y = raw.target.map(dict(enumerate(raw.target_names))).astype(str)
        return Dataset(
            "wine",
            raw.data,
            y,
            categorical=[],
            target_name="cultivar",
            domain="chemistry",
            source="sklearn.datasets.load_wine (UCI)",
        )

    if name == "breast_cancer":
        raw = load_breast_cancer(as_frame=True)
        y = raw.target.map(dict(enumerate(raw.target_names))).astype(str)
        return Dataset(
            "breast_cancer",
            raw.data,
            y,
            categorical=[],
            target_name="diagnosis",
            domain="oncology",
            source="sklearn.datasets.load_breast_cancer (WDBC, UCI)",
        )

    if name == "penguins":
        url, sha256 = DATASET_SOURCES["penguins"]
        path = fetch("penguins", url, sha256)
        df = pd.read_csv(path)
        num = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
        cat = ["island", "sex"]
        df = df.dropna(subset=num + cat + ["species"]).reset_index(drop=True)
        return Dataset(
            "penguins",
            df[num + cat].copy(),
            df["species"].astype(str),
            categorical=cat,
            target_name="species",
            domain="biology",
            source="seaborn palmerpenguins (Gorman et al. 2014, CC0)",
        )

    if name == "titanic":
        url, sha256 = DATASET_SOURCES["titanic"]
        path = fetch("titanic", url, sha256)
        df = pd.read_csv(path)
        num = ["pclass", "age", "sibsp", "parch", "fare"]
        cat = ["sex", "embarked"]
        df = df.dropna(subset=num + cat + ["survived"]).reset_index(drop=True)
        y = df["survived"].map({0: "died", 1: "survived"}).astype(str)
        return Dataset(
            "titanic",
            df[num + cat].copy(),
            y,
            categorical=cat,
            target_name="survived",
            domain="history/survival",
            source=(
                "seaborn titanic - no explicit licence upstream "
                "(redistributed via seaborn-data for teaching)"
            ),
        )

    raise ValueError(f"unknown dataset '{name}' (expected {', '.join(DATASETS)})")


# =============================================================================
# Pipeline steps (pure)
# =============================================================================


def explore_data(ds: Dataset) -> pd.Series:
    """Return the class distribution (counts, descending) of the target."""
    return ds.y.value_counts()


def preprocess(ds: Dataset) -> pd.DataFrame:
    """One-hot encode the categorical columns with pd.get_dummies.

    Numeric columns pass through unchanged. Returns the model-ready feature
    frame; ``list(x.columns)`` is exactly what plot_tree/export_text label with.
    """
    return pd.get_dummies(ds.X_raw, columns=ds.categorical)


def stratified_split(
    x: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 70/30 split. Returns the four splits; the caller can compare
    class proportions across train/test to see stratification held."""
    return train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


def class_proportions(y: pd.Series) -> pd.Series:
    """Normalised class frequencies, sorted by label for stable side-by-side view."""
    return y.value_counts(normalize=True).sort_index()


def train_tree(x_train: pd.DataFrame, y_train: pd.Series, max_depth: int) -> DecisionTreeClassifier:
    """Entropy-criterion decision tree at a given depth, fixed seed."""
    model = DecisionTreeClassifier(
        criterion="entropy", max_depth=max_depth, random_state=RANDOM_STATE
    )
    return model.fit(x_train, y_train)


def evaluate(
    model: DecisionTreeClassifier, x_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, Any]:
    y_pred = model.predict(x_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, zero_division=0),
    }


def extract_rules(model: DecisionTreeClassifier, feature_names: list[str]) -> str:
    """The tree as human-readable text - reproduce a prediction by hand."""
    return export_text(model, feature_names=list(feature_names))


def rank_importances(
    model: DecisionTreeClassifier, feature_names: list[str]
) -> list[tuple[str, float]]:
    pairs = list(zip(feature_names, model.feature_importances_, strict=True))
    return sorted(pairs, key=lambda kv: kv[1], reverse=True)


def _align_profile(
    profile: dict[str, Any], ds: Dataset, feature_columns: list[str]
) -> pd.DataFrame:
    """Turn a raw patient/example profile into a one-row frame aligned to the
    trained one-hot columns (missing dummy columns -> 0) - the manual step
    pd.get_dummies forces on us at predict time."""
    row = pd.DataFrame([profile])[ds.X_raw.columns]
    row = pd.get_dummies(row, columns=ds.categorical)
    return row.reindex(columns=feature_columns, fill_value=0)


def predict_patient(
    model: DecisionTreeClassifier, ds: Dataset, feature_columns: list[str], profile: dict[str, Any]
) -> str:
    """Recommend a class for a brand-new profile dict."""
    aligned = _align_profile(profile, ds, feature_columns)
    return str(model.predict(aligned)[0])


# =============================================================================
# Outputs
# =============================================================================


def plot_class_distribution(ds: Dataset, counts: pd.Series, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.index.astype(str), counts.to_numpy(), color="#4C72B0")
    ax.set_xlabel(ds.target_name)
    ax.set_ylabel("count")
    ax.set_title(f"{ds.name}: class distribution ({len(counts)} classes)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_feature_importance(
    ds: Dataset, ranking: list[tuple[str, float]], path: Path, top: int = 15
) -> None:
    top_pairs = [p for p in ranking if p[1] > 0][:top] or ranking[:top]
    names = [n for n, _ in top_pairs][::-1]
    vals = [v for _, v in top_pairs][::-1]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(names) + 1)))
    ax.barh(names, vals, color="#55A868")
    ax.set_xlabel("feature importance (entropy reduction)")
    ax.set_title(f"{ds.name}: decision-tree feature importances")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_tree_image(
    model: DecisionTreeClassifier,
    feature_names: list[str],
    class_names: list[str],
    depth: int,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    plot_tree(
        model,
        feature_names=list(feature_names),
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=7,
        ax=ax,
    )
    ax.set_title(f"Decision tree (criterion=entropy, max_depth={depth})")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def write_metrics(
    ds: Dataset,
    x: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    acc4: float,
    acc3: float,
    report4: str,
    rules: str,
    ranking: list[tuple[str, float]],
    path: Path,
) -> None:
    prop_tr = class_proportions(y_train)
    prop_te = class_proportions(y_test)
    lines = [
        "DECISION TREE CLASSIFIER - METRICS",
        "=" * 60,
        f"Dataset      : {ds.name}  ({ds.domain})",
        f"Source       : {ds.source}",
        f"Target       : {ds.target_name}   classes = {sorted(ds.y.unique())}",
        f"Rows         : {len(y_train)} train / {len(y_test)} test",
        f"Features     : {x.shape[1]} after one-hot "
        f"({len(ds.numeric)} numeric + {len(ds.categorical)} categorical -> dummies)",
        "",
        "One-hot feature columns:",
        f"  {list(x.columns)}",
        "",
        "Stratified split - class proportions:",
        f"  {'class':<14}{'train':>10}{'test':>10}",
    ]
    for cls in prop_tr.index:
        lines.append(f"  {str(cls):<14}{prop_tr[cls]:>10.3f}{prop_te.get(cls, 0):>10.3f}")
    lines += [
        "",
        "Accuracy on the held-out test set:",
        f"  max_depth = 4 : {acc4:.4f}",
        f"  max_depth = 3 : {acc3:.4f}",
        f"  delta (4 - 3) : {acc4 - acc3:+.4f}",
        "",
        "classification_report (max_depth=4):",
        report4,
        "Feature importances (desc):",
    ]
    for feat, imp in ranking:
        if imp > 0:
            lines.append(f"  {feat:<30} {imp:>8.4f}")
    lines += [
        "",
        "Decision rules - export_text (max_depth=4):",
        rules,
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_rules(rules: str, path: Path) -> None:
    """Standalone rules-only export - the same text embedded in metrics.txt,
    written on its own so a rule set can be diffed or read without the rest."""
    text = rules if rules.endswith("\n") else rules + "\n"
    path.write_text(text, encoding="utf-8")


# =============================================================================
# Orchestration
# =============================================================================


def run_one(name: str, suffix: str = "") -> dict[str, str | int | float]:
    """Run the full pipeline on one dataset.

    Writes ``metrics<tag>.txt``, ``rules<tag>.txt``, ``decision_tree<tag>.png``,
    ``class_distribution<tag>.png`` and ``feature_importance<tag>.png`` under
    ``OUT_DIR``, where ``tag`` is ``suffix`` if given, else ``_<name>`` - so
    every dataset gets its own set of output files by default. Returns a
    compact result dict.
    """
    ds = load_data(name)
    counts = explore_data(ds)
    x = preprocess(ds)
    class_names = sorted(ds.y.unique())

    x_train, x_test, y_train, y_test = stratified_split(x, ds.y)
    model4 = train_tree(x_train, y_train, max_depth=4)
    model3 = train_tree(x_train, y_train, max_depth=3)
    m4 = evaluate(model4, x_test, y_test)
    m3 = evaluate(model3, x_test, y_test)
    rules = extract_rules(model4, list(x.columns))
    ranking = rank_importances(model4, list(x.columns))

    tag = suffix or f"_{name}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tree_path = OUT_DIR / f"decision_tree{tag}.png"
    dist_path = OUT_DIR / f"class_distribution{tag}.png"
    imp_path = OUT_DIR / f"feature_importance{tag}.png"
    metrics_path = OUT_DIR / f"metrics{tag}.txt"
    rules_path = OUT_DIR / f"rules{tag}.txt"

    plot_tree_image(model4, list(x.columns), class_names, 4, tree_path)
    plot_class_distribution(ds, counts, dist_path)
    plot_feature_importance(ds, ranking, imp_path)
    write_metrics(
        ds,
        x,
        y_train,
        y_test,
        m4["accuracy"],
        m3["accuracy"],
        m4["report"],
        rules,
        ranking,
        metrics_path,
    )
    write_rules(rules, rules_path)

    log.info(
        "[%s] n=%d classes=%d feats=%d  acc(d4)=%.4f  acc(d3)=%.4f",
        ds.name,
        len(ds.y),
        len(class_names),
        x.shape[1],
        m4["accuracy"],
        m3["accuracy"],
    )
    log.info("    top feature: %s (%.3f)", ranking[0][0], ranking[0][1])
    log.info(
        "    wrote %s, %s, %s, %s, %s",
        metrics_path.name,
        rules_path.name,
        tree_path.name,
        dist_path.name,
        imp_path.name,
    )

    if name == "drug":
        profiles: list[dict[str, Any]] = [
            {"Age": 23, "Sex": "F", "BP": "HIGH", "Cholesterol": "HIGH", "Na_to_K": 9.5},
            {"Age": 61, "Sex": "M", "BP": "LOW", "Cholesterol": "NORMAL", "Na_to_K": 11.0},
            {"Age": 40, "Sex": "F", "BP": "NORMAL", "Cholesterol": "HIGH", "Na_to_K": 22.0},
        ]
        for prof in profiles:
            rec = predict_patient(model4, ds, list(x.columns), prof)
            log.info("    predict_patient(%s) -> %s", prof, rec)

    return {
        "name": ds.name,
        "n": len(ds.y),
        "classes": len(class_names),
        "feats": x.shape[1],
        "acc4": m4["accuracy"],
        "acc3": m3["accuracy"],
    }


def write_summary(results: list[dict[str, str | int | float]], path: Path) -> None:
    """Deterministic cross-dataset comparison table - no timestamps, no absolute paths."""
    header = f"{'Dataset':<15}{'n':>7}{'classes':>9}{'feats':>7}{'acc(d4)':>10}{'acc(d3)':>10}"
    lines = ["DECISION TREE - CROSS-DATASET SUMMARY", "=" * 58, header, "-" * 58]
    for r in results:
        lines.append(
            f"{r['name']:<15}{r['n']:>7}{r['classes']:>9}{r['feats']:>7}"
            f"{r['acc4']:>10.4f}{r['acc3']:>10.4f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_demo_metrics(results: list[dict[str, str | int | float]], path: Path) -> None:
    """Deterministic figures-only proof file: no timestamps, no absolute paths."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for r in results:
        lines.append(f"{r['name']}_acc4={r['acc4']:.3f}")
        lines.append(f"{r['name']}_acc3={r['acc3']:.3f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def demo() -> DemoResult:
    """Run the pipeline on drug + iris (both offline, deterministic) and write
    output/metrics.txt (figures) and output/summary.txt (comparison table)."""
    start = time.perf_counter()
    results = [run_one("drug"), run_one("iris")]
    _write_demo_metrics(results, OUT_DIR / "metrics.txt")
    write_summary(results, OUT_DIR / "summary.txt")
    figures = {f"{r['name']}_acc": f"{r['acc4']:.3f}" for r in results}
    return DemoResult(
        "p04-decision-tree",
        "ok",
        figures,
        seconds=round(time.perf_counter() - start, 2),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument(
        "--dataset",
        default="drug",
        choices=(*DATASETS, "all"),
        help="dataset to run (default: drug); 'all' runs every dataset.",
    )
    p.add_argument("--demo", action="store_true", help="Run the offline demo (drug + iris).")
    p.add_argument("--verbose", action="store_true", help="Log pipeline steps at INFO level.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s"
    )

    if args.demo:
        result = demo()
        print(f"p04-decision-tree: {result.status}")
        for key, value in result.figures.items():
            print(f"  {key}: {value}")
        print("  wrote: output/")
        print(f"  seconds: {result.seconds:.2f}")
        return 0 if result.status == "ok" else 1

    if args.dataset == "all":
        results = [run_one(name, suffix=f"_{name}") for name in DATASETS]
        write_summary(results, OUT_DIR / "summary_all.txt")
        print(f"p04-decision-tree: all ({len(results)} datasets)")
        print("  wrote: output/summary_all.txt")
        return 0

    dataset_result = run_one(args.dataset)
    print(f"p04-decision-tree: {dataset_result['name']}")
    print(
        f"  n={dataset_result['n']} classes={dataset_result['classes']} "
        f"feats={dataset_result['feats']}  acc(d4)={dataset_result['acc4']:.4f}  "
        f"acc(d3)={dataset_result['acc3']:.4f}"
    )
    print(f"  wrote: output/metrics_{dataset_result['name']}.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
