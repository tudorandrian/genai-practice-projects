"""ml_pipeline.py - a production-shaped ML pipeline with tuned model comparison,
run across several public tabular datasets.

The point is the *system*, not a single model: preprocessing (scale numerics +
one-hot categoricals) and the classifier live inside ONE ``Pipeline`` tuned with
``GridSearchCV`` over a ``StratifiedKFold``. Two models - Random Forest and
Logistic Regression - are compared on the EXACT same preprocessing stack, the
second swapped in with a single ``set_params(clf=...)`` so the comparison is
honest (same folds, same features, same metric). The winner is persisted with
joblib and round-trip-verified.

Two guarantees the architecture buys us:
  * No data leakage - every CV fold fits its OWN preprocessing (nothing is fit on
    test). The weather set's generator emits `rain_yesterday` directly as a
    feature and `rain_today` only as the target, so only information
    available *before* the prediction is used as a feature.
  * A fair, reproducible A/B of two models as one deployable artifact.

The same pipeline runs across six binary-classification datasets with mixed
numeric+categorical columns and (mostly) imbalanced targets:

    weather         (synthetic weatherAUS-style)  meteorology  PRIMARY, offline
    adult           (OpenML 1590)   census income >50K     ~24% positive
    credit_g        (OpenML 31)     German credit risk     ~30% "bad"
    churn           (OpenML 40701)  telecom churn          ~14% churn
    online_shoppers (UCI CSV)       e-commerce purchase    ~15% buy
    bank_marketing  (OpenML 1461)   term-deposit uptake    ~12% subscribe

PIPELINE  (pure functions)
    load_data -> build_pipeline -> stratified split
              -> tune_model(RF) -> tune_model(LR via set_params)
              -> evaluate (report + confusion) -> persist winner + round-trip
              -> write_metrics / write_model_card

HOW TO RUN
    uv run p06-ml-pipeline                       # default: weather (synthetic)
    uv run p06-ml-pipeline --dataset adult        # a real census set
    uv run p06-ml-pipeline --dataset all          # every dataset + summary
    uv run p06-ml-pipeline --demo                 # weather only, writes output/metrics.txt

DEPENDENCIES  numpy, pandas, scikit-learn, matplotlib, joblib
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

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.datasets import fetch_openml  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (  # noqa: E402
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

from shared.datasets import fetch  # noqa: E402
from shared.demo import DemoResult  # noqa: E402

log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output"
RANDOM_STATE = 42
TEST_SIZE = 0.2
SAMPLE_CAP = 8000  # stratified subsample cap to keep the full run fast

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
RF_GRID: dict[str, list[Any]] = {"clf__n_estimators": [100, 200], "clf__max_depth": [None, 10]}
LR_GRID: dict[str, list[Any]] = {"clf__C": [0.1, 1, 10], "clf__class_weight": [None, "balanced"]}

ONLINE_SHOPPERS_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv"
)
ONLINE_SHOPPERS_SHA256 = "b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9"

DATASETS = ("weather", "adult", "credit_g", "churn", "online_shoppers", "bank_marketing")


# =============================================================================
# Dataset registry
# =============================================================================


@dataclass
class Dataset:
    """A uniform container: raw features (numeric + categorical) + target."""

    name: str
    X: pd.DataFrame
    y: pd.Series
    domain: str
    source: str
    leakage_note: str
    pos_label: str = ""  # the minority / "event" class we care about
    num_cols: list[str] = field(default_factory=list)
    cat_cols: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Auto-detect column types with select_dtypes. bool -> categorical.
        self.num_cols = self.X.select_dtypes(include="number").columns.tolist()
        self.cat_cols = [c for c in self.X.columns if c not in self.num_cols]
        self.y = self.y.astype(str)
        if not self.pos_label:
            self.pos_label = str(self.y.value_counts().idxmin())  # minority class


def _subsample(x: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Stratified cap so big sets stay within the time budget (deterministic)."""
    if len(x) <= SAMPLE_CAP:
        return x, y
    x_s, _, y_s, _ = train_test_split(
        x, y, train_size=SAMPLE_CAP, random_state=RANDOM_STATE, stratify=y
    )
    return x_s.reset_index(drop=True), y_s.reset_index(drop=True)


def _openml_frame(data_id: int) -> tuple[pd.DataFrame, pd.Series]:
    """Fetch a dataset from OpenML as a frame; scikit-learn caches the download itself."""
    raw = fetch_openml(data_id=data_id, as_frame=True)
    return raw.data.copy(), raw.target


def _synthetic_weather(n: int = 800) -> tuple[pd.DataFrame, pd.Series]:
    """weatherAUS-style stand-in: same column *types*, deterministic, and framed
    with NO leakage - yesterday's rain is a legal feature; today's rain is the
    target and is never fed back in as a feature."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "temp_max": rng.normal(22, 7, n).round(1),
            "humidity_3pm": rng.uniform(20, 100, n).round(0),
            "pressure": rng.normal(1015, 8, n).round(1),
            "wind_dir": rng.choice(["N", "S", "E", "W"], n),
            "rain_yesterday": rng.choice(["Yes", "No"], n, p=[0.25, 0.75]),
        }
    )
    score = (
        (df["humidity_3pm"] - 60) / 15
        + (df["rain_yesterday"] == "Yes") * 1.2
        - (df["pressure"] - 1015) / 10
    )
    y = pd.Series(
        np.where(score + rng.normal(0, 1, n) > 0.5, "Yes", "No"),
        name="rain_today",
    )
    return df, y


def load_data(name: str) -> Dataset:
    if name == "weather":
        x, y = _synthetic_weather()
        return Dataset(
            "weather",
            x,
            y,
            "meteorology",
            "synthetic weatherAUS-style generator, written for this repository "
            "(fixed seed, no external data)",
            pos_label="Yes",
            leakage_note="target = today's rain; only prior/observational "
            "features used. 'rain_yesterday' is legal; today's rain never "
            "appears as a feature.",
        )

    if name == "adult":
        x, y = _openml_frame(1590)
        x = x.drop(columns=[c for c in ["fnlwgt"] if c in x.columns])  # sampling weight
        x, y = _subsample(x, y)
        return Dataset(
            "adult",
            x,
            y,
            "census/income",
            "OpenML 1590 (UCI Adult, Kohavi 1996)",
            pos_label=">50K",
            leakage_note="income is the outcome; all features are demographic "
            "attributes known independently of it.",
        )

    if name == "credit_g":
        x, y = _openml_frame(31)
        return Dataset(
            "credit_g",
            x,
            y,
            "credit risk",
            "OpenML 31 (Statlog German Credit)",
            pos_label="bad",
            leakage_note="target = credit good/bad; features are application "
            "attributes, none derived from the repayment outcome.",
        )

    if name == "churn":
        x, y = _openml_frame(40701)
        x = x.drop(columns=[c for c in ["phone_number"] if c in x.columns])  # unique id
        x, y = _subsample(x, y)
        return Dataset(
            "churn",
            x,
            y,
            "telecom",
            "OpenML 40701 (telecom churn)",
            pos_label="1",
            leakage_note="target = churned; usage/plan features precede the "
            "churn event. The unique phone_number id is dropped.",
        )

    if name == "online_shoppers":
        path = fetch("online_shoppers", ONLINE_SHOPPERS_URL, ONLINE_SHOPPERS_SHA256)
        df = pd.read_csv(path)
        y = df["Revenue"].astype(str)
        x = df.drop(columns=["Revenue"])
        x, y = _subsample(x, y)
        return Dataset(
            "online_shoppers",
            x,
            y,
            "e-commerce",
            "UCI Online Shoppers Purchasing Intention (Sakar 2018)",
            pos_label="True",
            leakage_note="target = purchase in session; features are "
            "clickstream/behaviour aggregates from the same session.",
        )

    if name == "bank_marketing":
        x, y = _openml_frame(1461)
        x, y = _subsample(x, y)
        return Dataset(
            "bank_marketing",
            x,
            y,
            "bank marketing",
            "OpenML 1461 (UCI Bank Marketing; columns anonymised V1..V16)",
            pos_label="2",
            leakage_note="target = subscribed term deposit; features are "
            "contact/campaign attributes recorded before the outcome.",
        )

    raise ValueError(f"unknown dataset '{name}' (expected {', '.join(DATASETS)})")


# =============================================================================
# Pipeline
# =============================================================================


def build_pipeline(ds: Dataset, clf: Any) -> Pipeline:
    """One ColumnTransformer (scale num + one-hot cat) + a classifier.
    Every fit of this object fits its OWN preprocessing - the leakage guarantee."""
    preproc = ColumnTransformer(
        [
            ("num", StandardScaler(), ds.num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ds.cat_cols),
        ]
    )
    return Pipeline([("preproc", preproc), ("clf", clf)])


def tune_model(
    pipe: Pipeline, grid: dict[str, list[Any]], x_train: pd.DataFrame, y_train: pd.Series
) -> GridSearchCV:
    """GridSearchCV over StratifiedKFold - all preprocessing fit inside CV."""
    search = GridSearchCV(pipe, grid, cv=CV, scoring="accuracy", n_jobs=-1)
    search.fit(x_train, y_train)
    return search


def evaluate(
    search: GridSearchCV, ds: Dataset, x_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, Any]:
    y_pred = search.predict(x_test)
    proba = search.predict_proba(x_test)
    pos = ds.pos_label
    pos_idx = list(search.classes_).index(pos)
    y_true_bin = (y_test == pos).astype(int)
    return {
        "best_params": search.best_params_,
        "cv_score": float(search.best_score_),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_pos": precision_score(y_test, y_pred, pos_label=pos, zero_division=0),
        "recall_pos": recall_score(y_test, y_pred, pos_label=pos, zero_division=0),
        "f1_pos": f1_score(y_test, y_pred, pos_label=pos, zero_division=0),
        "roc_auc": float(roc_auc_score(y_true_bin, proba[:, pos_idx])),
        "report": classification_report(y_test, y_pred, zero_division=0),
        "y_pred": y_pred,
    }


# =============================================================================
# Outputs
# =============================================================================


def plot_confusion(
    ds: Dataset, y_test: pd.Series, y_pred: np.ndarray, title: str, path: Path
) -> None:
    labels = sorted(ds.y.unique())
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, labels=labels, cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _feature_names(search: GridSearchCV) -> np.ndarray:
    return search.best_estimator_.named_steps["preproc"].get_feature_names_out()


def plot_rf_importances(search: GridSearchCV, path: Path, top: int = 15) -> None:
    names = _feature_names(search)
    imp = search.best_estimator_.named_steps["clf"].feature_importances_
    order = np.argsort(imp)[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(order) + 1)))
    ax.barh([names[i] for i in order], imp[order], color="#55A868")
    ax.set_xlabel("feature importance")
    ax.set_title("Random Forest - top feature importances")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_lr_coefficients(search: GridSearchCV, path: Path, top: int = 15) -> None:
    names = _feature_names(search)
    coef = search.best_estimator_.named_steps["clf"].coef_[0]
    order = np.argsort(np.abs(coef))[::-1][:top][::-1]
    colors = ["#C44E52" if coef[i] < 0 else "#4C72B0" for i in order]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(order) + 1)))
    ax.barh([names[i] for i in order], coef[order], color=colors)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("logistic-regression coefficient (log-odds)")
    ax.set_title("Logistic Regression - top |coefficients|")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_metrics(
    ds: Dataset, rf: dict[str, Any], lr: dict[str, Any], winner: str, roundtrip_ok: bool, path: Path
) -> None:
    def block(tag: str, m: dict[str, Any]) -> list[str]:
        return [
            f"[{tag}]",
            f"  best_params : {m['best_params']}",
            f"  CV accuracy : {m['cv_score']:.4f}",
            f"  test accuracy: {m['accuracy']:.4f}",
            f"  minority '{ds.pos_label}': precision={m['precision_pos']:.4f} "
            f"recall={m['recall_pos']:.4f} f1={m['f1_pos']:.4f} roc_auc={m['roc_auc']:.4f}",
            "  classification_report:",
            m["report"],
        ]

    lines = [
        "ML PIPELINE - RANDOM FOREST vs LOGISTIC REGRESSION",
        "=" * 62,
        f"Dataset      : {ds.name}  ({ds.domain})",
        f"Source       : {ds.source}",
        f"Rows         : {len(ds.X)}   Features: {len(ds.num_cols)} num + {len(ds.cat_cols)} cat",
        f"Target       : minority/event class = '{ds.pos_label}'  "
        f"(dist {ds.y.value_counts(normalize=True).round(3).to_dict()})",
        f"Anti-leakage : {ds.leakage_note}",
        f"Runtime      : scikit-learn {sklearn.__version__}, StratifiedKFold(5), scoring=accuracy",
        "",
        *block("Random Forest", rf),
        *block("Logistic Regression", lr),
        "HEAD-TO-HEAD:",
        f"  {'metric':<22}{'RandomForest':>14}{'LogisticReg':>14}",
        f"  {'test accuracy':<22}{rf['accuracy']:>14.4f}{lr['accuracy']:>14.4f}",
        f"  {'minority precision':<22}{rf['precision_pos']:>14.4f}{lr['precision_pos']:>14.4f}",
        f"  {'minority recall':<22}{rf['recall_pos']:>14.4f}{lr['recall_pos']:>14.4f}",
        f"  {'minority F1':<22}{rf['f1_pos']:>14.4f}{lr['f1_pos']:>14.4f}",
        f"  {'roc_auc':<22}{rf['roc_auc']:>14.4f}{lr['roc_auc']:>14.4f}",
        "",
        f"WINNER (by test accuracy): {winner}",
        f"Persisted model round-trip (reload -> identical predictions): "
        f"{'PASS' if roundtrip_ok else 'FAIL'}",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_model_card(
    ds: Dataset, rf: dict[str, Any], lr: dict[str, Any], winner: str, path: Path
) -> None:
    """A short, human-readable model card: what the data is, what was tried,
    who won, how it scored, and what the model should not be trusted with."""
    winner_metrics = rf if winner == "Random Forest" else lr
    dist = ds.y.value_counts(normalize=True).round(3).to_dict()
    lines = [
        f"# Model card - {ds.name}",
        "",
        "## Data",
        f"- Dataset: {ds.name} ({ds.domain})",
        f"- Source: {ds.source}",
        f"- Rows: {len(ds.X)}; features: {len(ds.num_cols)} numeric + "
        f"{len(ds.cat_cols)} categorical",
        f"- Positive/minority class: '{ds.pos_label}' (class distribution {dist})",
        f"- Anti-leakage: {ds.leakage_note}",
        "",
        "## Candidates and grid",
        "- Random Forest: ColumnTransformer(StandardScaler + OneHotEncoder) + "
        f"RandomForestClassifier, grid={RF_GRID}",
        "- Logistic Regression: the same preprocessing, classifier swapped via "
        f"`set_params(clf=...)`, grid={LR_GRID}",
        "- Both tuned with GridSearchCV(cv=StratifiedKFold(5, shuffle=True, "
        "random_state=42), scoring='accuracy')",
        "",
        "## Winner",
        f"{winner} - chosen by test-set accuracy "
        f"(Random Forest={rf['accuracy']:.4f}, Logistic Regression={lr['accuracy']:.4f})",
        "",
        "## Test-set scores",
        "| metric | Random Forest | Logistic Regression |",
        "|---|---:|---:|",
        f"| accuracy | {rf['accuracy']:.4f} | {lr['accuracy']:.4f} |",
        f"| precision (minority) | {rf['precision_pos']:.4f} | {lr['precision_pos']:.4f} |",
        f"| recall (minority) | {rf['recall_pos']:.4f} | {lr['recall_pos']:.4f} |",
        f"| f1 (minority) | {rf['f1_pos']:.4f} | {lr['f1_pos']:.4f} |",
        f"| roc_auc | {rf['roc_auc']:.4f} | {lr['roc_auc']:.4f} |",
        "",
        f"Winner f1 (minority class): {winner_metrics['f1_pos']:.3f}",
        f"Winner roc_auc: {winner_metrics['roc_auc']:.3f}",
        "",
        "## Limits",
        "- Trained and evaluated on one held-out split (test_size=0.2, "
        "random_state=42); no repeated-CV confidence interval is reported.",
        "- The grid searched is small (2-3 values per hyperparameter); a wider "
        "search could change the winner on some datasets.",
        "- `roc_auc`/precision/recall are reported for the minority class only "
        "- on heavily imbalanced sets, accuracy alone would be misleading.",
        "- The persisted model is tied to this scikit-learn version "
        f"({sklearn.__version__}); do not load it with a different one.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


# =============================================================================
# Orchestration
# =============================================================================


def run_one(name: str, suffix: str = "") -> dict[str, Any]:
    """Run the full pipeline on one dataset.

    Writes ``metrics<tag>.txt``, ``model_card<tag>.md`` and four plots under
    ``OUT_DIR``, where ``tag`` is ``suffix`` if given, else ``_<name>`` - so
    every dataset gets its own set of output files by default. Returns a
    compact result dict: ``{"dataset", "winner", "f1", "roc_auc"}``.
    """
    ds = load_data(name)
    x_train, x_test, y_train, y_test = train_test_split(
        ds.X, ds.y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=ds.y
    )

    rf_search = tune_model(
        build_pipeline(ds, RandomForestClassifier(random_state=RANDOM_STATE)),
        RF_GRID,
        x_train,
        y_train,
    )
    rf = evaluate(rf_search, ds, x_test, y_test)

    lr_pipe = build_pipeline(ds, RandomForestClassifier())  # same preprocessing...
    lr_pipe.set_params(clf=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    lr_search = tune_model(lr_pipe, LR_GRID, x_train, y_train)  # ...only clf swapped
    lr = evaluate(lr_search, ds, x_test, y_test)

    winner_name = "Random Forest" if rf["accuracy"] >= lr["accuracy"] else "Logistic Regression"
    winner_search = rf_search if winner_name == "Random Forest" else lr_search
    winner_metrics = rf if winner_name == "Random Forest" else lr

    tag = suffix or f"_{name}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUT_DIR / f"model_best{tag}.joblib"
    joblib.dump(winner_search.best_estimator_, model_path)  # persist
    # Safe here: this reloads the file this same run just wrote, not an
    # untrusted artifact - joblib/pickle deserialization from an external
    # source would be an arbitrary-code-execution risk.
    reloaded = joblib.load(model_path)
    roundtrip_ok = bool(
        np.array_equal(reloaded.predict(x_test), winner_search.predict(x_test))
    )  # round-trip

    plot_confusion(
        ds, y_test, rf["y_pred"], f"{ds.name}: Random Forest", OUT_DIR / f"01_confusion_rf{tag}.png"
    )
    plot_rf_importances(rf_search, OUT_DIR / f"02_rf_importances{tag}.png")
    plot_confusion(
        ds,
        y_test,
        lr["y_pred"],
        f"{ds.name}: Logistic Regression",
        OUT_DIR / f"03_confusion_lr{tag}.png",
    )
    plot_lr_coefficients(lr_search, OUT_DIR / f"04_lr_coefficients{tag}.png")
    metrics_path = OUT_DIR / f"metrics{tag}.txt"
    model_card_path = OUT_DIR / f"model_card{tag}.md"
    write_metrics(ds, rf, lr, winner_name, roundtrip_ok, metrics_path)
    write_model_card(ds, rf, lr, winner_name, model_card_path)

    log.info(
        "[%s] n=%d (%dnum+%dcat)  RF acc=%.4f rec=%.3f | LR acc=%.4f rec=%.3f | "
        "winner=%s roundtrip=%s",
        ds.name,
        len(ds.X),
        len(ds.num_cols),
        len(ds.cat_cols),
        rf["accuracy"],
        rf["recall_pos"],
        lr["accuracy"],
        lr["recall_pos"],
        winner_name,
        "OK" if roundtrip_ok else "FAIL",
    )
    log.info("    wrote %s, %s", metrics_path.name, model_card_path.name)
    return {
        "dataset": ds.name,
        "winner": winner_name,
        "f1": winner_metrics["f1_pos"],
        "roc_auc": winner_metrics["roc_auc"],
    }


def write_summary(results: list[dict[str, Any]], path: Path) -> None:
    """Deterministic cross-dataset comparison table - no timestamps, no absolute paths."""
    header = f"{'Dataset':<18}{'Winner':<22}{'F1':>8}{'ROC AUC':>10}"
    lines = ["ML PIPELINE - CROSS-DATASET SUMMARY (RF vs LR)", "=" * 60, header, "-" * 60]
    for r in results:
        lines.append(f"{r['dataset']:<18}{r['winner']:<22}{r['f1']:>8.4f}{r['roc_auc']:>10.4f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _write_demo_metrics(result: dict[str, Any], path: Path) -> None:
    """Deterministic figures-only proof file: no timestamps, no absolute paths."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"winner={result['winner']}",
        f"f1={result['f1']:.3f}",
        f"roc_auc={result['roc_auc']:.3f}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def demo() -> DemoResult:
    """Run the pipeline on weather (offline, deterministic) and write
    output/metrics.txt (figures) and output/summary.txt (the one-row transcript)."""
    start = time.perf_counter()
    result = run_one("weather")
    _write_demo_metrics(result, OUT_DIR / "metrics.txt")
    write_summary([result], OUT_DIR / "summary.txt")
    figures = {
        "winner": result["winner"],
        "f1": f"{result['f1']:.3f}",
        "roc_auc": f"{result['roc_auc']:.3f}",
    }
    return DemoResult(
        "p06-ml-pipeline",
        "ok",
        figures,
        seconds=round(time.perf_counter() - start, 2),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument(
        "--dataset",
        default="weather",
        choices=(*DATASETS, "all"),
        help="dataset to run (default: weather); 'all' runs every one.",
    )
    p.add_argument("--demo", action="store_true", help="Run the offline demo (weather).")
    p.add_argument("--verbose", action="store_true", help="Log pipeline steps at INFO level.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s"
    )

    if args.demo:
        result = demo()
        print(f"p06-ml-pipeline: {result.status}")
        for key, value in result.figures.items():
            print(f"  {key}: {value}")
        print("  wrote: output/")
        print(f"  seconds: {result.seconds:.2f}")
        return 0 if result.status == "ok" else 1

    if args.dataset == "all":
        results = [run_one(name, suffix=f"_{name}") for name in DATASETS]
        write_summary(results, OUT_DIR / "summary_all.txt")
        print(f"p06-ml-pipeline: all ({len(results)} datasets)")
        print("  wrote: output/summary_all.txt")
        return 0

    dataset_result = run_one(args.dataset)
    print(f"p06-ml-pipeline: {dataset_result['dataset']}")
    print(
        f"  winner={dataset_result['winner']}  f1={dataset_result['f1']:.4f}  "
        f"roc_auc={dataset_result['roc_auc']:.4f}"
    )
    print(f"  wrote: output/metrics_{dataset_result['dataset']}.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
