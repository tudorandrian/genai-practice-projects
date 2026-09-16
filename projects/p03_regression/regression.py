"""regression.py — a disciplined linear-regression predictor for tabular data.

The point is not the algorithm (one line: ``LinearRegression().fit(...)``) but
the discipline around it: a correct train/test split, standardization fitted
**only on train** (no leakage), metrics reported honestly on unseen data, and
coefficient interpretation — not just a score.

The same pipeline runs across seven public regression datasets so the
workflow is shown to be dataset-agnostic:

    diabetes    (sklearn, offline)       disease progression   — 442 rows, R2=0.453
    california  (sklearn, downloaded)    median house value    — 20,640 rows
    co2         (Government of Canada)   CO2 emissions g/km    — 1,067 rows
    mpg         (seaborn auto-mpg)       fuel efficiency (mpg) — 392 rows
    tips        (seaborn tips)           tip amount ($)        — 244 rows
    diamonds    (seaborn, downloaded)    price ($)             — 53,940 rows
    penguins    (seaborn palmerpenguins) body mass (g)         — 342 rows

PIPELINE  (pure functions)
    load_data -> explore_data -> preprocess -> train_model -> evaluate -> plot

HOW TO RUN
    python regression.py                      # default dataset: co2
    python regression.py --dataset diabetes    # offline reference (R2=0.453)
    python regression.py --dataset all         # run all seven + comparison table
    python regression.py --demo                # tips + co2, writes output/metrics.txt

DEPENDENCIES  numpy, pandas, scikit-learn, matplotlib, requests
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless backend — never plt.show()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.datasets import fetch_california_housing, load_diabetes  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from shared.datasets import fetch  # noqa: E402
from shared.demo import DemoResult  # noqa: E402

log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
OUT_DIR = HERE / "output"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Datasets fetched through shared.datasets.fetch (checksum-verified download,
# cached once under GENAI_DATA_DIR). Everything else is either bundled with
# scikit-learn or a small CSV committed under data/.
DATASET_SOURCES: dict[str, tuple[str, str]] = {
    "diamonds": (
        "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv",
        "9574730b03aba241d899c4a97511c5061b19358fab89510774fb6c24168345c4",
    ),
}


# =============================================================================
# Dataset registry
# =============================================================================


@dataclass
class Dataset:
    name: str
    X: pd.DataFrame
    y: np.ndarray
    target_name: str
    units: str
    domain: str
    source: str
    feature_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.feature_names = list(self.X.columns)


DATASETS = ("diabetes", "california", "co2", "mpg", "tips", "diamonds", "penguins")


def load_data(name: str) -> Dataset:
    """Load one regression dataset by name into a uniform Dataset container."""
    if name == "diabetes":
        raw = load_diabetes(as_frame=True)
        return Dataset(
            "diabetes",
            raw.data,
            raw.target.to_numpy(),
            "disease_progression",
            "index",
            "health",
            "sklearn.datasets.load_diabetes",
        )

    if name == "california":
        raw = fetch_california_housing(as_frame=True)  # sklearn caches this itself
        return Dataset(
            "california",
            raw.data,
            raw.target.to_numpy(),
            "MedHouseVal",
            "$100k",
            "real-estate",
            "sklearn.datasets.fetch_california_housing",
        )

    if name == "co2":
        df = pd.read_csv(DATA_DIR / "FuelConsumptionCo2.csv")
        feats = [
            "ENGINESIZE",
            "CYLINDERS",
            "FUELCONSUMPTION_CITY",
            "FUELCONSUMPTION_HWY",
            "FUELCONSUMPTION_COMB",
            "FUELCONSUMPTION_COMB_MPG",
        ]
        return Dataset(
            "co2",
            df[feats].copy(),
            df["CO2EMISSIONS"].to_numpy(),
            "CO2EMISSIONS",
            "g/km",
            "automotive/emissions",
            "Fuel consumption ratings — Government of Canada open data "
            "(Open Government Licence – Canada)",
        )

    if name == "mpg":
        df = pd.read_csv(DATA_DIR / "auto_mpg.csv")
        feats = [
            "cylinders",
            "displacement",
            "horsepower",
            "weight",
            "acceleration",
            "model_year",
        ]
        df = df.dropna(subset=feats + ["mpg"]).reset_index(drop=True)
        return Dataset(
            "mpg",
            df[feats].copy(),
            df["mpg"].to_numpy(),
            "mpg",
            "miles/gallon",
            "automotive/efficiency",
            "seaborn auto-mpg (UCI)",
        )

    if name == "tips":
        df = pd.read_csv(DATA_DIR / "tips.csv")
        feats = ["total_bill", "size"]
        return Dataset(
            "tips",
            df[feats].copy(),
            df["tip"].to_numpy(),
            "tip",
            "$",
            "dining/hospitality",
            "seaborn tips",
        )

    if name == "diamonds":
        url, sha256 = DATASET_SOURCES["diamonds"]
        path = fetch("diamonds", url, sha256)
        df = pd.read_csv(path)
        feats = ["carat", "depth", "table", "x", "y", "z"]
        return Dataset(
            "diamonds",
            df[feats].copy(),
            df["price"].to_numpy(),
            "price",
            "$",
            "retail/jewelry",
            "seaborn diamonds",
        )

    if name == "penguins":
        df = pd.read_csv(DATA_DIR / "penguins.csv")
        feats = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm"]
        df = df.dropna(subset=feats + ["body_mass_g"]).reset_index(drop=True)
        return Dataset(
            "penguins",
            df[feats].copy(),
            df["body_mass_g"].to_numpy(),
            "body_mass_g",
            "g",
            "biology",
            "seaborn palmerpenguins",
        )

    raise ValueError(f"unknown dataset '{name}' (expected {', '.join(DATASETS)})")


# =============================================================================
# Pipeline steps (pure)
# =============================================================================


def explore_data(ds: Dataset) -> pd.Series:
    """Return each feature's Pearson correlation with the target (|desc| sorted)."""
    corr = ds.X.apply(lambda col: np.corrcoef(col, ds.y)[0, 1])
    return corr.reindex(corr.abs().sort_values(ascending=False).index)


def preprocess(
    features: pd.DataFrame, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Split first, THEN fit the scaler on the train split only — the anti-leakage order.

    Returns (x_train_s, x_test_s, y_train, y_test, scaler). Asserts that the
    scaled train columns have ~zero mean, guarding against a fit that leaked
    into the test set.
    """
    x_train, x_test, y_train, y_test = train_test_split(
        features.to_numpy(), y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    scaler = StandardScaler().fit(x_train)  # fit ONLY on train
    x_train_s = scaler.transform(x_train)
    x_test_s = scaler.transform(x_test)

    assert np.abs(x_train_s.mean(axis=0)).max() < 1e-9, "train not centred — leakage?"
    return x_train_s, x_test_s, y_train, y_test, scaler


def train_model(x_train_s: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    return LinearRegression().fit(x_train_s, y_train)


def evaluate(model: LinearRegression, x_test_s: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    y_pred = model.predict(x_test_s)
    return {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "R2": r2_score(y_test, y_pred),
        "y_pred": y_pred,
    }


def rank_coefficients(model: LinearRegression, feature_names: list[str]) -> list[tuple[str, float]]:
    """Standardized coefficients sorted by |value| desc — 'which feature matters most'."""
    pairs = list(zip(feature_names, model.coef_, strict=True))
    return sorted(pairs, key=lambda kv: abs(kv[1]), reverse=True)


# =============================================================================
# Outputs
# =============================================================================


def write_metrics(
    ds: Dataset,
    metrics: dict[str, Any],
    model: LinearRegression,
    ranking: list[tuple[str, float]],
    n_train: int,
    n_test: int,
    path: Path,
) -> None:
    lines = [
        "LINEAR REGRESSION — METRICS",
        "=" * 44,
        f"Dataset      : {ds.name}  ({ds.domain})",
        f"Source       : {ds.source}",
        f"Target       : {ds.target_name} [{ds.units}]",
        f"Rows         : {n_train} train / {n_test} test   Features: {len(ds.feature_names)}",
        "",
        "Test-set metrics:",
        f"  MAE  : {metrics['MAE']:.3f}",
        f"  MSE  : {metrics['MSE']:.3f}",
        f"  RMSE : {metrics['RMSE']:.3f}  [{ds.units}]",
        f"  R2   : {metrics['R2']:.3f}",
        "",
        f"Intercept    : {model.intercept_:.3f}",
        "",
        "Standardized coefficients (|value| desc — most influential first):",
    ]
    for feat, coef in ranking:
        lines.append(f"  {feat:<26} {coef:>10.3f}")
    lines.append("")
    lines.append("Note: features standardized (fit on train only); coefficients are")
    lines.append("directly comparable across features in units of target-per-1-SD.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_predicted_vs_actual(
    ds: Dataset, y_test: np.ndarray, y_pred: np.ndarray, r2: float, path: Path
) -> None:
    lo = float(min(y_test.min(), y_pred.min()))
    hi = float(max(y_test.max(), y_pred.max()))
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, y_pred, alpha=0.4, s=18, edgecolor="none")
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=2, label="perfect prediction")
    ax.set_xlabel(f"Actual {ds.target_name} [{ds.units}]")
    ax.set_ylabel(f"Predicted {ds.target_name} [{ds.units}]")
    ax.set_title(f"{ds.name}: predicted vs actual  (R² = {r2:.3f})")
    ax.legend(loc="upper left")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# =============================================================================
# Orchestration
# =============================================================================


def run_one(name: str, suffix: str = "") -> dict[str, str | int | float]:
    """Run the full pipeline on one dataset.

    Writes ``metrics<tag>.txt`` and ``predicted_vs_actual<tag>.png`` under
    ``OUT_DIR``, where ``tag`` is ``suffix`` if given, else ``_<name>`` — so
    every dataset gets its own pair of output files by default. Returns a
    compact result dict.
    """
    ds = load_data(name)
    corr = explore_data(ds)
    x_train_s, x_test_s, y_train, y_test, _ = preprocess(ds.X, ds.y)
    model = train_model(x_train_s, y_train)
    metrics = evaluate(model, x_test_s, y_test)
    ranking = rank_coefficients(model, ds.feature_names)

    tag = suffix or f"_{name}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUT_DIR / f"metrics{tag}.txt"
    plot_path = OUT_DIR / f"predicted_vs_actual{tag}.png"
    write_metrics(ds, metrics, model, ranking, len(y_train), len(y_test), metrics_path)
    plot_predicted_vs_actual(ds, y_test, metrics["y_pred"], metrics["R2"], plot_path)

    log.info(
        "[%s] n=%d p=%d  MAE=%.3f  RMSE=%.3f  R2=%.3f",
        ds.name,
        len(ds.X),
        len(ds.feature_names),
        metrics["MAE"],
        metrics["RMSE"],
        metrics["R2"],
    )
    log.info(
        "    top feature by |coef|: %s (%+.3f) | top corr: %s (%+.3f)",
        ranking[0][0],
        ranking[0][1],
        corr.index[0],
        corr.iloc[0],
    )
    log.info("    wrote %s, %s", metrics_path.name, plot_path.name)
    return {
        "name": ds.name,
        "n": len(ds.X),
        "p": len(ds.feature_names),
        **{k: metrics[k] for k in ("MAE", "MSE", "RMSE", "R2")},
    }


def write_summary(results: list[dict[str, str | int | float]], path: Path) -> None:
    """Deterministic cross-dataset comparison table — no timestamps, no absolute paths."""
    header = f"{'Dataset':<12}{'n':>7}{'p':>4}{'MAE':>12}{'RMSE':>12}{'R2':>8}"
    lines = ["LINEAR REGRESSION — CROSS-DATASET SUMMARY", "=" * 55, header, "-" * 55]
    for r in results:
        lines.append(
            f"{r['name']:<12}{r['n']:>7}{r['p']:>4}"
            f"{r['MAE']:>12.3f}{r['RMSE']:>12.3f}{r['R2']:>8.3f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_demo_metrics(results: list[dict[str, str | int | float]], path: Path) -> None:
    """Deterministic figures-only proof file: no timestamps, no absolute paths."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for r in results:
        lines.append(f"{r['name']}_r2={r['R2']:.3f}")
        lines.append(f"{r['name']}_mae={r['MAE']:.3f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def demo() -> DemoResult:
    """Run the pipeline on tips + co2 (both offline, deterministic) and write
    output/metrics.txt (figures) and output/summary.txt (comparison table)."""
    start = time.perf_counter()
    results = [run_one("tips"), run_one("co2")]
    _write_demo_metrics(results, OUT_DIR / "metrics.txt")
    write_summary(results, OUT_DIR / "summary.txt")
    figures = {f"{r['name']}_r2": f"{r['R2']:.3f}" for r in results}
    return DemoResult(
        "p03-regression",
        "ok",
        figures,
        seconds=round(time.perf_counter() - start, 2),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument(
        "--dataset",
        default="co2",
        choices=(*DATASETS, "all"),
        help="dataset to run (default: co2); 'all' runs every dataset.",
    )
    p.add_argument("--demo", action="store_true", help="Run the offline demo (tips + co2).")
    p.add_argument("--verbose", action="store_true", help="Log pipeline steps at INFO level.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s"
    )

    if args.demo:
        result = demo()
        print(f"p03-regression: {result.status}")
        for key, value in result.figures.items():
            print(f"  {key}: {value}")
        print("  wrote: output/")
        print(f"  seconds: {result.seconds:.2f}")
        return 0 if result.status == "ok" else 1

    if args.dataset == "all":
        results = [run_one(name) for name in DATASETS]
        write_summary(results, OUT_DIR / "summary_all.txt")
        print(f"p03-regression: all ({len(results)} datasets)")
        print("  wrote: output/summary_all.txt")
        return 0

    dataset_result = run_one(args.dataset)
    print(f"p03-regression: {dataset_result['name']}")
    print(
        f"  n={dataset_result['n']} p={dataset_result['p']}  MAE={dataset_result['MAE']:.3f}  "
        f"RMSE={dataset_result['RMSE']:.3f}  R2={dataset_result['R2']:.3f}"
    )
    print(f"  wrote: output/metrics_{dataset_result['name']}.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
