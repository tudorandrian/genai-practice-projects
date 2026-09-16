"""segmentation.py — unsupervised K-Means + PCA customer segmentation.

The task is *unsupervised*: no labels are fed to the model. The discipline is
what matters — (1) StandardScaler before any distance-based clustering,
(2) k chosen on objective criteria (the elbow of inertia + the silhouette
score), not guessed, (3) a business profile per segment via groupby, and
(4) a PCA 2D projection with its explained variance reported so the reader
knows how much information the picture keeps.

The same pipeline runs across six datasets — one synthetic (with a known
ground truth to validate against) and five public real datasets spanning
retail, wholesale, transport, astronomy, wine chemistry and handwriting — so
the workflow is shown to be dataset-agnostic:

    customers    (synthetic, 3 known segments)  retail       — PRIMARY, silhouette peaks at k=3
    wholesale    (UCI Wholesale customers)       distribution — 440 clients, 6 spend features
    taxis        (seaborn taxis)                 transport    — 6433 NYC trips
    planets      (seaborn planets)                astronomy    — 498 exoplanets
    winequality  (UCI wine-quality-red)           chemistry    — 1599 wines, 11 features
    digits       (sklearn load_digits)            imaging      — 1797 8x8 digits, 64 features

PIPELINE  (pure functions)
    load_data -> preprocess (scale) -> choose_k (elbow+silhouette)
              -> fit_final -> segment_profile / validate_against_truth -> plot_* / pca_2d

HOW TO RUN
    uv run p05-segmentation                        # default: customers (synthetic)
    uv run p05-segmentation --dataset wholesale     # a real segmentation set
    uv run p05-segmentation --dataset all           # every dataset + a summary
    uv run p05-segmentation --demo                  # customers only, writes output/metrics.txt

DEPENDENCIES  numpy, pandas, scikit-learn, matplotlib, requests
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import matplotlib

matplotlib.use("Agg")  # headless backend — never plt.show()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.datasets import load_digits  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import adjusted_rand_score, silhouette_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from shared.datasets import fetch  # noqa: E402
from shared.demo import DemoResult  # noqa: E402

log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output"
RANDOM_STATE = 42  # KMeans / silhouette sampling
DATA_SEED = 7  # seeds the synthetic customer generator (np.random.default_rng)
SILHOUETTE_SAMPLE = 2000  # cap silhouette to keep it O(sample^2) on big sets

_SEABORN = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/"

# Datasets fetched through shared.datasets.fetch (checksum-verified download,
# cached once under GENAI_DATA_DIR). `customers` is a deterministic synthetic
# generator and `digits` ships with scikit-learn — both stay fully offline.
DATASET_SOURCES: dict[str, tuple[str, str]] = {
    "wholesale": (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "00292/Wholesale%20customers%20data.csv",
        "c3d018c643565b85cee733c4a2ac76dd76e080e857cb23f0ccfcc2e15a6c17ef",
    ),
    "taxis": (
        _SEABORN + "taxis.csv",
        "08d6d71784dbaa2651fee37fc03389754194c05d72d2d19cbc2c799dea6ac09d",
    ),
    "planets": (
        _SEABORN + "planets.csv",
        "a6d10044887e17396974525a366f5fa2e4b34df70f491e64eb9943de0e3d3825",
    ),
    "winequality": (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "wine-quality/winequality-red.csv",
        "4a402cf041b025d4566d954c3b9ba8635a3a8a01e039005d97d6a710278cf05e",
    ),
}

DATASETS = ("customers", "wholesale", "taxis", "planets", "winequality", "digits")


# =============================================================================
# Dataset registry
# =============================================================================


@dataclass
class Dataset:
    """Numeric feature frame for clustering + metadata. ``y_true`` is used ONLY
    for post-hoc validation (never fed to the model) — it exists for the
    synthetic set (known segments) and where a natural reference label exists.
    """

    name: str
    X: pd.DataFrame  # unscaled numeric features (also used for profiling)
    domain: str
    source: str
    k_max: int = 6  # search k in 2..k_max
    y_true: np.ndarray | None = None
    true_k: int | None = None
    feature_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.feature_names = list(self.X.columns)


class _SegmentRecipe(NamedTuple):
    income: tuple[float, float]  # (mean, std) of annual_income_k
    spend: tuple[float, float]  # (mean, std) of spending_score
    visits: tuple[float, float]  # (mean, std) of visits_per_month
    n: int


_SEGMENT_RECIPES: tuple[_SegmentRecipe, ...] = (
    _SegmentRecipe(income=(25, 5), spend=(70, 8), visits=(18, 3), n=120),  # young & active
    _SegmentRecipe(income=(85, 10), spend=(20, 6), visits=(3, 1.5), n=100),  # premium occasional
    _SegmentRecipe(income=(60, 8), spend=(55, 7), visits=(9, 2), n=140),  # balanced middle
)


def _synthetic_customers() -> Dataset:
    """Three real customer segments the model must rediscover without labels.

    Deterministic via ``default_rng(DATA_SEED)``; the silhouette score peaks
    at k=3. Returns the labelled ground truth as ``y_true`` for validation
    only — it is never passed to K-Means.
    """
    rng = np.random.default_rng(DATA_SEED)
    frames = []
    truth = []
    for i, s in enumerate(_SEGMENT_RECIPES):
        frames.append(
            pd.DataFrame(
                {
                    "annual_income_k": rng.normal(*s.income, s.n),
                    "spending_score": rng.normal(*s.spend, s.n),
                    "visits_per_month": rng.normal(*s.visits, s.n),
                }
            )
        )
        truth.append(np.full(s.n, i))
    x = pd.concat(frames, ignore_index=True)
    return Dataset(
        "customers",
        x,
        "retail",
        "synthetic 3-segment generator, written for this repository (fixed seed, no external data)",
        k_max=6,
        y_true=np.concatenate(truth),
        true_k=3,
    )


def load_data(name: str) -> Dataset:
    """Load one clustering dataset by name into a uniform Dataset container."""
    if name == "customers":
        return _synthetic_customers()

    if name == "wholesale":
        url, sha256 = DATASET_SOURCES["wholesale"]
        path = fetch("wholesale", url, sha256)
        df = pd.read_csv(path)
        feats = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
        return Dataset(
            "wholesale",
            df[feats].copy(),
            "wholesale distribution",
            "UCI Wholesale customers (Cardoso 2014)",
            k_max=6,
            y_true=df["Channel"].to_numpy(),  # Horeca vs Retail channel
        )

    if name == "taxis":
        url, sha256 = DATASET_SOURCES["taxis"]
        path = fetch("taxis", url, sha256)
        df = pd.read_csv(path)
        feats = ["passengers", "distance", "fare", "tip", "tolls", "total"]
        df = df.dropna(subset=feats).reset_index(drop=True)
        return Dataset(
            "taxis",
            df[feats].copy(),
            "urban transport",
            "seaborn taxis (NYC TLC 2019)",
            k_max=6,
        )

    if name == "planets":
        url, sha256 = DATASET_SOURCES["planets"]
        path = fetch("planets", url, sha256)
        df = pd.read_csv(path)
        feats = ["orbital_period", "mass", "distance", "year"]
        df = df.dropna(subset=feats).reset_index(drop=True)
        return Dataset(
            "planets",
            df[feats].copy(),
            "astronomy",
            "seaborn planets (NASA exoplanet archive)",
            k_max=6,
        )

    if name == "winequality":
        url, sha256 = DATASET_SOURCES["winequality"]
        path = fetch("winequality", url, sha256)
        df = pd.read_csv(path, sep=";")
        feats = [c for c in df.columns if c != "quality"]  # quality = held-out label
        return Dataset(
            "winequality",
            df[feats].copy(),
            "wine chemistry",
            "UCI wine-quality-red (Cortez et al. 2009)",
            k_max=8,
            y_true=df["quality"].to_numpy(),
        )

    if name == "digits":
        raw = load_digits()
        x = pd.DataFrame(raw.data, columns=[f"px_{i:02d}" for i in range(raw.data.shape[1])])
        return Dataset(
            "digits",
            x,
            "imaging / handwriting",
            "sklearn.datasets.load_digits (UCI, 8x8)",
            k_max=12,
            y_true=raw.target,
            true_k=10,
        )

    raise ValueError(f"unknown dataset '{name}' (expected {', '.join(DATASETS)})")


# =============================================================================
# Pipeline steps (pure)
# =============================================================================


def preprocess(ds: Dataset) -> np.ndarray:
    """Scale every feature to zero mean / unit variance.

    K-Means uses Euclidean distance, so unscaled features would let the
    largest-range column dominate the clustering. Asserts the result is
    actually standardized.
    """
    x_scaled = StandardScaler().fit_transform(ds.X.to_numpy(dtype=float))
    assert np.abs(x_scaled.mean(axis=0)).max() < 1e-9, "not centred"
    # Non-constant features must be unit-variance; constant columns (e.g. always-0
    # corner pixels in digits) are left at 0 by StandardScaler — tolerate those.
    non_constant = ds.X.std(axis=0, ddof=0).to_numpy() > 0
    scaled_std = x_scaled.std(axis=0)
    assert np.abs(scaled_std[non_constant] - 1).max() < 1e-6, "not unit-variance"
    return x_scaled


def _silhouette(x_scaled: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette with a fixed-seed sample cap so it stays O(sample^2) and
    deterministic on the larger datasets."""
    n = len(x_scaled)
    if n > SILHOUETTE_SAMPLE:
        return float(
            silhouette_score(
                x_scaled, labels, sample_size=SILHOUETTE_SAMPLE, random_state=RANDOM_STATE
            )
        )
    return float(silhouette_score(x_scaled, labels))


def choose_k(x_scaled: np.ndarray, k_max: int) -> tuple[list[dict[str, Any]], int]:
    """Run K-Means for k=2..k_max, collect inertia + silhouette.
    Returns (rows, best_k) where best_k maximises the silhouette score."""
    rows: list[dict[str, Any]] = []
    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(x_scaled)
        rows.append(
            {
                "k": k,
                "inertia": float(km.inertia_),
                "silhouette": _silhouette(x_scaled, km.labels_),
            }
        )
    best_k = max(rows, key=lambda r: r["silhouette"])["k"]
    return rows, best_k


def fit_final(x_scaled: np.ndarray, k: int) -> tuple[KMeans, np.ndarray]:
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(x_scaled)
    return km, km.labels_


def segment_profile(ds: Dataset, labels: np.ndarray) -> pd.DataFrame:
    """Mean feature values per segment plus an 'n' count column."""
    df = ds.X.copy()
    df["segment"] = labels
    profile = df.groupby("segment").mean()
    profile.insert(0, "n", df.groupby("segment").size())
    return profile


def validate_against_truth(ds: Dataset, labels: np.ndarray) -> float | None:
    """Adjusted Rand Index vs the held-out reference labels (validation only).
    None when the dataset has no natural ground truth."""
    if ds.y_true is None:
        return None
    return float(adjusted_rand_score(ds.y_true, labels))


def pca_2d(x_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project to 2D and return (points, explained_variance_ratio_[:2])."""
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pts = pca.fit_transform(x_scaled)
    return pts, pca.explained_variance_ratio_


# =============================================================================
# Outputs
# =============================================================================


def plot_elbow_silhouette(ds: Dataset, rows: list[dict[str, Any]], best_k: int, path: Path) -> None:
    ks = [r["k"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(ks, [r["inertia"] for r in rows], "o-", color="#4C72B0")
    ax1.axvline(best_k, color="grey", ls="--", lw=1)
    ax1.set_xlabel("k (clusters)")
    ax1.set_ylabel("inertia (within-cluster SS)")
    ax1.set_title("Elbow — inertia vs k")
    ax2.plot(ks, [r["silhouette"] for r in rows], "o-", color="#55A868")
    ax2.axvline(best_k, color="grey", ls="--", lw=1, label=f"chosen k={best_k}")
    ax2.set_xlabel("k (clusters)")
    ax2.set_ylabel("silhouette score")
    ax2.set_title("Silhouette vs k (higher = better)")
    ax2.legend(loc="best")
    fig.suptitle(f"{ds.name}: choosing k")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_pca_segments(
    ds: Dataset, pts: np.ndarray, labels: np.ndarray, evr: np.ndarray, path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(
        pts[:, 0], pts[:, 1], c=labels, cmap="viridis", alpha=0.6, s=18, edgecolor="none"
    )
    ax.set_xlabel(f"PC1 ({evr[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({evr[1] * 100:.1f}% var)")
    ax.set_title(f"{ds.name}: segments in PCA space (PC1+PC2 = {evr[:2].sum() * 100:.1f}% var)")
    legend = ax.legend(*sc.legend_elements(), title="segment", loc="best")
    ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_metrics(
    ds: Dataset,
    rows: list[dict[str, Any]],
    best_k: int,
    profile: pd.DataFrame,
    evr: np.ndarray,
    ari: float | None,
    path: Path,
) -> None:
    lines = [
        "K-MEANS + PCA SEGMENTATION — METRICS",
        "=" * 60,
        f"Dataset      : {ds.name}  ({ds.domain})",
        f"Source       : {ds.source}",
        f"Observations : {len(ds.X)}   Features: {len(ds.feature_names)} (numeric, scaled)",
        f"Features     : {ds.feature_names}",
        "",
        "Choosing k — inertia (elbow) + silhouette:",
        f"  {'k':>3}{'inertia':>14}{'silhouette':>13}",
    ]
    for r in rows:
        mark = "  <- chosen" if r["k"] == best_k else ""
        lines.append(f"  {r['k']:>3}{r['inertia']:>14.1f}{r['silhouette']:>13.4f}{mark}")
    lines += [
        "",
        f"Chosen k     : {best_k}  (max silhouette)",
    ]
    if ds.true_k is not None:
        lines.append(f"Reference k  : {ds.true_k}  (known ground truth)")
    if ari is not None:
        lines.append(
            f"Adjusted Rand vs reference labels : {ari:.4f}  "
            "(1.0 = perfect recovery; validation only, not used to train)"
        )
    lines += [
        "",
        f"PCA explained variance: PC1={evr[0]:.4f}  PC2={evr[1]:.4f}  sum={evr[:2].sum():.4f}",
        "",
        "Segment profile — groupby('segment').mean():",
        profile.round(2).to_string(),
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =============================================================================
# Orchestration
# =============================================================================


def run_one(name: str, suffix: str = "") -> dict[str, Any]:
    """Run the full pipeline on one dataset.

    Writes ``elbow_silhouette<tag>.png``, ``pca_segments<tag>.png`` and
    ``metrics<tag>.txt`` under ``OUT_DIR``, where ``tag`` is ``suffix`` if
    given, else ``_<name>`` — so every dataset gets its own set of output
    files by default. Returns a compact result dict.
    """
    ds = load_data(name)
    x_scaled = preprocess(ds)
    rows, best_k = choose_k(x_scaled, ds.k_max)
    _km, labels = fit_final(x_scaled, best_k)
    profile = segment_profile(ds, labels)
    ari = validate_against_truth(ds, labels)
    pts, evr = pca_2d(x_scaled)

    tag = suffix or f"_{name}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    elbow_path = OUT_DIR / f"elbow_silhouette{tag}.png"
    pca_path = OUT_DIR / f"pca_segments{tag}.png"
    metrics_path = OUT_DIR / f"metrics{tag}.txt"
    plot_elbow_silhouette(ds, rows, best_k, elbow_path)
    plot_pca_segments(ds, pts, labels, evr, pca_path)
    write_metrics(ds, rows, best_k, profile, evr, ari, metrics_path)

    best_sil = next(r["silhouette"] for r in rows if r["k"] == best_k)
    log.info(
        "[%s] n=%d feats=%d  best_k=%d  silhouette=%.4f  PC1+PC2=%.1f%%",
        ds.name,
        len(ds.X),
        len(ds.feature_names),
        best_k,
        best_sil,
        evr[:2].sum() * 100,
    )
    log.info("    wrote %s, %s, %s", metrics_path.name, elbow_path.name, pca_path.name)
    return {
        "name": ds.name,
        "n": len(ds.X),
        "feats": len(ds.feature_names),
        "best_k": best_k,
        "silhouette": best_sil,
        "pca2": float(evr[:2].sum()),
        "ari": ari,
    }


def write_summary(results: list[dict[str, Any]], path: Path) -> None:
    """Deterministic cross-dataset comparison table — no timestamps, no absolute paths."""
    header = (
        f"{'Dataset':<13}{'n':>7}{'feat':>6}{'best_k':>8}"
        f"{'silhouette':>12}{'PC1+PC2':>10}{'ARI':>8}"
    )
    lines = ["K-MEANS + PCA — CROSS-DATASET SUMMARY", "=" * 64, header, "-" * 64]
    for r in results:
        ari = f"{r['ari']:.3f}" if r["ari"] is not None else "n/a"
        lines.append(
            f"{r['name']:<13}{r['n']:>7}{r['feats']:>6}{r['best_k']:>8}"
            f"{r['silhouette']:>12.4f}{r['pca2'] * 100:>9.1f}%{ari:>8}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_demo_metrics(result: dict[str, Any], path: Path) -> None:
    """Deterministic figures-only proof file: no timestamps, no absolute paths."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"k={result['best_k']}", f"silhouette={result['silhouette']:.3f}"]
    if result["ari"] is not None:
        lines.append(f"ground_truth_agreement={result['ari']:.3f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def demo() -> DemoResult:
    """Run the pipeline on customers (offline, deterministic, known ground
    truth) and write output/metrics.txt (figures) and output/summary.txt
    (the one-row transcript)."""
    start = time.perf_counter()
    result = run_one("customers")
    _write_demo_metrics(result, OUT_DIR / "metrics.txt")
    write_summary([result], OUT_DIR / "summary.txt")
    ari = result["ari"]
    figures = {
        "k": str(result["best_k"]),
        "silhouette": f"{result['silhouette']:.3f}",
        "ground_truth_agreement": f"{ari:.3f}" if ari is not None else "n/a",
    }
    return DemoResult(
        "p05-segmentation",
        "ok",
        figures,
        seconds=round(time.perf_counter() - start, 2),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument(
        "--dataset",
        default="customers",
        choices=(*DATASETS, "all"),
        help="dataset to run (default: customers); 'all' runs every dataset.",
    )
    p.add_argument("--demo", action="store_true", help="Run the offline demo (customers).")
    p.add_argument("--verbose", action="store_true", help="Log pipeline steps at INFO level.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s"
    )

    if args.demo:
        result = demo()
        print(f"p05-segmentation: {result.status}")
        for key, value in result.figures.items():
            print(f"  {key}: {value}")
        print("  wrote: output/")
        print(f"  seconds: {result.seconds:.2f}")
        return 0 if result.status == "ok" else 1

    if args.dataset == "all":
        results = [run_one(name, suffix=f"_{name}") for name in DATASETS]
        write_summary(results, OUT_DIR / "summary_all.txt")
        print(f"p05-segmentation: all ({len(results)} datasets)")
        print("  wrote: output/summary_all.txt")
        return 0

    dataset_result = run_one(args.dataset)
    print(f"p05-segmentation: {dataset_result['name']}")
    print(
        f"  n={dataset_result['n']} feats={dataset_result['feats']}  "
        f"best_k={dataset_result['best_k']}  silhouette={dataset_result['silhouette']:.4f}"
    )
    print(f"  wrote: output/metrics_{dataset_result['name']}.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
