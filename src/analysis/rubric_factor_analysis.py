# src/analysis/rubric_factor_analysis.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA, FactorAnalysis


# -------------------------
# Rubrics (must match CSV columns)
# -------------------------
RUBRICS: List[str] = [
    "accuracy",
    "safety",
    "helpfulness",
    "empathy",
    "completeness",
    "bias_avoidance",
    "limitation_awareness",
    "communication",
]


# -------------------------
# Utils
# -------------------------
def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def detect_cols(df: pd.DataFrame) -> Tuple[str, str, str, Optional[str]]:
    # scenario uid
    scenario_col = None
    for c in ["scenario_uid", "scenario_id", "id"]:
        if c in df.columns:
            scenario_col = c
            break
    if not scenario_col:
        raise ValueError("Cannot find scenario column: expected one of scenario_uid/scenario_id/id")

    # answer model
    model_col = None
    for c in ["answer_model", "model"]:
        if c in df.columns:
            model_col = c
            break
    if not model_col:
        raise ValueError("Cannot find model column: expected answer_model/model")

    # judge model
    judge_col = None
    for c in ["judge_model", "grader_model"]:
        if c in df.columns:
            judge_col = c
            break
    if not judge_col:
        raise ValueError("Cannot find judge column: expected judge_model/grader_model")

    language_col = "language" if "language" in df.columns else None
    return scenario_col, model_col, judge_col, language_col


def zscore(X: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, keepdims=True) + eps
    return (X - mu) / sd


def aggregate_to_samples(
    df: pd.DataFrame,
    scenario_col: str,
    model_col: str,
    judge_col: str,
    language_col: Optional[str],
    rubrics: List[str],
    language: str = "all",
    judge_filter: str = "",
) -> pd.DataFrame:
    """
    Build samples at (scenario, model) level.
    - If multiple judges exist, we average over judge to avoid judge-specific correlation structures.
      (You can also filter to a single judge with --judge.)
    - If language filtering is requested and language column exists, apply it.

    Returns DataFrame with columns: scenario_col, model_col, [rubrics...]
    """
    dfx = df.copy()

    # filter judge
    if judge_filter.strip():
        dfx = dfx[dfx[judge_col].astype(str) == judge_filter.strip()].copy()

    # filter language
    lang = (language or "all").lower().strip()
    if lang != "all":
        if not language_col:
            print("[WARN] --language provided but CSV has no 'language' column; language filter skipped.")
        else:
            dfx = dfx[dfx[language_col].astype(str).str.lower().str.strip() == lang].copy()

    avail = [r for r in rubrics if r in dfx.columns]
    if not avail:
        raise ValueError("No rubric columns found in CSV. Expected: " + ", ".join(rubrics))

    # average over judge (and language if not filtered) to get stable per (scenario, model)
    group_keys = [scenario_col, model_col]
    out = dfx[group_keys + avail].groupby(group_keys, dropna=False)[avail].mean().reset_index()
    return out


def varimax(Phi: np.ndarray, gamma: float = 1.0, q: int = 50, tol: float = 1e-6) -> np.ndarray:
    """
    Varimax rotation for loadings matrix.
    Phi: (n_features, n_factors)
    Returns rotated loadings matrix same shape.
    """
    p, k = Phi.shape
    R = np.eye(k)
    d = 0.0
    for _ in range(q):
        d_old = d
        Lambda = Phi @ R
        # compute gradient-like update
        u, s, vt = np.linalg.svd(
            Phi.T @ (Lambda ** 3 - (gamma / p) * Lambda @ np.diag(np.diag(Lambda.T @ Lambda))),
            full_matrices=False,
        )
        R = u @ vt
        d = float(np.sum(s))
        if d_old != 0 and (d - d_old) < tol:
            break
    return Phi @ R


def plot_scree(explained_ratio: np.ndarray, out_path: Path, title: str) -> None:
    xs = np.arange(1, len(explained_ratio) + 1)
    plt.figure(figsize=(7, 4.5))
    ax = plt.gca()
    ax.plot(xs, explained_ratio, marker="o")
    ax.set_xticks(xs)
    ax.set_xlabel("Component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_rubric_loading_scatter(
    loadings: pd.DataFrame,
    out_path: Path,
    title: str,
    xlab: str,
    ylab: str,
) -> None:
    """
    loadings: index=rubrics, columns like ["C1","C2"] or ["F1","F2"]
    """
    plt.figure(figsize=(6.8, 6.0))
    ax = plt.gca()

    x = loadings.iloc[:, 0].values.astype(float)
    y = loadings.iloc[:, 1].values.astype(float)
    ax.scatter(x, y)

    for i, name in enumerate(loadings.index.astype(str).tolist()):
        ax.annotate(name, (x[i], y[i]), fontsize=9, xytext=(4, 4), textcoords="offset points")

    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Merged judge scores CSV (e.g. results/merged/all_judge_scores.csv)")
    ap.add_argument("--out", type=str, default="results/analysis/rubric_factor_analysis", help="Output directory")
    ap.add_argument("--judge", type=str, default="", help="Optional: filter to a specific judge_model")
    ap.add_argument("--language", type=str, default="all", help="Optional: all|en|zh (requires language column)")
    ap.add_argument("--min_samples", type=int, default=200, help="Warn if total samples < this (factor analysis becomes unstable)")
    ap.add_argument("--efa_factors", type=int, default=2, help="Number of EFA factors (default 2)")
    ap.add_argument("--pca_components", type=int, default=8, help="How many PCA components to compute (<= #rubrics)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    safe_mkdir(out_dir)

    df = pd.read_csv(Path(args.csv))
    scenario_col, model_col, judge_col, language_col = detect_cols(df)

    # ---- Build sample matrix (scenario, model) x rubrics
    samples = aggregate_to_samples(
        df=df,
        scenario_col=scenario_col,
        model_col=model_col,
        judge_col=judge_col,
        language_col=language_col,
        rubrics=RUBRICS,
        language=args.language,
        judge_filter=args.judge,
    )

    avail = [r for r in RUBRICS if r in samples.columns]
    if len(avail) < 2:
        raise ValueError("Need at least 2 rubric columns.")

    X = samples[avail].values.astype(float)
    # drop rows with any NaN to keep clean factor model
    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    if X.shape[0] < args.min_samples:
        print(f"[WARN] Only {X.shape[0]} samples after NaN filtering. Consider using more data or relaxing filters.")

    # standardize
    Xz = zscore(X)

    # =========================================================
    # 1) PCA
    # =========================================================
    n_comp = min(args.pca_components, Xz.shape[1])
    pca = PCA(n_components=n_comp, random_state=0)
    pca.fit(Xz)

    explained = pca.explained_variance_ratio_
    pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(len(explained))],
        "explained_variance_ratio": explained,
        "explained_variance_cum": np.cumsum(explained),
    }).to_csv(out_dir / "pca_explained_variance.csv", index=False, encoding="utf-8")

    plot_scree(
        explained_ratio=explained,
        out_path=out_dir / "pca_scree.png",
        title="PCA Scree Plot (Rubric Space)",
    )

    # PCA loadings: components_ has shape (n_components, n_features)
    # Interpret loadings as feature weights on principal axes.
    load_pca_2d = pd.DataFrame(
        pca.components_[:2, :].T,
        index=avail,
        columns=["PC1_loading", "PC2_loading"],
    )
    load_pca_2d.to_csv(out_dir / "pca_rubric_loadings_2d.csv", index=True, encoding="utf-8")
    plot_rubric_loading_scatter(
        loadings=load_pca_2d.rename(columns={"PC1_loading": "C1", "PC2_loading": "C2"}),
        out_path=out_dir / "pca_rubric_loading_scatter.png",
        title="PCA Rubric Loading Scatter (PC1 vs PC2)",
        xlab="PC1 loading",
        ylab="PC2 loading",
    )

    # =========================================================
    # 2) EFA (Exploratory Factor Analysis) + Varimax rotation
    # =========================================================
    k = int(args.efa_factors)
    if k < 1 or k > len(avail):
        raise ValueError("--efa_factors must be between 1 and number of rubric columns.")

    fa = FactorAnalysis(n_components=k, random_state=0)
    fa.fit(Xz)

    # sklearn FactorAnalysis.components_ is (n_components, n_features)
    # Convert to (n_features, n_components) loadings
    loadings = fa.components_.T  # (features, factors)

    # varimax rotation for interpretability
    loadings_rot = varimax(loadings)

    cols = [f"F{i+1}" for i in range(k)]
    efa_load = pd.DataFrame(loadings_rot, index=avail, columns=cols)
    efa_load.to_csv(out_dir / "efa_loadings_varimax.csv", index=True, encoding="utf-8")

    # also export unrotated for reference
    efa_load_raw = pd.DataFrame(loadings, index=avail, columns=[f"F{i+1}_raw" for i in range(k)])
    efa_load_raw.to_csv(out_dir / "efa_loadings_raw.csv", index=True, encoding="utf-8")

    # 2D scatter if k>=2
    if k >= 2:
        plot_rubric_loading_scatter(
            loadings=efa_load.iloc[:, :2],
            out_path=out_dir / "efa_rubric_loading_scatter.png",
            title="EFA Rubric Loading Scatter (Varimax Rotated)",
            xlab="Factor 1 loading",
            ylab="Factor 2 loading",
        )

    # Factor scores per sample (optional outputs; useful for model-level plots)
    # sklearn FA has transform -> scores in factor space
    scores = fa.transform(Xz)  # (n_samples, k)
    scores_df = pd.DataFrame(scores, columns=[f"Factor{i+1}_score" for i in range(k)])
    scores_df.to_csv(out_dir / "efa_factor_scores_per_sample.csv", index=False, encoding="utf-8")

    print("[DONE]")
    print("Out dir:", out_dir.resolve())
    print("Key outputs:")
    print("- pca_explained_variance.csv, pca_scree.png, pca_rubric_loadings_2d.csv, pca_rubric_loading_scatter.png")
    print("- efa_loadings_raw.csv, efa_loadings_varimax.csv, efa_rubric_loading_scatter.png (if factors>=2)")
    print("- efa_factor_scores_per_sample.csv")


if __name__ == "__main__":
    main()
