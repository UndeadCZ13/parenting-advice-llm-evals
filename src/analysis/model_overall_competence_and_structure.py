from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler


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

FACTOR1_NAME = "Structure Factor 1"
FACTOR2_NAME = "Structure Factor 2"


# -------------------------
# Utils
# -------------------------
def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def detect_cols(df: pd.DataFrame) -> Tuple[str, str, str, Optional[str]]:
    scenario = None
    for c in ["scenario_uid", "scenario_id", "id"]:
        if c in df.columns:
            scenario = c
            break
    if not scenario:
        raise ValueError("Cannot find scenario column: scenario_uid/scenario_id/id")

    model = None
    for c in ["answer_model", "model"]:
        if c in df.columns:
            model = c
            break
    if not model:
        raise ValueError("Cannot find model column: answer_model/model")

    judge = None
    for c in ["judge_model", "grader_model"]:
        if c in df.columns:
            judge = c
            break
    if not judge:
        raise ValueError("Cannot find judge column: judge_model/grader_model")

    language = "language" if "language" in df.columns else None
    return scenario, model, judge, language


def preprocess_missing(df: pd.DataFrame, rubrics: List[str], strategy: str) -> pd.DataFrame:
    """
    strategy: drop | median | mean
    """
    if strategy == "drop":
        return df.dropna(subset=rubrics).copy()

    dfx = df.copy()
    if strategy == "median":
        fills = {c: float(dfx[c].median()) for c in rubrics}
    elif strategy == "mean":
        fills = {c: float(dfx[c].mean()) for c in rubrics}
    else:
        raise ValueError("Unknown missing strategy. Use drop|median|mean")
    dfx[rubrics] = dfx[rubrics].fillna(fills)
    return dfx


def zscore_cols(X: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(X)


def run_fa(X: np.ndarray, n_factors: int = 2, seed: int = 0) -> np.ndarray:
    """
    Return factor scores (n_samples, n_factors) from FactorAnalysis on z-scored columns.
    """
    Z = zscore_cols(X)
    fa = FactorAnalysis(n_components=n_factors, random_state=seed)
    fa.fit(Z)
    return fa.transform(Z)


def row_center(X: np.ndarray) -> np.ndarray:
    """
    Remove per-sample overall level: X - row_mean
    """
    return X - np.mean(X, axis=1, keepdims=True)


# -------------------------
# Plotters
# -------------------------
def plot_overall_bar(df_model: pd.DataFrame, out_path: Path, title: str) -> None:
    """
    df_model: columns = model, overall
    """
    dfx = df_model.sort_values("overall", ascending=True).reset_index(drop=True)

    plt.figure(figsize=(max(10, 0.45 * len(dfx) + 4), 6))
    ax = plt.gca()
    ax.barh(np.arange(len(dfx)), dfx["overall"].values)
    ax.set_yticks(np.arange(len(dfx)))
    ax.set_yticklabels(dfx["model"].astype(str).tolist(), fontsize=9)
    ax.set_xlabel("Overall competence (mean rubric score)")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_rubric_breakdown(df_model_rubrics: pd.DataFrame, out_path: Path, title: str) -> None:
    """
    Stacked-ish grouped bars: show per-model mean for each rubric (optional big figure)
    """
    dfx = df_model_rubrics.set_index("model")
    plt.figure(figsize=(max(12, 0.55 * len(dfx) + 4), 6))
    ax = plt.gca()
    dfx.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Mean rubric score")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_structure_scatter(df_model: pd.DataFrame, out_path: Path, title: str) -> None:
    """
    df_model: columns = model, S1, S2
    """
    plt.figure(figsize=(7.2, 6.2))
    ax = plt.gca()
    ax.scatter(df_model["S1"], df_model["S2"])

    for _, r in df_model.iterrows():
        ax.annotate(str(r["model"]), (r["S1"], r["S2"]), fontsize=8, xytext=(4, 4), textcoords="offset points")

    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    ax.set_xlabel(FACTOR1_NAME)
    ax.set_ylabel(FACTOR2_NAME)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Merged judge scores CSV, e.g. results/merged/all_judge_scores.csv")
    ap.add_argument("--out", default="results/analysis/model_overall_competence", help="Output directory")
    ap.add_argument("--judge", default="", help="Optional: filter judge_model")
    ap.add_argument("--language", default="all", help="Optional: all|en|zh (requires language column)")
    ap.add_argument("--missing_strategy", default="drop", help="drop|median|mean (default drop)")
    ap.add_argument("--min_samples", type=int, default=200, help="Warn if total samples < this")
    ap.add_argument("--save_rubric_breakdown", action="store_true", help="Also save per-rubric per-model bar chart (bigger)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    safe_mkdir(out_dir)

    df = pd.read_csv(args.csv)
    scenario_col, model_col, judge_col, language_col = detect_cols(df)

    # Filters
    if args.judge.strip():
        df = df[df[judge_col].astype(str) == args.judge.strip()].copy()

    lang = args.language.lower().strip()
    if lang != "all":
        if not language_col:
            print("[WARN] CSV has no language column; language filter skipped.")
        else:
            df = df[df[language_col].astype(str).str.lower().str.strip() == lang].copy()

    # Aggregate to (scenario, model) level (average over judge/language repeats)
    group_cols = [scenario_col, model_col]
    # If you want language-specific competence, keep language in group_cols by running with --language filter.
    df = df[group_cols + RUBRICS].groupby(group_cols, dropna=False)[RUBRICS].mean().reset_index()

    # Missing handling
    miss_report = pd.DataFrame({
        "missing_count": df[RUBRICS].isna().sum(),
        "missing_frac": df[RUBRICS].isna().mean(),
    }).sort_values("missing_frac", ascending=False)
    miss_report.to_csv(out_dir / "missing_report.csv", index=True, encoding="utf-8")

    df = preprocess_missing(df, RUBRICS, args.missing_strategy)

    # Sanity
    X = df[RUBRICS].values.astype(float)
    if np.isnan(X).any():
        raise ValueError("Still has NaN after missing handling. Check missing_strategy and input data.")
    if X.shape[0] < args.min_samples:
        print(f"[WARN] Only {X.shape[0]} samples. Overall score is fine; structure FA may be noisy.")

    # ============================================================
    # A) Overall competence (THIS is the ability level plot you want)
    # ============================================================
    df["overall"] = df[RUBRICS].mean(axis=1)

    model_overall = (
        df.groupby(model_col)["overall"]
        .mean()
        .reset_index()
        .rename(columns={model_col: "model"})
    )
    model_overall.to_csv(out_dir / "model_overall_competence.csv", index=False, encoding="utf-8")

    plot_overall_bar(
        model_overall,
        out_dir / "model_overall_competence_bar.png",
        title=f"Overall Model Competence (mean over {len(RUBRICS)} rubrics){' | judge='+args.judge if args.judge else ''}{' | lang='+lang if lang!='all' else ''}",
    )

    # Optional: per-rubric breakdown
    model_rubrics = (
        df.groupby(model_col)[RUBRICS]
        .mean()
        .reset_index()
        .rename(columns={model_col: "model"})
    )
    model_rubrics.to_csv(out_dir / "model_mean_rubrics.csv", index=False, encoding="utf-8")

    if args.save_rubric_breakdown:
        plot_rubric_breakdown(
            model_rubrics,
            out_dir / "model_rubric_breakdown_bars.png",
            title="Per-rubric Mean Scores by Model",
        )

    # ============================================================
    # B) Structure-only factor map (remove overall level first)
    #    This avoids misreading "left = worse".
    # ============================================================
    X_centered = row_center(X)
    # After row-centering, factors capture relative pattern, not absolute level.
    S = run_fa(X_centered, n_factors=2, seed=0)
    df["S1"] = S[:, 0]
    df["S2"] = S[:, 1]

    model_structure = (
        df.groupby(model_col)[["S1", "S2"]]
        .mean()
        .reset_index()
        .rename(columns={model_col: "model"})
    )
    model_structure.to_csv(out_dir / "model_structure_factor_scores.csv", index=False, encoding="utf-8")

    plot_structure_scatter(
        model_structure,
        out_dir / "model_structure_scatter.png",
        title="Model Structure Map (FA on row-centered rubric scores)",
    )

    # Join overall + structure for combined table
    combined = model_overall.merge(model_structure, on="model", how="inner")
    combined.to_csv(out_dir / "model_overall_plus_structure.csv", index=False, encoding="utf-8")

    with (out_dir / "README.txt").open("w", encoding="utf-8") as f:
        f.write("Key outputs:\n")
        f.write("- model_overall_competence_bar.png : overall ability level (what you asked for)\n")
        f.write("- model_overall_competence.csv : numeric overall level per model\n")
        f.write("- model_structure_scatter.png : structure-only (row-centered) factor map (NOT a quality axis)\n")
        f.write("- model_overall_plus_structure.csv : combined table\n")
        f.write("\nNotes:\n")
        f.write("- Overall competence = mean rubric score per sample, then averaged per model.\n")
        f.write("- Structure FA uses row-centered X to remove overall level; it captures relative rubric trade-offs.\n")

    print("[DONE] Outputs in:", out_dir.resolve())


if __name__ == "__main__":
    main()
