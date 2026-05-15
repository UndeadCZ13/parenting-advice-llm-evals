from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

try:
    from scipy.spatial import procrustes
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


RUBRICS = [
    "accuracy",
    "safety",
    "helpfulness",
    "empathy",
    "completeness",
    "bias_avoidance",
    "limitation_awareness",
    "communication",
]

FACTOR1_NAME = "Factor 1: Cognitive–Constructive Competence"
FACTOR2_NAME = "Factor 2: Alignment & Social Responsibility"


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


def zscore(X: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(X)


def run_fa_scores_and_loadings(X: np.ndarray, n_factors: int = 2, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      scores: (n_samples, n_factors)
      loadings: (n_rubrics, n_factors)  (unrotated, but adequate for stability comparison)
    """
    Z = zscore(X)
    fa = FactorAnalysis(n_components=n_factors, random_state=seed)
    fa.fit(Z)
    scores = fa.transform(Z)
    loadings = fa.components_.T
    return scores, loadings


def report_missing(df: pd.DataFrame, rubrics: List[str], out_path: Path) -> None:
    miss = df[rubrics].isna().sum().sort_values(ascending=False)
    frac = (df[rubrics].isna().mean()).sort_values(ascending=False)
    rep = pd.DataFrame({"missing_count": miss, "missing_frac": frac})
    rep.to_csv(out_path, index=True, encoding="utf-8")


def preprocess_missing(
    df: pd.DataFrame,
    rubrics: List[str],
    strategy: str,
    out_dir: Path,
) -> pd.DataFrame:
    """
    strategy:
      - drop: drop any row with NaN in rubrics
      - median: fill NaN with column median
      - mean: fill NaN with column mean
    """
    report_missing(df, rubrics, out_dir / "missing_report.csv")

    if strategy == "drop":
        before = len(df)
        dfx = df.dropna(subset=rubrics).copy()
        after = len(dfx)
        with (out_dir / "missing_handling.txt").open("w", encoding="utf-8") as f:
            f.write(f"strategy=drop\nbefore_rows={before}\nafter_rows={after}\ndropped_rows={before-after}\n")
        return dfx

    dfx = df.copy()
    if strategy == "median":
        fills = {c: float(dfx[c].median()) for c in rubrics}
    elif strategy == "mean":
        fills = {c: float(dfx[c].mean()) for c in rubrics}
    else:
        raise ValueError("Unknown --missing_strategy. Use: drop|median|mean")

    dfx[rubrics] = dfx[rubrics].fillna(fills)
    with (out_dir / "missing_handling.txt").open("w", encoding="utf-8") as f:
        f.write(f"strategy={strategy}\nfilled_with={fills}\n")
    return dfx


def flip_factor_signs_for_readability(loadings: np.ndarray, rubric_names: List[str]) -> np.ndarray:
    """
    Factor directions are arbitrary (sign can flip).
    For readability, we make Factor1 have positive loading on 'accuracy' if present.
    """
    L = loadings.copy()
    if "accuracy" in rubric_names:
        i = rubric_names.index("accuracy")
        if L[i, 0] < 0:
            L[:, 0] *= -1
    # For Factor2, make 'empathy' positive if present
    if "empathy" in rubric_names and L.shape[1] >= 2:
        j = rubric_names.index("empathy")
        if L[j, 1] < 0:
            L[:, 1] *= -1
    return L


# -------------------------
# Plot helpers
# -------------------------
def plot_model_scatter(df_model: pd.DataFrame, out_path: Path, title: str) -> None:
    plt.figure(figsize=(7.2, 6.2))
    ax = plt.gca()
    ax.scatter(df_model["Factor1"], df_model["Factor2"])

    for _, r in df_model.iterrows():
        ax.annotate(str(r["model"]), (r["Factor1"], r["Factor2"]), fontsize=8, xytext=(4, 4), textcoords="offset points")

    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    ax.set_xlabel(FACTOR1_NAME)
    ax.set_ylabel(FACTOR2_NAME)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_model_bars(df_model: pd.DataFrame, out_path: Path, title: str) -> None:
    dfx = df_model.set_index("model")[["Factor1", "Factor2"]]
    plt.figure(figsize=(max(10, 0.45 * len(dfx) + 4), 5.5))
    ax = plt.gca()
    dfx.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Mean factor score")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_loadings_table(loadings_df: pd.DataFrame, out_path: Path, title: str) -> None:
    """
    Save as CSV already; this makes a quick heatmap-ish image for sharing.
    """
    mat = loadings_df.values.astype(float)
    plt.figure(figsize=(6.8, 5.8))
    ax = plt.gca()
    im = ax.imshow(mat, aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_title(title)
    ax.set_xticks(range(loadings_df.shape[1]))
    ax.set_xticklabels(loadings_df.columns.tolist(), rotation=0)
    ax.set_yticks(range(loadings_df.shape[0]))
    ax.set_yticklabels(loadings_df.index.tolist(), fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Merged judge scores CSV")
    ap.add_argument("--out", default="results/analysis/model_factor_maps", help="Output directory")
    ap.add_argument("--judge", default="", help="Optional: filter to a specific judge_model")
    ap.add_argument("--language", default="all", help="Optional: all|en|zh (requires language column)")
    ap.add_argument("--missing_strategy", default="drop", help="drop|median|mean (default drop)")
    ap.add_argument("--min_samples", type=int, default=200, help="Warn if < this many samples for FA")
    args = ap.parse_args()

    out_dir = Path(args.out)
    safe_mkdir(out_dir)

    df = pd.read_csv(args.csv)
    scenario_col, model_col, judge_col, language_col = detect_cols(df)

    # ---- Filter judge/language (optional)
    if args.judge.strip():
        df = df[df[judge_col].astype(str) == args.judge.strip()].copy()

    lang = args.language.lower().strip()
    if lang != "all":
        if not language_col:
            print("[WARN] CSV has no language column; language filter skipped.")
        else:
            df = df[df[language_col].astype(str).str.lower().str.strip() == lang].copy()

    # ---- Aggregate to (scenario, model, judge, language?) then mean
    group_cols = [scenario_col, model_col]
    if language_col:
        group_cols.append(language_col)
    group_cols.append(judge_col)

    df = df[group_cols + RUBRICS].groupby(group_cols, dropna=False)[RUBRICS].mean().reset_index()

    # Save a quick info file
    with (out_dir / "data_slice_info.txt").open("w", encoding="utf-8") as f:
        f.write(f"csv={args.csv}\n")
        f.write(f"judge_filter={args.judge.strip() or 'NONE'}\n")
        f.write(f"language_filter={lang}\n")
        f.write(f"rows_after_groupby={len(df)}\n")

    # ---- Handle missing
    df = preprocess_missing(df, RUBRICS, args.missing_strategy, out_dir)

    X = df[RUBRICS].values.astype(float)
    if X.shape[0] < args.min_samples:
        print(f"[WARN] Only {X.shape[0]} samples. FA stability may be weak; consider more data or less filtering.")

    # ============================================================
    # 1) Global factor model -> scores per sample
    # ============================================================
    scores, loadings = run_fa_scores_and_loadings(X, n_factors=2, seed=0)
    loadings = flip_factor_signs_for_readability(loadings, RUBRICS)

    df["Factor1"] = scores[:, 0]
    df["Factor2"] = scores[:, 1]

    # Export loadings
    load_df = pd.DataFrame(loadings, index=RUBRICS, columns=["Factor1_loading", "Factor2_loading"])
    load_df.to_csv(out_dir / "global_fa_loadings.csv", index=True, encoding="utf-8")
    plot_loadings_table(load_df, out_dir / "global_fa_loadings_heatmap.png", "Global Factor Loadings (unrotated)")

    # Model-level means
    model_scores = df.groupby(model_col)[["Factor1", "Factor2"]].mean().reset_index().rename(columns={model_col: "model"})
    model_scores.to_csv(out_dir / "model_factor_scores.csv", index=False, encoding="utf-8")

    # Scatter + bars
    plot_model_scatter(model_scores, out_dir / "model_factor_scatter.png", "Model Capability Map (Global FA)")
    plot_model_bars(model_scores.sort_values("Factor1", ascending=False), out_dir / "model_factor_bars_sorted_by_F1.png",
                    "Model Factor Scores (sorted by Factor1)")
    plot_model_bars(model_scores.sort_values("Factor2", ascending=False), out_dir / "model_factor_bars_sorted_by_F2.png",
                    "Model Factor Scores (sorted by Factor2)")

    # Add quadrant labels file
    q = model_scores.copy()
    q["quadrant"] = np.select(
        [
            (q["Factor1"] >= 0) & (q["Factor2"] >= 0),
            (q["Factor1"] >= 0) & (q["Factor2"] < 0),
            (q["Factor1"] < 0) & (q["Factor2"] >= 0),
            (q["Factor1"] < 0) & (q["Factor2"] < 0),
        ],
        ["High Cog / High Align", "High Cog / Low Align", "Low Cog / High Align", "Low Cog / Low Align"],
        default="NA"
    )
    q.to_csv(out_dir / "model_quadrants.csv", index=False, encoding="utf-8")

    # ============================================================
    # 2) Stability: language & judge
    #    We compare factor LOADING structures using Procrustes disparity.
    # ============================================================
    stability_dir = out_dir / "stability"
    safe_mkdir(stability_dir)

    rows = []

    def compute_loadings_for_slice(dfs: pd.DataFrame) -> Optional[np.ndarray]:
        # same missing handling strategy in slice
        dfs2 = dfs.copy()
        dfs2 = preprocess_missing(dfs2, RUBRICS, args.missing_strategy, stability_dir)  # writes over missing files; OK
        Xs = dfs2[RUBRICS].values.astype(float)
        if Xs.shape[0] < 50:
            return None
        _, L = run_fa_scores_and_loadings(Xs, n_factors=2, seed=0)
        L = flip_factor_signs_for_readability(L, RUBRICS)
        return L

    def procrustes_disparity(A: np.ndarray, B: np.ndarray) -> float:
        if not SCIPY_OK:
            return float("nan")
        _, _, d = procrustes(A, B)
        return float(d)

    # ---- Language stability: pairwise across all languages present
    if language_col:
        langs = sorted(df[language_col].dropna().astype(str).unique().tolist())
        if len(langs) >= 2:
            lang_loadings: Dict[str, np.ndarray] = {}
            for l in langs:
                dfs_l = df[df[language_col].astype(str) == l].copy()
                L = compute_loadings_for_slice(dfs_l)
                if L is not None:
                    lang_loadings[l] = L
                    pd.DataFrame(L, index=RUBRICS, columns=["F1", "F2"]).to_csv(stability_dir / f"loadings_language_{l}.csv")

            for a, b in itertools.combinations(sorted(lang_loadings.keys()), 2):
                d = procrustes_disparity(lang_loadings[a], lang_loadings[b])
                rows.append({"type": "language", "A": a, "B": b, "procrustes_disparity": d})

    # ---- Judge stability: pairwise across all judges present
    judges = sorted(df[judge_col].dropna().astype(str).unique().tolist())
    if len(judges) >= 2:
        judge_loadings: Dict[str, np.ndarray] = {}
        for j in judges:
            dfs_j = df[df[judge_col].astype(str) == j].copy()
            L = compute_loadings_for_slice(dfs_j)
            if L is not None:
                judge_loadings[j] = L
                pd.DataFrame(L, index=RUBRICS, columns=["F1", "F2"]).to_csv(stability_dir / f"loadings_judge_{sanitize(j)}.csv")

        for a, b in itertools.combinations(sorted(judge_loadings.keys()), 2):
            d = procrustes_disparity(judge_loadings[a], judge_loadings[b])
            rows.append({"type": "judge", "A": a, "B": b, "procrustes_disparity": d})

    stab = pd.DataFrame(rows)
    stab.to_csv(stability_dir / "factor_structure_stability.csv", index=False, encoding="utf-8")

    # Quick summary by type
    if not stab.empty:
        summary = stab.groupby("type")["procrustes_disparity"].agg(["count", "mean", "median", "max"]).reset_index()
        summary.to_csv(stability_dir / "factor_structure_stability_summary.csv", index=False, encoding="utf-8")

    with (out_dir / "README.txt").open("w", encoding="utf-8") as f:
        f.write("Outputs:\n")
        f.write("- model_factor_scatter.png : model capability map in (Factor1, Factor2)\n")
        f.write("- model_factor_scores.csv : numeric model factor means\n")
        f.write("- model_quadrants.csv : quadrant labels for quick interpretation\n")
        f.write("- global_fa_loadings.csv (+ heatmap png)\n")
        f.write("- stability/factor_structure_stability.csv : Procrustes disparity across languages/judges\n")
        if not SCIPY_OK:
            f.write("\nNOTE: scipy not available; Procrustes disparity will be NaN.\n")

    print("[DONE] Outputs written to:", out_dir.resolve())


def sanitize(s: str) -> str:
    s = str(s)
    for ch in ['/', '\\', ':', '|', '*', '?', '"', '<', '>', ' ']:
        s = s.replace(ch, "_")
    return s


if __name__ == "__main__":
    main()
