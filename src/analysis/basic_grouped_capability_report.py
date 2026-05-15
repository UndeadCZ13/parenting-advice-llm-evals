from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Group definitions (YOUR NEW DEFAULT)
# -------------------------
CONSTRUCTIVE = ["accuracy", "communication", "completeness", "helpfulness"]
RESPONSIBLE = ["empathy", "bias_avoidance", "limitation_awareness", "safety"]
ALL_RUBRICS = CONSTRUCTIVE + RESPONSIBLE

GROUP1_NAME_EN = "Constructive Answering"
GROUP2_NAME_EN = "Responsible Alignment"
OVERALL_NAME_EN = "Overall"

# If you want bilingual labels in plots, swap to these:
# GROUP1_NAME_EN = "Constructive Answering（构建式回答）"
# GROUP2_NAME_EN = "Responsible Alignment（责任对齐）"
# OVERALL_NAME_EN = "Overall（综合）"


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


def handle_missing(df: pd.DataFrame, rubrics: List[str], strategy: str) -> pd.DataFrame:
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
        raise ValueError("Unknown --missing_strategy. Use drop|median|mean")
    dfx[rubrics] = dfx[rubrics].fillna(fills)
    return dfx


def compute_model_scores(
    df: pd.DataFrame,
    scenario_col: str,
    model_col: str,
    rubrics: List[str],
) -> pd.DataFrame:
    """
    Aggregate to (scenario, model) then compute per-model mean scores:
      - constructive_mean
      - responsible_mean
      - overall_mean
    """
    # scenario-model mean first (avoid repeats inflating)
    dfx = df.groupby([scenario_col, model_col], dropna=False)[rubrics].mean().reset_index()

    dfx["constructive_mean"] = dfx[CONSTRUCTIVE].mean(axis=1)
    dfx["responsible_mean"] = dfx[RESPONSIBLE].mean(axis=1)
    dfx["overall_mean"] = dfx[rubrics].mean(axis=1)

    out = (
        dfx.groupby(model_col)[["constructive_mean", "responsible_mean", "overall_mean"]]
        .mean()
        .reset_index()
        .rename(columns={model_col: "model"})
    )
    out = out.sort_values("overall_mean", ascending=False).reset_index(drop=True)
    return out


def plot_three_bars_per_model(
    model_scores: pd.DataFrame,
    out_path: Path,
    title: str,
    max_models: int = 30,
) -> None:
    """
    One chart: x=model, 3 bars each (constructive, responsible, overall)
    """
    dfx = model_scores.copy()
    if len(dfx) > max_models:
        dfx = dfx.head(max_models)

    x = np.arange(len(dfx))
    w = 0.26

    plt.figure(figsize=(max(12, 0.55 * len(dfx) + 4), 6))
    ax = plt.gca()

    ax.bar(x - w, dfx["constructive_mean"].values, width=w, label=GROUP1_NAME_EN)
    ax.bar(x,       dfx["responsible_mean"].values, width=w, label=GROUP2_NAME_EN)
    ax.bar(x + w, dfx["overall_mean"].values, width=w, label=OVERALL_NAME_EN)

    ax.set_xticks(x)
    ax.set_xticklabels(dfx["model"].astype(str).tolist(), rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean score")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_and_plot_slice(
    df_slice: pd.DataFrame,
    slice_name: str,
    out_dir: Path,
    scenario_col: str,
    model_col: str,
    rubrics: List[str],
    max_models: int,
) -> None:
    if df_slice.empty:
        return
    scores = compute_model_scores(df_slice, scenario_col, model_col, rubrics)
    scores.to_csv(out_dir / f"model_scores__{slice_name}.csv", index=False, encoding="utf-8")
    plot_three_bars_per_model(
        scores,
        out_dir / f"model_3bar__{slice_name}.png",
        title=f"{slice_name}: {GROUP1_NAME_EN} vs {GROUP2_NAME_EN} vs {OVERALL_NAME_EN} (sorted by Overall)",
        max_models=max_models,
    )


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Merged judge scores CSV")
    ap.add_argument("--out", default="results/analysis/basic_capability_report", help="Output directory")
    ap.add_argument("--missing_strategy", default="drop", help="drop|median|mean (default drop)")
    ap.add_argument("--max_models", type=int, default=30, help="Max models to show in plots (top by overall)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    safe_mkdir(out_dir)

    df = pd.read_csv(args.csv)
    scenario_col, model_col, judge_col, language_col = detect_cols(df)

    # Keep only needed columns
    need_cols = [scenario_col, model_col, judge_col] + (["language"] if language_col else []) + ALL_RUBRICS
    df = df[need_cols].copy()

    # Missing report
    miss = pd.DataFrame({
        "missing_count": df[ALL_RUBRICS].isna().sum(),
        "missing_frac": df[ALL_RUBRICS].isna().mean(),
    }).sort_values("missing_frac", ascending=False)
    miss.to_csv(out_dir / "missing_report.csv", index=True, encoding="utf-8")

    # Handle missing
    df = handle_missing(df, ALL_RUBRICS, args.missing_strategy)

    # -------------------------
    # 0) GLOBAL (all judges, all languages)
    # -------------------------
    global_dir = out_dir / "global"
    safe_mkdir(global_dir)
    save_and_plot_slice(
        df_slice=df,
        slice_name="GLOBAL",
        out_dir=global_dir,
        scenario_col=scenario_col,
        model_col=model_col,
        rubrics=ALL_RUBRICS,
        max_models=args.max_models,
    )

    # -------------------------
    # 1) By LANGUAGE
    # -------------------------
    by_lang_dir = out_dir / "by_language"
    safe_mkdir(by_lang_dir)
    if language_col:
        langs = sorted(df[language_col].dropna().astype(str).unique().tolist())
        for lang in langs:
            dfl = df[df[language_col].astype(str) == lang].copy()
            save_and_plot_slice(
                df_slice=dfl,
                slice_name=f"LANG={lang}",
                out_dir=by_lang_dir,
                scenario_col=scenario_col,
                model_col=model_col,
                rubrics=ALL_RUBRICS,
                max_models=args.max_models,
            )
    else:
        with (by_lang_dir / "README.txt").open("w", encoding="utf-8") as f:
            f.write("No language column in CSV; language slices not generated.\n")

    # -------------------------
    # 2) By JUDGE
    # -------------------------
    by_judge_dir = out_dir / "by_judge"
    safe_mkdir(by_judge_dir)
    judges = sorted(df[judge_col].dropna().astype(str).unique().tolist())
    for j in judges:
        dfj = df[df[judge_col].astype(str) == j].copy()
        # For judge slices, we keep all languages together (unless you want to add further filters)
        save_and_plot_slice(
            df_slice=dfj,
            slice_name=f"JUDGE={sanitize(j)}",
            out_dir=by_judge_dir,
            scenario_col=scenario_col,
            model_col=model_col,
            rubrics=ALL_RUBRICS,
            max_models=args.max_models,
        )

    # -------------------------
    # 3) Language × Judge (cross slice)
    # -------------------------
    cross_dir = out_dir / "by_language_x_judge"
    safe_mkdir(cross_dir)
    if language_col:
        langs = sorted(df[language_col].dropna().astype(str).unique().tolist())
        for lang in langs:
            for j in judges:
                dfx = df[(df[language_col].astype(str) == lang) & (df[judge_col].astype(str) == j)].copy()
                if dfx.empty:
                    continue
                save_and_plot_slice(
                    df_slice=dfx,
                    slice_name=f"LANG={lang}__JUDGE={sanitize(j)}",
                    out_dir=cross_dir,
                    scenario_col=scenario_col,
                    model_col=model_col,
                    rubrics=ALL_RUBRICS,
                    max_models=args.max_models,
                )
    else:
        with (cross_dir / "README.txt").open("w", encoding="utf-8") as f:
            f.write("No language column in CSV; language×judge slices not generated.\n")

    # -------------------------
    # Summary index file (what to look at)
    # -------------------------
    with (out_dir / "README.txt").open("w", encoding="utf-8") as f:
        f.write("This report uses simple mean scores (no PCA/FA) for maximum interpretability.\n\n")
        f.write(f"Groups:\n- {GROUP1_NAME_EN}: {', '.join(CONSTRUCTIVE)}\n- {GROUP2_NAME_EN}: {', '.join(RESPONSIBLE)}\n\n")
        f.write("Outputs:\n")
        f.write("- global/: GLOBAL ranking and 3-bar chart\n")
        f.write("- by_language/: per-language ranking and 3-bar charts\n")
        f.write("- by_judge/: per-judge ranking and 3-bar charts\n")
        f.write("- by_language_x_judge/: per (language, judge) slice charts\n")

    print("[DONE] Outputs in:", out_dir.resolve())


def sanitize(s: str) -> str:
    s = str(s)
    for ch in ['/', '\\', ':', '|', '*', '?', '"', '<', '>', ' ']:
        s = s.replace(ch, "_")
    return s


if __name__ == "__main__":
    main()
