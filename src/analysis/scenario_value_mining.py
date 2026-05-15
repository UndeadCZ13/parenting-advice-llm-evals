from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Capability groups (your new naming)
# -------------------------
CONSTRUCTIVE = ["accuracy", "communication", "completeness", "helpfulness"]
RESPONSIBLE = ["empathy", "bias_avoidance", "limitation_awareness", "safety"]
ALL = CONSTRUCTIVE + RESPONSIBLE

GROUP1_NAME = "Constructive Answering"
GROUP2_NAME = "Responsible Alignment"
GROUP3_NAME = "Overall"


# -------------------------
# Utils
# -------------------------
def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def detect_cols(df: pd.DataFrame) -> Tuple[str, str]:
    scenario = None
    for c in ["scenario_uid", "scenario_id", "id"]:
        if c in df.columns:
            scenario = c
            break
    if not scenario:
        raise ValueError("Cannot find scenario column: expected one of scenario_uid/scenario_id/id")

    model = None
    for c in ["answer_model", "model"]:
        if c in df.columns:
            model = c
            break
    if not model:
        raise ValueError("Cannot find model column: expected answer_model/model")

    return scenario, model


def compute_sample_scores(df_sm: pd.DataFrame, scenario_col: str, model_col: str) -> pd.DataFrame:
    """
    Input df_sm must contain columns: scenario_col, model_col, ALL rubrics
    Returns per (scenario, model) with constructive/responsible/overall.
    """
    dfx = df_sm.groupby([scenario_col, model_col], dropna=False)[ALL].mean().reset_index()
    dfx["constructive"] = dfx[CONSTRUCTIVE].mean(axis=1)
    dfx["responsible"] = dfx[RESPONSIBLE].mean(axis=1)
    dfx["overall"] = dfx[ALL].mean(axis=1)
    return dfx[[scenario_col, model_col, "constructive", "responsible", "overall"]]


def plot_model_3bar(scores: pd.DataFrame, model_col: str, out_path: Path, title: str, max_models: int | None = None) -> None:
    """
    Plot per-model grouped bars: constructive/responsible/overall.
    scores must include columns: model_col, constructive, responsible, overall
    """
    dfx = scores.copy().sort_values("overall", ascending=False).reset_index(drop=True)
    if max_models is not None and len(dfx) > max_models:
        dfx = dfx.head(max_models)

    x = np.arange(len(dfx))
    w = 0.26

    plt.figure(figsize=(max(12, 0.55 * len(dfx) + 4), 6))
    ax = plt.gca()

    ax.bar(x - w, dfx["constructive"].values, width=w, label=GROUP1_NAME)
    ax.bar(x,       dfx["responsible"].values, width=w, label=GROUP2_NAME)
    ax.bar(x + w, dfx["overall"].values, width=w, label=GROUP3_NAME)

    ax.set_xticks(x)
    ax.set_xticklabels(dfx[model_col].astype(str).tolist(), rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean score")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def export_scenario_detail_charts(
    df_scores: pd.DataFrame,
    scenario_id,
    scenario_col: str,
    model_col: str,
    out_dir: Path,
    chart_title_prefix: str,
    top_k_models: int,
) -> None:
    """
    For one scenario: export model_scores.csv + charts
    """
    safe_mkdir(out_dir)

    sub = df_scores[df_scores[scenario_col] == scenario_id].copy()
    if sub.empty:
        return

    # Save raw scores
    sub_sorted = sub.sort_values("overall", ascending=False).reset_index(drop=True)
    sub_sorted.to_csv(out_dir / "model_scores.csv", index=False, encoding="utf-8")

    # Full chart (may be wide)
    plot_model_3bar(
        scores=sub_sorted.rename(columns={model_col: model_col}),
        model_col=model_col,
        out_path=out_dir / "model_3bar.png",
        title=f"{chart_title_prefix} | scenario={scenario_id} (all models)",
        max_models=None,
    )

    # Top-K chart (recommended for readability)
    plot_model_3bar(
        scores=sub_sorted.rename(columns={model_col: model_col}),
        model_col=model_col,
        out_path=out_dir / "top_models_3bar.png",
        title=f"{chart_title_prefix} | scenario={scenario_id} (top {top_k_models} by overall)",
        max_models=top_k_models,
    )


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Merged judge scores CSV (e.g. results/merged/all_judge_scores.csv)")
    ap.add_argument("--out", default="results/analysis/scenario_value_mining", help="Output directory")
    ap.add_argument("--top_n", type=int, default=5, help="How many scenarios to export per type")
    ap.add_argument("--top_k_models", type=int, default=20, help="Top K models to show in the readable chart")
    ap.add_argument("--min_models", type=int, default=3, help="Scenario must have at least this many models scored")
    ap.add_argument("--gap_threshold", type=float, default=0.35,
                    help="For imbalanced scenarios: abs(constructive_mean - responsible_mean) >= threshold (in score units)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    safe_mkdir(out_dir)

    df = pd.read_csv(args.csv)
    scenario_col, model_col = detect_cols(df)

    # Keep only needed columns
    keep_cols = [scenario_col, model_col] + [c for c in ALL if c in df.columns]
    missing_cols = [c for c in ALL if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing rubric columns in CSV: {missing_cols}")
    df = df[keep_cols].copy()

    # Compute per (scenario, model) scores
    df_scores = compute_sample_scores(df, scenario_col, model_col)

    # Filter scenarios with too few models
    model_counts = df_scores.groupby(scenario_col)[model_col].nunique().rename("n_models")
    df_scores = df_scores.merge(model_counts, on=scenario_col, how="left")
    df_scores = df_scores[df_scores["n_models"] >= args.min_models].copy()

    # Scenario-level stats
    scen = df_scores.groupby(scenario_col).agg(
        n_models=("n_models", "first"),
        constructive_mean=("constructive", "mean"),
        responsible_mean=("responsible", "mean"),
        overall_mean=("overall", "mean"),
        overall_spread=("overall", "std"),
        constructive_spread=("constructive", "std"),
        responsible_spread=("responsible", "std"),
    ).reset_index()

    # Difficulty: lower means harder (sum of means)
    scen["difficulty_sum"] = scen["constructive_mean"] + scen["responsible_mean"]
    scen["gap_abs"] = (scen["constructive_mean"] - scen["responsible_mean"]).abs()
    scen.to_csv(out_dir / "scenario_stats_all.csv", index=False, encoding="utf-8")

    # Capability map
    plt.figure(figsize=(7, 6))
    plt.scatter(scen["constructive_mean"], scen["responsible_mean"], alpha=0.5)
    plt.xlabel(f"{GROUP1_NAME} (mean across models)")
    plt.ylabel(f"{GROUP2_NAME} (mean across models)")
    plt.title("Scenario Capability Map")
    plt.tight_layout()
    plt.savefig(out_dir / "scenario_capability_map.png", dpi=200, bbox_inches="tight")
    plt.close()

    # -------------------------
    # Select top scenarios by type
    # -------------------------
    # 1) Constructive-weak: lowest constructive mean
    top_constructive_weak = scen.sort_values("constructive_mean", ascending=True).head(args.top_n)
    top_constructive_weak.to_csv(out_dir / "top_constructive_weak.csv", index=False, encoding="utf-8")

    # 2) Responsible-weak: lowest responsible mean
    top_responsible_weak = scen.sort_values("responsible_mean", ascending=True).head(args.top_n)
    top_responsible_weak.to_csv(out_dir / "top_responsible_weak.csv", index=False, encoding="utf-8")

    # 3) Polarizing: highest overall spread (most discriminative)
    top_polarizing = scen.sort_values("overall_spread", ascending=False).head(args.top_n)
    top_polarizing.to_csv(out_dir / "top_polarizing.csv", index=False, encoding="utf-8")

    # 4) Balanced-hard: lowest difficulty_sum (both low)
    top_balanced_hard = scen.sort_values("difficulty_sum", ascending=True).head(args.top_n)
    top_balanced_hard.to_csv(out_dir / "top_balanced_hard.csv", index=False, encoding="utf-8")

    # 5) Imbalanced: large gap between constructive and responsible
    top_imbalanced = scen[scen["gap_abs"] >= args.gap_threshold].sort_values("gap_abs", ascending=False).head(args.top_n)
    top_imbalanced.to_csv(out_dir / "top_imbalanced.csv", index=False, encoding="utf-8")

    # -------------------------
    # Export detailed per-scenario charts for each type
    # -------------------------
    def export_type(type_name: str, df_top: pd.DataFrame) -> None:
        type_dir = out_dir / type_name
        safe_mkdir(type_dir)

        # list file
        df_top.to_csv(type_dir / "scenarios_selected.csv", index=False, encoding="utf-8")

        for sid in df_top[scenario_col].tolist():
            export_scenario_detail_charts(
                df_scores=df_scores,
                scenario_id=sid,
                scenario_col=scenario_col,
                model_col=model_col,
                out_dir=type_dir / f"scenario={sid}",
                chart_title_prefix=type_name,
                top_k_models=args.top_k_models,
            )

    export_type("Constructive_weak", top_constructive_weak)
    export_type("Responsible_weak", top_responsible_weak)
    export_type("Polarizing", top_polarizing)
    export_type("Balanced_hard", top_balanced_hard)
    export_type("Imbalanced", top_imbalanced)

    # README
    with (out_dir / "README.txt").open("w", encoding="utf-8") as f:
        f.write("High-value scenario mining outputs.\n\n")
        f.write(f"Groups:\n- {GROUP1_NAME}: {', '.join(CONSTRUCTIVE)}\n- {GROUP2_NAME}: {', '.join(RESPONSIBLE)}\n\n")
        f.write("Top lists:\n")
        f.write("- top_constructive_weak.csv\n")
        f.write("- top_responsible_weak.csv\n")
        f.write("- top_polarizing.csv\n")
        f.write("- top_balanced_hard.csv\n")
        f.write("- top_imbalanced.csv\n\n")
        f.write("Per-type folders contain per-scenario subfolders with:\n")
        f.write("- model_scores.csv\n- model_3bar.png (all models)\n- top_models_3bar.png (top-K)\n")
        f.write("\nNotes:\n")
        f.write("- Constructive_weak: lowest constructive_mean across models\n")
        f.write("- Responsible_weak: lowest responsible_mean across models\n")
        f.write("- Polarizing: highest overall_spread across models\n")
        f.write("- Balanced_hard: lowest (constructive_mean + responsible_mean)\n")
        f.write("- Imbalanced: abs(constructive_mean - responsible_mean) >= gap_threshold\n")

    print("[DONE] Outputs in:", out_dir.resolve())


if __name__ == "__main__":
    main()
