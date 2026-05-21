#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("results/merged/all_judge_scores.csv")
DEFAULT_OUTPUT_DIR = Path("results/analysis/language_separated_judge_agreement")

PRIMARY_JUDGE = "gpt-5.2"
SECONDARY_JUDGE = "deepseek-v3.1:671b-cloud"

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

LANGUAGE_LABELS = {"en": "English", "zh": "Chinese"}
LANGUAGE_COLORS = {"en": "#4C78A8", "zh": "#C44E52"}


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._=-]+", "_", value)


def load_scores(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in ["scenario_uid", "answer_model", "language", "judge_model", *RUBRICS] if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")
    df["overall_mean"] = df[RUBRICS].mean(axis=1)
    return df.dropna(subset=["scenario_uid", "answer_model", "language", "judge_model", "overall_mean"])


def prepare_points(df: pd.DataFrame, primary_judge: str, secondary_judge: str) -> pd.DataFrame:
    grouped = (
        df[df["judge_model"].isin([primary_judge, secondary_judge])]
        .groupby(["scenario_uid", "answer_model", "language", "judge_model"], dropna=False)["overall_mean"]
        .mean()
        .reset_index()
    )
    pivot = grouped.pivot_table(
        index=["scenario_uid", "answer_model", "language"],
        columns="judge_model",
        values="overall_mean",
        aggfunc="mean",
    )
    if primary_judge not in pivot.columns or secondary_judge not in pivot.columns:
        raise SystemExit(f"Missing one of the requested judges: {primary_judge}, {secondary_judge}")
    return (
        pivot[[primary_judge, secondary_judge]]
        .dropna()
        .reset_index()
        .rename(columns={primary_judge: "primary", secondary_judge: "secondary"})
    )


def pearson_r(x: pd.Series, y: pd.Series) -> float:
    valid = pd.concat([x, y], axis=1).dropna()
    if len(valid) < 2:
        return float("nan")
    return float(valid.iloc[:, 0].corr(valid.iloc[:, 1]))


def limits(points: pd.DataFrame) -> tuple[float, float]:
    vals = points[["primary", "secondary"]].to_numpy().ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 100.0
    span = max(float(vals.max()) - float(vals.min()), 1.0)
    pad = span * 0.05
    return max(0.0, float(vals.min()) - pad), min(100.0, float(vals.max()) + pad)


def select_points(points: pd.DataFrame, language: str) -> pd.DataFrame:
    if language == "en_zh":
        return points.copy()
    return points[points["language"].astype(str) == language].copy()


def language_title(language: str) -> str:
    if language == "en_zh":
        return "English and Chinese"
    return LANGUAGE_LABELS.get(language, language)


def write_stats(points: pd.DataFrame, out_csv: Path) -> None:
    rows = []
    for language in ["en", "zh", "en_zh"]:
        sub = select_points(points, language)
        rows.append(
            {
                "language": language,
                "n": int(sub.shape[0]),
                "pearson_r": pearson_r(sub["primary"], sub["secondary"]),
                "mae": float((sub["primary"] - sub["secondary"]).abs().mean()),
                "primary_mean": float(sub["primary"].mean()),
                "secondary_mean": float(sub["secondary"].mean()),
            }
        )
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def make_plot(points: pd.DataFrame, language: str, out_png: Path, primary_judge: str, secondary_judge: str) -> None:
    plot_points = select_points(points, language)
    if plot_points.empty:
        raise SystemExit(f"No points available for language={language}")
    fig, ax = plt.subplots(figsize=(7.2, 6.8), facecolor="white")
    lo, hi = limits(plot_points)
    for lang in sorted(plot_points["language"].dropna().astype(str).unique()):
        sub = plot_points[plot_points["language"].astype(str) == lang]
        ax.scatter(
            sub["primary"],
            sub["secondary"],
            s=13,
            alpha=0.52,
            linewidths=0,
            color=LANGUAGE_COLORS.get(lang, "#777777"),
            label=LANGUAGE_LABELS.get(lang, lang),
        )
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.1, color="#555555", alpha=0.9)
    ax.text(
        0.03,
        0.96,
        f"Pearson r = {pearson_r(plot_points['primary'], plot_points['secondary']):.3f}\nMAE = {(plot_points['primary'] - plot_points['secondary']).abs().mean():.2f}\nn = {len(plot_points)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(primary_judge)
    ax.set_ylabel(secondary_judge)
    ax.set_title(f"Judge Agreement | Overall Rubric Score | {language_title(language)}", pad=12)
    ax.grid(color="#E3E3E3", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if language == "en_zh":
        ax.legend(frameon=False, loc="lower right", title="Language")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot language-separated rubric judge agreement.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--primary-judge", default=PRIMARY_JUDGE)
    parser.add_argument("--secondary-judge", default=SECONDARY_JUDGE)
    args = parser.parse_args()

    points = prepare_points(load_scores(args.input), args.primary_judge, args.secondary_judge)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"rubric_judge_agreement_by_language__primary={safe_name(args.primary_judge)}"
        f"__secondary={safe_name(args.secondary_judge)}"
    )
    out_points = args.output_dir / f"{stem}__points.csv"
    out_stats = args.output_dir / f"{stem}__stats.csv"

    points.to_csv(out_points, index=False)
    write_stats(points, out_stats)
    for language in ["en", "zh", "en_zh"]:
        out_png = args.output_dir / f"{stem}__language={language}.png"
        make_plot(points, language, out_png, args.primary_judge, args.secondary_judge)
        print(f"[DONE] plot: {out_png}")
    print(f"[DONE] points: {out_points}")
    print(f"[DONE] stats: {out_stats}")


if __name__ == "__main__":
    main()
