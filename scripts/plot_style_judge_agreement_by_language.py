#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT_DIR = Path("data/judge_outputs/style_2d")
DEFAULT_OUTPUT_DIR = Path("results/analysis/language_separated_judge_agreement")

PRIMARY_JUDGE = "gpt-5.2"
SECONDARY_JUDGE = "deepseek-v3.1:671b-cloud"

LANGUAGE_LABELS = {"en": "English", "zh": "Chinese"}
LANGUAGE_COLORS = {"en": "#4C78A8", "zh": "#C44E52"}


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._=-]+", "_", value)


def normalise_model_name(raw: Any, source_path: Path | None = None) -> str:
    s = str(raw or "").strip()
    if not s and source_path is not None:
        s = source_path.name.split("__dims__judge=", 1)[0]
    s = re.sub(r"^(en|zh)_", "", s)
    s = re.sub(r"^(openai|ollama|groq)_", "", s)
    return s


def load_style_rows(input_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("*__dims__judge=*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                rows.append(
                    {
                        "source_file": path.name,
                        "scenario_uid": rec.get("scenario_uid"),
                        "language": rec.get("language"),
                        "answer_model": normalise_model_name(rec.get("model_name"), path),
                        "judge_model": rec.get("judge_model"),
                        "judge_backend": rec.get("judge_backend"),
                        "responsiveness": pd.to_numeric(rec.get("responsiveness"), errors="coerce"),
                        "demandingness": pd.to_numeric(rec.get("demandingness"), errors="coerce"),
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No style judge rows found under {input_dir}")
    return df.dropna(subset=["scenario_uid", "answer_model", "language", "judge_model", "responsiveness", "demandingness"])


def prepare_points(df: pd.DataFrame, primary_judge: str, secondary_judge: str) -> pd.DataFrame:
    grouped = (
        df[df["judge_model"].isin([primary_judge, secondary_judge])]
        .groupby(["scenario_uid", "answer_model", "language", "judge_model"], dropna=False)[["responsiveness", "demandingness"]]
        .mean()
        .reset_index()
    )
    pivots: list[pd.DataFrame] = []
    for metric in ["responsiveness", "demandingness"]:
        pivot = grouped.pivot_table(
            index=["scenario_uid", "answer_model", "language"],
            columns="judge_model",
            values=metric,
            aggfunc="mean",
        )
        if primary_judge not in pivot.columns or secondary_judge not in pivot.columns:
            raise SystemExit(f"Missing one of the requested judges for {metric}: {primary_judge}, {secondary_judge}")
        pivots.append(
            pivot[[primary_judge, secondary_judge]]
            .dropna()
            .reset_index()
            .rename(columns={primary_judge: f"{metric}_primary", secondary_judge: f"{metric}_secondary"})
        )
    points = pivots[0].merge(pivots[1], on=["scenario_uid", "answer_model", "language"], how="inner")
    if points.empty:
        raise SystemExit("No paired judge-agreement points could be created.")
    return points


def pearson_r(x: pd.Series, y: pd.Series) -> float:
    valid = pd.concat([x, y], axis=1).dropna()
    if len(valid) < 2:
        return float("nan")
    return float(valid.iloc[:, 0].corr(valid.iloc[:, 1]))


def metric_limits(points: pd.DataFrame) -> tuple[float, float]:
    cols = [c for c in points.columns if c.endswith("_primary") or c.endswith("_secondary")]
    vals = points[cols].to_numpy().ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    lo = max(0.0, float(vals.min()) - 0.04)
    hi = min(1.0, float(vals.max()) + 0.04)
    return lo, hi


def write_stats(points: pd.DataFrame, out_csv: Path) -> None:
    rows = []
    for metric in ["responsiveness", "demandingness"]:
        x_col = f"{metric}_primary"
        y_col = f"{metric}_secondary"
        rows.append(
            {
                "metric": metric,
                "group": "all",
                "n": int(points[[x_col, y_col]].dropna().shape[0]),
                "pearson_r": pearson_r(points[x_col], points[y_col]),
                "mae": float((points[x_col] - points[y_col]).abs().mean()),
                "primary_mean": float(points[x_col].mean()),
                "secondary_mean": float(points[y_col].mean()),
            }
        )
        for lang in sorted(points["language"].dropna().astype(str).unique()):
            sub = points[points["language"].astype(str) == lang]
            rows.append(
                {
                    "metric": metric,
                    "group": lang,
                    "n": int(sub[[x_col, y_col]].dropna().shape[0]),
                    "pearson_r": pearson_r(sub[x_col], sub[y_col]),
                    "mae": float((sub[x_col] - sub[y_col]).abs().mean()),
                    "primary_mean": float(sub[x_col].mean()),
                    "secondary_mean": float(sub[y_col].mean()),
                }
            )
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def select_points(points: pd.DataFrame, language: str) -> pd.DataFrame:
    if language == "en_zh":
        return points.copy()
    return points[points["language"].astype(str) == language].copy()


def language_title(language: str) -> str:
    if language == "en_zh":
        return "English and Chinese"
    return LANGUAGE_LABELS.get(language, language)


def make_metric_plot(points: pd.DataFrame, metric: str, metric_label: str, language: str, out_png: Path, primary_judge: str, secondary_judge: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.8), facecolor="white")
    x_col = f"{metric}_primary"
    y_col = f"{metric}_secondary"
    metric_points = select_points(points, language)[["scenario_uid", "answer_model", "language", x_col, y_col]].dropna().copy()
    if metric_points.empty:
        raise SystemExit(f"No points available for metric={metric}, language={language}")
    lo, hi = metric_limits(metric_points)
    for lang in sorted(metric_points["language"].dropna().astype(str).unique()):
        sub = metric_points[metric_points["language"].astype(str) == lang]
        ax.scatter(
            sub[x_col],
            sub[y_col],
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
        f"Pearson r = {pearson_r(metric_points[x_col], metric_points[y_col]):.3f}\nMAE = {(metric_points[x_col] - metric_points[y_col]).abs().mean():.2f}\nn = {len(metric_points)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(primary_judge)
    ax.set_ylabel(secondary_judge)
    ax.set_title(f"Judge Agreement | {metric_label} | {language_title(language)}", pad=12)
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
    parser = argparse.ArgumentParser(description="Plot language-separated R/D judge agreement.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--primary-judge", default=PRIMARY_JUDGE)
    parser.add_argument("--secondary-judge", default=SECONDARY_JUDGE)
    args = parser.parse_args()

    points = prepare_points(load_style_rows(args.input_dir), args.primary_judge, args.secondary_judge)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"style_judge_agreement_by_language__primary={safe_name(args.primary_judge)}"
        f"__secondary={safe_name(args.secondary_judge)}"
    )
    out_points = args.output_dir / f"{stem}__points.csv"
    out_stats = args.output_dir / f"{stem}__stats.csv"

    points.to_csv(out_points, index=False)
    write_stats(points, out_stats)
    for metric, label in [("responsiveness", "Responsiveness"), ("demandingness", "Demandingness")]:
        for language in ["en", "zh", "en_zh"]:
            out_png = args.output_dir / f"{stem}__metric={metric}__language={language}.png"
            make_metric_plot(points, metric, label, language, out_png, args.primary_judge, args.secondary_judge)
            print(f"[DONE] plot: {out_png}")
    print(f"[DONE] points: {out_points}")
    print(f"[DONE] stats: {out_stats}")


if __name__ == "__main__":
    main()
