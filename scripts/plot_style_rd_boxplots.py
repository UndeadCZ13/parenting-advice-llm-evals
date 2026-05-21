#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import gridspec


DEFAULT_INPUT_DIR = Path("data/judge_outputs/style_2d")
DEFAULT_OUTPUT_DIR = Path("results/analysis/style_2d_plots")

PRIMARY_JUDGE = "gpt-5.2"

MODEL_ORDER = [
    "gpt-5.2",
    "gpt-5-nano",
    "kimi_k2",
    "deepseek_v3",
    "gpt_oss",
    "minimax_m2",
    "ministral3_14b",
    "qwen3_32b",
    "qwen3_8b",
    "llama33_70b",
    "glm46",
    "gpt-4o-mini",
    "llama31_8b",
    "glm4_9b",
    "deepseek_r1",
]

MODEL_LABELS = {
    "gpt-5.2": "GPT-5.2",
    "gpt-5-nano": "GPT-5 Nano",
    "gpt-4o-mini": "GPT-4o Mini",
    "gpt_oss": "GPT-OSS 20B",
    "deepseek_v3": "DeepSeek V3.1",
    "deepseek_r1": "DeepSeek R1",
    "kimi_k2": "Kimi K2 Thinking",
    "minimax_m2": "MiniMax M2",
    "ministral3_14b": "Ministral 3 14B",
    "qwen3_32b": "Qwen3 32B",
    "qwen3_8b": "Qwen3 8B",
    "llama31_8b": "Llama 3.1 8B",
    "llama33_70b": "Llama 3.3 70B",
    "glm4_9b": "GLM-4 9B",
    "glm46": "GLM-4.6",
}

MODEL_COLORS = {
    "gpt-5.2": "#2C5AA0",
    "gpt-5-nano": "#5B7DBE",
    "gpt-4o-mini": "#8CA7D8",
    "gpt_oss": "#6CA6CD",
    "deepseek_v3": "#2E8B8B",
    "deepseek_r1": "#66B2A3",
    "kimi_k2": "#8E63A9",
    "minimax_m2": "#C9A227",
    "ministral3_14b": "#8B6F47",
    "qwen3_32b": "#2E8B57",
    "qwen3_8b": "#77B255",
    "llama31_8b": "#E07A3F",
    "llama33_70b": "#B45F06",
    "glm4_9b": "#C44E52",
    "glm46": "#A23B72",
}


def normalise_model_name(raw: Any, source_path: Path | None = None) -> str:
    s = str(raw or "").strip()
    if not s and source_path is not None:
        s = source_path.name.split("__dims__judge=", 1)[0]
    s = re.sub(r"^(en|zh)_", "", s)
    s = re.sub(r"^(openai|ollama|groq)_", "", s)
    return s


def display_name(model: str) -> str:
    return MODEL_LABELS.get(model, model.replace("_", " "))


def ordered_models(present: list[str]) -> list[str]:
    present_set = set(present)
    ordered = [m for m in MODEL_ORDER if m in present_set]
    ordered += sorted(m for m in present_set if m not in ordered)
    return ordered


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
    return df.dropna(subset=["scenario_uid", "answer_model", "judge_model", "responsiveness", "demandingness"])


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby("answer_model", dropna=False)
        .agg(
            n=("scenario_uid", "size"),
            responsiveness_mean=("responsiveness", "mean"),
            responsiveness_median=("responsiveness", "median"),
            responsiveness_std=("responsiveness", "std"),
            demandingness_mean=("demandingness", "mean"),
            demandingness_median=("demandingness", "median"),
            demandingness_std=("demandingness", "std"),
        )
        .reset_index()
    )
    out["display_name"] = out["answer_model"].map(display_name)
    order = {m: i for i, m in enumerate(ordered_models(out["answer_model"].astype(str).tolist()))}
    out["model_order"] = out["answer_model"].map(order)
    return out.sort_values("model_order").drop(columns=["model_order"])


def apply_base_theme() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 20,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 12,
        }
    )


def metric_summary(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in ordered_models(df["answer_model"].dropna().astype(str).unique().tolist()):
        scores = df.loc[df["answer_model"].astype(str) == model, metric].dropna()
        if scores.empty:
            continue
        rows.append(
            {
                "model": model,
                "display_name": display_name(model),
                "mean": round(float(scores.mean()), 3),
                "median": round(float(scores.median()), 3),
                "std": round(float(scores.std(ddof=1)), 3),
                "n": int(scores.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def make_boxplot_with_table(df: pd.DataFrame, metric: str, metric_label: str, out_png: Path, title: str) -> pd.DataFrame:
    apply_base_theme()
    summary = metric_summary(df, metric)

    fig = plt.figure(figsize=(17.2, 10.2), facecolor="white")
    gs = gridspec.GridSpec(1, 2, width_ratios=[3.72, 1.88], wspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    ax_tbl = fig.add_subplot(gs[0, 1])

    plot_rows = []
    for _, row in summary.iterrows():
        model = row["model"]
        scores = df.loc[df["answer_model"].astype(str) == model, metric].dropna()
        plot_rows.append((model, scores))

    data = [scores.values for _, scores in plot_rows]
    labels = [display_name(model) for model, _ in plot_rows]
    positions = list(range(1, len(plot_rows) + 1))

    bp = ax.boxplot(
        data,
        vert=False,
        positions=positions,
        widths=0.62,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#1f1f1f", "linewidth": 1.6},
        whiskerprops={"color": "#444444", "linewidth": 1.2},
        capprops={"color": "#444444", "linewidth": 1.2},
        boxprops={"edgecolor": "#444444", "linewidth": 1.2},
    )

    for patch, (model, _) in zip(bp["boxes"], plot_rows):
        patch.set_facecolor(MODEL_COLORS.get(model, "#8C8C8C"))
        patch.set_alpha(0.9)

    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel(metric_label)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for pos, (_, row) in zip(positions, summary.iterrows()):
        ax.plot(row["mean"], pos, marker="o", markersize=4.8, color="#111111", zorder=4)

    fig.suptitle(title, fontsize=20, y=0.975)

    ax_tbl.axis("off")
    ax_tbl.set_title("Numeric Summary", fontsize=15, pad=10)
    cell_text = [
        [row["display_name"], f"{row['mean']:.2f}", f"{row['median']:.2f}"]
        for _, row in summary.iterrows()
    ]
    table = ax_tbl.table(
        cellText=cell_text,
        colLabels=["Model", "Mean", "Median"],
        cellLoc="left",
        colLoc="left",
        loc="center",
        bbox=[0.0, 0.02, 1.0, 0.92],
        colWidths=[0.42, 0.26, 0.32],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.0)

    summary_records = summary.to_dict("records")
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor("#F2F2F2")
            cell.set_text_props(weight="bold", color="#111111")
        else:
            cell.set_facecolor("white")
            idx = r - 1
            if c == 0 and 0 <= idx < len(summary_records):
                cell.set_text_props(
                    color=MODEL_COLORS.get(summary_records[idx]["model"], "#333333"),
                    weight="bold",
                )

    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.06, wspace=0.05)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot R/D boxplots by answer model using one style judge.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--judge-model", default=PRIMARY_JUDGE)
    parser.add_argument("--language", choices=["all", "en", "zh"], default="all")
    args = parser.parse_args()

    df = load_style_rows(args.input_dir)
    df = df[df["judge_model"].astype(str) == args.judge_model].copy()
    if args.language != "all":
        df = df[df["language"].astype(str) == args.language].copy()
    if df.empty:
        raise SystemExit("No rows left after filtering. Check --judge-model and --language.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lang_label = "English + Chinese" if args.language == "all" else args.language.upper()
    stem = f"rd_boxplots_by_model__judge={args.judge_model.replace(':', '_')}__language={args.language}"
    out_rows = args.output_dir / f"{stem}__rows.csv"
    out_summary = args.output_dir / f"{stem}__summary.csv"

    df.to_csv(out_rows, index=False)
    summarise(df).to_csv(out_summary, index=False)
    for metric, metric_label in [
        ("responsiveness", "Responsiveness"),
        ("demandingness", "Demandingness"),
    ]:
        metric_stem = (
            f"rd_boxplot_{metric}_by_model__judge={args.judge_model.replace(':', '_')}"
            f"__language={args.language}"
        )
        metric_png = args.output_dir / f"{metric_stem}.png"
        metric_summary_csv = args.output_dir / f"{metric_stem}__summary.csv"
        metric_summary_df = make_boxplot_with_table(
            df,
            metric,
            metric_label,
            metric_png,
            title=f"{metric_label} by Model ({lang_label}; judge={args.judge_model})",
        )
        metric_summary_df.to_csv(metric_summary_csv, index=False)
        print(f"[DONE] plot: {metric_png}")
        print(f"[DONE] summary: {metric_summary_csv}")

    print(f"[DONE] rows: {out_rows}")
    print(f"[DONE] combined summary: {out_summary}")


if __name__ == "__main__":
    main()
