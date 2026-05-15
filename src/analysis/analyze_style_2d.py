# src/analysis/analyze_style_2d.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


STYLE_KEYS = ["authoritative", "authoritarian", "permissive", "neglectful"]


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sanitize_filename(s: str) -> str:
    s = str(s)
    for ch in ['/', '\\', ':', '|', '*', '?', '"', '<', '>', ' ']:
        s = s.replace(ch, "_")
    return s


def short_model_name(name: str) -> str:
    """
    Convert names like:
        en_openai_gpt-5.2
        zh_openai_gpt-4o
        en_groq_llama33_70b
    to:
        gpt-5.2
        gpt-4o
        llama33_70b
    Heuristic: split by '_' and take the last segment when pattern looks like lang_backend_model.
    """
    if not isinstance(name, str):
        return str(name)
    parts = name.split("_")
    ans=""
    for i in range(len(parts)):
        if i>=2:
            ans+=parts[i]
    return ans


def display_model_name(row: pd.Series) -> str:
    if "canonical_model" in row and isinstance(row["canonical_model"], str) and row["canonical_model"].strip():
        return str(row["canonical_model"]).strip()
    if "model_name_short" in row and isinstance(row["model_name_short"], str) and row["model_name_short"].strip():
        return str(row["model_name_short"]).strip()
    return short_model_name(row.get("model_name", ""))


def plot_rd_scatter_means(df: pd.DataFrame, out_dir: Path) -> None:
    safe_mkdir(out_dir)
    grp = (
        df.groupby(["model_name", "judge_model", "language"], dropna=False)[["responsiveness", "demandingness"]]
        .mean()
        .reset_index()
    )
    if "canonical_model" in grp.columns:
        grp["model_name_short"] = grp["canonical_model"].fillna(grp["model_name"]).astype(str)
    else:
        grp["model_name_short"] = grp["model_name"].apply(short_model_name)

    judges = sorted(grp["judge_model"].dropna().unique().tolist())
    langs = sorted(grp["language"].dropna().unique().tolist())

    for lang in langs:
        for j in judges:
            sub = grp[(grp["language"] == lang) & (grp["judge_model"] == j)].copy()
            if sub.empty:
                continue

            plt.figure(figsize=(7, 7))
            plt.scatter(sub["demandingness"].values, sub["responsiveness"].values, s=35)
            for _, r in sub.iterrows():
                plt.text(
                    float(r["demandingness"]),
                    float(r["responsiveness"]),
                    str(r["model_name_short"]),
                    fontsize=8,
                )

            plt.axhline(0.5, linestyle="--", linewidth=1)
            plt.axvline(0.5, linestyle="--", linewidth=1)

            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.xlabel("Demandingness (D)")
            plt.ylabel("Responsiveness (R)")
            plt.title(f"Mean (R,D) per Model | lang={lang} | judge={j}")
            plt.tight_layout()
            plt.savefig(out_dir / f"scatter_mean_rd__lang={lang}__judge={sanitize_filename(j)}.png", dpi=180)
            plt.close()


def plot_boxplots_rd(df: pd.DataFrame, out_dir: Path) -> None:
    safe_mkdir(out_dir)

    judges = sorted(df["judge_model"].dropna().unique().tolist())
    langs = sorted(df["language"].dropna().unique().tolist())
    group_col = "canonical_model" if "canonical_model" in df.columns else "model_name"
    models = sorted(df[group_col].dropna().unique().tolist())

    for lang in langs:
        for j in judges:
            sub = df[(df["language"] == lang) & (df["judge_model"] == j)].copy()
            if sub.empty:
                continue

            for metric in ["responsiveness", "demandingness"]:
                data, labels = [], []
                for m in models:
                    s = sub[sub[group_col] == m][metric].dropna()
                    if len(s) == 0:
                        continue
                    data.append(s.values)
                    labels.append(str(m))

                if len(data) < 2:
                    continue

                plt.figure(figsize=(max(10, 0.4 * len(labels)), 6))
                plt.boxplot(data, tick_labels=labels, showfliers=False)
                plt.xticks(rotation=45, ha="right")
                plt.ylim(0, 1)
                plt.title(f"Boxplot | {metric} | lang={lang} | judge={j}")
                plt.tight_layout()
                plt.savefig(out_dir / f"boxplot_{metric}__lang={lang}__judge={sanitize_filename(j)}.png", dpi=180)
                plt.close()


def plot_style_bars(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Stacked bar of mean style probs per model, with numeric labels on each stack segment.
    """
    safe_mkdir(out_dir)

    group_col = "canonical_model" if "canonical_model" in df.columns else "model_name"
    grp = df.groupby([group_col, "judge_model", "language"], dropna=False)[STYLE_KEYS].mean().reset_index()
    grp["model_name_short"] = grp[group_col].astype(str)

    judges = sorted(grp["judge_model"].dropna().unique().tolist())
    langs = sorted(grp["language"].dropna().unique().tolist())

    for lang in langs:
        for j in judges:
            sub = grp[(grp["language"] == lang) & (grp["judge_model"] == j)].copy()
            if sub.empty:
                continue

            sub = sub.sort_values("authoritative", ascending=False)
            xs = sub["model_name_short"].astype(str).tolist()
            x_idx = np.arange(len(xs))

            plt.figure(figsize=(max(10, 0.6 * len(xs)), 6))
            bottom = np.zeros(len(xs), dtype=float)

            for k in STYLE_KEYS:
                vals = sub[k].values.astype(float)
                plt.bar(x_idx, vals, bottom=bottom, label=k)

                # numeric labels (skip tiny segments to avoid clutter)
                for i, v in enumerate(vals):
                    if v > 0.05:
                        plt.text(
                            x_idx[i],
                            bottom[i] + v / 2,
                            f"{v:.2f}",
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="black",
                        )

                bottom += vals

            plt.xticks(x_idx, xs, rotation=45, ha="right", fontsize=9)
            plt.ylim(0, 1)
            plt.title(f"Mean Style (derived from R,D) | lang={lang} | judge={j}")
            plt.ylabel("Mean probability")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f"mean_style_stacked__lang={lang}__judge={sanitize_filename(j)}.png", dpi=180)
            plt.close()


def plot_dominant_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Heatmap of dominant style share per model.
    """
    safe_mkdir(out_dir)

    judges = sorted(df["judge_model"].dropna().unique().tolist())
    langs = sorted(df["language"].dropna().unique().tolist())

    for lang in langs:
        for j in judges:
            sub = df[(df["language"] == lang) & (df["judge_model"] == j)].copy()
            if sub.empty:
                continue

            # Use short names for display
            sub = sub.copy()
            if "canonical_model" in sub.columns:
                sub["model_name_short"] = sub["canonical_model"].astype(str)
            else:
                sub["model_name_short"] = sub["model_name"].apply(short_model_name)

            tab = (
                sub.groupby("model_name_short")["dominant_style"]
                .value_counts(normalize=True)
                .rename("share")
                .reset_index()
            )
            piv = tab.pivot(index="model_name_short", columns="dominant_style", values="share").fillna(0.0)

            for k in STYLE_KEYS:
                if k not in piv.columns:
                    piv[k] = 0.0
            piv = piv[STYLE_KEYS]

            mat = piv.values.astype(float)

            plt.figure(figsize=(8, 0.45 * len(piv) + 2))
            ax = plt.gca()
            im = ax.imshow(mat, aspect="auto")

            ax.set_title(f"Dominant Style Share | lang={lang} | judge={j}")
            ax.set_yticks(range(len(piv.index)))
            ax.set_yticklabels(piv.index.astype(str).tolist(), fontsize=9)
            ax.set_xticks(range(len(piv.columns)))
            ax.set_xticklabels(piv.columns.astype(str).tolist(), rotation=30, ha="right", fontsize=9)

            for r_i in range(mat.shape[0]):
                for c_i in range(mat.shape[1]):
                    ax.text(c_i, r_i, f"{mat[r_i, c_i]:.2f}", ha="center", va="center", fontsize=8)

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.savefig(
                out_dir / f"dominant_share_heatmap__lang={lang}__judge={sanitize_filename(j)}.png",
                dpi=180,
                bbox_inches="tight",
            )
            plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-csv", default="results/analysis/style_2d_tables/style_2d_rows.csv")
    ap.add_argument("--out-dir", default="results/analysis")
    ap.add_argument("--language", default="all", choices=["all", "en", "zh"])
    ap.add_argument("--judge-model", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.rows_csv)
    if df.empty:
        raise ValueError("Empty CSV.")

    if args.judge_model:
        df = df[df["judge_model"].astype(str) == str(args.judge_model)].copy()

    if args.language != "all":
        df = df[df["language"] == args.language].copy()

    # Add short model name column (used in outputs/plots)
    if "canonical_model" in df.columns:
        df["model_name_short"] = df["canonical_model"].fillna(df["model_name"]).astype(str)
    else:
        df["model_name_short"] = df["model_name"].apply(short_model_name)

    out_root = Path(args.out_dir) / args.language / "style_2d"
    stats_dir = out_root / "stats"
    box_dir = out_root / "boxplots"
    scat_dir = out_root / "scatter"
    heat_dir = out_root / "heatmaps"
    bars_dir = out_root / "style_bars"
    for p in [stats_dir, box_dir, scat_dir, heat_dir, bars_dir]:
        safe_mkdir(p)

    # Save a quick model-level mean table (using short names for readability)
    model_mean = (
        df.groupby(["model_name_short", "judge_model", "language"], dropna=False)[
            ["responsiveness", "demandingness", "confidence"] + STYLE_KEYS
        ]
        .mean(numeric_only=True)
        .reset_index()
        .rename(columns={"model_name_short": "model_name"})
    )
    model_mean.to_csv(stats_dir / "model_means.csv", index=False, encoding="utf-8")

    # Plots (internally use short names)
    plot_rd_scatter_means(df.rename(columns={"model_name_short": "model_name_short"}), scat_dir)
    plot_boxplots_rd(df, box_dir)
    plot_style_bars(df, bars_dir)
    plot_dominant_heatmap(df, heat_dir)

    print(f"[DONE] outputs -> {out_root.resolve()}")


if __name__ == "__main__":
    main()
