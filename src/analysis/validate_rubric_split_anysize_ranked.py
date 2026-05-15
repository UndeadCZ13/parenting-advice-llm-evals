# src/analysis/validate_rubric_split_anysize_ranked.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import itertools
import json
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Rubrics
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

ANCHOR_RUBRIC = "accuracy"  # must be in TECH

# Your current default 4v4 split (kept for reference in outputs)
DEFAULT_TECH: List[str] = [
    "accuracy",
    "helpfulness",
    "completeness",
    "communication",
]
DEFAULT_HUMAN: List[str] = [
    "empathy",
    "bias_avoidance",
    "safety",
    "limitation_awareness",
]


# -------------------------
# Utils
# -------------------------
def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sanitize_filename(s: str) -> str:
    s = str(s)
    for ch in ['/', '\\', ':', '|', '*', '?', '"', '<', '>', ' ']:
        s = s.replace(ch, "_")
    return s


def detect_cols(df: pd.DataFrame) -> Tuple[str, str, str, Optional[str]]:
    scenario_col = None
    for c in ["scenario_uid", "scenario_id", "id"]:
        if c in df.columns:
            scenario_col = c
            break
    if not scenario_col:
        raise ValueError("Cannot find scenario column: expected scenario_uid/scenario_id/id")

    model_col = None
    for c in ["answer_model", "model"]:
        if c in df.columns:
            model_col = c
            break
    if not model_col:
        raise ValueError("Cannot find model column: expected answer_model/model")

    judge_col = None
    for c in ["judge_model", "grader_model"]:
        if c in df.columns:
            judge_col = c
            break
    if not judge_col:
        raise ValueError("Cannot find judge column: expected judge_model/grader_model")

    language_col = "language" if "language" in df.columns else None
    return scenario_col, model_col, judge_col, language_col


def aggregate_to_scenario_level(
    df: pd.DataFrame,
    scenario_col: str,
    model_col: str,
    judge_col: str,
    language_col: Optional[str],
    rubrics: List[str],
) -> pd.DataFrame:
    keys = [scenario_col, model_col, judge_col]
    if language_col:
        keys.append(language_col)

    avail = [r for r in rubrics if r in df.columns]
    if not avail:
        raise ValueError("No rubric columns found in CSV.")

    out = (
        df[keys + avail]
        .copy()
        .groupby(keys, dropna=False)[avail]
        .mean()
        .reset_index()
    )
    return out


def build_pairs(within: List[str]) -> List[Tuple[str, str]]:
    return [(a, b) for a, b in itertools.combinations(within, 2)]


def build_between_pairs(a: List[str], b: List[str]) -> List[Tuple[str, str]]:
    return [(x, y) for x in a for y in b]


def pairwise_mean_corr(corr: pd.DataFrame, pairs: List[Tuple[str, str]]) -> float:
    vals = []
    for a, b in pairs:
        if a in corr.columns and b in corr.columns:
            v = corr.loc[a, b]
            if pd.notna(v):
                vals.append(float(v))
    return float(np.mean(vals)) if vals else float("nan")


def compute_signal(corr: pd.DataFrame, tech: List[str], human: List[str]) -> Dict[str, float]:
    """
    within_tech: mean of pairwise correlations inside tech (NaN if tech size < 2)
    within_human: mean of pairwise correlations inside human (NaN if human size < 2)
    between_groups: mean correlation across tech x human pairs (NaN only if a side empty, which we disallow)
    within_avg: mean([within_tech, within_human]) ignoring NaN
    signal: within_avg - between_groups
    """
    tech_pairs = build_pairs(tech)
    human_pairs = build_pairs(human)
    between_pairs = build_between_pairs(tech, human)

    within_tech = pairwise_mean_corr(corr, tech_pairs)  # NaN if <2
    within_human = pairwise_mean_corr(corr, human_pairs)  # NaN if <2
    between = pairwise_mean_corr(corr, between_pairs)  # should exist if both non-empty

    within_avg = float(np.nanmean([within_tech, within_human]))
    signal = within_avg - between

    return {
        "within_tech": within_tech,
        "within_human": within_human,
        "between_groups": between,
        "within_avg": within_avg,
        "signal": signal,
    }


def enumerate_anysize_splits_anchor(
    rubrics: List[str],
    anchor: str,
    allow_all_tech: bool = False,
) -> List[Tuple[Tuple[str, ...], Tuple[str, ...]]]:
    """
    Enumerate all splits where:
    - anchor must be in TECH
    - TECH non-empty
    - HUMAN non-empty (unless allow_all_tech=True)
    This yields 2^(n-1)-1 = 127 splits for n=8 (default).
    """
    rubrics = [r for r in rubrics]
    if anchor not in rubrics:
        raise ValueError(f"Anchor rubric '{anchor}' not found in rubrics list.")

    others = [r for r in rubrics if r != anchor]
    out = []

    # choose any subset of "others" to join TECH
    # tech = {anchor} ∪ subset
    # human = remaining others not in subset
    # exclude subset == all_others if not allow_all_tech (would make human empty)
    for r in range(0, len(others) + 1):
        for subset in itertools.combinations(others, r):
            subset_set = set(subset)
            human = [x for x in others if x not in subset_set]
            if (not allow_all_tech) and (len(human) == 0):
                continue
            tech = [anchor] + list(subset)
            if len(tech) == 0 or len(human) == 0:
                continue
            out.append((tuple(sorted(tech)), tuple(sorted(human))))

    # stable order
    out = sorted(out, key=lambda th: (len(th[0]), ",".join(th[0]), ",".join(th[1])))
    return out


def split_id(tech: Tuple[str, ...], human: Tuple[str, ...]) -> str:
    return "TECH:[" + ",".join(tech) + "]__HUMAN:[" + ",".join(human) + "]"


def is_default_split(tech: Tuple[str, ...], human: Tuple[str, ...]) -> bool:
    return (set(tech) == set(DEFAULT_TECH) and set(human) == set(DEFAULT_HUMAN)) or (
        set(tech) == set(DEFAULT_HUMAN) and set(human) == set(DEFAULT_TECH)
    )


def plot_split_comparison_bar(
    rows: List[Dict[str, float]],
    labels: List[str],
    out_path: Path,
    title: str,
) -> None:
    """
    Compare DEFAULT vs top alternatives: within_tech, within_human, between, signal
    """
    if not rows:
        return

    x = np.arange(len(rows))
    within_tech = np.array([r["within_tech"] for r in rows], dtype=float)
    within_human = np.array([r["within_human"] for r in rows], dtype=float)
    between = np.array([r["between_groups"] for r in rows], dtype=float)
    signal = np.array([r["signal"] for r in rows], dtype=float)

    plt.figure(figsize=(10, 5))
    ax = plt.gca()

    w = 0.18
    ax.bar(x - 1.5 * w, within_tech, width=w, label="within_tech")
    ax.bar(x - 0.5 * w, within_human, width=w, label="within_human")
    ax.bar(x + 0.5 * w, between, width=w, label="between_groups")
    ax.bar(x + 1.5 * w, signal, width=w, label="signal (within_avg - between)")

    ax.set_title(title)
    ax.set_ylabel("Pearson correlation (avg)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(0, linewidth=1)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_corr_heatmap(corr: pd.DataFrame, out_path: Path, title: str) -> None:
    cols = corr.columns.astype(str).tolist()
    mat = corr.values.astype(float)

    plt.figure(figsize=(7.5, 6.5))
    ax = plt.gca()
    im = ax.imshow(mat, aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_title(title)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols, fontsize=9)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Merged judge scores CSV path (e.g. results/merged/all_judge_scores.csv)")
    ap.add_argument("--out", type=str, default="results/analysis/rubric_split_anysize_ranked", help="Output directory")
    ap.add_argument("--judge", type=str, default="", help="Optional: filter by judge_model")
    ap.add_argument("--language", type=str, default="all", help="Optional: all|en|zh (requires language column)")
    ap.add_argument("--min_scenarios", type=int, default=20, help="Skip model if < this many scenarios")
    ap.add_argument("--topk_alt", type=int, default=2, help="How many best alternative splits (besides default if found) to export")
    ap.add_argument("--allow_all_tech", action="store_true", help="Allow TECH=all rubrics (HUMAN empty). Default: False.")
    args = ap.parse_args()

    in_path = Path(args.csv)
    out_dir = Path(args.out)
    safe_mkdir(out_dir)

    df = pd.read_csv(in_path)
    scenario_col, model_col, judge_col, language_col = detect_cols(df)

    avail_rubrics = [r for r in RUBRICS if r in df.columns]
    if not avail_rubrics:
        raise ValueError("No rubric columns found. Expected: " + ", ".join(RUBRICS))
    if ANCHOR_RUBRIC not in avail_rubrics:
        raise ValueError(f"Anchor rubric '{ANCHOR_RUBRIC}' not found in CSV columns.")

    # Filters
    if args.judge.strip():
        df = df[df[judge_col].astype(str) == args.judge.strip()].copy()

    lang = args.language.strip().lower()
    if lang != "all":
        if not language_col:
            print("[WARN] CSV has no 'language' column; language filter skipped.")
        else:
            df = df[df[language_col].astype(str).str.lower().str.strip() == lang].copy()

    # Scenario-level aggregation
    agg = aggregate_to_scenario_level(
        df=df,
        scenario_col=scenario_col,
        model_col=model_col,
        judge_col=judge_col,
        language_col=language_col if (language_col and lang != "all") else None,
        rubrics=avail_rubrics,
    )

    models = sorted(agg[model_col].dropna().astype(str).unique().tolist())
    if not models:
        raise ValueError("No models found after filtering.")

    # Enumerate all splits once
    splits = enumerate_anysize_splits_anchor(
        rubrics=sorted(avail_rubrics),
        anchor=ANCHOR_RUBRIC,
        allow_all_tech=args.allow_all_tech,
    )
    # Should be 127 when allow_all_tech=False
    all_splits_path = out_dir / "all_splits_anchor_accuracy.json"
    with all_splits_path.open("w", encoding="utf-8") as f:
        json.dump(
            [
                {"tech": list(t), "human": list(h), "tech_size": len(t), "human_size": len(h), "id": split_id(t, h)}
                for t, h in splits
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Output dirs
    per_model_dir = out_dir / "per_model"
    safe_mkdir(per_model_dir)

    # For global aggregation
    global_rows = []

    for m in models:
        sub = agg[agg[model_col].astype(str) == m].copy()

        # average over judges per scenario (if multiple judges exist)
        sub2 = sub.groupby(scenario_col, dropna=False)[avail_rubrics].mean()
        if sub2.shape[0] < args.min_scenarios:
            continue

        corr = sub2.corr(method="pearson")

        model_out = per_model_dir / f"model={sanitize_filename(m)}"
        safe_mkdir(model_out)

        corr.to_csv(model_out / "corr_matrix.csv", index=True)
        plot_corr_heatmap(corr, model_out / "corr_matrix_heatmap.png", title=f"Rubric Correlations | model={m}")

        rows = []
        for tech, human in splits:
            stats = compute_signal(corr, list(tech), list(human))
            sid = split_id(tech, human)
            rows.append(
                {
                    "split_id": sid,
                    "tech": ",".join(tech),
                    "human": ",".join(human),
                    "tech_size": len(tech),
                    "human_size": len(human),
                    **stats,
                }
            )
            global_rows.append(
                {
                    "split_id": sid,
                    "model": m,
                    "tech_size": len(tech),
                    "human_size": len(human),
                    **stats,
                }
            )

        rank_df = pd.DataFrame(rows).sort_values("signal", ascending=False).reset_index(drop=True)
        rank_df["rank"] = np.arange(1, len(rank_df) + 1)
        rank_df.to_csv(model_out / "split_rankings_all.csv", index=False, encoding="utf-8")
        rank_df.head(15).to_csv(model_out / "split_rankings_top15.csv", index=False, encoding="utf-8")

        # Pick DEFAULT (if present) + top alternatives
        chosen = []
        labels = []

        default_rows = rank_df[rank_df.apply(lambda r: is_default_split(tuple(r["tech"].split(",")), tuple(r["human"].split(","))), axis=1)]
        if not default_rows.empty:
            drow = default_rows.iloc[0].to_dict()
            chosen.append(drow)
            labels.append("DEFAULT")

        # top alternatives excluding DEFAULT row(s)
        alts = []
        for _, r in rank_df.iterrows():
            t = tuple(r["tech"].split(","))
            h = tuple(r["human"].split(","))
            if is_default_split(t, h):
                continue
            alts.append(r.to_dict())
            if len(alts) >= args.topk_alt:
                break

        for i, a in enumerate(alts, start=1):
            chosen.append(a)
            labels.append(f"ALT{i}")

        pd.DataFrame(chosen).to_csv(model_out / "default_and_top_alternatives.csv", index=False, encoding="utf-8")

        plot_split_comparison_bar(
            rows=[
                {
                    "within_tech": float(r.get("within_tech", np.nan)),
                    "within_human": float(r.get("within_human", np.nan)),
                    "between_groups": float(r.get("between_groups", np.nan)),
                    "signal": float(r.get("signal", np.nan)),
                }
                for r in chosen
            ],
            labels=labels,
            out_path=model_out / "default_vs_alternatives_bar.png",
            title=f"Default vs Top Alternatives | model={m}",
        )

    # -------------------------
    # Global aggregation across models
    # -------------------------
    gdf = pd.DataFrame(global_rows)
    if gdf.empty:
        raise ValueError("No models passed min_scenarios threshold; global summary empty.")

    agg_split = (
        gdf.groupby("split_id", dropna=False)
        .agg(
            n_models=("model", "nunique"),
            tech_size=("tech_size", "first"),
            human_size=("human_size", "first"),
            signal_mean=("signal", "mean"),
            signal_median=("signal", "median"),
            signal_std=("signal", "std"),
            within_tech_mean=("within_tech", "mean"),
            within_human_mean=("within_human", "mean"),
            between_mean=("between_groups", "mean"),
        )
        .reset_index()
        .sort_values(["signal_mean", "signal_median"], ascending=False)
        .reset_index(drop=True)
    )
    agg_split["rank_by_signal_mean"] = np.arange(1, len(agg_split) + 1)

    # Attach tech/human composition
    id_to_comp = {split_id(t, h): (",".join(t), ",".join(h), len(t), len(h)) for t, h in splits}
    agg_split["tech"] = agg_split["split_id"].map(lambda x: id_to_comp.get(x, ("", "", 0, 0))[0])
    agg_split["human"] = agg_split["split_id"].map(lambda x: id_to_comp.get(x, ("", "", 0, 0))[1])

    agg_split.to_csv(out_dir / "global_split_rankings_by_mean_signal.csv", index=False, encoding="utf-8")

    # Global DEFAULT + top2 alternatives
    default_sid = None
    for t, h in splits:
        if is_default_split(t, h):
            default_sid = split_id(t, h)
            break

    chosen_global = []
    labels = []

    if default_sid and (agg_split["split_id"] == default_sid).any():
        d = agg_split[agg_split["split_id"] == default_sid].iloc[0].to_dict()
        chosen_global.append(d)
        labels.append("DEFAULT")

    # top2 alternatives
    alts = []
    for _, r in agg_split.iterrows():
        if default_sid and r["split_id"] == default_sid:
            continue
        alts.append(r.to_dict())
        if len(alts) >= 2:
            break
    for i, a in enumerate(alts, start=1):
        chosen_global.append(a)
        labels.append(f"ALT{i}")

    pd.DataFrame(chosen_global).to_csv(out_dir / "global_default_and_top2_alternatives.csv", index=False, encoding="utf-8")

    # Plot global chosen comparison (means)
    plot_split_comparison_bar(
        rows=[
            {
                "within_tech": float(r.get("within_tech_mean", np.nan)),
                "within_human": float(r.get("within_human_mean", np.nan)),
                "between_groups": float(r.get("between_mean", np.nan)),
                "signal": float(r.get("signal_mean", np.nan)),
            }
            for r in chosen_global
        ],
        labels=labels,
        out_path=out_dir / "global_default_vs_top2_alternatives_bar.png",
        title="Global: Default vs Top2 Alternative Splits (mean across models)",
    )

    print("[DONE]")
    print("Out dir:", out_dir.resolve())
    print("Total splits enumerated:", len(splits))
    if not args.allow_all_tech:
        print("Expected (n=8, anchor fixed, exclude all-tech): 127")
    print("Per-model outputs ->", (per_model_dir).resolve())
    print("Global split ranking ->", (out_dir / "global_split_rankings_by_mean_signal.csv").resolve())
    print("Global chosen splits ->", (out_dir / "global_default_and_top2_alternatives.csv").resolve())


if __name__ == "__main__":
    main()
