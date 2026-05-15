# src/analysis/analyze_scores.py
from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path
from typing import List, Tuple, Optional, Dict

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


# -------------------------
# Utils
# -------------------------
def ensure_overall_mean(df: pd.DataFrame, rubrics: List[str]) -> pd.DataFrame:
    if "overall_mean" not in df.columns:
        avail = [r for r in rubrics if r in df.columns]
        if not avail:
            raise ValueError("No rubric columns found to compute overall_mean.")
        df["overall_mean"] = df[avail].mean(axis=1)
    return df


def detect_cols(df: pd.DataFrame) -> Tuple[str, str, str]:
    # scenario uid
    scenario_col = "scenario_uid" if "scenario_uid" in df.columns else ("scenario_id" if "scenario_id" in df.columns else None)
    if not scenario_col:
        raise ValueError("Cannot find scenario_uid/scenario_id column.")
    model_col = "answer_model" if "answer_model" in df.columns else ("model" if "model" in df.columns else None)
    if not model_col:
        raise ValueError("Cannot find answer_model/model column.")
    judge_col = "judge_model" if "judge_model" in df.columns else ("grader_model" if "grader_model" in df.columns else None)
    if not judge_col:
        raise ValueError("Cannot find judge_model/grader_model column.")
    return scenario_col, model_col, judge_col


def filter_language(df: pd.DataFrame, language: str) -> pd.DataFrame:
    lang = (language or "all").lower().strip()
    if lang == "all":
        return df
    if "language" not in df.columns:
        print("[WARN] CSV has no 'language' column; language filtering skipped.")
        return df
    return df[df["language"].astype(str).str.lower() == lang].copy()


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def build_model_order_by_reference_judge(
    df: pd.DataFrame,
    model_col: str,
    judge_col: str,
    ref_judge: Optional[str] = None,
) -> List[str]:
    """
    Keep old spirit: rank models by median overall_mean under a reference judge (prefer GPT judge).
    """
    judges = sorted(df[judge_col].dropna().unique().tolist())
    if not judges:
        return sorted(df[model_col].dropna().unique().tolist())

    if ref_judge and ref_judge in judges:
        j = ref_judge
    else:
        # prefer a judge with "gpt" in name
        gpts = [x for x in judges if "gpt" in str(x).lower()]
        j = gpts[0] if gpts else judges[0]

    sub = df[df[judge_col] == j].copy()
    if sub.empty:
        return sorted(df[model_col].dropna().unique().tolist())

    med = sub.groupby(model_col)["overall_mean"].median().sort_values(ascending=False)
    order = med.index.astype(str).tolist()
    return order


def compute_stats(df: pd.DataFrame, model_col: str, judge_col: str, rubrics: List[str]) -> pd.DataFrame:
    cols = [c for c in rubrics if c in df.columns] + (["overall_mean"] if "overall_mean" in df.columns else [])
    g = df.groupby([model_col, judge_col])[cols]
    stats = g.agg(["mean", "std", "min", "max", "count"])
    stats.columns = ["_".join([a, b]) for (a, b) in stats.columns]
    stats = stats.reset_index()
    return stats


# -------------------------
# Plots
# -------------------------
def plot_boxplots(df: pd.DataFrame, out_dir: Path, model_order: List[str], model_col: str, judge_col: str, rubrics: List[str]) -> None:
    safe_mkdir(out_dir)
    judges = sorted(df[judge_col].dropna().unique().tolist())
    for j in judges:
        sub = df[df[judge_col] == j].copy()
        if sub.empty:
            continue
        for r in rubrics + ["overall_mean"]:
            if r not in sub.columns:
                continue

            data = []
            labels = []
            for m in model_order:
                s = sub[sub[model_col].astype(str) == m][r].dropna()
                if len(s) == 0:
                    continue
                data.append(s.values)
                labels.append(m)

            if len(data) < 2:
                continue

            plt.figure(figsize=(max(10, 0.4 * len(labels)), 6))
            plt.boxplot(data, labels=labels, showfliers=False)
            plt.xticks(rotation=45, ha="right")
            plt.title(f"Boxplot | judge={j} | metric={r}")
            plt.tight_layout()
            plt.savefig(out_dir / f"boxplot_{sanitize_filename(j)}_{r}.png", dpi=180)
            plt.close()


def radar_plot(ax, labels: List[str], values: List[float], title: str) -> None:
    n = len(labels)
    if n == 0:
        return
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    values2 = values + values[:1]
    angles2 = angles + angles[:1]
    ax.plot(angles2, values2)
    ax.fill(angles2, values2, alpha=0.15)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels([])
    ax.set_title(title, fontsize=10)


def plot_radar(
    df: pd.DataFrame,
    out_dir: Path,
    model_order: List[str],
    model_col: str,
    judge_col: str,
    rubrics: List[str],
    top_k: int = -1,
) -> None:
    safe_mkdir(out_dir)
    judges = sorted(df[judge_col].dropna().unique().tolist())
    avail = [r for r in rubrics if r in df.columns]

    def _nice_step(span: float) -> float:
        # span 是归一化后的跨度（0~1）
        if span <= 0.20:
            return 0.05   # 5分一格
        if span <= 0.50:
            return 0.10   # 10分一格
        return 0.20       # 20分一格（跨度很大时减少刻度密度）

    def _set_radar_scale(ax, rmin: float, rmax: float) -> None:
        rmin = max(0.0, float(rmin))
        rmax = min(1.0, float(rmax))
        if rmax <= rmin:
            rmin, rmax = 0.0, 1.0

        span = rmax - rmin
        step = _nice_step(span)

        # 向外留一点边距
        pad = min(0.05, span * 0.15)
        rmin2 = max(0.0, rmin - pad)
        rmax2 = min(1.0, rmax + pad)

        # 对齐到 step 网格
        rmin_grid = math.floor(rmin2 / step) * step
        rmax_grid = math.ceil(rmax2 / step) * step
        rmin_grid = max(0.0, rmin_grid)
        rmax_grid = min(1.0, rmax_grid)

        ax.set_ylim(rmin_grid, rmax_grid)

        # 生成刻度（包含 rmax）
        ticks = []
        t = rmin_grid
        # 防浮点误差
        while t < rmax_grid - 1e-9:
            ticks.append(round(t, 4))
            t += step
        ticks.append(round(rmax_grid, 4))

        # 显示 100/90/...
        tick_labels = [str(int(round(x * 100))) for x in ticks]
        ax.set_yticks(ticks)
        ax.set_yticklabels(tick_labels, fontsize=8)

    for j in judges:
        sub = df[df[judge_col] == j].copy()
        if sub.empty:
            continue

        means = sub.groupby(model_col)[avail].mean()
        means["overall_mean"] = sub.groupby(model_col)["overall_mean"].mean()

        # 排序（按 overall_mean）
        ranked_all = means["overall_mean"].sort_values(ascending=False).index.astype(str).tolist()
        # 保持 model_order 里有的（避免杂项）
        ranked_all = [m for m in ranked_all if m in model_order]

        # -------- 单模型雷达：仍用 top_k（可调） --------
        ranked_single = ranked_all

        for m in ranked_single:
            vals = means.loc[m, avail].values.astype(float).tolist()
            vals01 = [v / 100.0 for v in vals]

            # 单模型默认仍显示 0~100 全刻度（更直观）
            fig = plt.figure(figsize=(6, 6))
            ax = plt.subplot(111, polar=True)
            radar_plot(ax, avail, vals01, title=f"Radar | judge={j} | model={m}")

            _set_radar_scale(ax, 0.0, 1.0)  # 单图固定 0-100
            plt.tight_layout()
            plt.savefig(out_dir / f"radar_{sanitize_filename(j)}_{sanitize_filename(m)}.png", dpi=180)
            plt.close()

        # -------- 多模型 overlay：画“全部模型”，并自适应范围 --------
                # -------- 多模型 overlay：同时输出 ALL + TOP8，并自适应范围 --------
        overlay_all = ranked_all  # 全部模型（比如 13）
        overlay_top = ranked_all[:6]  # Top 8

        def _plot_overlay(overlay_models: List[str], tag: str) -> None:
            if len(overlay_models) < 2:
                return

            fig = plt.figure(figsize=(10.5, 8.0))  # 适当加大画布，给标签/legend留空间
            ax = plt.subplot(111, polar=True)

            n = len(avail)
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
            angles2 = angles + angles[:1]

            all_vals: List[float] = []

            for m in overlay_models:
                vals = means.loc[m, avail].values.astype(float).tolist()
                vals01_closed = [v / 100.0 for v in vals] + [vals[0] / 100.0]
                ax.plot(angles2, vals01_closed, label=m)
                all_vals.extend(vals01_closed)

            ax.set_xticks(angles)
            ax.set_xticklabels(avail, fontsize=10)

            # 让轴标签离图更远一点，避免挤到图外/被裁
            ax.tick_params(axis="x", pad=14)

            ax.set_title(f"Radar Overlay ({tag}) | judge={j}", fontsize=12)

            # ✅ 自适应范围（尤其 overlay）
            if all_vals:
                rmin = min(all_vals)
                rmax = max(all_vals)
            else:
                rmin, rmax = 0.0, 1.0
            _set_radar_scale(ax, rmin, rmax)

            # legend：模型多时自动多列，放右侧
            n_models = len(overlay_models)
            ncol = 1 if n_models <= 10 else 2 if n_models <= 20 else 3
            leg = ax.legend(
                loc="center left",
                bbox_to_anchor=(1.18, 0.5),
                ncol=ncol,
                fontsize=9,
                frameon=False,
            )

            # ✅ 关键：给右侧 legend 留边距；同时保存时 bbox_inches 防裁剪
            fig.subplots_adjust(left=0.08, right=0.72, top=0.90, bottom=0.08)

            plt.savefig(
                out_dir / f"radar_overlay_{tag.lower()}_{sanitize_filename(j)}.png",
                dpi=180,
                bbox_inches="tight",
                bbox_extra_artists=(leg,),
            )
            plt.close()

        _plot_overlay(overlay_all, f"All {len(overlay_all)}")
        _plot_overlay(overlay_top, "Top 6")

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional


def cross_language_analysis(
    df: pd.DataFrame,
    out_dir: Path,
    model_col: str,
    judge_col: str,
    lang_col: str,
    rubrics: List[str],
) -> None:
    """
    Cross-language comparison (EN vs ZH) per judge.
    Only meaningful when df includes BOTH languages and language filter == 'all'.

    Outputs per judge:
      - cross_lang_summary__judge=<J>.csv
      - delta_overall__judge=<J>.png         (ZH-EN overall)
      - scatter_overall_zh_vs_en__judge=<J>.png
      - heatmap_delta_rubrics__judge=<J>.png (ZH-EN per rubric)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # normalize language labels
    dfx = df.copy()
    dfx[lang_col] = dfx[lang_col].astype(str).str.lower().str.strip()
    dfx.loc[dfx[lang_col].isin(["zh-cn", "zh-hans", "chinese"]), lang_col] = "zh"
    dfx.loc[dfx[lang_col].isin(["en-us", "english"]), lang_col] = "en"

    if "en" not in set(dfx[lang_col].unique()) or "zh" not in set(dfx[lang_col].unique()):
        print("[WARN] cross_language_analysis: need both 'en' and 'zh' in data; skipped.")
        return

    avail = [r for r in rubrics if r in dfx.columns]
    if "overall_mean" not in dfx.columns:
        print("[WARN] cross_language_analysis: missing overall_mean; skipped.")
        return

    judges = sorted(dfx[judge_col].dropna().unique().tolist())

    for j in judges:
        sub = dfx[dfx[judge_col] == j].copy()
        if sub.empty:
            continue

        # mean scores per model+language
        grp = sub.groupby([model_col, lang_col], dropna=False)[avail + ["overall_mean"]].mean().reset_index()

        # pivot: rows=models, columns=(metric, lang)
        # overall
        piv_overall = grp.pivot(index=model_col, columns=lang_col, values="overall_mean")
        if "en" not in piv_overall.columns or "zh" not in piv_overall.columns:
            continue

        # keep only models that have BOTH languages
        both_models = piv_overall.dropna(subset=["en", "zh"]).index.astype(str).tolist()
        if len(both_models) < 2:
            continue

        piv_overall = piv_overall.loc[both_models].copy()
        piv_overall["delta_zh_minus_en"] = piv_overall["zh"] - piv_overall["en"]

        # rubric deltas
        delta_rubrics = {}
        for r in avail:
            piv_r = grp.pivot(index=model_col, columns=lang_col, values=r)
            if "en" in piv_r.columns and "zh" in piv_r.columns:
                piv_r = piv_r.loc[both_models]
                delta_rubrics[r] = (piv_r["zh"] - piv_r["en"])

        delta_rub_df = pd.DataFrame(delta_rubrics, index=both_models)

        # ---- 1) export summary CSV
        # Build a wide summary table: overall_en/overall_zh/delta + rubric_en/zh/delta
        summary = pd.DataFrame(index=both_models)
        summary["overall_en"] = piv_overall["en"]
        summary["overall_zh"] = piv_overall["zh"]
        summary["overall_delta_zh_minus_en"] = piv_overall["delta_zh_minus_en"]

        for r in avail:
            piv_r = grp.pivot(index=model_col, columns=lang_col, values=r).loc[both_models]
            summary[f"{r}_en"] = piv_r.get("en")
            summary[f"{r}_zh"] = piv_r.get("zh")
            summary[f"{r}_delta_zh_minus_en"] = summary[f"{r}_zh"] - summary[f"{r}_en"]

        summary = summary.sort_values("overall_delta_zh_minus_en", ascending=False)
        csv_path = out_dir / f"cross_lang_summary__judge={sanitize_filename(str(j))}.csv"
        summary.to_csv(csv_path, index=True)

        # ---- 2) bar: overall delta (ZH-EN)
        fig = plt.figure(figsize=(10, 6))
        ax = plt.gca()
        xs = summary.index.astype(str).tolist()
        ys = summary["overall_delta_zh_minus_en"].values.astype(float)
        ax.bar(xs, ys)
        ax.axhline(0, linewidth=1)
        ax.set_title(f"Overall Δ (ZH - EN) by Model | judge={j}")
        ax.set_ylabel("Δ overall_mean (points)")
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs, rotation=45, ha="right", fontsize=9)
        plt.tight_layout()
        plt.savefig(out_dir / f"delta_overall__judge={sanitize_filename(str(j))}.png", dpi=180, bbox_inches="tight")
        plt.close()

        # ---- 3) scatter: EN vs ZH overall
        fig = plt.figure(figsize=(7, 7))
        ax = plt.gca()
        x = piv_overall["en"].values.astype(float)
        y = piv_overall["zh"].values.astype(float)
        ax.scatter(x, y)
        # y=x line
        mn = min(x.min(), y.min())
        mx = max(x.max(), y.max())
        pad = (mx - mn) * 0.05 if mx > mn else 1.0
        ax.plot([mn - pad, mx + pad], [mn - pad, mx + pad], linestyle="--")
        ax.set_xlim(mn - pad, mx + pad)
        ax.set_ylim(mn - pad, mx + pad)
        ax.set_xlabel("EN overall_mean")
        ax.set_ylabel("ZH overall_mean")
        ax.set_title(f"EN vs ZH Overall | judge={j}")
        # label points
        for i, m in enumerate(both_models):
            ax.annotate(m, (x[i], y[i]), fontsize=8, xytext=(3, 3), textcoords="offset points")
        plt.tight_layout()
        plt.savefig(out_dir / f"scatter_overall_zh_vs_en__judge={sanitize_filename(str(j))}.png", dpi=180, bbox_inches="tight")
        plt.close()

        # ---- 4) heatmap: rubric deltas (ZH-EN)
        if not delta_rub_df.empty:
            # sort models by overall delta for readability
            delta_rub_df = delta_rub_df.loc[summary.index.tolist()]
            mat = delta_rub_df.values.astype(float)

            fig = plt.figure(figsize=(10, 0.45 * len(delta_rub_df) + 2))
            ax = plt.gca()
            im = ax.imshow(mat, aspect="auto")
            ax.set_title(f"Rubric Δ (ZH - EN) Heatmap | judge={j}")
            ax.set_yticks(range(len(delta_rub_df.index)))
            ax.set_yticklabels(delta_rub_df.index.astype(str).tolist(), fontsize=9)
            ax.set_xticks(range(len(delta_rub_df.columns)))
            ax.set_xticklabels(delta_rub_df.columns.astype(str).tolist(), rotation=30, ha="right", fontsize=9)

            # annotate cells
            for r_i in range(mat.shape[0]):
                for c_i in range(mat.shape[1]):
                    v = mat[r_i, c_i]
                    ax.text(c_i, r_i, f"{v:+.1f}", ha="center", va="center", fontsize=8)

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.savefig(out_dir / f"heatmap_delta_rubrics__judge={sanitize_filename(str(j))}.png", dpi=180, bbox_inches="tight")
            plt.close()

        print(f"[CROSS-LANG] judge={j} models_with_both_langs={len(both_models)} -> {out_dir}")

def plot_scenario_rubric_delta_heatmaps(
    df: pd.DataFrame,
    out_dir: Path,
    scenario_col: str,
    model_col: str,
    judge_col: str,
    lang_col: str,
    rubrics: List[str],
    max_scenarios: int = 200,
    annotate: bool = True,
) -> None:
    """
    For each judge and each model, output a heatmap:
      rows   = scenarios
      cols   = rubrics
      value  = (zh - en) per scenario per rubric

    Outputs (per judge, per model):
      - scenario_delta__judge=<J>__model=<M>.csv
      - scenario_delta__judge=<J>__model=<M>.png
    """

    safe_mkdir(out_dir)

    if lang_col not in df.columns:
        print(f"[WARN] No '{lang_col}' column, scenario delta heatmaps skipped.")
        return

    # normalize language values
    dfx = df.copy()
    dfx[lang_col] = dfx[lang_col].astype(str).str.lower().str.strip()

    judges = sorted(dfx[judge_col].dropna().unique().tolist())
    models = sorted(dfx[model_col].dropna().unique().tolist())

    avail_rubrics = [r for r in rubrics if r in dfx.columns]
    if not avail_rubrics:
        print("[WARN] No rubric columns available for heatmaps.")
        return

    total_png = 0
    for j in judges:
        sub_j = dfx[dfx[judge_col] == j].copy()
        if sub_j.empty:
            continue

        # judge folder
        judge_dir = out_dir / f"judge={sanitize_filename(str(j))}"
        safe_mkdir(judge_dir)

        for m in models:
            sub = sub_j[sub_j[model_col] == m].copy()
            if sub.empty:
                continue

            # pivot by scenario x language for each rubric, then compute delta
            # only keep scenarios that have both en and zh rows
            # (we still allow NaN in specific rubric if missing)
            scenarios = sub[scenario_col].dropna().astype(str).unique().tolist()
            if not scenarios:
                continue

            # Build delta matrix: index=scenario, columns=rubrics
            delta_mat = pd.DataFrame(index=sorted(map(str, scenarios)), columns=avail_rubrics, dtype=float)

            # Speed: compute per rubric using pivot_table
            for r in avail_rubrics:
                piv = sub.pivot_table(index=scenario_col, columns=lang_col, values=r, aggfunc="mean")
                if "en" not in piv.columns or "zh" not in piv.columns:
                    continue
                both = piv.dropna(subset=["en", "zh"], how="any")
                if both.empty:
                    continue
                delta = both["zh"] - both["en"]
                delta_mat.loc[delta.index.astype(str), r] = delta.values.astype(float)

            # filter to rows having at least one non-NaN
            delta_mat = delta_mat.dropna(how="all")
            if delta_mat.empty:
                continue

            # limit scenarios to avoid huge figures (keep original order as much as possible)
            if len(delta_mat) > max_scenarios:
                delta_mat = delta_mat.iloc[:max_scenarios].copy()

            # Save CSV
            csv_path = judge_dir / f"scenario_delta__judge={sanitize_filename(str(j))}__model={sanitize_filename(str(m))}.csv"
            delta_mat.to_csv(csv_path, index=True, encoding="utf-8")

            # Plot heatmap (scenario x rubric)
            n_rows = len(delta_mat)
            n_cols = len(delta_mat.columns)

            # dynamic figsize: keep readable
            fig_w = max(8, min(18, 1.2 * n_cols + 6))
            fig_h = max(10, min(40, 0.22 * n_rows + 6))

            fig = plt.figure(figsize=(fig_w, fig_h))
            ax = plt.gca()

            mat = delta_mat.values.astype(float)

            # robust vmin/vmax for symmetric diverging colormap
            finite_vals = mat[np.isfinite(mat)]
            if finite_vals.size == 0:
                plt.close()
                continue
            vmax = float(np.nanpercentile(np.abs(finite_vals), 95))
            vmax = max(vmax, 1.0)
            vmin = -vmax

            im = ax.imshow(mat, aspect="auto", vmin=vmin, vmax=vmax, cmap="coolwarm")

            ax.set_title(f"Rubric Δ (ZH - EN) by scenario | judge={j} | model={m}", fontsize=12)

            # x ticks: rubrics
            ax.set_xticks(range(n_cols))
            ax.set_xticklabels(delta_mat.columns.astype(str).tolist(), rotation=30, ha="right", fontsize=10)

            # y ticks: scenarios
            ax.set_yticks(range(n_rows))
            ax.set_yticklabels(delta_mat.index.astype(str).tolist(), fontsize=8)

            # annotate cells
            if annotate:
                for r_i in range(n_rows):
                    for c_i in range(n_cols):
                        v = mat[r_i, c_i]
                        if np.isfinite(v):
                            ax.text(c_i, r_i, f"{v:+.1f}", ha="center", va="center", fontsize=7)

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()

            png_path = judge_dir / f"scenario_delta__judge={sanitize_filename(str(j))}__model={sanitize_filename(str(m))}.png"
            plt.savefig(png_path, dpi=220, bbox_inches="tight")
            plt.close()

            total_png += 1

    print(f"[CROSS-LANG] scenario×rubric delta heatmaps exported: {total_png} png(s) -> {out_dir}")


def compute_winrate_matrix(df: pd.DataFrame, scenario_col: str, model_col: str, judge_col: str) -> Dict[str, pd.DataFrame]:
    """
    Pairwise win-rate by scenario (overall_mean). For each judge.
    winrate[A,B] = fraction of scenarios where score(A) > score(B)
    """
    out = {}
    judges = sorted(df[judge_col].dropna().unique().tolist())
    for j in judges:
        sub = df[df[judge_col] == j].copy()
        if sub.empty:
            continue
        pivot = sub.pivot_table(index=scenario_col, columns=model_col, values="overall_mean", aggfunc="mean")
        models = pivot.columns.astype(str).tolist()
        win = pd.DataFrame(np.nan, index=models, columns=models)
        for a in models:
            for b in models:
                if a == b:
                    win.loc[a, b] = 0.5
                    continue
                pa = pivot[a]
                pb = pivot[b]
                mask = pa.notna() & pb.notna()
                if mask.sum() == 0:
                    continue
                win.loc[a, b] = (pa[mask] > pb[mask]).mean()
        out[j] = win
    return out


def plot_heatmap(mat: pd.DataFrame, title: str, out_path: Path) -> None:
    plt.figure(figsize=(max(8, 0.35 * mat.shape[1]), max(7, 0.35 * mat.shape[0])))
    plt.imshow(mat.values, aspect="auto")
    plt.colorbar()
    plt.xticks(range(mat.shape[1]), mat.columns.tolist(), rotation=45, ha="right", fontsize=7)
    plt.yticks(range(mat.shape[0]), mat.index.tolist(), fontsize=7)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_winrates(df: pd.DataFrame, out_dir: Path, scenario_col: str, model_col: str, judge_col: str, model_order: List[str]) -> None:
    safe_mkdir(out_dir)
    win_mats = compute_winrate_matrix(df, scenario_col, model_col, judge_col)
    for j, mat in win_mats.items():
        # reorder
        order = [m for m in model_order if m in mat.index]
        mat2 = mat.loc[order, order]
        plot_heatmap(mat2, title=f"Pairwise Win-Rate | judge={j}", out_path=out_dir / f"winrate_{sanitize_filename(j)}.png")


def find_high_value_scenarios(df: pd.DataFrame, scenario_col: str, model_col: str, judge_col: str, top_n: int = 5) -> Dict[str, List[str]]:
    """
    Per judge: pick scenarios that are "hard + discriminative".
    score = (1 - normalized mean) + normalized std
    """
    out = {}
    judges = sorted(df[judge_col].dropna().unique().tolist())
    for j in judges:
        sub = df[df[judge_col] == j].copy()
        if sub.empty:
            continue
        pivot = sub.pivot_table(index=scenario_col, columns=model_col, values="overall_mean", aggfunc="mean")
        scen_mean = pivot.mean(axis=1)
        scen_std = pivot.std(axis=1)

        # normalize
        mmin, mmax = scen_mean.min(), scen_mean.max()
        smin, smax = scen_std.min(), scen_std.max()
        mean_norm = (scen_mean - mmin) / (mmax - mmin + 1e-9)
        std_norm = (scen_std - smin) / (smax - smin + 1e-9)

        score = (1.0 - mean_norm) + std_norm
        picked = score.sort_values(ascending=False).head(top_n).index.astype(str).tolist()
        out[j] = picked
    return out


def plot_high_value_scenarios(df: pd.DataFrame, out_dir: Path, scenario_col: str, model_col: str, judge_col: str, model_order: List[str], top_n: int = 5) -> None:
    safe_mkdir(out_dir)
    picked_map = find_high_value_scenarios(df, scenario_col, model_col, judge_col, top_n=top_n)

    for j, scen_ids in picked_map.items():
        sub = df[df[judge_col] == j].copy()
        pivot = sub.pivot_table(index=scenario_col, columns=model_col, values="overall_mean", aggfunc="mean")
        pivot = pivot.loc[[s for s in scen_ids if s in pivot.index]]

        # order models
        order = [m for m in model_order if m in pivot.columns]
        pivot = pivot[order]

        if pivot.shape[0] == 0 or pivot.shape[1] == 0:
            continue

        plt.figure(figsize=(max(10, 0.45 * pivot.shape[1]), max(4, 0.7 * pivot.shape[0])))
        plt.imshow(pivot.values, aspect="auto")
        plt.colorbar()
        plt.xticks(range(pivot.shape[1]), pivot.columns.tolist(), rotation=45, ha="right", fontsize=7)
        plt.yticks(range(pivot.shape[0]), pivot.index.tolist(), fontsize=7)
        plt.title(f"High-Value Scenarios Heatmap | judge={j}")
        plt.tight_layout()
        plt.savefig(out_dir / f"high_value_scenarios_{sanitize_filename(j)}.png", dpi=180)
        plt.close()
def find_high_lang_shift_scenarios(
    df: pd.DataFrame,
    scenario_col: str,
    model_col: str,
    judge_col: str,
    lang_col: str = "language",
    top_n: int = 5,
    use_abs_mean: bool = True,
) -> Dict[str, List[str]]:
    """
    Per judge: pick scenarios with large EN/ZH shift.

    For each judge and each scenario:
      for each model, compute delta = mean(zh overall_mean) - mean(en overall_mean)
      scenario_score = mean(|delta|) across models (only models having both en & zh)
    Then pick top_n scenarios by score.

    Returns: {judge: [scenario_uid,...]}
    """
    out: Dict[str, List[str]] = {}
    if lang_col not in df.columns:
        print(f"[WARN] '{lang_col}' not found, skip high_lang_shift_scenarios.")
        return out

    dfx = df.copy()
    dfx[lang_col] = dfx[lang_col].astype(str).str.lower().str.strip()

    judges = sorted(dfx[judge_col].dropna().unique().tolist())
    for j in judges:
        sub = dfx[dfx[judge_col] == j].copy()
        if sub.empty:
            continue

        # Pivot: index=(scenario, model), columns=language, values=overall_mean
        piv = sub.pivot_table(
            index=[scenario_col, model_col],
            columns=lang_col,
            values="overall_mean",
            aggfunc="mean",
        )

        if "en" not in piv.columns or "zh" not in piv.columns:
            continue

        both = piv.dropna(subset=["en", "zh"], how="any").copy()
        if both.empty:
            continue

        both["delta"] = both["zh"] - both["en"]

        # scenario score: mean(abs(delta)) over models
        if use_abs_mean:
            scen_score = both["delta"].abs().groupby(level=0).mean()
        else:
            scen_score = both["delta"].groupby(level=0).mean().abs()

        picked = scen_score.sort_values(ascending=False).head(top_n).index.astype(str).tolist()
        out[str(j)] = picked

    return out


def plot_high_lang_shift_scenarios(
    df: pd.DataFrame,
    out_dir: Path,
    scenario_col: str,
    model_col: str,
    judge_col: str,
    model_order: List[str],
    lang_col: str = "language",
    top_n: int = 5,
) -> None:
    """
    Export per-judge:
      - TXT list of picked scenarios
      - CSV: scenario x model delta_overall (ZH-EN)
      - PNG: heatmap scenario x model of delta_overall (ZH-EN)
    """
    safe_mkdir(out_dir)

    picked_map = find_high_lang_shift_scenarios(
        df=df,
        scenario_col=scenario_col,
        model_col=model_col,
        judge_col=judge_col,
        lang_col=lang_col,
        top_n=top_n,
        use_abs_mean=True,
    )

    if not picked_map:
        print("[CROSS-LANG] No high_lang_shift_scenarios found (maybe missing en/zh).")
        return

    dfx = df.copy()
    dfx[lang_col] = dfx[lang_col].astype(str).str.lower().str.strip()

    for j, scen_ids in picked_map.items():
        sub = dfx[dfx[judge_col] == j].copy()
        if sub.empty:
            continue

        # write scenario list
        txt_path = out_dir / f"high_lang_shift_scenarios_{sanitize_filename(j)}.txt"
        txt_path.write_text("\n".join([str(s) for s in scen_ids]) + "\n", encoding="utf-8")

        # build scenario x model delta matrix (overall)
        piv = sub.pivot_table(
            index=[scenario_col, model_col],
            columns=lang_col,
            values="overall_mean",
            aggfunc="mean",
        )
        if "en" not in piv.columns or "zh" not in piv.columns:
            continue

        both = piv.dropna(subset=["en", "zh"], how="any").copy()
        both["delta"] = both["zh"] - both["en"]

        # scenario x model
        mat = both["delta"].unstack(model_col)

        # keep only picked scenarios
        mat = mat.loc[[s for s in scen_ids if s in mat.index]]
        if mat.empty:
            continue

        # order models
        order = [m for m in model_order if m in mat.columns]
        # also keep any remaining models not in model_order (just in case)
        tail = [m for m in mat.columns if m not in order]
        mat = mat[order + tail]

        # export csv
        csv_path = out_dir / f"high_lang_shift_scenarios_delta_overall_{sanitize_filename(j)}.csv"
        mat.to_csv(csv_path, index=True, encoding="utf-8")

        # plot heatmap
        plt.figure(figsize=(max(10, 0.45 * mat.shape[1]), max(4, 0.8 * mat.shape[0])))
        vals = mat.values.astype(float)
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            plt.close()
            continue
        vmax = float(np.nanpercentile(np.abs(finite), 95))
        vmax = max(vmax, 1.0)
        vmin = -vmax

        plt.imshow(vals, aspect="auto", cmap="coolwarm", vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.xticks(range(mat.shape[1]), mat.columns.astype(str).tolist(), rotation=45, ha="right", fontsize=7)
        plt.yticks(range(mat.shape[0]), mat.index.astype(str).tolist(), fontsize=8)
        plt.title(f"High Lang-Shift Scenarios | Δ overall (ZH-EN) | judge={j}")
        plt.tight_layout()
        plt.savefig(out_dir / f"high_lang_shift_scenarios_{sanitize_filename(j)}.png", dpi=180)
        plt.close()

    print(f"[CROSS-LANG] high_lang_shift_scenarios exported -> {out_dir}")


def plot_judge_agreement(df: pd.DataFrame, out_dir: Path, scenario_col: str, model_col: str, judge_col: str) -> None:
    safe_mkdir(out_dir)
    judges = sorted(df[judge_col].dropna().unique().tolist())
    if len(judges) < 2:
        return

    # agreement on (scenario, model)
    pivot = df.pivot_table(index=[scenario_col, model_col], columns=judge_col, values="overall_mean", aggfunc="mean")

    for a, b in itertools.combinations(judges, 2):
        if a not in pivot.columns or b not in pivot.columns:
            continue
        x = pivot[a]
        y = pivot[b]
        mask = x.notna() & y.notna()
        if mask.sum() < 10:
            continue

        plt.figure(figsize=(6, 6))
        plt.scatter(x[mask].values, y[mask].values, s=8, alpha=0.6)
        plt.xlabel(str(a))
        plt.ylabel(str(b))
        corr = np.corrcoef(x[mask].values, y[mask].values)[0, 1]
        plt.title(f"Judge Agreement | corr={corr:.3f}")
        plt.tight_layout()
        plt.savefig(out_dir / f"judge_agreement_{sanitize_filename(a)}__{sanitize_filename(b)}.png", dpi=180)
        plt.close()


def pca_2d(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # deterministic PCA via eigen decomposition of covariance
    n = X.shape[0]
    if n < 2:
        raise ValueError("PCA needs >= 2 samples.")
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = (Xc.T @ Xc) / (n - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    W = eigvecs[:, :2]
    X2 = Xc @ W
    ratio = eigvals[:2] / (eigvals.sum() + 1e-12)
    return X2, ratio


def plot_embedding_pca(df: pd.DataFrame, out_dir: Path, scenario_col: str, model_col: str, judge_col: str, rubrics: List[str], ref_judge: Optional[str]) -> None:
    safe_mkdir(out_dir)

    # Use a reference judge for embeddings (same spirit as old script)
    judges = sorted(df[judge_col].dropna().unique().tolist())
    if not judges:
        return

    if ref_judge and ref_judge in judges:
        j = ref_judge
    else:
        gpts = [x for x in judges if "gpt" in str(x).lower()]
        j = gpts[0] if gpts else judges[0]

    sub = df[df[judge_col] == j].copy()
    if sub.empty:
        return

    # --- Rubric space: model x rubrics ---
    avail = [r for r in rubrics if r in sub.columns]
    if len(avail) >= 2:
        mat = sub.groupby(model_col)[avail].mean()
        X = mat.values.astype(float)
        X2, ratio = pca_2d(X)

        plt.figure(figsize=(7, 6))
        plt.scatter(X2[:, 0], X2[:, 1], s=25)
        for i, name in enumerate(mat.index.astype(str).tolist()):
            plt.text(X2[i, 0], X2[i, 1], name, fontsize=8)
        plt.title(f"PCA (Rubric Space) | judge={j} | var={ratio[0]:.2f},{ratio[1]:.2f}")
        plt.tight_layout()
        plt.savefig(out_dir / f"pca_rubric_space_{sanitize_filename(j)}.png", dpi=180)
        plt.close()

    # --- Scenario space (robust): model x scenarios ---
    pivot = sub.pivot_table(index=model_col, columns=scenario_col, values="overall_mean", aggfunc="mean")
    if pivot.shape[0] >= 2 and pivot.shape[1] >= 2:
        X = pivot.values.astype(float)
        # robust normalize per-model: (x - median) / MAD
        med = np.nanmedian(X, axis=1, keepdims=True)
        mad = np.nanmedian(np.abs(X - med), axis=1, keepdims=True) + 1e-9
        Xr = (X - med) / mad
        # fill NaN with 0 after normalization
        Xr = np.nan_to_num(Xr, nan=0.0)

        X2, ratio = pca_2d(Xr)

        plt.figure(figsize=(7, 6))
        plt.scatter(X2[:, 0], X2[:, 1], s=25)
        for i, name in enumerate(pivot.index.astype(str).tolist()):
            plt.text(X2[i, 0], X2[i, 1], name, fontsize=8)
        plt.title(f"PCA (Scenario Space Robust) | judge={j} | var={ratio[0]:.2f},{ratio[1]:.2f}")
        plt.tight_layout()
        plt.savefig(out_dir / f"pca_scenario_space_robust_{sanitize_filename(j)}.png", dpi=180)
        plt.close()


def sanitize_filename(s: str) -> str:
    s = str(s)
    for ch in ['/', '\\', ':', '|', '*', '?', '"', '<', '>', ' ']:
        s = s.replace(ch, "_")
    return s


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze ParentBench merged scores with language slicing.")
    ap.add_argument("--input", default="results/merged/all_judge_scores.csv")
    ap.add_argument("--out-dir", default="results/analysis")
    ap.add_argument("--language", default="all", choices=["all", "en", "zh"])
    ap.add_argument("--ref-judge", default=None, help="Optional: choose judge model name for ranking/embeddings.")
    ap.add_argument("--topk-radar", type=int, default=-1)
    ap.add_argument("--topn-highvalue", type=int, default=5)
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Not found: {in_path}")

    df = pd.read_csv(in_path)
    scenario_col, model_col, judge_col = detect_cols(df)

    df = ensure_overall_mean(df, RUBRICS)
    df = filter_language(df, args.language)

    # language-sliced output root
    out_root = Path(args.out_dir) / args.language
    stats_dir = out_root / "stats"
    box_dir = out_root / "boxplots"
    radar_dir = out_root / "radar"
    win_dir = out_root / "winrate"
    scen_dir = out_root / "scenarios"
    agree_dir = out_root / "judge_agreement"
    embed_dir = out_root / "embedding"

    for p in [stats_dir, box_dir, radar_dir, win_dir, scen_dir, agree_dir, embed_dir]:
        safe_mkdir(p)
    
    # cross-language outputs (only when language == all)
    cross_dir = out_root / "cross_language"
    if args.language == "all":
        safe_mkdir(cross_dir)

    # model order
    model_order = build_model_order_by_reference_judge(df, model_col, judge_col, ref_judge=args.ref_judge)
    print(f"[INFO] language={args.language} | models={len(model_order)} | rows={len(df)}")

    # stats table
    stats = compute_stats(df, model_col, judge_col, RUBRICS)
    stats.to_csv(stats_dir / "model_stats.csv", index=False, encoding="utf-8")
    print(f"[INFO] stats -> {stats_dir / 'model_stats.csv'}")

    # plots
    plot_boxplots(df, box_dir, model_order, model_col, judge_col, RUBRICS)
    plot_radar(df, radar_dir, model_order, model_col, judge_col, RUBRICS, top_k=args.topk_radar)
    plot_winrates(df, win_dir, scenario_col, model_col, judge_col, model_order)
    plot_high_value_scenarios(df, scen_dir, scenario_col, model_col, judge_col, model_order, top_n=args.topn_highvalue)
    plot_judge_agreement(df, agree_dir, scenario_col, model_col, judge_col)
    plot_embedding_pca(df, embed_dir, scenario_col, model_col, judge_col, RUBRICS, ref_judge=args.ref_judge)
    # cross-language comparison (EN vs ZH) — only when language == all
    if args.language == "all":
        if "language" not in df.columns:
            print("[WARN] --language=all but merged CSV has no 'language' column; cross-language skipped.")
        else:
            cross_language_analysis(
                df=df,
                out_dir=cross_dir,
                model_col=model_col,
                judge_col=judge_col,
                lang_col="language",
                rubrics=RUBRICS,
            )
                    # scenario × rubric delta (ZH-EN) heatmaps per judge per model
            scenario_delta_dir = cross_dir / "scenario_delta_heatmaps"
            plot_scenario_rubric_delta_heatmaps(
                df=df,
                out_dir=scenario_delta_dir,
                scenario_col=scenario_col,
                model_col=model_col,
                judge_col=judge_col,
                lang_col="language",
                rubrics=RUBRICS,
                max_scenarios=200,   # 你也可以改大/改小
                annotate=True,
            )
                    # NEW: find + export 5 scenarios with large ZH/EN shift
            lang_shift_dir = cross_dir / "high_lang_shift_scenarios"
            plot_high_lang_shift_scenarios(
                df=df,
                out_dir=lang_shift_dir,
                scenario_col=scenario_col,
                model_col=model_col,
                judge_col=judge_col,
                model_order=model_order,
                lang_col="language",
                top_n=5,
            )




    print(f"[DONE] analysis outputs -> {out_root.resolve()}")


if __name__ == "__main__":
    main()
