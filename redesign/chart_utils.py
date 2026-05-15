from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors, gridspec

from chart_config import (
    PRIMARY_JUDGE_LABEL,
    PRIMARY_JUDGE_RAW,
    RESPONSIBLE_RUBRICS,
    CONSTRUCTIVE_RUBRICS,
    RUBRICS,
    SECONDARY_JUDGE_LABEL,
    SECONDARY_JUDGE_RAW,
    display_name,
    load_model_order,
    model_color,
    results_csv_path,
    METRIC_COLORS,
)


def apply_base_theme() -> None:
    plt.rcParams.update(
        {
            'font.family': 'serif',
            'font.size': 11,
            'axes.titlesize': 20,
            'axes.labelsize': 12,
            'xtick.labelsize': 11,
            'ytick.labelsize': 12,
        }
    )


def load_scores() -> pd.DataFrame:
    df = pd.read_csv(results_csv_path())
    df['overall_mean'] = df[RUBRICS].mean(axis=1)
    df['constructive_mean'] = df[CONSTRUCTIVE_RUBRICS].mean(axis=1)
    df['responsible_mean'] = df[RESPONSIBLE_RUBRICS].mean(axis=1)
    return df


def ordered_models(present_models: Iterable[str] | None = None) -> list[str]:
    order = load_model_order()
    if present_models is None:
        return order
    present = {str(x) for x in present_models}
    return [m for m in order if m in present]


def judge_subset(df: pd.DataFrame, judge: str) -> pd.DataFrame:
    return df[df['judge_model'].astype(str) == judge].copy()


def model_summary(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for model in ordered_models(df['answer_model'].dropna().astype(str).unique().tolist()):
        scores = df.loc[df['answer_model'].astype(str) == model, metric].dropna()
        if scores.empty:
            continue
        rows.append(
            {
                'model': model,
                'display_name': display_name(model),
                'mean': round(float(scores.mean()), 2),
                'median': round(float(scores.median()), 2),
                'std': round(float(scores.std(ddof=1)), 2),
                'n': int(scores.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def make_boxplot_with_table(
    df: pd.DataFrame,
    metric: str,
    title: str,
    out_png: Path,
    out_csv: Path,
    xlim: tuple[float, float] | None = None,
) -> None:
    apply_base_theme()
    summary = model_summary(df, metric)
    summary.to_csv(out_csv, index=False)

    fig = plt.figure(figsize=(17.2, 10.2), facecolor='white')
    gs = gridspec.GridSpec(1, 2, width_ratios=[3.72, 1.88], wspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    ax_tbl = fig.add_subplot(gs[0, 1])

    plot_rows = []
    for _, row in summary.iterrows():
        model = row['model']
        scores = df.loc[df['answer_model'].astype(str) == model, metric].dropna()
        plot_rows.append((model, scores))

    data = [scores.values for _, scores in plot_rows]
    labels = [display_name(model) for model, _ in plot_rows]
    positions = list(range(1, len(plot_rows) + 1))

    bp = ax.boxplot(
        data,
        vert=False,
        positions=positions,
        patch_artist=True,
        widths=0.62,
        showfliers=False,
        medianprops={'color': '#1f1f1f', 'linewidth': 1.6},
        whiskerprops={'color': '#444444', 'linewidth': 1.2},
        capprops={'color': '#444444', 'linewidth': 1.2},
        boxprops={'edgecolor': '#444444', 'linewidth': 1.2},
    )

    for patch, (model, _) in zip(bp['boxes'], plot_rows):
        patch.set_facecolor(model_color(model))
        patch.set_alpha(0.9)

    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_xlabel(metric.replace('_', ' ').title())
    ax.grid(axis='x', color='#D9D9D9', linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for pos, (_, row) in zip(positions, summary.iterrows()):
        ax.plot(row['mean'], pos, marker='o', markersize=4.8, color='#111111', zorder=4)

    fig.suptitle(title, fontsize=20, y=0.975)

    ax_tbl.axis('off')
    ax_tbl.set_title('Numeric Summary', fontsize=15, pad=10)
    cell_text = [[row['display_name'], f"{row['mean']:.2f}", f"{row['median']:.2f}"] for _, row in summary.iterrows()]
    table = ax_tbl.table(
        cellText=cell_text,
        colLabels=['Model', 'Mean', 'Median'],
        cellLoc='left',
        colLoc='left',
        loc='center',
        bbox=[0.0, 0.02, 1.0, 0.92],
        colWidths=[0.42, 0.26, 0.32],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.0)

    summary_records = summary.to_dict('records')
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#DDDDDD')
        cell.set_linewidth(0.6)
        if c == 0:
            cell.set_width(0.42)
        elif c == 1:
            cell.set_width(0.26)
        elif c == 2:
            cell.set_width(0.32)
        if r == 0:
            cell.set_facecolor('#F2F2F2')
            cell.set_text_props(weight='bold', color='#111111')
        else:
            cell.set_facecolor('white')
            idx = r - 1
            if c == 0 and 0 <= idx < len(summary_records):
                cell.set_text_props(color=model_color(summary_records[idx]['model']), weight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    plt.savefig(out_png, dpi=220, bbox_inches='tight')
    plt.close(fig)


def pairwise_winrate_matrix(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    pivot = df.pivot_table(index='scenario_uid', columns='answer_model', values=metric, aggfunc='mean')
    models = ordered_models(pivot.columns.astype(str).tolist())
    pivot = pivot[models]
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
            win.loc[a, b] = float((pa[mask] > pb[mask]).mean())
    return win


def draw_heatmap(
    mat: pd.DataFrame,
    title: str,
    out_png: Path,
    cmap: str = 'RdBu_r',
    vmin: float | None = None,
    vmax: float | None = None,
    fmt_display_names: bool = True,
    annotate: bool = False,
    annotation_fmt: str = '{:.2f}',
    center: float | None = None,
) -> None:
    apply_base_theme()
    fig, ax = plt.subplots(figsize=(13.6, 11.4), facecolor='white')
    if center is not None and vmin is not None and vmax is not None:
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
        im = ax.imshow(mat.values, aspect='auto', cmap=cmap, norm=norm)
    else:
        im = ax.imshow(mat.values, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    labels_x = [display_name(x) if fmt_display_names else str(x) for x in mat.columns]
    labels_y = [display_name(x) if fmt_display_names else str(x) for x in mat.index]
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(labels_x, rotation=45, ha='right')
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(labels_y)
    ax.set_title(title, pad=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=11)

    if annotate:
        vals = mat.values.astype(float)
        finite = vals[np.isfinite(vals)]
        if finite.size:
            mid = center if center is not None else ((float(np.nanmin(finite)) + float(np.nanmax(finite))) / 2.0)
            spread = max(abs(float(np.nanmax(finite)) - mid), abs(float(np.nanmin(finite)) - mid))
            threshold = mid + 0.35 * spread
            threshold_low = mid - 0.35 * spread
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    val = vals[i, j]
                    if not np.isfinite(val):
                        continue
                    color = 'white' if (val >= threshold or val <= threshold_low) else '#111111'
                    ax.text(j, i, annotation_fmt.format(val), ha='center', va='center', fontsize=8.2, color=color)

    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches='tight')
    plt.close(fig)


def make_profile_plot(df: pd.DataFrame, models: Sequence[str], title: str, out_png: Path) -> None:
    apply_base_theme()
    means = df.groupby('answer_model')[RUBRICS].mean()
    labels = [r.replace('_', ' ').title() for r in RUBRICS]
    angles = np.linspace(0, 2 * np.pi, len(RUBRICS), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(10.6, 8.4), facecolor='white')
    ax = plt.subplot(111, polar=True)

    for model in models:
        if model not in means.index:
            continue
        vals = means.loc[model, RUBRICS].values.astype(float).tolist()
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2.2, color=model_color(model), label=display_name(model))
        ax.fill(angles, vals, color=model_color(model), alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(40, 100)
    ax.set_yticks([40, 55, 70, 85, 100])
    ax.set_yticklabels(['40', '55', '70', '85', '100'], fontsize=9)
    ax.set_title(title, pad=22)
    ax.grid(color='#D9D9D9', linewidth=0.8, alpha=0.8)
    ax.legend(frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.10))
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches='tight')
    plt.close(fig)


def make_profile_plot_with_table(
    df: pd.DataFrame,
    models: Sequence[str],
    title: str,
    out_png: Path,
    out_csv: Path,
) -> None:
    apply_base_theme()
    means = df.groupby('answer_model')[RUBRICS].mean()
    rows = []
    plot_models = [m for m in models if m in means.index]
    for model in plot_models:
        row = {'model': model, 'display_name': display_name(model)}
        for rubric in RUBRICS:
            row[rubric] = float(means.loc[model, rubric])
        rows.append(row)
    table_df = pd.DataFrame(rows)
    table_df.to_csv(out_csv, index=False)

    labels = [r.replace('_', ' ').title() for r in RUBRICS]
    angles = np.linspace(0, 2 * np.pi, len(RUBRICS), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(17.2, 9.6), facecolor='white')
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.65, 2.15], wspace=0.08)
    ax = fig.add_subplot(gs[0, 0], polar=True)
    ax_tbl = fig.add_subplot(gs[0, 1])

    for model in plot_models:
        vals = means.loc[model, RUBRICS].values.astype(float).tolist()
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2.2, color=model_color(model), label=display_name(model))
        ax.fill(angles, vals, color=model_color(model), alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(40, 100)
    ax.set_yticks([40, 55, 70, 85, 100])
    ax.set_yticklabels(['40', '55', '70', '85', '100'], fontsize=9)
    ax.set_title(title, pad=22)
    ax.grid(color='#D9D9D9', linewidth=0.8, alpha=0.8)
    ax.legend(frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.10))

    ax_tbl.axis('off')
    ax_tbl.set_title('Numeric Summary', fontsize=15, pad=10)
    rubric_cols = ['display_name'] + RUBRICS
    cell_text = []
    for _, row in table_df[rubric_cols].iterrows():
        cell_text.append(
            [
                row['display_name'],
                f"{row['accuracy']:.1f}",
                f"{row['safety']:.1f}",
                f"{row['helpfulness']:.1f}",
                f"{row['empathy']:.1f}",
                f"{row['completeness']:.1f}",
                f"{row['bias_avoidance']:.1f}",
                f"{row['limitation_awareness']:.1f}",
                f"{row['communication']:.1f}",
            ]
        )
    table = ax_tbl.table(
        cellText=cell_text,
        colLabels=['Model', 'Acc', 'Saf', 'Help', 'Emp', 'Comp', 'Bias', 'Limit', 'Comm'],
        cellLoc='center',
        colLoc='center',
        loc='center',
        bbox=[0.0, 0.05, 1.0, 0.9],
        colWidths=[0.26, 0.082, 0.082, 0.09, 0.082, 0.09, 0.082, 0.095, 0.095],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.8)

    summary_records = table_df.to_dict('records')
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#DDDDDD')
        cell.set_linewidth(0.55)
        if r == 0:
            cell.set_facecolor('#F2F2F2')
            cell.set_text_props(weight='bold', color='#111111')
        else:
            cell.set_facecolor('white')
            idx = r - 1
            if c == 0 and 0 <= idx < len(summary_records):
                cell.set_text_props(color=model_color(summary_records[idx]['model']), weight='bold', ha='left')

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(out_png, dpi=220, bbox_inches='tight')
    plt.close(fig)


def make_judge_agreement_scatter(df: pd.DataFrame, title: str, out_png: Path) -> pd.DataFrame:
    apply_base_theme()
    pivot = df.pivot_table(index=['scenario_uid', 'answer_model'], columns='judge_model', values='overall_mean', aggfunc='mean')
    x = pivot[PRIMARY_JUDGE_RAW]
    y = pivot[SECONDARY_JUDGE_RAW]
    mask = x.notna() & y.notna()
    x = x[mask]
    y = y[mask]
    corr = float(np.corrcoef(x.values, y.values)[0, 1])
    mae = float(np.mean(np.abs(x.values - y.values)))

    fig, ax = plt.subplots(figsize=(7.2, 6.8), facecolor='white')
    ax.scatter(x.values, y.values, s=12, alpha=0.55, color='#4E7FC7', edgecolors='none')
    low = min(x.min(), y.min())
    high = max(x.max(), y.max())
    ax.plot([low, high], [low, high], linestyle='--', linewidth=1.1, color='#555555')
    ax.set_xlabel(PRIMARY_JUDGE_LABEL)
    ax.set_ylabel(SECONDARY_JUDGE_LABEL)
    ax.set_title(title, pad=12)
    ax.text(0.03, 0.96, f'Pearson r = {corr:.3f}\nMAE = {mae:.2f}', transform=ax.transAxes, ha='left', va='top', fontsize=11)
    ax.grid(color='#E3E3E3', linewidth=0.8, alpha=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches='tight')
    plt.close(fig)

    return pd.DataFrame({'pearson_r': [corr], 'mae': [mae], 'n': [int(mask.sum())]})


def make_rubric_agreement_bar(df: pd.DataFrame, title: str, out_png: Path, out_csv: Path) -> None:
    apply_base_theme()
    pivot = df.pivot_table(index=['scenario_uid', 'answer_model'], columns='judge_model', values=RUBRICS, aggfunc='mean')
    rows = []
    for rubric in RUBRICS:
        if rubric not in pivot.columns.get_level_values(0):
            continue
        x = pivot[(rubric, PRIMARY_JUDGE_RAW)]
        y = pivot[(rubric, SECONDARY_JUDGE_RAW)]
        mask = x.notna() & y.notna()
        if mask.sum() == 0:
            continue
        corr = float(np.corrcoef(x[mask].values, y[mask].values)[0, 1])
        mae = float(np.mean(np.abs(x[mask].values - y[mask].values)))
        rows.append({'rubric': rubric, 'pearson_r': corr, 'mae': mae})
    stats = pd.DataFrame(rows).sort_values('pearson_r', ascending=True)
    stats.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(9.0, 5.8), facecolor='white')
    y_pos = np.arange(len(stats))
    ax.barh(y_pos, stats['pearson_r'], color='#2E5FA7', alpha=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([r.replace('_', ' ').title() for r in stats['rubric']])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel('Pearson correlation')
    ax.set_title(title, pad=12)
    ax.grid(axis='x', color='#D9D9D9', linewidth=0.8, alpha=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for yv, val, mae in zip(y_pos, stats['pearson_r'], stats['mae']):
        ax.text(min(val + 0.015, 0.985), yv, f'{val:.2f} | MAE {mae:.1f}', va='center', ha='left', fontsize=10)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches='tight')
    plt.close(fig)


def pca_2d(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = (Xc.T @ Xc) / max(n - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    W = eigvecs[:, :2]
    X2 = Xc @ W
    ratio = eigvals[:2] / (eigvals.sum() + 1e-12)
    return X2, ratio






def make_scenario_grouped_bar(
    df: pd.DataFrame,
    scenario_uid: str,
    title: str,
    out_png: Path,
    out_csv: Path,
) -> None:
    apply_base_theme()
    sub = df[df['scenario_uid'].astype(str) == str(scenario_uid)].copy()
    rows = []
    for model in ordered_models(sub['answer_model'].dropna().astype(str).unique().tolist()):
        msub = sub[sub['answer_model'].astype(str) == model]
        if msub.empty:
            continue
        rows.append({
            'model': model,
            'display_name': display_name(model),
            'constructive': float(msub['constructive_mean'].mean()),
            'responsible': float(msub['responsible_mean'].mean()),
            'overall': float(msub['overall_mean'].mean()),
        })
    plot_df = pd.DataFrame(rows)
    plot_df.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(14.8, 8.6), facecolor='white')
    y = np.arange(len(plot_df))
    h = 0.22
    ax.barh(y - h, plot_df['constructive'], height=h, color=METRIC_COLORS['constructive'], label='Constructive')
    ax.barh(y, plot_df['responsible'], height=h, color=METRIC_COLORS['responsible'], label='Responsible')
    ax.barh(y + h, plot_df['overall'], height=h, color=METRIC_COLORS['overall'], label='Overall')

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df['display_name'])
    ax.invert_yaxis()
    ax.set_xlim(50, 95)
    ax.set_xlabel('Mean score')
    ax.set_title(title, pad=12)
    ax.grid(axis='x', color='#D9D9D9', linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, loc='lower right')

    for vals, yy in [(plot_df['constructive'], y - h), (plot_df['responsible'], y), (plot_df['overall'], y + h)]:
        for xv, yv in zip(vals, yy):
            ax.text(xv + 0.35, yv, f'{xv:.1f}', va='center', ha='left', fontsize=8.5, color='#222222')

    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches='tight')
    plt.close(fig)

def make_two_metric_dumbbell(
    df: pd.DataFrame,
    metric_a: str,
    metric_b: str,
    label_a: str,
    label_b: str,
    title: str,
    out_png: Path,
    out_csv: Path,
) -> None:
    apply_base_theme()
    rows = []
    for model in ordered_models(df['answer_model'].dropna().astype(str).unique().tolist()):
        sub = df[df['answer_model'].astype(str) == model]
        a = float(sub[metric_a].mean())
        b = float(sub[metric_b].mean())
        rows.append({
            'model': model,
            'display_name': display_name(model),
            metric_a: a,
            metric_b: b,
            'delta': b - a,
        })
    plot_df = pd.DataFrame(rows)
    plot_df.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(10.4, 8.6), facecolor='white')
    y = np.arange(len(plot_df))
    for i, row in plot_df.iterrows():
        c = model_color(row['model'])
        ax.plot([row[metric_a], row[metric_b]], [i, i], color=c, linewidth=2.2, alpha=0.9)
        ax.scatter(row[metric_a], i, s=48, color='white', edgecolors=c, linewidths=1.8, zorder=3)
        ax.scatter(row[metric_b], i, s=52, color=c, edgecolors='white', linewidths=0.8, zorder=3)
        ax.text(max(row[metric_a], row[metric_b]) + 0.45, i, row['display_name'], va='center', fontsize=9)

    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_xlabel('Mean rubric score')
    ax.set_title(title, pad=12)
    ax.grid(axis='x', color='#D9D9D9', linewidth=0.8, alpha=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    low = min(plot_df[metric_a].min(), plot_df[metric_b].min())
    high = max(plot_df[metric_a].max(), plot_df[metric_b].max())
    ax.set_xlim(low - 2.5, high + 6.5)
    ax.scatter([], [], s=48, color='white', edgecolors='#333333', linewidths=1.6, label=label_a)
    ax.scatter([], [], s=52, color='#333333', edgecolors='white', linewidths=0.8, label=label_b)
    ax.legend(frameon=False, loc='lower right')
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches='tight')
    plt.close(fig)

def make_grouped_indices_scatter(df: pd.DataFrame, title: str, out_png: Path, out_csv: Path) -> None:
    apply_base_theme()
    means = df.groupby('answer_model')[['constructive_mean', 'responsible_mean']].mean().reset_index()
    means['order_key'] = means['answer_model'].apply(lambda x: ordered_models([*means['answer_model'].tolist()]).index(x) if x in ordered_models(means['answer_model'].tolist()) else 999)
    means = means.sort_values('order_key')
    means.to_csv(out_csv, index=False)

    fig = plt.figure(figsize=(15.8, 8.4), facecolor='white')
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.75, 1.7], wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    ax_tbl = fig.add_subplot(gs[0, 1])

    for _, row in means.iterrows():
        model = row['answer_model']
        ax.scatter(row['constructive_mean'], row['responsible_mean'], s=90, color=model_color(model), edgecolors='white', linewidths=0.8)
        ax.text(row['constructive_mean'] + 0.16, row['responsible_mean'] + 0.04, display_name(model), fontsize=9)
    ax.set_xlabel('Constructive index')
    ax.set_ylabel('Responsible index')
    ax.set_title(title, pad=12)
    ax.grid(color='#E3E3E3', linewidth=0.8, alpha=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax_tbl.axis('off')
    ax_tbl.set_title('Numeric Summary', fontsize=15, pad=10)
    cell_text = []
    for _, row in means.iterrows():
        cell_text.append([
            display_name(row['answer_model']),
            f"{row['constructive_mean']:.2f}",
            f"{row['responsible_mean']:.2f}",
        ])
    table = ax_tbl.table(
        cellText=cell_text,
        colLabels=['Model', 'Constructive', 'Responsible'],
        cellLoc='left',
        colLoc='left',
        loc='center',
        bbox=[0.0, 0.06, 1.0, 0.88],
        colWidths=[0.48, 0.26, 0.26],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    records = means.to_dict('records')
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#DDDDDD')
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor('#F2F2F2')
            cell.set_text_props(weight='bold', color='#111111')
        else:
            cell.set_facecolor('white')
            idx = r - 1
            if c == 0 and 0 <= idx < len(records):
                cell.set_text_props(color=model_color(records[idx]['answer_model']), weight='bold')

    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches='tight')
    plt.close(fig)


def make_pca_rubric_space(df: pd.DataFrame, title: str, out_png: Path, out_csv: Path) -> None:
    apply_base_theme()
    mat = df.groupby('answer_model')[RUBRICS].mean()
    order = ordered_models(mat.index.astype(str).tolist())
    mat = mat.loc[order]
    X2, ratio = pca_2d(mat.values.astype(float))
    export = pd.DataFrame({'model': mat.index, 'pc1': X2[:, 0], 'pc2': X2[:, 1]})
    export.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(8.4, 6.8), facecolor='white')
    for i, model in enumerate(mat.index.astype(str).tolist()):
        ax.scatter(X2[i, 0], X2[i, 1], s=95, color=model_color(model), edgecolors='white', linewidths=0.8)
        ax.text(X2[i, 0] + 0.12, X2[i, 1] + 0.04, display_name(model), fontsize=9)
    ax.set_xlabel(f'PC1 ({ratio[0]*100:.1f}% variance)')
    ax.set_ylabel(f'PC2 ({ratio[1]*100:.1f}% variance)')
    ax.set_title(title, pad=12)
    ax.axhline(0, color='#CCCCCC', linewidth=0.8)
    ax.axvline(0, color='#CCCCCC', linewidth=0.8)
    ax.grid(color='#E8E8E8', linewidth=0.8, alpha=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches='tight')
    plt.close(fig)




def make_rubric_loading_pca(df: pd.DataFrame, title: str, out_png: Path, out_csv: Path) -> None:
    apply_base_theme()
    X = df[RUBRICS].dropna().astype(float)
    # standardize rubric columns so loadings reflect shared structure rather than scale
    Xz = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    Xz = Xz.replace([np.inf, -np.inf], np.nan).dropna()

    cov = np.cov(Xz.values, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    loadings = eigvecs[:, :2] * np.sqrt(np.maximum(eigvals[:2], 0.0))
    ratio = eigvals[:2] / (eigvals.sum() + 1e-12)

    rows = []
    for i, rubric in enumerate(RUBRICS):
        group = 'Constructive' if rubric in CONSTRUCTIVE_RUBRICS else 'Responsible'
        rows.append({
            'rubric': rubric,
            'group': group,
            'pc1_loading': float(loadings[i, 0]),
            'pc2_loading': float(loadings[i, 1]),
        })
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(9.4, 7.6), facecolor='white')
    color_map = {'Constructive': '#123B7A', 'Responsible': '#0F766E'}
    for _, row in out.iterrows():
        ax.scatter(row['pc1_loading'], row['pc2_loading'], s=95, color=color_map[row['group']], edgecolors='white', linewidths=0.8)
        ax.text(row['pc1_loading'] + 0.012, row['pc2_loading'] + 0.012, row['rubric'].replace('_', ' '), fontsize=10)

    ax.axhline(0, color='#7F8C8D', linewidth=1.0)
    ax.axvline(0, color='#7F8C8D', linewidth=1.0)
    ax.set_xlabel(f'PC1 loading ({ratio[0]*100:.1f}% variance)')
    ax.set_ylabel(f'PC2 loading ({ratio[1]*100:.1f}% variance)')
    ax.set_title(title, pad=12)
    ax.grid(color='#E6E6E6', linewidth=0.8, alpha=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', label='Constructive', markerfacecolor='#123B7A', markersize=9),
        plt.Line2D([0], [0], marker='o', color='w', label='Responsible', markerfacecolor='#0F766E', markersize=9),
    ]
    ax.legend(handles=handles, frameon=False, loc='lower left')
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches='tight')
    plt.close(fig)

def make_grouping_figure(title: str, out_png: Path) -> None:
    apply_base_theme()
    fig, ax = plt.subplots(figsize=(10.0, 6.0), facecolor='white')
    ax.axis('off')

    left_x, right_x = 0.14, 0.58
    top_y = 0.80
    gap = 0.11

    ax.text(left_x + 0.13, 0.92, 'Constructive', ha='center', va='center', fontsize=16, fontweight='bold', color='#123B7A')
    ax.text(right_x + 0.13, 0.92, 'Responsible', ha='center', va='center', fontsize=16, fontweight='bold', color='#0F766E')

    for i, rubric in enumerate(CONSTRUCTIVE_RUBRICS):
        y = top_y - i * gap
        rect = plt.Rectangle((left_x, y), 0.26, 0.075, facecolor='#EAF1FB', edgecolor='#123B7A', linewidth=1.2)
        ax.add_patch(rect)
        ax.text(left_x + 0.13, y + 0.0375, rubric.replace('_', ' ').title(), ha='center', va='center', fontsize=12)

    for i, rubric in enumerate(RESPONSIBLE_RUBRICS):
        y = top_y - i * gap
        rect = plt.Rectangle((right_x, y), 0.26, 0.075, facecolor='#E5F5F2', edgecolor='#0F766E', linewidth=1.2)
        ax.add_patch(rect)
        ax.text(right_x + 0.13, y + 0.0375, rubric.replace('_', ' ').title(), ha='center', va='center', fontsize=12)

    ax.set_title(title, pad=12)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches='tight')
    plt.close(fig)
