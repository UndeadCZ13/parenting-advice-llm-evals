#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate a long-form analysis report with image understanding.

Pipeline:
1) Load prompt & environment
2) Collect evidence from results/analysis (+ optional results/merged)
3) Run vision model (gpt-4o-mini) to caption images
4) Run gpt-5-nano to generate long-form report
5) Write Markdown output
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import List, Tuple, Optional

from dotenv import load_dotenv

# ---- ensure src is importable ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_caller import call_model
from scripts.vision_model_caller import caption_images, VisionCaption


# -----------------------------
# Config
# -----------------------------
TEXT_EXTS = {".txt", ".md", ".json", ".jsonl", ".yaml", ".yml"}
TABLE_EXTS = {".csv", ".tsv"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

PRIORITY_KEYWORDS = [
    "/stats/",
    "/radar/",
    "/boxplots/",
    "/winrate/",
    "/judge_agreement/",
    "/cross_language/",
    "/scenario_delta_heatmaps/",
    "/embedding/",
    "/scenarios/",
    "/generated_plots/",
]


# -----------------------------
# Helpers
# -----------------------------
def score_path_priority(rel: str) -> int:
    rel = rel.lower().replace("\\", "/")
    for i, kw in enumerate(PRIORITY_KEYWORDS):
        if kw in rel:
            return i
    return 999


def read_text_sample(path: Path, max_chars: int = 8000) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n\n...[TRUNCATED]...\n\n" + text[-half:]


def summarize_csv(path: Path) -> str:
    try:
        import pandas as pd  # type: ignore
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
        lines = []
        lines.append(f"shape: {df.shape[0]} rows x {df.shape[1]} cols")
        lines.append(f"columns: {list(df.columns)}")

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols:
            desc = df[numeric_cols].describe().round(4)
            lines.append("numeric_describe:\n" + desc.to_string())

        head = df.head(8).to_string(index=False)
        tail = df.tail(6).to_string(index=False) if df.shape[0] > 8 else ""
        lines.append("\nhead:\n" + head)
        if tail:
            lines.append("\ntail:\n" + tail)
        return "\n".join(lines)
    except Exception:
        return read_text_sample(path, max_chars=9000)


def collect_candidates(
    project_root: Path,
    analysis_dir: Path,
    merged_dir: Optional[Path],
) -> Tuple[List[Path], List[Path], List[Path]]:
    files: List[Path] = []

    for p in analysis_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in (TEXT_EXTS | TABLE_EXTS | IMAGE_EXTS):
            files.append(p)

    if merged_dir and merged_dir.exists():
        for p in merged_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in (TEXT_EXTS | TABLE_EXTS):
                files.append(p)

    scored = []
    for p in files:
        rel = p.relative_to(project_root).as_posix()
        scored.append((score_path_priority(rel), rel, p))
    scored.sort(key=lambda x: (x[0], x[1]))

    tables = [p for _, _, p in scored if p.suffix.lower() in TABLE_EXTS]
    texts = [p for _, _, p in scored if p.suffix.lower() in TEXT_EXTS]
    images = [p for _, _, p in scored if p.suffix.lower() in IMAGE_EXTS]

    return tables, texts, images


def build_evidence_text(
    project_root: Path,
    tables: List[Path],
    texts: List[Path],
    max_tables: int,
    max_texts: int,
) -> str:
    blocks: List[str] = []

    blocks.append("# FILE INDEX（证据索引，节选）\n")
    for p in tables[:max_tables]:
        blocks.append(f"- {p.relative_to(project_root).as_posix()}")
    for p in texts[:max_texts]:
        blocks.append(f"- {p.relative_to(project_root).as_posix()}")

    if tables:
        blocks.append("\n# TABLE SUMMARIES\n")
        for p in tables[:max_tables]:
            blocks.append(f"## {p.relative_to(project_root).as_posix()}")
            blocks.append("```text")
            blocks.append(summarize_csv(p))
            blocks.append("```")

    if texts:
        blocks.append("\n# TEXT SUMMARIES\n")
        for p in texts[:max_texts]:
            blocks.append(f"## {p.relative_to(project_root).as_posix()}")
            blocks.append("```text")
            blocks.append(read_text_sample(p))
            blocks.append("```")

    return "\n".join(blocks)


def build_image_section(project_root: Path, caps: List[VisionCaption]) -> str:
    lines: List[str] = []
    lines.append("\n# IMAGE EVIDENCE（图像证据 + 读图分析）\n")

    for idx, c in enumerate(caps, 1):
        p = Path(c.image_path).resolve()
        rel = p.relative_to(project_root).as_posix()
        lines.append(f"## Figure {idx}: {rel}\n")
        lines.append(f"![]({rel})\n")
        lines.append("**Vision model 图像解读（证据性段落）**：\n")
        lines.append(c.caption.strip() + "\n")

    return "\n".join(lines)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--prompt", default="scripts/report_prompt/report_prompt.txt")
    ap.add_argument("--analysis-dir", default="results/analysis")
    ap.add_argument("--merged-dir", default="results/merged")
    ap.add_argument("--include-merged", action="store_true")

    ap.add_argument("--vision-model", default="gpt-4o-mini")
    ap.add_argument("--report-model", default="gpt-5-nano")

    ap.add_argument("--max-tables", type=int, default=50)
    ap.add_argument("--max-texts", type=int, default=5)
    ap.add_argument("--max-images", type=int, default=250)

    ap.add_argument("--max-tokens", type=int, default=20000)
    ap.add_argument("--temperature", type=float, default=0.2)

    ap.add_argument("--out", default="")
    args = ap.parse_args()

    # ---------- Stage 1 ----------
    print("[1/5] Loading environment and prompt...")
    root = Path(args.project_root).resolve()
    load_dotenv(root / ".env")
    prompt_template = (root / args.prompt).read_text(encoding="utf-8")
    analysis_dir = root / args.analysis_dir
    merged_dir = root / args.merged_dir if args.include_merged else None

    # ---------- Stage 2 ----------
    print("[2/5] Collecting and summarizing experiment artifacts...")
    tables, texts, images = collect_candidates(root, analysis_dir, merged_dir)
    print(
        f"      Found {len(tables)} tables, {len(texts)} text files, "
        f"{len(images)} images (before filtering)"
    )

    evidence_text = build_evidence_text(
        root, tables, texts, args.max_tables, args.max_texts
    )

    # ---------- Stage 3 ----------
    selected_images = images[: args.max_images]
    print(
        f"[3/5] Running vision model on {len(selected_images)} images "
        f"(model={args.vision_model})..."
    )
    print("      This step may take some time. Please wait...")

    vision_prompt = (
        "你将看到若干张来自 ParentBench-LLM-Evals 的评测图表。\n"
        "请对每一张图生成“可直接写入论文或技术报告的分析性段落”，每张不少于 200 字，包含：\n"
        "- 图的评测含义\n"
        "- 关键趋势与对比\n"
        "- 对模型能力 / 稳定性 / 语言差异的解释\n"
        "- 局限性说明\n"
        "请严格按顺序输出，对应输入图片顺序。"
    )

    image_captions = caption_images(
        selected_images,
        prompt=vision_prompt,
        model=args.vision_model,
        max_images_per_request=4,
        temperature=0.2,
    )

    print(f"      Vision stage complete. Generated captions for {len(image_captions)} images.")

    image_section = build_image_section(root, image_captions)

    # ---------- Stage 4 ----------
    print(
        f"[4/5] Generating long-form report with {args.report_model} "
        f"(max_tokens={args.max_tokens})..."
    )
    print("      This is the longest step. Please wait...")

    final_prompt = (
        prompt_template
        + "\n\n---\n\n"
        + "以下为自动收集的实验结果证据材料（表格摘要 + 图像分析）：\n\n"
        + evidence_text
        + "\n\n---\n\n"
        + image_section
    )

    result = call_model(
        prompt=final_prompt,
        backend="openai",
        model=args.report_model,
        system_prompt=None,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        reasoning_effort="low",
    )

    # ---------- Stage 5 ----------
    print("[5/5] Writing report to disk...")

    report_text = (result.get("text") or "").strip()
    meta = {
        "vision_model": args.vision_model,
        "report_model": args.report_model,
        "max_tokens": args.max_tokens,
        "finish_reason": result.get("finish_reason"),
        "error": result.get("error"),
        "generated_at": dt.datetime.now().isoformat(),
    }

    if args.out:
        out_path = root / args.out
    else:
        out_dir = root / "results" / "analysis" / "report"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"llm_report_with_vision_{args.report_model}_{ts}.md"

    out_path.write_text(
        "# (AUTO-GENERATED REPORT)\n\n"
        "```json\n" + json.dumps(meta, ensure_ascii=False, indent=2) + "\n```\n\n"
        + report_text,
        encoding="utf-8",
    )

    print(f"[DONE] Long-form report written to: {out_path}")


if __name__ == "__main__":
    main()
