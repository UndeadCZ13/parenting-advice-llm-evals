#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Vision model caller (standalone):
- Reads local images
- Encodes as base64 data URL
- Calls OpenAI Responses API with input_image blocks
- Returns captions / insights for each image

Docs:
- Responses API supports image inputs (input_image) and text outputs. :contentReference[oaicite:1]{index=1}
- Images can be provided as a fully qualified URL or base64-encoded data URL. :contentReference[oaicite:2]{index=2}
"""

from __future__ import annotations

import base64
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv

try:
    from openai import OpenAI
except Exception as e:
    raise RuntimeError(
        "Missing dependency: openai. Install with: pip install openai"
    ) from e


@dataclass
class VisionCaption:
    image_path: str
    caption: str


def _guess_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "image/png"


def image_to_data_url(path: Path) -> str:
    mime = _guess_mime(path)
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def caption_images(
    image_paths: List[Path],
    prompt: str,
    model: str = "gpt-4o-mini",
    max_images_per_request: int = 4,
    temperature: float = 0.2,
) -> List[VisionCaption]:
    """
    Caption images using Responses API.
    We batch images to reduce overhead.

    The model will receive:
      - a text instruction
      - N images as input_image blocks
    and should output a structured caption per image.
    """
    client = OpenAI()

    results: List[VisionCaption] = []

    # batch
    for i in range(0, len(image_paths), max_images_per_request):
        batch = image_paths[i : i + max_images_per_request]

        content_blocks: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        for p in batch:
            data_url = image_to_data_url(p)
            content_blocks.append({"type": "input_image", "image_url": data_url})

        # Responses API call (text output)
        resp = client.responses.create(
            model=model,
            input=[{"role": "user", "content": content_blocks}],
            temperature=temperature,
        )

        # We ask the model to output per-image captions; parse as best-effort.
        text = getattr(resp, "output_text", None) or ""
        text = text.strip()

        # Best-effort split: we’ll map captions back to images in order.
        # Recommended: prompt the model to output numbered items 1..N.
        captions = _split_numbered_captions(text, len(batch))

        for p, cap in zip(batch, captions):
            results.append(VisionCaption(image_path=str(p), caption=cap.strip()))

    return results


def _split_numbered_captions(text: str, n: int) -> List[str]:
    """
    Best-effort parser for numbered outputs:
      1) ...
      2) ...
    Falls back to whole text if cannot parse.
    """
    import re

    # Match "1) ..." or "1. ..."
    parts = re.split(r"\n\s*(?=\d+[\.\)])", text)
    parts = [p.strip() for p in parts if p.strip()]

    # If we got n parts and each starts with number, strip numbering
    if len(parts) >= n and all(re.match(r"^\d+[\.\)]\s*", p) for p in parts[:n]):
        cleaned = [re.sub(r"^\d+[\.\)]\s*", "", p).strip() for p in parts[:n]]
        return cleaned

    # Fallback: return same text for first image, empties for rest
    if n <= 1:
        return [text]
    return [text] + ["(no caption parsed)"] * (n - 1)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".", help="Repo root (default: .)")
    ap.add_argument("--model", default="gpt-4o-mini", help="Vision model")
    ap.add_argument("--glob", default="results/analysis/**/*.png", help="Image glob (relative to root)")
    ap.add_argument("--max", type=int, default=8, help="Max images to caption")
    ap.add_argument("--out", default="results/analysis/report/image_captions.md", help="Output markdown")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    load_dotenv(root / ".env")

    paths = sorted(root.glob(args.glob))
    paths = [p for p in paths if p.is_file()][: args.max]

    prompt = (
        "你将看到若干张来自 LLM 评测工程的图表（雷达图/箱线图/胜率/热力图/PCA 等）。\n"
        "请对每张图做“可用于报告的证据性解读”，严格输出编号 1..N：\n"
        "- 这张图的类型与在评估中表示什么\n"
        "- 主要可观察到的趋势/对比（不要编造具体数值）\n"
        "- 支持的结论（1-2条）\n"
        "- 局限/需要配套表格的地方（如果有）\n"
    )

    caps = caption_images(paths, prompt=prompt, model=args.model)

    out_path = (root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# Image Captions (Vision Model)\n"]
    for c in caps:
        rel = Path(c.image_path).resolve().relative_to(root).as_posix()
        lines.append(f"## {rel}\n")
        lines.append(c.caption + "\n")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Wrote captions to: {out_path}")


if __name__ == "__main__":
    main()
