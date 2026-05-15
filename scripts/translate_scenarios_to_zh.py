# scripts/translate_scenarios_to_zh.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from src.model_caller import call_model
from src.config import sanitize_tag


TRANSLATE_PROMPT = """你是一名专业育儿场景数据集编辑。
请把下面的英文育儿场景翻译成自然、真实的中文表达，要求：
- 保持语义一致，不要添加或删除关键事实
- 保持风险级别与语气（不要“变更严重程度”）
- 不引入额外文化假设（除非原文包含）
- 输出只包含翻译后的中文文本（不要解释，不要加引号）

英文原文：
{en_text}
"""


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/scenarios/views/en/parentbench_v0.jsonl")
    ap.add_argument("--output", default="data/scenarios/views/zh/parentbench_v0_zh.jsonl")
    ap.add_argument("--backend", default="openai", choices=["openai", "ollama", "local"])
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(in_path)
    if args.max_items is not None:
        rows = rows[: args.max_items]

    with out_path.open("w", encoding="utf-8") as f:
        for i, r in enumerate(rows, start=1):
            uid = r.get("scenario_uid") or r.get("id") or ""
            en_text = r.get("scenario_text") or r.get("prompt") or ""
            if not en_text.strip():
                continue

            print(f"[{i}/{len(rows)}] translate uid={uid}")

            if args.dry_run:
                zh_text = "【DRY_RUN】" + en_text
            else:
                zh_text = call_model(
                    prompt=TRANSLATE_PROMPT.format(en_text=en_text),
                    backend=args.backend,
                    model=args.model,
                    temperature=0.2,
                    max_tokens=1200,
                ) or ""
                zh_text = str(zh_text).strip()

            out_rec: Dict[str, Any] = {
                "scenario_uid": uid,
                "language": "zh",
                "source": "translated",
                "scenario_text": zh_text,
                "metadata": r.get("metadata", {}),
                "origin": {"from_language": "en", "translator_backend": args.backend, "translator_model": args.model},
            }
            f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    print("✅ wrote zh view ->", out_path.resolve())


if __name__ == "__main__":
    main()
