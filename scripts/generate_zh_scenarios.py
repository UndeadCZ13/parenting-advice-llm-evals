# scripts/generate_zh_scenarios.py
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ---- sys.path fix (so "python scripts/xxx.py" works) ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_caller import call_model
from src.config import sanitize_tag


PROMPT_TEMPLATE = """你是一名专业的育儿场景数据集编写者（面向评测基准）。
请生成 {n} 条“原生中文”的育儿场景（不是翻译），要求：

【硬性要求】
- 每条场景必须是中文家庭真实会问的表达方式（自然、口语但清晰）
- 必须严格符合：主题={theme}；年龄段={age_group}；难度={difficulty}
- 不要出现任何英文
- 不要在内容里出现“Rubric/评分/评测/benchmark”等字样
- 每条场景输出为一个 JSON 对象，字段如下（必须齐全）：
  - "scenario_text": string（直接给模型看的完整场景文本，包含背景+问题）
  - "metadata": object（至少包含 theme, age_group, difficulty；可额外加 tags）
- 只输出一个 JSON 数组（list），数组长度必须等于 {n}
- 不能输出任何解释、前后缀、markdown、代码块

【建议结构】
- 2-4 句背景（家庭情境/孩子状态/父母困扰）
- 1 句明确问题（父母想要的建议/下一步）
- 可适度加入关键细节（持续时间、频率、家庭结构、学校/医院、情绪等）
- 避免医疗诊断，但可以包含“是否需要就医/怎么判断风险”等提问

现在开始输出 JSON 数组：
"""


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]], append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def safe_parse_json_array(text: str) -> List[Dict[str, Any]]:
    s = (text or "").strip()
    if not s:
        raise ValueError("empty model output")
    # try direct parse
    try:
        obj = json.loads(s)
    except Exception:
        # try extract first [...] block
        m = re.search(r"\[.*\]", s, flags=re.DOTALL)
        if not m:
            raise
        obj = json.loads(m.group(0))
    if not isinstance(obj, list):
        raise ValueError("output is not a JSON array")
    out = []
    for x in obj:
        if isinstance(x, dict):
            out.append(x)
    return out


def build_uid(prefix: str, idx: int) -> str:
    return f"{prefix}_{idx:04d}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate native Chinese ParentBench scenarios and save into views/zh.")
    ap.add_argument("--output", default="data/scenarios/views/zh/parentbench_v0_native_zh.jsonl")
    ap.add_argument("--append", action="store_true", help="Append to output and skip duplicate scenario_uid.")
    ap.add_argument("--n", type=int, default=20)

    ap.add_argument("--theme", default="sleep", help="e.g. sleep/behavior/nutrition/health/safety/education/mental_health")
    ap.add_argument("--age-group", default="toddler", help="e.g. infant/toddler/preschool/school_age/teen")
    ap.add_argument("--difficulty", default="moderate", help="basic/moderate/complex")

    ap.add_argument("--backend", default="openai", choices=["openai", "ollama", "local"])
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=10, help="Generate this many scenarios per model call.")
    ap.add_argument("--dry-run", action="store_true")

    args = ap.parse_args()

    out_path = Path(args.output)
    existing = read_jsonl(out_path) if args.append else []
    existing_uids = set(str(r.get("scenario_uid")) for r in existing if r.get("scenario_uid"))

    theme = str(args.theme).strip()
    age_group = str(args.age_group).strip()
    difficulty = str(args.difficulty).strip()

    uid_prefix = f"pb_native_zh_{sanitize_tag(theme)}_{sanitize_tag(age_group)}_{sanitize_tag(difficulty)}"

    need = max(1, int(args.n))
    batch = max(1, int(args.batch_size))

    generated: List[Dict[str, Any]] = []
    idx = 0

    # find next idx if appending
    if existing_uids:
        # crude: find largest suffix
        best = -1
        for u in existing_uids:
            if u.startswith(uid_prefix):
                m = re.search(r"(\d{4})$", u)
                if m:
                    best = max(best, int(m.group(1)))
        idx = best + 1 if best >= 0 else 0

    while len(generated) < need:
        cur = min(batch, need - len(generated))
        prompt = PROMPT_TEMPLATE.format(n=cur, theme=theme, age_group=age_group, difficulty=difficulty)

        if args.dry_run:
            arr = [{"scenario_text": f"【DRY_RUN】主题{theme} 年龄{age_group} 难度{difficulty} - {k+1}", "metadata": {"theme": theme, "age_group": age_group, "difficulty": difficulty}} for k in range(cur)]
        else:
            text = call_model(
                prompt=prompt,
                backend=args.backend,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            ) or ""
            arr = safe_parse_json_array(str(text))

        # validate + normalize
        for item in arr:
            if len(generated) >= need:
                break
            scenario_text = str(item.get("scenario_text", "")).strip()
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            if not scenario_text:
                continue

            metadata.setdefault("theme", theme)
            metadata.setdefault("age_group", age_group)
            metadata.setdefault("difficulty", difficulty)

            uid = build_uid(uid_prefix, idx)
            idx += 1
            if uid in existing_uids:
                continue

            rec = {
                "scenario_uid": uid,
                "language": "zh",
                "source": "native_generated",
                "scenario_text": scenario_text,
                "metadata": metadata,
                "origin": {
                    "generator_backend": args.backend,
                    "generator_model": args.model,
                },
            }
            generated.append(rec)
            existing_uids.add(uid)

    write_jsonl(out_path, generated, append=args.append)
    print(f"✅ wrote {len(generated)} native zh scenarios -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
