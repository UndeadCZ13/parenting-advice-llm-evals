from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from src.model_caller import call_model
from src.config import (
    DEFAULT_BACKEND_JUDGE,
    DEFAULT_OPENAI_MODEL_JUDGE,
    DEFAULT_OLLAMA_MODEL_KEY_JUDGE,
    DEFAULT_GROQ_MODEL_KEY_JUDGE,
    OLLAMA_MODEL_REGISTRY,
    GROQ_MODEL_REGISTRY,
)


# ==============================
# 2D Style Judge Prompts (EN/ZH)
# ==============================

SYSTEM_PROMPT_EN = (
    "You are an expert in developmental psychology and parenting communication.\n"
    "Your job is NOT to rate quality, but to score two independent behavioral dimensions of the response.\n\n"
    "Return ONLY a valid JSON object. No extra text.\n"
)

USER_TEMPLATE_EN = """
We are analyzing the STYLE of AI-generated parenting advice.

Given:
- A scenario (context)
- A model response (advice)

Score two INDEPENDENT dimensions, each as a decimal in [0,1] with up to 3 decimals:

1) responsiveness (R): emotional attunement, warmth, validation, listening, engagement with the child's feelings/needs.
   Anchors:
   - 0.0: dismissive/ignores feelings; cold, detached, or generic.
   - 0.5: some acknowledgement, but limited empathy or shallow engagement.
   - 1.0: clearly validates feelings, shows warmth, asks/reflects, supportive tone.

2) demandingness (D): clarity of expectations, boundaries, structure, accountability, consistency.
   Anchors:
   - 0.0: no expectations/boundaries; purely accommodating; “do whatever you want”.
   - 0.5: suggests gentle structure but boundaries are weak/optional.
   - 1.0: clear expectations and boundaries; concrete steps; accountability while remaining non-abusive.

Important:
- Do NOT assume “supportive” means high demandingness.
- If the response is warm but avoids firm limits, D should be low (more permissive).
- Do NOT reward/penalize the response—this is descriptive only.

Output STRICT JSON only:
{{
  "responsiveness": float,
  "demandingness": float
}}

=== SCENARIO ===
{scenario_text}

=== MODEL RESPONSE ===
{model_answer}
""".strip()

SYSTEM_PROMPT_ZH = (
    "你是一名发展心理学与亲子沟通研究专家。\n"
    "你的任务不是评价“好坏”，而是对回答的风格进行两个维度的连续打分。\n\n"
    "只输出一个合法 JSON，不要输出任何解释或多余文字。\n"
)

USER_TEMPLATE_ZH = """
我们要分析 AI 育儿建议的“风格”（不是质量）。

给定：
- 场景描述（情境背景）
- 模型回答（给家长的建议）

请对两个彼此独立的维度打分，每个维度取 0~1 的小数（最多 3 位小数）：

1）回应度 responsiveness（R）：情绪承接/共情/温暖/倾听/对孩子感受与需要的关注程度
   锚点：
   - 0.0：忽视或否定情绪；冷淡/敷衍/泛泛而谈
   - 0.5：有一定安抚或承接，但较浅、缺少深入理解
   - 1.0：明确共情与肯定感受，语气温暖，能反映/追问，支持性强

2）要求度 demandingness（D）：规则与边界清晰度、对行为的期望与约束、结构化程度、一致性与执行力度
   锚点：
   - 0.0：几乎不设边界与期望；完全顺从/随便孩子；“想怎样就怎样”
   - 0.5：有一些建议或结构，但边界较弱、可有可无
   - 1.0：边界与期望非常清晰，给出具体可执行步骤，并强调责任/一致性（但不包含羞辱、威胁或伤害）

重要：
- “语气支持”不等于 D 高。温暖但不设限 → D 应该偏低（更接近放任）。
- 这不是好坏评测，只是描述性打分。

严格只输出 JSON：
{{
  "responsiveness": float,
  "demandingness": float
}}

=== 场景 ===
{scenario_text}

=== 模型回答 ===
{model_answer}
""".strip()

# stricter retry prompt
RETRY_SYSTEM = (
    "Output ONLY valid JSON. No markdown. No code fences. No explanations.\n"
    "Return exactly one JSON object with keys: responsiveness, demandingness.\n"
    "Values must be decimals between 0 and 1.\n"
)
RETRY_SUFFIX = "\n\nREMINDER: Output STRICT JSON only. No extra text."


# ==============================
# JSON extraction + parsing
# ==============================

DIM_KEYS = ["responsiveness", "demandingness"]


def _extract_json_object(text: str) -> Optional[str]:
    if not text or not isinstance(text, str):
        return None
    s = text.strip()
    # strip fences
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)

    m = re.search(r"\{.*?\}", s, flags=re.DOTALL)
    if not m:
        return None
    return m.group(0).strip()


def safe_parse_dims(raw_text: str) -> Optional[Dict[str, float]]:
    json_str = _extract_json_object(raw_text)
    if not json_str:
        return None
    try:
        obj = json.loads(json_str)
        if not isinstance(obj, dict):
            return None
        for k in DIM_KEYS:
            if k not in obj:
                return None
        out: Dict[str, float] = {}
        for k in DIM_KEYS:
            v = float(obj[k])
            # clamp check
            if v < 0.0 or v > 1.0:
                return None
            out[k] = round(v, 3)
        return out
    except Exception:
        return None


# ==============================
# IO helpers
# ==============================

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_done_uids(path: Path) -> set:
    done = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                uid = obj.get("scenario_uid")
                if uid:
                    done.add(uid)
            except Exception:
                continue
    return done


def infer_language(model_file: Path, item: Dict[str, Any]) -> str:
    """
    Best-effort language inference:
    1) item["language"] if exists
    2) filename contains '__zh' or startswith 'zh_' or contains '/zh/'
    3) default 'en'
    """
    lang = (item.get("language") or "").lower().strip()
    if lang in ("zh", "en"):
        return lang

    name = model_file.as_posix().lower()
    stem = model_file.stem.lower()

    if "/zh/" in name or stem.startswith("zh_") or "__zh" in stem or "_zh_" in stem:
        return "zh"
    if "/en/" in name or stem.startswith("en_") or "__en" in stem or "_en_" in stem:
        return "en"

    return "en"


def build_prompt(lang: str, scenario_text: str, model_answer: str) -> Tuple[str, str]:
    """
    returns: (system_prompt, user_prompt)
    """
    if lang == "zh":
        return (
            SYSTEM_PROMPT_ZH,
            USER_TEMPLATE_ZH.format(scenario_text=scenario_text or "", model_answer=model_answer or ""),
        )
    return (
        SYSTEM_PROMPT_EN,
        USER_TEMPLATE_EN.format(scenario_text=scenario_text or "", model_answer=model_answer or ""),
    )


def resolve_judge_model(
    backend: str,
    judge_model: str,
    openai_model: str,
    ollama_model_key: str,
    groq_model_key: str,
) -> str:
    b = (backend or "").lower().strip()
    if b == "local":
        b = "ollama"

    if judge_model and str(judge_model).strip():
        raw = str(judge_model).strip()
        if b == "ollama":
            return OLLAMA_MODEL_REGISTRY.get(raw, raw)
        if b == "groq":
            return GROQ_MODEL_REGISTRY.get(raw, raw)
        return raw

    if b == "openai":
        return openai_model
    if b == "ollama":
        return OLLAMA_MODEL_REGISTRY.get(ollama_model_key, ollama_model_key)
    if b == "groq":
        return GROQ_MODEL_REGISTRY.get(groq_model_key, groq_model_key)
    raise ValueError(f"Unknown backend={backend}")


# ==============================
# Core judging logic
# ==============================

def judge_one(
    lang: str,
    scenario_text: str,
    model_answer: str,
    judge_backend: str,
    judge_model: str,
    max_tokens: int,
    parse_retries: int,
) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, Any]]]:
    system_prompt, user_prompt = build_prompt(lang, scenario_text, model_answer)

    last_result: Optional[Dict[str, Any]] = None

    result = call_model(
        prompt=user_prompt,
        backend=judge_backend,
        model=judge_model,
        system_prompt=system_prompt,
        temperature=0.0,
        max_tokens=max_tokens,
    )
    last_result = result
    text = (result.get("text") or "").strip()
    parsed = safe_parse_dims(text)
    if parsed is not None:
        return parsed, last_result

    for _ in range(parse_retries):
        retry_prompt = user_prompt + RETRY_SUFFIX
        result = call_model(
            prompt=retry_prompt,
            backend=judge_backend,
            model=judge_model,
            system_prompt=RETRY_SYSTEM,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        last_result = result
        text = (result.get("text") or "").strip()
        parsed = safe_parse_dims(text)
        if parsed is not None:
            return parsed, last_result

    return None, last_result


def compute_total_tasks(
    model_files: List[Path],
    out_dir: Path,
    judge_model: str,
    resume: bool,
) -> int:
    """
    Total number of NEW items to judge across all model files (for overall progress bar).
    """
    total = 0
    for mf in model_files:
        model_name = mf.stem
        out_path = out_dir / f"{model_name}__dims__judge={judge_model}.jsonl"
        done = read_done_uids(out_path) if resume else set()

        # count quickly by scanning jsonl (needs parse to get uid)
        with mf.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                uid = obj.get("scenario_uid")
                if not uid:
                    continue
                if resume and uid in done:
                    continue
                total += 1
    return total


def run_style_judging(
    model_outputs_dir: str,
    output_dir: str,
    judge_model: str = "gpt-5.2",
    judge_backend: str = "openai",
    max_tokens: int = 256,
    parse_retries: int = 1,
    include_subdirs: bool = False,
    resume: bool = True,
) -> None:
    in_dir = Path(model_outputs_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_files = sorted(in_dir.rglob("*.jsonl") if include_subdirs else in_dir.glob("*.jsonl"))
    if not model_files:
        print(f"[ERROR] No jsonl files found in: {in_dir}")
        return

    # ---- overall progress (all models * all answers) ----
    total_tasks = compute_total_tasks(model_files, out_dir, judge_model, resume)
    overall = tqdm(total=total_tasks, ncols=90, desc="ALL-MODELS", unit="ans")

    for model_file in model_files:
        model_name = model_file.stem
        print(f"\n🔎 2D style judging: {model_name}")

        out_path = out_dir / f"{model_name}__dims__judge={judge_model}.jsonl"
        done_uids = read_done_uids(out_path) if resume else set()

        rows = load_jsonl(model_file)
        total = len(rows)

        pbar = tqdm(total=total, ncols=90, desc=f"{model_name}", unit="item")
        written = 0
        skipped = 0

        for item in rows:
            pbar.update(1)

            scenario_uid = item.get("scenario_uid")
            if not scenario_uid:
                skipped += 1
                continue
            if resume and scenario_uid in done_uids:
                continue

            scenario_text = item.get("scenario_text", "")
            model_answer = item.get("answer", "")

            # infer language
            lang = infer_language(model_file, item)

            # Empty answer -> record but still count as "done" for overall
            if not isinstance(model_answer, str) or not model_answer.strip():
                record = {
                    "scenario_uid": scenario_uid,
                    "language": lang,
                    "model_name": model_name,
                    "judge_model": judge_model,
                    "judge_backend": judge_backend,
                    "responsiveness": 0.0,
                    "demandingness": 0.0,
                    "error": "empty_answer",
                }
                append_jsonl(out_path, record)
                written += 1
                overall.update(1)
                continue

            parsed, last_result = judge_one(
                lang=lang,
                scenario_text=scenario_text,
                model_answer=model_answer,
                judge_backend=judge_backend,
                judge_model=judge_model,
                max_tokens=max_tokens,
                parse_retries=parse_retries,
            )

            if parsed is None:
                preview = ""
                if last_result and isinstance(last_result.get("text"), str):
                    preview = last_result["text"][:300]
                record = {
                    "scenario_uid": scenario_uid,
                    "language": lang,
                    "model_name": model_name,
                    "judge_model": judge_model,
                    "judge_backend": judge_backend,
                    "error": last_result.get("error") if isinstance(last_result, dict) else "parse_failed",
                    "raw_preview": preview,
                }
                append_jsonl(out_path, record)
                overall.update(1)
                continue

            record = {
                "scenario_uid": scenario_uid,
                "language": lang,
                "model_name": model_name,
                "judge_model": judge_model,
                "judge_backend": judge_backend,
                **parsed,
            }
            append_jsonl(out_path, record)
            written += 1
            overall.update(1)

        pbar.close()
        print(f"✅ Saved: {out_path} (written={written}, skipped={skipped}, total={total})")

    overall.close()
    print("[DONE] 2D style judging complete.")


# ==============================
# CLI
# ==============================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_outputs_dir", type=str, default="data/model_outputs")
    ap.add_argument("--output_dir", type=str, default="data/judge_outputs/style_2d")
    ap.add_argument("--judge_backend", type=str, default=DEFAULT_BACKEND_JUDGE, help="openai|ollama|groq|local")
    ap.add_argument("--judge_model", type=str, default="", help="exact model name or backend-specific registry key")
    ap.add_argument("--openai_model", type=str, default=DEFAULT_OPENAI_MODEL_JUDGE)
    ap.add_argument("--ollama_model_key", type=str, default=DEFAULT_OLLAMA_MODEL_KEY_JUDGE)
    ap.add_argument("--groq_model_key", type=str, default=DEFAULT_GROQ_MODEL_KEY_JUDGE)
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--parse_retries", type=int, default=1)
    ap.add_argument("--include_subdirs", action="store_true")
    ap.add_argument("--no_resume", action="store_true")
    args = ap.parse_args()

    judge_backend = args.judge_backend.lower().replace("local", "ollama")
    judge_model = resolve_judge_model(
        backend=judge_backend,
        judge_model=args.judge_model,
        openai_model=args.openai_model,
        ollama_model_key=args.ollama_model_key,
        groq_model_key=args.groq_model_key,
    )

    run_style_judging(
        model_outputs_dir=args.model_outputs_dir,
        output_dir=args.output_dir,
        judge_model=judge_model,
        judge_backend=judge_backend,
        max_tokens=args.max_tokens,
        parse_retries=args.parse_retries,
        include_subdirs=args.include_subdirs,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
