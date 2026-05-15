from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import time
from pathlib import Path
from typing import Dict, Any, Optional

from src.model_caller import call_model
from src.config import (
    STRIP_REASONING,
    ENABLE_TRUNCATION_RETRY,
    MAX_RETRIES_ON_TRUNCATION,
    MAX_TOKENS_DEFAULT,
    MAX_TOKENS_REASONING_MODEL,
    REASONING_MODEL_HINTS,
    RETRY_TOKEN_MULTIPLIER,
    OLLAMA_MODEL_REGISTRY,
    GROQ_MODEL_REGISTRY,
)


def is_reasoning_model(model_name: str) -> bool:
    name = (model_name or "").lower()
    return any(h in name for h in REASONING_MODEL_HINTS)


def build_final_only_prompt(user_prompt: str, lang: str) -> str:
    """
    Model acts as an AI assistant advising a parent (not speaking as the parent).
    Language-aligned output.
    """
    l = (lang or "en").lower().strip()

    if l in ("zh", "zh-cn", "zh-hans", "chinese"):
        return (
            "你是一名 AI 助手，正在为家长提供建议。\n\n"
            "不要输出任何思考过程、推理过程或草稿。\n"
            "请在内部完成推理，只输出最终、完整、可执行的建议。\n\n"
            "请基于下述情境，为家长提供如何回应孩子的指导性建议。\n"
            "你的建议应当是现实可行、具有支持性的，\n"
            "体现家长在该情境下可以采取的回应方式或做法。\n\n"
            "请用中文作答。\n\n"
            f"{user_prompt}"
        )

    if l in ("en_zh", "zh_en", "bilingual"):
        return (
            "You are an AI assistant providing advice to a parent.\n\n"
            "Do NOT output any chain-of-thought, reasoning, or draft.\n"
            "Think privately, and output ONLY the final, complete, actionable advice.\n\n"
            "Based on the situation described below, provide guidance on how a parent\n"
            "could respond to their child. Your advice should be practical and supportive,\n"
            "reflecting what a parent might reasonably say or do in this situation.\n\n"
            "Output in TWO sections:\n"
            "1) English\n"
            "2) 中文\n\n"
            f"{user_prompt}"
        )

    return (
        "You are an AI assistant providing advice to a parent.\n\n"
        "Do NOT output any chain-of-thought, reasoning, or draft.\n"
        "Think privately, and output ONLY the final, complete, actionable advice.\n\n"
        "Based on the situation described below, provide guidance on how a parent\n"
        "could respond to their child. Your advice should be practical and supportive,\n"
        "reflecting what a parent might reasonably say or do in this situation.\n\n"
        "Write your answer in English.\n\n"
        f"{user_prompt}"
    )


def strip_reasoning(text: str) -> Dict[str, Any]:
    raw = text or ""
    stripped = raw

    patterns = [
        r"<think>.*?</think>",
        r"思考：.*",
        r"推理：.*",
        r"Reasoning:.*",
        r"Thoughts:.*",
    ]
    for p in patterns:
        stripped = re.sub(p, "", stripped, flags=re.DOTALL | re.IGNORECASE)

    for key in ["Final:", "Answer:", "最终:", "最终建议:"]:
        if key in stripped:
            stripped = stripped.split(key, 1)[1]

    stripped = stripped.strip()

    return {
        "final": stripped,
        "stripped": stripped != raw,
        "removed_chars": max(0, len(raw) - len(stripped)),
    }


def suspected_truncation(text: str, finish_reason: str | None) -> bool:
    if finish_reason == "length":
        return True
    if not text:
        return True
    if len(text) > 20 and text[-1] not in "。.!?！？":
        return True
    return False


def _get_language(s: Dict[str, Any]) -> str:
    lang = (s.get("language") or s.get("lang") or "").strip()
    if lang:
        return lang
    src = (s.get("source") or "").lower()
    if "zh" in src:
        return "zh"
    return "en"


def _resolve_ollama_model(key_or_name: str) -> str:
    """
    Accept either registry key (e.g., qwen3_8b) or direct name (e.g., qwen3:8b).
    Returns the final model name to call.
    """
    s = (key_or_name or "").strip()
    if not s:
        return s
    if s in OLLAMA_MODEL_REGISTRY:
        return OLLAMA_MODEL_REGISTRY[s]
    # also accept a value directly equal to one of registry values
    if s in set(OLLAMA_MODEL_REGISTRY.values()):
        return s
    # fallback: treat as direct model name
    print(f"[INFO] Using ollama model name directly: '{s}'")
    return s


def _resolve_groq_model(key_or_name: str) -> str:
    """
    Accept either registry key (e.g., llama31_8b) or direct name (e.g., llama-3.1-8b-instant).
    Returns the final model name to call.
    """
    s = (key_or_name or "").strip()
    if not s:
        return s
    if s in GROQ_MODEL_REGISTRY:
        return GROQ_MODEL_REGISTRY[s]
    if s in set(GROQ_MODEL_REGISTRY.values()):
        return s
    print(f"[INFO] Using groq model name directly: '{s}'")
    return s


def _fmt_hms(sec: float) -> str:
    sec = max(0, int(sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--backend", required=True, choices=["openai", "ollama", "groq", "local"])
    ap.add_argument("--openai-model", default=None)
    ap.add_argument("--ollama-model-key", default=None)
    ap.add_argument("--groq-model-key", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-items", type=int, default=None, help="only run first N scenarios")
    args = ap.parse_args()

    backend = args.backend.lower()
    if backend == "local":
        backend = "ollama"

    scenarios = [json.loads(l) for l in Path(args.scenarios).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.max_items is not None:
        scenarios = scenarios[: args.max_items]

    t0 = time.time()
    total = len(scenarios)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve model name from backend + args
    if backend == "openai":
        if not args.openai_model:
            raise ValueError("--openai-model required for backend=openai")
        model_name = args.openai_model
    elif backend == "ollama":
        if not args.ollama_model_key:
            raise ValueError("--ollama-model-key required for backend=ollama")
        model_name = _resolve_ollama_model(args.ollama_model_key)
    elif backend == "groq":
        if not args.groq_model_key:
            raise ValueError("--groq-model-key required for backend=groq")
        model_name = _resolve_groq_model(args.groq_model_key)
    else:
        raise ValueError(f"Unknown backend={backend}")

    reasoning = is_reasoning_model(model_name)
    base_max_tokens = MAX_TOKENS_REASONING_MODEL if reasoning else MAX_TOKENS_DEFAULT

    print(f"[INFO] Start generation: {total} items | backend={backend} | model={model_name} | base_max_tokens={base_max_tokens}")
    print(f"[INFO] Output: {out_path}")

    with out_path.open("w", encoding="utf-8") as fout:
        for idx, s in enumerate(scenarios, start=1):
            uid = s.get("scenario_uid") or s.get("scenario_id") or s.get("id") or ""
            lang = _get_language(s).lower()

            user_prompt = s.get("scenario_text") or s.get("prompt") or s.get("question") or ""
            if not user_prompt:
                print(f"[WARN] empty scenario_text, skipped uid={uid}")
                continue

            # ---- Progress + ETA ----
            now = time.time()
            elapsed = now - t0
            done = idx
            remaining = max(0, total - done)

            avg_per_item = (elapsed / done) if done > 0 else 0.0
            eta_sec = avg_per_item * remaining
            finish_ts = now + eta_sec
            pct = (done / total * 100) if total else 0.0
            finish_local = dt.datetime.fromtimestamp(finish_ts).strftime("%Y-%m-%d %H:%M:%S")

            print(
                f"\n[{done}/{total}] ({pct:.1f}%) uid={uid} lang={lang} | "
                f"elapsed={_fmt_hms(elapsed)} avg/item={avg_per_item:.2f}s | "
                f"ETA={_fmt_hms(eta_sec)} finish~{finish_local}"
            )

            prompt = build_final_only_prompt(user_prompt, lang)

            max_tokens = base_max_tokens
            retries = 0

            while True:
                resp = call_model(
                    prompt=prompt,
                    backend=backend,
                    model=model_name,
                    max_tokens=max_tokens,
                )

                raw = resp.get("text", "")
                finish_reason = resp.get("finish_reason")
                api_error = resp.get("error")

                cleaned = strip_reasoning(raw) if STRIP_REASONING else {
                    "final": raw,
                    "stripped": False,
                    "removed_chars": 0,
                }

                trunc = suspected_truncation(cleaned["final"], finish_reason)

                print(f"  ↳ finish_reason={finish_reason} len={len(cleaned['final'])} trunc={trunc} api_error={'Y' if api_error else 'N'}")

                if trunc and ENABLE_TRUNCATION_RETRY and retries < MAX_RETRIES_ON_TRUNCATION:
                    retries += 1
                    max_tokens = int(max_tokens * RETRY_TOKEN_MULTIPLIER)
                    print(f"  ↳ trunc_retry {retries}/{MAX_RETRIES_ON_TRUNCATION}, max_tokens={max_tokens}")
                    continue

                record = {
                    **s,
                    "answer_raw": raw,
                    "answer": cleaned["final"],
                    "reasoning_stripped": cleaned["stripped"],
                    "removed_reasoning_chars": cleaned["removed_chars"],
                    "suspected_truncation": trunc,
                    "retry_used": retries > 0,
                    "model": model_name,
                    "backend": backend,
                    "finish_reason": finish_reason,
                    "api_error": api_error,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                break


if __name__ == "__main__":
    main()
