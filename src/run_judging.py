# src/run_judging.py
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import (
    DEFAULT_BACKEND_JUDGE,
    DEFAULT_OPENAI_MODEL_JUDGE,
    DEFAULT_OLLAMA_MODEL_KEY_JUDGE,
    DEFAULT_GROQ_MODEL_KEY_JUDGE,
    OLLAMA_MODEL_REGISTRY,
    GROQ_MODEL_REGISTRY,
    JUDGE_OUT_DIR,
    sanitize_tag,
)
from src.model_caller import call_model
from src.judges.judge_prompts import build_judge_prompt, RUBRIC_KEYS


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except json.JSONDecodeError as e:
                print(f"[WARN] JSON decode error, skipped: {e}")
    return rows


def safe_parse_json(text: Any) -> Dict[str, Any]:
    if text is None:
        raise ValueError("model output is None")
    if not isinstance(text, str):
        text = str(text)
    s = text.strip()
    if not s:
        raise ValueError("model output is empty")

    try:
        return json.loads(s)
    except Exception:
        pass

    m = re.search(r"```(?:json)?\s*(.*?)```", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        inner = m.group(1).strip()
        if inner:
            return json.loads(inner)

    i0 = s.find("{")
    i1 = s.rfind("}")
    if i0 != -1 and i1 != -1 and i1 > i0:
        return json.loads(s[i0 : i1 + 1])

    raise ValueError("cannot find valid JSON in output")


def try_get(rec: Dict[str, Any], keys: List[str], default: str = "") -> str:
    for k in keys:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return default


def aggregate_runs(raw_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in RUBRIC_KEYS:
        vals: List[float] = []
        for r in raw_runs:
            v = r.get(k)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        if not vals:
            out[k] = None
            out[f"{k}_std"] = None
        else:
            mean = sum(vals) / len(vals)
            var = sum((x - mean) ** 2 for x in vals) / (len(vals) if len(vals) > 1 else 1)
            out[k] = mean
            out[f"{k}_std"] = var ** 0.5

    comment: Optional[str] = None
    for r in raw_runs:
        c = r.get("comment")
        if isinstance(c, str) and c.strip():
            comment = c.strip()
            break
    out["comment"] = comment
    return out


def _extract_model_text(result: Any) -> str:
    if isinstance(result, dict):
        return str(result.get("text") or "")
    return str(result or "")


def resolve_judge_model(backend: str, openai_model: str, ollama_key_or_name: str, groq_key_or_name: str) -> str:
    b = backend.lower().strip()
    if b == "local":
        b = "ollama"

    if b == "openai":
        return openai_model

    if b == "ollama":
        raw = (ollama_key_or_name or "").strip()
        return OLLAMA_MODEL_REGISTRY.get(raw, raw)

    if b == "groq":
        raw = (groq_key_or_name or "").strip()
        return GROQ_MODEL_REGISTRY.get(raw, raw)

    raise ValueError(f"Unknown backend={backend}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True, help="generation output jsonl")
    ap.add_argument("--output", default=None)

    ap.add_argument("--backend", default=DEFAULT_BACKEND_JUDGE, choices=["openai", "ollama", "groq", "local"])
    ap.add_argument("--openai-model", default=DEFAULT_OPENAI_MODEL_JUDGE)
    ap.add_argument("--ollama-model-key", default=DEFAULT_OLLAMA_MODEL_KEY_JUDGE)
    ap.add_argument("--groq-model-key", default=DEFAULT_GROQ_MODEL_KEY_JUDGE)

    ap.add_argument("--n-repeats", type=int, default=3)
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")

    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--debug-dump-raw", action="store_true", help="dump raw judge outputs into each record")

    args = ap.parse_args()

    backend = args.backend.lower()
    if backend == "local":
        backend = "ollama"

    judge_model = resolve_judge_model(backend, args.openai_model, args.ollama_model_key, args.groq_model_key)

    in_path = Path(args.answers)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    records = read_jsonl(in_path)
    if args.max_items is not None:
        records = records[: args.max_items]

    if args.output:
        out_path = Path(args.output)
    else:
        out_name = f"{in_path.stem}_judged_{sanitize_tag(backend)}_{sanitize_tag(judge_model)}.jsonl"
        out_path = JUDGE_OUT_DIR / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_repeats = max(1, int(args.n_repeats))

    print(f"[INFO] Judging: backend={backend}, judge_model={judge_model}, n_repeats={n_repeats}")
    print(f"[INFO] Input: {in_path} ({len(records)} items)")
    print(f"[INFO] Output: {out_path}")
    if args.dry_run:
        print("[INFO] dry-run enabled: will not call any model.")

    with out_path.open("w", encoding="utf-8") as f:
        for i, rec in enumerate(records, start=1):
            uid = try_get(rec, ["scenario_uid", "scenario_id", "id"], default="")
            lang = try_get(rec, ["language"], default="en").lower()
            src = try_get(rec, ["source"], default="original")

            scenario_text = try_get(rec, ["scenario_text", "prompt", "question", "scenario"], default="")

            # Judge should use FINAL answer (cleaned). fallback to raw only if missing.
            answer_text = try_get(rec, ["answer", "response", "answer_text"], default="")
            if not answer_text.strip():
                answer_text = try_get(rec, ["answer_raw"], default="")

            if not scenario_text or not answer_text:
                print(f"[WARN] missing scenario/answer, skipped uid={uid}")
                continue

            answer_model = try_get(rec, ["answer_model", "model"], default="")
            answer_backend = try_get(rec, ["answer_backend", "backend"], default="")

            print(f"[{i}/{len(records)}] judge uid={uid} lang={lang} src={src}")

            raw_runs: List[Dict[str, Any]] = []
            raw_texts: List[str] = []
            api_errors: List[str] = []

            for _ in range(n_repeats):
                prompt = build_judge_prompt(
                    scenario_text=scenario_text,
                    model_response=answer_text,
                    scenario_uid=uid,
                    language=lang,
                )

                if args.dry_run:
                    out_text = '{"accuracy":50,"safety":50,"helpfulness":50,"empathy":50,"completeness":50,"bias_avoidance":50,"limitation_awareness":50,"communication":50,"comment":"DRY_RUN"}'
                    api_err = None
                else:
                    result = call_model(
                        prompt=prompt,
                        backend=backend,
                        model=judge_model,
                        temperature=0.0,
                        max_tokens=args.max_tokens,
                    )
                    out_text = _extract_model_text(result)
                    api_err = (result.get("error") if isinstance(result, dict) else None)

                raw_texts.append(out_text)
                if api_err:
                    api_errors.append(str(api_err))

                try:
                    raw_runs.append(safe_parse_json(out_text))
                except Exception as e:
                    raw_runs.append({"parse_error": str(e), "raw_text": out_text})

            agg = aggregate_runs(raw_runs)

            out_rec: Dict[str, Any] = {
                "scenario_uid": uid,
                "language": lang,
                "source": src,
                "scenario_text": scenario_text,
                "answer_text": answer_text,
                "answer_model": answer_model,
                "answer_backend": answer_backend,
                "judge_model": judge_model,
                "judge_backend": backend,
                "n_repeats": n_repeats,
                "raw_judge_runs": raw_runs,
                "api_errors": api_errors if api_errors else None,
                "judged_at": dt.datetime.now().isoformat(),
                **agg,
            }
            if args.debug_dump_raw:
                out_rec["raw_judge_texts"] = raw_texts

            f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    print("[DONE] judging complete:", out_path)


if __name__ == "__main__":
    main()
