from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from src.run_style_judging import (
    judge_one,
    resolve_judge_model,
)
from src.config import (
    DEFAULT_BACKEND_JUDGE,
    DEFAULT_OPENAI_MODEL_JUDGE,
    DEFAULT_OLLAMA_MODEL_KEY_JUDGE,
    DEFAULT_GROQ_MODEL_KEY_JUDGE,
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def needs_retry(rec: Dict[str, Any]) -> bool:
    if rec.get("error"):
        return True
    return not (
        isinstance(rec.get("responsiveness"), (int, float))
        and isinstance(rec.get("demandingness"), (int, float))
    )


def parse_model_name_from_output(path: Path, judge_model: str) -> str:
    suffix = f"__dims__judge={judge_model}.jsonl"
    name = path.name
    if not name.endswith(suffix):
        raise ValueError(f"Unexpected style output filename: {path.name}")
    return name[: -len(suffix)]


def build_model_output_index(model_outputs_dir: Path, include_subdirs: bool) -> Dict[str, Path]:
    files = model_outputs_dir.rglob("*.jsonl") if include_subdirs else model_outputs_dir.glob("*.jsonl")
    out: Dict[str, Path] = {}
    for p in sorted(files):
        out[p.stem] = p
    return out


def build_input_row_index(model_output_path: Path) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for row in load_jsonl(model_output_path):
        uid = row.get("scenario_uid")
        if isinstance(uid, str) and uid:
            index[uid] = row
    return index


def rerun_file(
    out_path: Path,
    input_rows: Dict[str, Dict[str, Any]],
    judge_backend: str,
    judge_model: str,
    max_tokens: int,
    parse_retries: int,
) -> Tuple[int, int, int]:
    rows = load_jsonl(out_path)
    retried = 0
    fixed = 0
    still_bad = 0
    new_rows: List[Dict[str, Any]] = []

    for rec in rows:
        if not needs_retry(rec):
            new_rows.append(rec)
            continue

        uid = rec.get("scenario_uid")
        retried += 1
        src = input_rows.get(str(uid)) if uid is not None else None
        if not src:
            still_bad += 1
            new_rows.append(rec)
            continue

        scenario_text = src.get("scenario_text", "")
        model_answer = src.get("answer", "")
        lang = (src.get("language") or rec.get("language") or "en").lower()

        if not isinstance(model_answer, str) or not model_answer.strip():
            new_rows.append(
                {
                    "scenario_uid": uid,
                    "language": lang,
                    "model_name": rec.get("model_name") or src.get("model") or src.get("answer_model"),
                    "judge_model": judge_model,
                    "judge_backend": judge_backend,
                    "responsiveness": 0.0,
                    "demandingness": 0.0,
                    "error": "empty_answer",
                }
            )
            fixed += 1
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
            if isinstance(last_result, dict) and isinstance(last_result.get("text"), str):
                preview = last_result["text"][:300]
            new_rows.append(
                {
                    "scenario_uid": uid,
                    "language": lang,
                    "model_name": rec.get("model_name") or src.get("model") or src.get("answer_model"),
                    "judge_model": judge_model,
                    "judge_backend": judge_backend,
                    "error": last_result.get("error") if isinstance(last_result, dict) else "parse_failed",
                    "raw_preview": preview,
                }
            )
            still_bad += 1
            continue

        new_rows.append(
            {
                "scenario_uid": uid,
                "language": lang,
                "model_name": rec.get("model_name") or src.get("model") or src.get("answer_model"),
                "judge_model": judge_model,
                "judge_backend": judge_backend,
                **parsed,
            }
        )
        fixed += 1

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in new_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(out_path)
    return retried, fixed, still_bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_outputs_dir", default="data/model_outputs")
    ap.add_argument("--output_dir", default="data/judge_outputs/style_2d")
    ap.add_argument("--judge_backend", default=DEFAULT_BACKEND_JUDGE)
    ap.add_argument("--judge_model", default="")
    ap.add_argument("--openai_model", default=DEFAULT_OPENAI_MODEL_JUDGE)
    ap.add_argument("--ollama_model_key", default=DEFAULT_OLLAMA_MODEL_KEY_JUDGE)
    ap.add_argument("--groq_model_key", default=DEFAULT_GROQ_MODEL_KEY_JUDGE)
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--parse_retries", type=int, default=2)
    ap.add_argument("--include_subdirs", action="store_true")
    ap.add_argument("--only_file", default="")
    args = ap.parse_args()

    judge_backend = args.judge_backend.lower().replace("local", "ollama")
    judge_model = resolve_judge_model(
        backend=judge_backend,
        judge_model=args.judge_model,
        openai_model=args.openai_model,
        ollama_model_key=args.ollama_model_key,
        groq_model_key=args.groq_model_key,
    )

    model_outputs_dir = Path(args.model_outputs_dir)
    output_dir = Path(args.output_dir)
    model_index = build_model_output_index(model_outputs_dir, args.include_subdirs)

    pattern = f"*__dims__judge={judge_model}.jsonl"
    out_files = sorted(output_dir.glob(pattern))
    if args.only_file:
        out_files = [p for p in out_files if p.name == args.only_file or p.stem == args.only_file]

    if not out_files:
        print(f"[ERROR] No style judge output files found for judge={judge_model}")
        return

    total_files = 0
    total_retried = 0
    total_fixed = 0
    total_still_bad = 0

    for out_path in tqdm(out_files, desc="STYLE-ERROR-RERUN", ncols=90, unit="file"):
        model_name = parse_model_name_from_output(out_path, judge_model)
        input_path = model_index.get(model_name)
        if input_path is None:
            print(f"[WARN] Missing source model output for {out_path.name}")
            continue

        rows = load_jsonl(out_path)
        if not any(needs_retry(r) for r in rows):
            continue

        input_rows = build_input_row_index(input_path)
        retried, fixed, still_bad = rerun_file(
            out_path=out_path,
            input_rows=input_rows,
            judge_backend=judge_backend,
            judge_model=judge_model,
            max_tokens=args.max_tokens,
            parse_retries=args.parse_retries,
        )
        total_files += 1
        total_retried += retried
        total_fixed += fixed
        total_still_bad += still_bad
        print(
            f"[DONE] {out_path.name}: retried={retried}, fixed={fixed}, still_bad={still_bad}"
        )

    print(
        f"[SUMMARY] files={total_files}, retried={total_retried}, fixed={total_fixed}, still_bad={total_still_bad}"
    )


if __name__ == "__main__":
    main()
