#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from answer_repair.lib.common import ensure_dir, iter_jsonl, read_jsonl, write_csv, write_jsonl
from src.model_caller import call_model


def normalize_backend(raw: Any) -> str:
    backend = str(raw or "").strip().lower()
    if backend == "local":
        return "ollama"
    return backend


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Optional batch generation path: call the model directly from exported prompt JSONL files."
    )
    ap.add_argument("--prompt-jsonl-dir", default="answer_repair/workspace/prompt_jsonl")
    ap.add_argument("--output-dir", default="answer_repair/workspace/manual_replies/jsonl")
    ap.add_argument("--manifest-csv", default="answer_repair/workspace/manifests/batch_generation_manifest.csv")
    ap.add_argument("--only-file", action="append", default=[], help="Limit to one or more source files")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing scenario_uid replies in output JSONL")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--sleep-sec", type=float, default=0.0)
    args = ap.parse_args()

    prompt_dir = ROOT / args.prompt_jsonl_dir
    output_dir = ROOT / args.output_dir
    manifest_csv = ROOT / args.manifest_csv
    ensure_dir(output_dir)

    only_files = set(args.only_file or [])
    prompt_files = sorted(prompt_dir.glob("*.jsonl"))
    if only_files:
        prompt_files = [path for path in prompt_files if path.name in only_files]
    if not prompt_files:
        raise SystemExit(f"No prompt JSONL files found in {prompt_dir}")

    manifest_rows: List[Dict[str, Any]] = []

    for prompt_path in prompt_files:
        source_file = prompt_path.name
        output_path = output_dir / source_file
        existing_rows = read_jsonl(output_path) if output_path.exists() else []
        existing_by_uid = {
            str(row.get("scenario_uid") or "").strip(): row
            for row in existing_rows
            if str(row.get("scenario_uid") or "").strip()
        }

        for row in iter_jsonl(prompt_path):
            uid = str(row.get("scenario_uid") or "").strip()
            if not uid:
                continue
            if uid in existing_by_uid and not args.overwrite:
                manifest_rows.append(
                    {
                        "source_file": source_file,
                        "scenario_uid": uid,
                        "status": "skipped_existing",
                        "backend": str(row.get("backend") or ""),
                        "model": str(row.get("model") or ""),
                        "error": "",
                    }
                )
                continue

            backend = normalize_backend(row.get("backend"))
            model = str(row.get("model") or "").strip()
            prompt_text = str(row.get("prompt_text") or "")
            result = call_model(
                prompt=prompt_text,
                backend=backend,
                model=model,
                temperature=float(args.temperature),
                max_tokens=int(args.max_tokens),
                max_retries=int(args.max_retries),
            )
            answer_raw = str(result.get("text") or "")
            error = str(result.get("error") or "").strip()
            finish_reason = result.get("finish_reason")
            status = "generated" if answer_raw.strip() and not error else "failed"

            existing_by_uid[uid] = {
                "source_file": source_file,
                "scenario_uid": uid,
                "answer_raw": answer_raw,
                "backend": backend,
                "model": model,
                "finish_reason": finish_reason,
                "api_error": error or None,
            }
            manifest_rows.append(
                {
                    "source_file": source_file,
                    "scenario_uid": uid,
                    "status": status,
                    "backend": backend,
                    "model": model,
                    "error": error,
                }
            )

            if args.sleep_sec > 0:
                time.sleep(float(args.sleep_sec))

        merged_rows = sorted(existing_by_uid.values(), key=lambda r: str(r.get("scenario_uid") or ""))
        write_jsonl(output_path, merged_rows)

    write_csv(
        manifest_csv,
        manifest_rows,
        ["source_file", "scenario_uid", "status", "backend", "model", "error"],
    )
    generated = sum(1 for row in manifest_rows if row["status"] == "generated")
    failed = sum(1 for row in manifest_rows if row["status"] == "failed")
    print(f"[DONE] output_dir={output_dir}")
    print(f"[SUMMARY] generated={generated} failed={failed}")


if __name__ == "__main__":
    main()
