#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from answer_repair.lib.common import ROOT, ensure_dir, iter_jsonl, now_iso, read_jsonl, write_csv, write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply validated manual replies to the original answer output files in place.")
    ap.add_argument("--normalized-dir", default="answer_repair/workspace/normalized_replies")
    ap.add_argument("--inputs", default="data/model_outputs")
    ap.add_argument("--manifest-csv", default="answer_repair/workspace/manifests/applied_manual_repairs.csv")
    ap.add_argument("--backup-dir", default="answer_repair/workspace/manifests/original_rows")
    ap.add_argument("--allow-warnings", action="store_true", help="Allow replies with validation_status=warning")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    normalized_dir = ROOT / args.normalized_dir
    inputs_dir = ROOT / args.inputs
    manifest_csv = ROOT / args.manifest_csv
    backup_dir = ROOT / args.backup_dir
    ensure_dir(backup_dir)

    normalized_files = sorted(normalized_dir.glob("*.jsonl"))
    if not normalized_files:
        raise SystemExit(f"No normalized replies found in {normalized_dir}")

    manifest_rows: List[Dict[str, Any]] = []

    for norm_path in normalized_files:
        source_file = norm_path.name
        source_path = inputs_dir / source_file
        if not source_path.exists():
            manifest_rows.append(
                {
                    "source_file": source_file,
                    "scenario_uid": "",
                    "status": "missing_source_file",
                    "old_char_count": "",
                    "new_char_count": "",
                }
            )
            continue

        replacements = {row["scenario_uid"]: row for row in read_jsonl(norm_path)}
        source_rows = list(iter_jsonl(source_path))
        replaced_originals: List[Dict[str, Any]] = []
        updated_rows: List[Dict[str, Any]] = []

        for rec in source_rows:
            uid = str(rec.get("scenario_uid") or "").strip()
            replacement = replacements.get(uid)
            if not replacement:
                updated_rows.append(rec)
                continue

            if replacement.get("validation_status") == "warning" and not args.allow_warnings:
                manifest_rows.append(
                    {
                        "source_file": source_file,
                        "scenario_uid": uid,
                        "status": "skipped_warning_reply",
                        "old_char_count": len(str(rec.get("answer") or "").strip()),
                        "new_char_count": len(str(replacement.get("answer") or "").strip()),
                    }
                )
                updated_rows.append(rec)
                continue

            replaced_originals.append(rec)
            new_answer = str(replacement.get("answer") or "").strip()
            new_raw = str(replacement.get("answer_raw") or new_answer)
            updated = {
                **rec,
                "answer_raw": new_raw,
                "answer": new_answer,
                "reasoning_stripped": bool(replacement.get("reasoning_stripped", False)),
                "removed_reasoning_chars": int(replacement.get("removed_reasoning_chars", 0)),
                "suspected_truncation": False,
                "retry_used": True,
                "finish_reason": "manual_repair",
                "api_error": None,
                "manual_repair_applied": True,
                "manual_repair_timestamp": now_iso(),
                "manual_repair_source": str(replacement.get("reply_source") or ""),
            }
            updated_rows.append(updated)
            manifest_rows.append(
                {
                    "source_file": source_file,
                    "scenario_uid": uid,
                    "status": "applied" if not args.dry_run else "dry_run_applied",
                    "old_char_count": len(str(rec.get("answer") or "").strip()),
                    "new_char_count": len(new_answer),
                }
            )

        if replaced_originals:
            write_jsonl(backup_dir / source_file, replaced_originals)
        if not args.dry_run:
            write_jsonl(source_path, updated_rows)

    write_csv(
        manifest_csv,
        manifest_rows,
        ["source_file", "scenario_uid", "status", "old_char_count", "new_char_count"],
    )
    applied = sum(1 for row in manifest_rows if row["status"] in {"applied", "dry_run_applied"})
    print(f"[DONE] manifest={manifest_csv}")
    print(f"[SUMMARY] applied_rows={applied}")


if __name__ == "__main__":
    main()
