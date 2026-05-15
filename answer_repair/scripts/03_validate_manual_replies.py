#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from answer_repair.lib.common import (
    ROOT,
    analyze_text_for_truncation,
    ensure_dir,
    iter_jsonl,
    normalize_manual_reply,
    read_toml,
    selected_review_rows,
    write_csv,
    write_jsonl,
)


def infer_source_file_from_txt_parent(parent_name: str, expected_files: set[str]) -> str:
    if parent_name in expected_files:
        return parent_name
    if parent_name.endswith(".jsonl") and parent_name in expected_files:
        return parent_name
    candidate = f"{parent_name}.jsonl"
    if candidate in expected_files:
        return candidate
    return ""


def collect_jsonl_replies(jsonl_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not jsonl_dir.exists():
        return rows
    for fp in sorted(jsonl_dir.glob("*.jsonl")):
        for rec in iter_jsonl(fp):
            rows.append(
                {
                    "reply_source": f"jsonl:{fp.name}",
                    "source_file": str(rec.get("source_file") or fp.name).strip(),
                    "scenario_uid": str(rec.get("scenario_uid") or "").strip(),
                    "answer_raw": rec.get("answer_raw"),
                    "answer": rec.get("answer"),
                }
            )
    return rows


def collect_txt_replies(txt_dir: Path, expected_files: set[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not txt_dir.exists():
        return rows
    for fp in sorted(txt_dir.rglob("*.txt")):
        parent_name = fp.parent.name
        source_file = infer_source_file_from_txt_parent(parent_name, expected_files)
        rows.append(
            {
                "reply_source": f"txt:{fp.relative_to(txt_dir)}",
                "source_file": source_file,
                "scenario_uid": fp.stem,
                "answer_raw": fp.read_text(encoding="utf-8"),
                "answer": None,
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate manual repair replies and normalize them into per-source-file JSONL files.")
    ap.add_argument("--rules", default="answer_repair/config/truncation_rules.toml")
    ap.add_argument("--review-file", default="answer_repair/workspace/review_lists/truncation_candidates.csv")
    ap.add_argument("--jsonl-dir", default="answer_repair/workspace/manual_replies/jsonl")
    ap.add_argument("--txt-dir", default="answer_repair/workspace/manual_replies/txt")
    ap.add_argument("--normalized-dir", default="answer_repair/workspace/normalized_replies")
    ap.add_argument("--manifest-csv", default="answer_repair/workspace/manifests/manual_reply_validation.csv")
    args = ap.parse_args()

    rules = read_toml(Path(args.rules))
    review_file = ROOT / args.review_file
    expected = selected_review_rows(review_file, rules)
    expected_files = set(expected.keys())
    expected_pairs = {(file_name, uid) for file_name, uids in expected.items() for uid in uids}

    jsonl_replies = collect_jsonl_replies(ROOT / args.jsonl_dir)
    txt_replies = collect_txt_replies(ROOT / args.txt_dir, expected_files)
    all_replies = jsonl_replies + txt_replies

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in all_replies:
        grouped.setdefault((row["source_file"], row["scenario_uid"]), []).append(row)

    normalized_rows: Dict[str, List[Dict[str, Any]]] = {}
    manifest_rows: List[Dict[str, Any]] = []

    for source_file, uid in sorted(expected_pairs):
        reply_options = grouped.get((source_file, uid), [])
        if not reply_options:
            manifest_rows.append(
                {
                    "source_file": source_file,
                    "scenario_uid": uid,
                    "status": "missing_reply",
                    "reply_source": "",
                    "char_count": "",
                    "validation_status": "",
                    "validation_reasons": "",
                }
            )
            continue
        if len(reply_options) > 1:
            manifest_rows.append(
                {
                    "source_file": source_file,
                    "scenario_uid": uid,
                    "status": "duplicate_reply",
                    "reply_source": "|".join(r["reply_source"] for r in reply_options),
                    "char_count": "",
                    "validation_status": "",
                    "validation_reasons": "",
                }
            )
            continue

        reply = reply_options[0]
        raw_answer, final_answer, stripped, removed_chars = normalize_manual_reply(
            raw_answer=str(reply.get("answer_raw") or reply.get("answer") or ""),
            final_answer=(None if reply.get("answer") is None else str(reply.get("answer") or "")),
        )

        analysis = analyze_text_for_truncation(
            {
                "scenario_uid": uid,
                "answer": final_answer,
                "finish_reason": "manual_reply",
                "api_error": None,
                "suspected_truncation": False,
                "language": ("zh" if source_file.startswith("zh_") else "en"),
            },
            source_file,
            rules,
        )
        validation_status = "warning" if analysis["candidate"] else "ok"
        validation_reasons = analysis["all_reasons"]

        normalized_rows.setdefault(source_file, []).append(
            {
                "source_file": source_file,
                "scenario_uid": uid,
                "answer_raw": raw_answer,
                "answer": final_answer,
                "reasoning_stripped": stripped,
                "removed_reasoning_chars": removed_chars,
                "validation_status": validation_status,
                "validation_reasons": validation_reasons,
                "reply_source": reply["reply_source"],
            }
        )
        manifest_rows.append(
            {
                "source_file": source_file,
                "scenario_uid": uid,
                "status": "validated",
                "reply_source": reply["reply_source"],
                "char_count": len(final_answer.strip()),
                "validation_status": validation_status,
                "validation_reasons": validation_reasons,
            }
        )

    # record extra replies that are not expected
    for (source_file, uid), rows in sorted(grouped.items()):
        if (source_file, uid) in expected_pairs:
            continue
        manifest_rows.append(
            {
                "source_file": source_file,
                "scenario_uid": uid,
                "status": "unexpected_reply",
                "reply_source": "|".join(r["reply_source"] for r in rows),
                "char_count": "",
                "validation_status": "",
                "validation_reasons": "",
            }
        )

    normalized_dir = ROOT / args.normalized_dir
    ensure_dir(normalized_dir)
    for source_file, rows in normalized_rows.items():
        rows.sort(key=lambda r: r["scenario_uid"])
        write_jsonl(normalized_dir / source_file, rows)

    write_csv(
        ROOT / args.manifest_csv,
        manifest_rows,
        ["source_file", "scenario_uid", "status", "reply_source", "char_count", "validation_status", "validation_reasons"],
    )

    missing = sum(1 for row in manifest_rows if row["status"] == "missing_reply")
    duplicates = sum(1 for row in manifest_rows if row["status"] == "duplicate_reply")
    warnings = sum(1 for row in manifest_rows if row.get("validation_status") == "warning")

    print(f"[DONE] normalized_dir={normalized_dir}")
    print(f"[SUMMARY] missing={missing} duplicates={duplicates} warnings={warnings}")
    if missing or duplicates:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
