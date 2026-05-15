#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from answer_repair.lib.common import (
    ROOT,
    WORKSPACE,
    build_prompt_exact,
    ensure_dir,
    iter_jsonl,
    load_scenarios,
    normalize_language,
    read_toml,
    selected_review_rows,
    write_csv,
    write_jsonl,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export exact generation prompts for manually selected repair rows.")
    ap.add_argument("--rules", default="answer_repair/config/truncation_rules.toml")
    ap.add_argument("--review-file", default="answer_repair/workspace/review_lists/truncation_candidates.csv")
    ap.add_argument("--inputs", default=None, help="Override answer input directory")
    ap.add_argument("--prompt-jsonl-dir", default="answer_repair/workspace/prompt_jsonl")
    ap.add_argument("--prompt-txt-dir", default="answer_repair/workspace/prompt_txt")
    ap.add_argument("--prompt-packet-dir", default="answer_repair/workspace/prompt_packets")
    ap.add_argument("--manifest-csv", default="answer_repair/workspace/manifests/prompt_export_manifest.csv")
    args = ap.parse_args()

    rules = read_toml(Path(args.rules))
    input_dir = Path(args.inputs) if args.inputs else (ROOT / str(rules["global"]["inputs_dir"]))
    review_file = ROOT / args.review_file
    selected = selected_review_rows(review_file, rules)
    scenarios = load_scenarios()

    prompt_jsonl_dir = ROOT / args.prompt_jsonl_dir
    prompt_txt_dir = ROOT / args.prompt_txt_dir
    prompt_packet_dir = ROOT / args.prompt_packet_dir
    manifest_csv = ROOT / args.manifest_csv
    ensure_dir(prompt_jsonl_dir)
    ensure_dir(prompt_txt_dir)
    ensure_dir(prompt_packet_dir)

    manifest_rows: List[Dict[str, Any]] = []

    for source_file, selected_uids in sorted(selected.items()):
        in_path = input_dir / source_file
        packet_path = prompt_packet_dir / f"{in_path.stem}.txt"
        if not in_path.exists():
            manifest_rows.append(
                {
                    "source_file": source_file,
                    "scenario_uid": "",
                    "status": "missing_source_file",
                    "prompt_txt": "",
                    "prompt_jsonl": "",
                    "prompt_packet": "",
                }
            )
            continue

        prompt_rows: List[Dict[str, Any]] = []
        packet_blocks: List[str] = []
        txt_subdir = ensure_dir(prompt_txt_dir / in_path.stem)

        for rec in iter_jsonl(in_path):
            uid = str(rec.get("scenario_uid") or "").strip()
            if uid not in selected_uids:
                continue
            lang = normalize_language(rec.get("language"), file_name=source_file, rules=rules)
            scenario_text = str(rec.get("scenario_text") or "").strip()
            if not scenario_text:
                scenario_row = scenarios.get((lang, uid), {})
                scenario_text = str(scenario_row.get("scenario_text") or "").strip()

            if not scenario_text:
                manifest_rows.append(
                    {
                        "source_file": source_file,
                        "scenario_uid": uid,
                        "status": "missing_scenario_text",
                        "prompt_txt": "",
                        "prompt_jsonl": str(prompt_jsonl_dir / source_file),
                        "prompt_packet": str(packet_path.relative_to(ROOT)),
                    }
                )
                continue

            prompt_text = build_prompt_exact(scenario_text, lang)
            txt_path = txt_subdir / f"{uid}.txt"
            txt_path.write_text(prompt_text, encoding="utf-8")

            prompt_row = {
                "source_file": source_file,
                "scenario_uid": uid,
                "language": lang,
                "backend": str(rec.get("backend") or "").strip(),
                "model": str(rec.get("model") or "").strip(),
                "scenario_text": scenario_text,
                "prompt_text": prompt_text,
            }
            prompt_rows.append(prompt_row)
            packet_blocks.append(
                "\n".join(
                    [
                        f"===== {source_file} | {uid} | {lang} | {prompt_row['backend']} | {prompt_row['model']} =====",
                        prompt_text,
                    ]
                )
            )
            manifest_rows.append(
                {
                    "source_file": source_file,
                    "scenario_uid": uid,
                    "status": "exported",
                    "prompt_txt": str(txt_path.relative_to(ROOT)),
                    "prompt_jsonl": str((prompt_jsonl_dir / source_file).relative_to(ROOT)),
                    "prompt_packet": str(packet_path.relative_to(ROOT)),
                }
            )

        if prompt_rows:
            write_jsonl(prompt_jsonl_dir / source_file, prompt_rows)
            packet_path.write_text("\n\n".join(packet_blocks).strip() + "\n", encoding="utf-8")

    manifest_fields = ["source_file", "scenario_uid", "status", "prompt_txt", "prompt_jsonl", "prompt_packet"]
    write_csv(manifest_csv, manifest_rows, manifest_fields)

    exported = sum(1 for row in manifest_rows if row["status"] == "exported")
    print(f"[DONE] exported_prompts={exported}")
    print(f"[DONE] manifest_csv={manifest_csv}")


if __name__ == "__main__":
    main()
