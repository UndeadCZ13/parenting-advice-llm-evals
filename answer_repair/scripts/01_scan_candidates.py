#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from answer_repair.lib.common import ROOT, analyze_text_for_truncation, iter_jsonl, read_toml, write_csv


def main() -> None:
    ap = argparse.ArgumentParser(description="Scan answer outputs and produce a truncation-candidate review list.")
    ap.add_argument(
        "--rules",
        default="answer_repair/config/truncation_rules.toml",
        help="Path to truncation rules TOML",
    )
    ap.add_argument(
        "--inputs",
        default=None,
        help="Override input directory (default comes from the rules file)",
    )
    ap.add_argument(
        "--out-csv",
        default=None,
        help="Override candidate review CSV output path",
    )
    ap.add_argument(
        "--summary-csv",
        default=None,
        help="Override summary CSV output path",
    )
    args = ap.parse_args()

    rules_path = Path(args.rules)
    rules = read_toml(rules_path)
    input_dir = Path(args.inputs) if args.inputs else (ROOT / str(rules["global"]["inputs_dir"]))
    out_csv = Path(args.out_csv) if args.out_csv else (ROOT / str(rules["global"]["review_csv"]))
    summary_csv = Path(args.summary_csv) if args.summary_csv else (ROOT / str(rules["global"]["summary_csv"]))

    files = sorted(input_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"No jsonl files found in {input_dir}")

    review_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for fp in files:
        total_rows = 0
        candidate_rows = 0
        hard_rows = 0

        for rec in iter_jsonl(fp):
            total_rows += 1
            scenario_uid = str(rec.get("scenario_uid") or "").strip()
            analysis = analyze_text_for_truncation(rec, fp.name, rules)
            if not analysis["candidate"]:
                continue
            candidate_rows += 1
            if analysis["hard_reasons"]:
                hard_rows += 1
            review_rows.append(
                {
                    "source_file": fp.name,
                    "scenario_uid": scenario_uid,
                    "language": analysis["language"],
                    "backend": str(rec.get("backend") or "").strip(),
                    "model": str(rec.get("model") or "").strip(),
                    "char_count": analysis["char_count"],
                    "finish_reason": analysis["finish_reason"],
                    "api_error": analysis["api_error"],
                    "legacy_truncation_flag": analysis["legacy_truncation_flag"],
                    "hard_reasons": analysis["hard_reasons"],
                    "soft_reasons": analysis["soft_reasons"],
                    "all_reasons": analysis["all_reasons"],
                    "nonterminal_end": analysis["nonterminal_end"],
                    "short_fragment": analysis["short_fragment"],
                    "answer_preview": analysis["answer_preview"],
                    "answer_tail": analysis["answer_tail"],
                    "last_fragment": analysis["last_fragment"],
                    "manual_action": "",
                    "manual_notes": "",
                }
            )

        summary_rows.append(
            {
                "source_file": fp.name,
                "total_rows": total_rows,
                "candidate_rows": candidate_rows,
                "hard_candidate_rows": hard_rows,
            }
        )

    review_rows.sort(key=lambda r: (r["source_file"], r["scenario_uid"]))
    summary_rows.sort(key=lambda r: r["source_file"])

    review_fields = [
        "source_file",
        "scenario_uid",
        "language",
        "backend",
        "model",
        "char_count",
        "finish_reason",
        "api_error",
        "legacy_truncation_flag",
        "hard_reasons",
        "soft_reasons",
        "all_reasons",
        "nonterminal_end",
        "short_fragment",
        "answer_preview",
        "answer_tail",
        "last_fragment",
        "manual_action",
        "manual_notes",
    ]
    summary_fields = ["source_file", "total_rows", "candidate_rows", "hard_candidate_rows"]

    write_csv(out_csv, review_rows, review_fields)
    write_csv(summary_csv, summary_rows, summary_fields)

    print(f"[DONE] review_csv={out_csv}")
    print(f"[DONE] summary_csv={summary_csv}")
    print(f"[SUMMARY] files={len(files)} candidates={len(review_rows)}")


if __name__ == "__main__":
    main()
