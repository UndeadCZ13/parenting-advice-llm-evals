# scripts/convert_scenarios_from_excel.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def build_scenario_text(question: str, description: str, tags: str) -> str:
    parts = []
    if description.strip():
        parts.append(f"Context: {description.strip()}")
    if tags.strip():
        parts.append(f"Tags: {tags.strip()}")
    parts.append(f"Question: {question.strip()}")
    return "\n".join(parts).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/scenarios/v0_ParentBench_Scenarios.xlsx")
    ap.add_argument("--output", default="data/scenarios/views/en/parentbench_v0.jsonl")
    ap.add_argument("--source", default="original", help="original|translated|native_generated")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(in_path)
    expected = {"Question", "Description", "Tags"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in Excel: {missing}")

    rows: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        q = str(row["Question"]).strip()
        desc = "" if pd.isna(row["Description"]) else str(row["Description"]).strip()
        tags = "" if pd.isna(row["Tags"]) else str(row["Tags"]).strip()

        uid = f"pb_v0_{idx:04d}"
        scenario_text = build_scenario_text(q, desc, tags)

        rows.append(
            {
                "scenario_uid": uid,
                "language": "en",
                "source": args.source,
                "scenario_text": scenario_text,
                "metadata": {
                    "tags_raw": tags,
                },
            }
        )

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {len(rows)} scenarios -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
