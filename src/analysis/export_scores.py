# src/analysis/export_scores.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.analysis.scoring_utils import aggregate_runs
from src.judges.judge_prompts import RUBRIC_KEYS
from src.config import SCORES_DIR


def process_jsonl(in_path: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rec = json.loads(s)

            raw_runs = rec.get("raw_judge_runs") or []
            agg = aggregate_runs(raw_runs)

            row: Dict[str, Any] = {
                "scenario_uid": rec.get("scenario_uid") or rec.get("scenario_id"),
                "language": rec.get("language", "en"),
                "source": rec.get("source", "original"),
                "answer_model": rec.get("answer_model"),
                "answer_backend": rec.get("answer_backend"),
                "judge_model": rec.get("judge_model"),
                "judge_backend": rec.get("judge_backend"),
                "n_repeats": rec.get("n_repeats", len(raw_runs)),
                "comment": agg.comment,
            }

            for k in RUBRIC_KEYS:
                row[k] = agg.avg_scores.get(k)
                row[f"{k}_std"] = agg.std_scores.get(k)

            rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    df = process_jsonl(in_path)

    if args.output:
        out_path = Path(args.output)
    else:
        SCORES_DIR.mkdir(parents=True, exist_ok=True)
        out_path = SCORES_DIR / (in_path.stem + ".csv")

    df.to_csv(out_path, index=False, encoding="utf-8")
    print("Exported:", out_path.resolve())


if __name__ == "__main__":
    main()
