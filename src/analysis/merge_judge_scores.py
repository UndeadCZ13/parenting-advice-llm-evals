# src/analysis/merge_judge_scores.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from src.config import MERGED_DIR
from src.judges.judge_prompts import RUBRIC_KEYS


def collect_inputs(inputs: List[str]) -> List[Path]:
    paths: List[Path] = []
    for it in inputs:
        p = Path(it)
        if p.is_dir():
            paths += sorted(p.glob("*.csv"))
        elif any(ch in it for ch in ["*", "?", "["]):
            paths += sorted(Path(".").glob(it))
        else:
            paths.append(p)
    return [p for p in paths if p.exists()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="CSV files, directories, or glob patterns")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    files = collect_inputs(args.inputs)
    if not files:
        raise ValueError("No input CSV found.")

    all_dfs = []
    for fp in files:
        df = pd.read_csv(fp)
        df["source_file"] = fp.name

        for r in RUBRIC_KEYS:
            if r not in df.columns:
                df[r] = pd.NA

        all_dfs.append(df)

    merged = pd.concat(all_dfs, ignore_index=True)

    # column order
    meta_first = [c for c in ["scenario_uid", "scenario_id", "language", "source", "answer_model", "answer_backend", "judge_model", "judge_backend", "comment", "source_file"] if c in merged.columns]
    rubric_cols = [r for r in RUBRIC_KEYS if r in merged.columns]
    std_cols = [f"{r}_std" for r in RUBRIC_KEYS if f"{r}_std" in merged.columns]
    rest = [c for c in merged.columns if c not in (meta_first + rubric_cols + std_cols)]
    merged = merged[meta_first + rubric_cols + std_cols + rest]

    if args.output:
        out_path = Path(args.output)
    else:
        MERGED_DIR.mkdir(parents=True, exist_ok=True)
        out_path = MERGED_DIR / "all_judge_scores.csv"

    merged.to_csv(out_path, index=False, encoding="utf-8")
    print(f"merged {len(files)} files -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
