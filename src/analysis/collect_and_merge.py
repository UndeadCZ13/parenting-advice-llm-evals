# src/analysis/collect_and_merge.py
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd

from src.analysis.export_scores import process_jsonl


def merge_csvs(csv_paths: list[Path]) -> pd.DataFrame:
    """
    Merge multiple CSV files into a single DataFrame.
    This mirrors the behavior of merge_judge_scores.py (row-wise concat).
    """
    dfs = []
    for p in csv_paths:
        df = pd.read_csv(p)
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Collect all judge jsonl files in a directory, export to CSV, and merge into one table."
    )
    ap.add_argument(
        "--judge-dir",
        required=True,
        help="Directory containing judge jsonl files (e.g. data/judge_outputs)",
    )
    ap.add_argument(
        "--scores-dir",
        default="results/scores",
        help="Directory to store per-file CSVs (default: results/scores)",
    )
    ap.add_argument(
        "--merged-output",
        default="results/merged/all_judge_scores.csv",
        help="Output merged CSV path",
    )
    args = ap.parse_args()

    judge_dir = Path(args.judge_dir)
    if not judge_dir.exists():
        print(f"[ERROR] judge-dir not found: {judge_dir}")
        sys.exit(1)

    scores_dir = Path(args.scores_dir)
    scores_dir.mkdir(parents=True, exist_ok=True)

    merged_output = Path(args.merged_output)
    merged_output.parent.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(judge_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"[WARN] No jsonl files found in {judge_dir}")
        sys.exit(0)

    csv_paths: list[Path] = []

    print(f"[INFO] Found {len(jsonl_files)} judge files.")
    for jf in jsonl_files:
        out_csv = scores_dir / f"{jf.stem}.csv"
        print(f"[EXPORT] {jf.name} -> {out_csv}")
        df = process_jsonl(jf)
        df.to_csv(out_csv, index=False)
        csv_paths.append(out_csv)

    print(f"[INFO] Merging {len(csv_paths)} CSV files...")
    merged_df = merge_csvs(csv_paths)
    merged_df.to_csv(merged_output, index=False)

    print(f"[DONE] Merged table written to: {merged_output}")
    print("[NEXT] Run analyze_scores.py with --language en | zh | all")


if __name__ == "__main__":
    main()
