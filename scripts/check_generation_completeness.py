#!/usr/bin/env python3
# scripts/check_generation_completeness.py

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception:
                continue


def pick_str(rec: Dict[str, Any], keys: List[str], default: str = "") -> str:
    for k in keys:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return default


def pick_bool(rec: Dict[str, Any], keys: List[str], default: bool = False) -> bool:
    for k in keys:
        v = rec.get(k)
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)) and v in (0, 1):
            return bool(v)
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("true", "t", "1", "yes", "y"):
                return True
            if s in ("false", "f", "0", "no", "n"):
                return False
    return default


def is_empty_text(s: Any) -> bool:
    if s is None:
        return True
    if not isinstance(s, str):
        s = str(s)
    return len(s.strip()) == 0


def get_uid(rec: Dict[str, Any]) -> str:
    return pick_str(rec, ["scenario_uid", "scenario_id", "id"], default="")


def normalize_for_terminal_punct(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[\s\"'“”‘’`*_)\]）】〉》」』]+$", "", s)
    s = re.sub(r"\([^()]*\)$", "", s).rstrip()
    s = re.sub(r"（[^（）]*）$", "", s).rstrip()
    return s


def infer_truncation_from_text(text: Any, finish_reason: Any) -> bool:
    if isinstance(finish_reason, str) and finish_reason.strip().lower() == "length":
        return True
    if is_empty_text(text):
        return True
    s = normalize_for_terminal_punct(str(text))
    if not s:
        return True
    if len(s) > 20 and s[-1] not in "。.!?！？":
        return True
    return False


def make_preview(text: Any, limit: int = 240) -> str:
    if text is None:
        return ""
    s = str(text).replace("\r", " ").replace("\n", " ").strip()
    s = re.sub(r"\s+", " ", s)
    if len(s) <= limit:
        return s
    return s[: limit - 3] + "..."


def make_reason_list(
    empty: bool,
    pipeline_trunc_flag: bool,
    diagnostic_trunc: bool,
    finish_reason: Any,
) -> str:
    reasons: List[str] = []
    if empty:
        reasons.append("empty_answer")
    if isinstance(finish_reason, str) and finish_reason.strip().lower() == "length":
        reasons.append("finish_reason_length")
    if pipeline_trunc_flag:
        reasons.append("pipeline_trunc_flag")
    if diagnostic_trunc:
        reasons.append("diagnostic_trunc")
    return "|".join(reasons)


def get_last_fragment(text: Any) -> str:
    s = str(text or "").strip()
    if not s:
        return ""
    parts = re.split(r"[。.!?！？\n\r]+", s)
    frag = parts[-1].strip() if parts else s
    return frag


def fragment_word_count(text: str) -> int:
    return len([w for w in re.split(r"\s+", text.strip()) if w])


def has_obvious_cut_marker(text: Any) -> bool:
    s = str(text or "").rstrip()
    if not s:
        return False
    if s.endswith(("**", "__", "```")):
        return True
    if s[-1] in {":", ";", ",", "：", "，", "、", "(", "（", "[", "{", "/", "\\", "-", "—"}:
        return True
    return False


def conservative_review_candidate(row: Dict[str, Any]) -> bool:
    if row["empty"]:
        return True
    finish_reason = str(row.get("finish_reason") or "").strip().lower()
    if finish_reason == "length":
        return True
    if not row["diagnostic_trunc"]:
        return False
    if has_obvious_cut_marker(row.get("answer_tail", "")):
        return True

    frag = normalize_for_terminal_punct(get_last_fragment(row.get("answer_tail", "")))
    if not frag:
        return False

    lang = str(row.get("language") or "").strip().lower()
    if lang.startswith("zh"):
        return len(frag) <= 4
    return fragment_word_count(frag) <= 3


def should_include_candidate(row: Dict[str, Any], candidate_mode: str) -> bool:
    if candidate_mode == "conservative":
        return conservative_review_candidate(row)
    if candidate_mode == "pipeline":
        return bool(row["pipeline_trunc_flag"] or row["empty"])
    if candidate_mode == "diagnostic":
        return bool(row["defective"])
    return bool(row["pipeline_trunc_flag"] or row["defective"])


def suggested_action(row: Dict[str, Any]) -> str:
    if row["empty"]:
        return "retry"
    finish_reason = str(row.get("finish_reason") or "").strip().lower()
    if finish_reason == "length":
        return "retry"
    if row["diagnostic_trunc"]:
        return "review"
    return "review"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Scan generation jsonl files and produce summary plus a manual review list for reruns."
    )
    ap.add_argument(
        "--dir",
        default="data/model_outputs",
        help="Directory containing generation jsonl files (default: data/model_outputs)",
    )
    ap.add_argument(
        "--pattern",
        default="*.jsonl",
        help="Glob pattern under --dir (default: *.jsonl)",
    )
    ap.add_argument(
        "--out",
        default="results/analysis/generation_completeness.csv",
        help="Summary CSV path (default: results/analysis/generation_completeness.csv)",
    )
    ap.add_argument(
        "--details-out",
        default="results/analysis/generation_completeness_details.csv",
        help="Detailed diagnostic-defect CSV path (default: results/analysis/generation_completeness_details.csv)",
    )
    ap.add_argument(
        "--review-out",
        default="results/analysis/generation_rerun_review_list.csv",
        help="Manual review CSV path for later reruns (default: results/analysis/generation_rerun_review_list.csv)",
    )
    ap.add_argument(
        "--candidate-mode",
        choices=["conservative", "diagnostic", "pipeline", "union"],
        default="conservative",
        help="Which automatic candidates to put into the manual review file (default: conservative)",
    )
    ap.add_argument(
        "--min-total",
        type=int,
        default=1,
        help="Only show models with at least N items (default: 1)",
    )
    args = ap.parse_args()

    in_dir = Path(args.dir)
    files = sorted(in_dir.rglob(args.pattern)) if "**" in args.pattern else sorted(in_dir.glob(args.pattern))
    if not files:
        print(f"[WARN] No files matched: dir={in_dir} pattern={args.pattern}")
        return

    rows: List[Dict[str, Any]] = []

    for fp in files:
        fname = fp.name.lower()
        inferred_lang = "zh" if "_zh" in fname or fname.startswith("zh_") else ("en" if "_en" in fname or fname.startswith("en_") else "")

        for rec in iter_jsonl(fp):
            model = pick_str(rec, ["model", "answer_model"], default="(unknown_model)")
            backend = pick_str(rec, ["backend", "answer_backend"], default="(unknown_backend)")
            lang = pick_str(rec, ["language", "lang"], default=inferred_lang or "(unknown_lang)")
            uid = get_uid(rec)

            ans = rec.get("answer")
            ans_raw = rec.get("answer_raw")
            finish_reason = rec.get("finish_reason")

            empty = is_empty_text(ans)
            pipeline_trunc_flag = pick_bool(rec, ["suspected_truncation", "trunc", "is_trunc"], default=False)
            diagnostic_trunc = infer_truncation_from_text(ans, finish_reason)
            incomplete = diagnostic_trunc
            defective = bool(empty or incomplete)

            api_error = rec.get("api_error")
            has_api_error = not is_empty_text(api_error)

            row = {
                "scenario_uid": uid,
                "file": fp.name,
                "language": lang,
                "backend": backend,
                "model": model,
                "empty": empty,
                "pipeline_trunc_flag": pipeline_trunc_flag,
                "diagnostic_trunc": diagnostic_trunc,
                "pipeline_only_flag": bool(pipeline_trunc_flag and not diagnostic_trunc),
                "diagnostic_only_flag": bool(diagnostic_trunc and not pipeline_trunc_flag),
                "incomplete": incomplete,
                "defective": defective,
                "empty_and_incomplete": bool(empty and incomplete),
                "api_error": has_api_error,
                "finish_reason": finish_reason,
                "retry_used": pick_bool(rec, ["retry_used"], default=False),
                "answer_len": 0 if ans is None else len(str(ans)),
                "answer_raw_len": 0 if ans_raw is None else len(str(ans_raw)),
                "answer_preview": make_preview(ans, limit=240),
                "answer_tail": "" if ans is None else str(ans)[-120:],
            }
            row["auto_reasons"] = make_reason_list(
                empty=empty,
                pipeline_trunc_flag=pipeline_trunc_flag,
                diagnostic_trunc=diagnostic_trunc,
                finish_reason=finish_reason,
            )
            row["auto_suggested_action"] = suggested_action(row)
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        print("[WARN] No records parsed from matched files.")
        return

    def _agg(group_cols: List[str]) -> pd.DataFrame:
        out = (
            df.groupby(group_cols, dropna=False)
            .agg(
                n_total=("defective", "size"),
                n_empty=("empty", "sum"),
                n_pipeline_trunc_flag=("pipeline_trunc_flag", "sum"),
                n_incomplete=("incomplete", "sum"),
                n_pipeline_only_flag=("pipeline_only_flag", "sum"),
                n_diagnostic_only_flag=("diagnostic_only_flag", "sum"),
                n_empty_and_incomplete=("empty_and_incomplete", "sum"),
                n_defective=("defective", "sum"),
                n_api_error=("api_error", "sum"),
                n_retry_used=("retry_used", "sum"),
                avg_answer_len=("answer_len", "mean"),
                p50_answer_len=("answer_len", "median"),
            )
            .reset_index()
        )
        out["empty_rate"] = (out["n_empty"] / out["n_total"]).round(4)
        out["pipeline_trunc_flag_rate"] = (out["n_pipeline_trunc_flag"] / out["n_total"]).round(4)
        out["incomplete_rate"] = (out["n_incomplete"] / out["n_total"]).round(4)
        out["defect_rate"] = (out["n_defective"] / out["n_total"]).round(4)
        out["api_error_rate"] = (out["n_api_error"] / out["n_total"]).round(4)
        out["retry_used_rate"] = (out["n_retry_used"] / out["n_total"]).round(4)
        return out

    summary = _agg(["backend", "model", "language"])
    summary = summary[summary["n_total"] >= int(args.min_total)].copy()
    summary = summary.sort_values(
        by=["defect_rate", "incomplete_rate", "empty_rate", "api_error_rate", "n_total"],
        ascending=[False, False, False, False, False],
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)

    details_path = Path(args.details_out)
    details_path.parent.mkdir(parents=True, exist_ok=True)
    details = df[df["defective"]].copy().sort_values(
        by=["backend", "model", "language", "scenario_uid", "file"]
    )
    details.to_csv(details_path, index=False)

    review_path = Path(args.review_out)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review = df[df.apply(lambda r: should_include_candidate(r, args.candidate_mode), axis=1)].copy()
    review = review.sort_values(by=["backend", "model", "language", "scenario_uid", "file"])
    review["manual_action"] = ""
    review["review_notes"] = ""
    review = review[
        [
            "file",
            "scenario_uid",
            "backend",
            "model",
            "language",
            "finish_reason",
            "retry_used",
            "answer_len",
            "empty",
            "pipeline_trunc_flag",
            "diagnostic_trunc",
            "pipeline_only_flag",
            "diagnostic_only_flag",
            "auto_reasons",
            "auto_suggested_action",
            "manual_action",
            "review_notes",
            "answer_preview",
            "answer_tail",
        ]
    ]
    review.to_csv(review_path, index=False)

    print(f"[INFO] Scanned files: {len(files)} | records: {len(df)}")
    print(f"[DONE] Wrote summary CSV: {out_path}")
    print(f"[DONE] Wrote details CSV: {details_path}")
    print(f"[DONE] Wrote manual review CSV: {review_path}\n")

    show_cols = [
        "backend",
        "model",
        "language",
        "n_total",
        "n_empty",
        "empty_rate",
        "n_pipeline_trunc_flag",
        "pipeline_trunc_flag_rate",
        "n_incomplete",
        "incomplete_rate",
        "n_defective",
        "defect_rate",
        "n_pipeline_only_flag",
        "n_empty_and_incomplete",
        "n_api_error",
        "api_error_rate",
        "avg_answer_len",
        "p50_answer_len",
    ]
    with pd.option_context("display.max_rows", 60, "display.max_columns", 50, "display.width", 160):
        print(summary[show_cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
