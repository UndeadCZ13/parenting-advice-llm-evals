#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from answer_repair.lib.common import ensure_dir, iter_table_rows, read_jsonl, write_csv, write_jsonl
from src.judges.judge_prompts import RUBRIC_KEYS, build_judge_prompt
from src.model_caller import call_model
from src.run_judging import aggregate_runs, safe_parse_json, try_get
from src.run_style_judging import build_prompt as build_style_prompt, safe_parse_dims

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def normalize_cell(value: Any) -> str:
    return str(value or "").strip()


def progress(iterable: Sequence[Any], desc: str, leave: bool = True) -> Sequence[Any]:
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, leave=leave)


def merge_rows_by_uid(existing_rows: List[Dict[str, Any]], updated_by_uid: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for row in existing_rows:
        uid = try_get(row, ["scenario_uid", "scenario_id", "id"], default="")
        if uid and uid in updated_by_uid:
            merged.append(updated_by_uid[uid])
            seen.add(uid)
        else:
            merged.append(row)
    for uid, row in updated_by_uid.items():
        if uid not in seen:
            merged.append(row)
    return merged


def load_applied_selection(path: Path, allowed_statuses: Set[str]) -> Dict[str, Set[str]]:
    selected: Dict[str, Set[str]] = {}
    for row in iter_table_rows(path):
        status = normalize_cell(row.get("status")).lower()
        if status not in allowed_statuses:
            continue
        source_file = normalize_cell(row.get("source_file") or row.get("file"))
        uid = normalize_cell(row.get("scenario_uid"))
        if not source_file or not uid:
            continue
        selected.setdefault(source_file, set()).add(uid)
    return selected


def rerun_rubric_record(
    answer_rec: Dict[str, Any],
    judge_backend: str,
    judge_model: str,
    n_repeats: int,
    max_tokens: int,
) -> Tuple[Optional[Dict[str, Any]], str]:
    uid = try_get(answer_rec, ["scenario_uid", "scenario_id", "id"], default="")
    lang = try_get(answer_rec, ["language"], default="en").lower()
    src = try_get(answer_rec, ["source"], default="")
    scenario_text = try_get(answer_rec, ["scenario_text"], default="")
    answer_text = try_get(answer_rec, ["answer"], default="")
    answer_model = try_get(answer_rec, ["model", "answer_model"], default="")
    answer_backend = try_get(answer_rec, ["backend", "answer_backend"], default="")

    raw_runs: List[Dict[str, Any]] = []
    api_errors: List[str] = []
    parse_errors = 0

    for _ in range(max(1, n_repeats)):
        prompt = build_judge_prompt(
            scenario_text=scenario_text,
            model_response=answer_text,
            scenario_uid=uid,
            language=lang,
        )
        result = call_model(
            prompt=prompt,
            backend=judge_backend,
            model=judge_model,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        out_text = str(result.get("text") or "")
        api_err = result.get("error")
        if api_err:
            api_errors.append(str(api_err))
        try:
            raw_runs.append(safe_parse_json(out_text))
        except Exception:
            retry_result = call_model(
                prompt=prompt + "\n\nReturn ONLY one JSON object.",
                backend=judge_backend,
                model=judge_model,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            retry_text = str(retry_result.get("text") or "")
            retry_err = retry_result.get("error")
            if retry_err:
                api_errors.append(str(retry_err))
            try:
                raw_runs.append(safe_parse_json(retry_text))
            except Exception as exc:
                parse_errors += 1
                raw_runs.append({"parse_error": str(exc), "raw_text": out_text[:500]})

    agg = aggregate_runs(raw_runs)
    success = any(isinstance(agg.get(key), (int, float)) for key in RUBRIC_KEYS)
    if not success:
        reasons: List[str] = []
        if api_errors:
            reasons.append(f"api_errors={len(api_errors)}")
        if parse_errors:
            reasons.append(f"parse_errors={parse_errors}")
        return None, ",".join(reasons) or "no_valid_rubric_scores"

    return (
        {
            "scenario_uid": uid,
            "language": lang,
            "source": src,
            "scenario_text": scenario_text,
            "answer_text": answer_text,
            "answer_model": answer_model,
            "answer_backend": answer_backend,
            "judge_model": judge_model,
            "judge_backend": judge_backend,
            "n_repeats": max(1, n_repeats),
            "raw_judge_runs": raw_runs,
            "api_errors": api_errors if api_errors else None,
            "judged_at": dt.datetime.now().isoformat(),
            **agg,
        },
        "",
    )


def rerun_style_record(
    answer_rec: Dict[str, Any],
    judge_backend: str,
    judge_model: str,
    source_file: str,
    max_tokens: int,
    parse_retries: int,
) -> Tuple[Optional[Dict[str, Any]], str]:
    uid = try_get(answer_rec, ["scenario_uid", "scenario_id", "id"], default="")
    lang = try_get(answer_rec, ["language"], default="en").lower()
    scenario_text = try_get(answer_rec, ["scenario_text"], default="")
    answer_text = try_get(answer_rec, ["answer"], default="")
    model_name = Path(source_file).stem or try_get(answer_rec, ["model"], default="")

    if not answer_text.strip():
        return None, "empty_answer"

    system_prompt, user_prompt = build_style_prompt(lang, scenario_text, answer_text)
    last_error = ""
    parsed = None
    for attempt in range(parse_retries + 1):
        prompt = user_prompt if attempt == 0 else user_prompt + "\n\nREMINDER: Output STRICT JSON only."
        system = system_prompt if attempt == 0 else "Output ONLY valid JSON. No markdown. No explanations."
        result = call_model(
            prompt=prompt,
            backend=judge_backend,
            model=judge_model,
            system_prompt=system,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        text = str(result.get("text") or "")
        last_error = str(result.get("error") or "").strip()
        parsed = safe_parse_dims(text)
        if parsed is not None:
            break

    if parsed is None:
        return None, last_error or "parse_failed"

    return (
        {
            "scenario_uid": uid,
            "language": lang,
            "model_name": model_name,
            "judge_model": judge_model,
            "judge_backend": judge_backend,
            **parsed,
        },
        "",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Rerun rubric/style judges only for answers repaired by answer_repair step 04.")
    ap.add_argument("--apply-manifest", default="answer_repair/workspace/manifests/applied_manual_repairs.csv")
    ap.add_argument("--status", action="append", default=["applied"], help="Manifest status values to include")
    ap.add_argument("--answers-dir", default="data/model_outputs")
    ap.add_argument("--judge-dir", default="data/judge_outputs")
    ap.add_argument("--style-dir", default="data/judge_outputs/style_2d")
    ap.add_argument("--report-dir", default="answer_repair/workspace/judge_rerun")
    ap.add_argument("--skip-rubric", action="store_true")
    ap.add_argument("--skip-style", action="store_true")
    ap.add_argument("--max-tokens-rubric", type=int, default=1024)
    ap.add_argument("--max-tokens-style", type=int, default=256)
    ap.add_argument("--style-parse-retries", type=int, default=1)
    args = ap.parse_args()

    apply_manifest = ROOT / args.apply_manifest
    answers_dir = ROOT / args.answers_dir
    judge_dir = ROOT / args.judge_dir
    style_dir = ROOT / args.style_dir
    report_dir = ROOT / args.report_dir
    ensure_dir(report_dir)

    allowed_statuses = {normalize_cell(value).lower() for value in args.status}
    selected_by_file = load_applied_selection(apply_manifest, allowed_statuses)
    if not selected_by_file:
        raise SystemExit(f"No repaired rows selected from {apply_manifest}")

    rubric_manifest: List[Dict[str, Any]] = []
    style_manifest: List[Dict[str, Any]] = []

    for source_file, uids in progress(sorted(selected_by_file.items()), desc="Source files"):
        answer_path = answers_dir / source_file
        if not answer_path.exists():
            for uid in sorted(uids):
                rubric_manifest.append(
                    {"source_file": source_file, "output_file": "", "scenario_uid": uid, "status": "missing_answer_file", "error": ""}
                )
            continue

        answer_rows = read_jsonl(answer_path)
        answers_by_uid = {try_get(row, ["scenario_uid", "scenario_id", "id"], default=""): row for row in answer_rows}
        source_stem = Path(source_file).stem

        if not args.skip_rubric:
            rubric_files = sorted(judge_dir.glob(f"{source_stem}_judged_*.jsonl"))
            if not rubric_files:
                for uid in sorted(uids):
                    rubric_manifest.append(
                        {"source_file": source_file, "output_file": "", "scenario_uid": uid, "status": "missing_rubric_file", "error": ""}
                    )
            for judge_path in rubric_files:
                existing = read_jsonl(judge_path)
                if not existing:
                    continue
                sample = existing[0]
                judge_backend = normalize_cell(sample.get("judge_backend")).lower()
                judge_model = normalize_cell(sample.get("judge_model"))
                n_repeats = int(sample.get("n_repeats") or 3)
                updated_by_uid: Dict[str, Dict[str, Any]] = {}
                rerun_only_rows: List[Dict[str, Any]] = []

                for uid in progress(sorted(uids), desc=f"Rubric {judge_path.name}", leave=False):
                    answer_rec = answers_by_uid.get(uid)
                    if answer_rec is None:
                        rubric_manifest.append(
                            {
                                "source_file": source_file,
                                "output_file": judge_path.name,
                                "scenario_uid": uid,
                                "status": "missing_answer_row",
                                "error": "",
                            }
                        )
                        continue
                    rerun_row, error = rerun_rubric_record(
                        answer_rec=answer_rec,
                        judge_backend=judge_backend,
                        judge_model=judge_model,
                        n_repeats=n_repeats,
                        max_tokens=int(args.max_tokens_rubric),
                    )
                    if rerun_row is None:
                        rubric_manifest.append(
                            {
                                "source_file": source_file,
                                "output_file": judge_path.name,
                                "scenario_uid": uid,
                                "status": "rerun_failed",
                                "error": error,
                            }
                        )
                        continue
                    updated_by_uid[uid] = rerun_row
                    rerun_only_rows.append(rerun_row)
                    rubric_manifest.append(
                        {
                            "source_file": source_file,
                            "output_file": judge_path.name,
                            "scenario_uid": uid,
                            "status": "rerun_applied",
                            "error": "",
                        }
                    )

                if updated_by_uid:
                    merged = merge_rows_by_uid(existing, updated_by_uid)
                    write_jsonl(judge_path, merged)
                    rerun_only_path = report_dir / "rubric_rerun_only" / judge_path.name
                    ensure_dir(rerun_only_path.parent)
                    write_jsonl(rerun_only_path, rerun_only_rows)

        if not args.skip_style:
            style_files = sorted(style_dir.glob(f"{source_stem}__dims__judge=*.jsonl"))
            if not style_files:
                for uid in sorted(uids):
                    style_manifest.append(
                        {"source_file": source_file, "output_file": "", "scenario_uid": uid, "status": "missing_style_file", "error": ""}
                    )
            for style_path in style_files:
                existing = read_jsonl(style_path)
                if not existing:
                    continue
                sample = existing[0]
                judge_backend = normalize_cell(sample.get("judge_backend")).lower()
                judge_model = normalize_cell(sample.get("judge_model"))
                updated_by_uid = {}
                rerun_only_rows: List[Dict[str, Any]] = []

                for uid in progress(sorted(uids), desc=f"Style {style_path.name}", leave=False):
                    answer_rec = answers_by_uid.get(uid)
                    if answer_rec is None:
                        style_manifest.append(
                            {
                                "source_file": source_file,
                                "output_file": style_path.name,
                                "scenario_uid": uid,
                                "status": "missing_answer_row",
                                "error": "",
                            }
                        )
                        continue
                    rerun_row, error = rerun_style_record(
                        answer_rec=answer_rec,
                        judge_backend=judge_backend,
                        judge_model=judge_model,
                        source_file=source_file,
                        max_tokens=int(args.max_tokens_style),
                        parse_retries=int(args.style_parse_retries),
                    )
                    if rerun_row is None:
                        style_manifest.append(
                            {
                                "source_file": source_file,
                                "output_file": style_path.name,
                                "scenario_uid": uid,
                                "status": "rerun_failed",
                                "error": error,
                            }
                        )
                        continue
                    updated_by_uid[uid] = rerun_row
                    rerun_only_rows.append(rerun_row)
                    style_manifest.append(
                        {
                            "source_file": source_file,
                            "output_file": style_path.name,
                            "scenario_uid": uid,
                            "status": "rerun_applied",
                            "error": "",
                        }
                    )

                if updated_by_uid:
                    merged = merge_rows_by_uid(existing, updated_by_uid)
                    write_jsonl(style_path, merged)
                    rerun_only_path = report_dir / "style_rerun_only" / style_path.name
                    ensure_dir(rerun_only_path.parent)
                    write_jsonl(rerun_only_path, rerun_only_rows)

    write_csv(
        report_dir / "rubric_rerun_manifest.csv",
        rubric_manifest,
        ["source_file", "output_file", "scenario_uid", "status", "error"],
    )
    write_csv(
        report_dir / "style_rerun_manifest.csv",
        style_manifest,
        ["source_file", "output_file", "scenario_uid", "status", "error"],
    )

    rubric_ok = sum(1 for row in rubric_manifest if row["status"] == "rerun_applied")
    rubric_fail = sum(1 for row in rubric_manifest if row["status"] == "rerun_failed")
    style_ok = sum(1 for row in style_manifest if row["status"] == "rerun_applied")
    style_fail = sum(1 for row in style_manifest if row["status"] == "rerun_failed")
    print(f"[DONE] report_dir={report_dir}")
    print(f"[SUMMARY] rubric_ok={rubric_ok} rubric_fail={rubric_fail} style_ok={style_ok} style_fail={style_fail}")


if __name__ == "__main__":
    main()
