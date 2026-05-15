#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.judges.judge_prompts import build_judge_prompt
from src.model_caller import call_model
from src.run_judging import aggregate_runs, safe_parse_json, try_get


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


def normalize_cell(v: Any) -> str:
    return str(v or "").strip()


def normalize_boolish(v: Any) -> str:
    return normalize_cell(v).lower()


def is_selected_value(v: Any) -> bool:
    return normalize_boolish(v) in {"retry", "rerun", "rejudge", "1", "true", "yes", "y"}


def iter_table_rows(path: Path) -> Iterable[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield row
        return

    if suffix == ".numbers":
        try:
            from numbers_parser import Document
        except Exception as e:
            raise RuntimeError(
                "Reading .numbers files requires the 'numbers-parser' package."
            ) from e

        doc = Document(str(path))
        for sheet in doc.sheets:
            for table in sheet.tables:
                rows = list(table.rows(values_only=True))
                if not rows:
                    continue
                header = [normalize_cell(x) for x in rows[0]]
                for raw_row in rows[1:]:
                    row: Dict[str, Any] = {}
                    for i, col in enumerate(header):
                        row[col] = raw_row[i] if i < len(raw_row) else None
                    yield row
                return
        raise ValueError(f"No table rows found in {path}")

    raise ValueError(f"Unsupported selection file format: {path}")


def detect_columns(rows: List[Dict[str, Any]], requested_decision_column: str) -> Tuple[str, str, Optional[str], str]:
    if not rows:
        raise ValueError("selection file is empty")

    keys = set()
    for row in rows[:5]:
        keys.update(row.keys())

    file_candidates: Sequence[str] = ("source_file", "file", "answer_file")
    uid_candidates: Sequence[str] = ("scenario_uid", "uid", "id")
    decision_candidates: Sequence[str] = (
        requested_decision_column,
        "retry",
        "manual_action",
        "rejudge",
        "judge_retry",
        "action",
    )

    file_col = next((c for c in file_candidates if c and c in keys), None)
    uid_col = next((c for c in uid_candidates if c and c in keys), None)
    if not file_col or not uid_col:
        raise ValueError("selection file must contain file/source_file and scenario_uid columns")

    decision_col: Optional[str] = None
    if requested_decision_column != "auto" and requested_decision_column in keys:
        decision_col = requested_decision_column
    else:
        for cand in decision_candidates:
            if cand and cand in keys:
                decision_col = cand
                break

    source_kind = "review_list" if decision_col in {"retry", "manual_action", "rejudge", "judge_retry", "action"} else "manifest"
    return file_col, uid_col, decision_col, source_kind


def load_selection(path: Path, decision_column: str) -> Tuple[Dict[str, Set[str]], List[Dict[str, Any]]]:
    rows = list(iter_table_rows(path))
    file_col, uid_col, detected_decision_col, source_kind = detect_columns(rows, decision_column)

    selected: Dict[str, Set[str]] = {}
    normalized_rows: List[Dict[str, Any]] = []

    for row in rows:
        file_name = normalize_cell(row.get(file_col))
        uid = normalize_cell(row.get(uid_col))
        if not file_name or not uid:
            continue

        include = True
        if detected_decision_col:
            include = is_selected_value(row.get(detected_decision_col))
        if "dry_run" in row and normalize_boolish(row.get("dry_run")) in {"true", "1", "yes"}:
            include = False

        normalized_rows.append(
            {
                "source_file": file_name,
                "scenario_uid": uid,
                "selected": include,
                "decision_column": detected_decision_col or "",
                "source_kind": source_kind,
            }
        )

        if include:
            selected.setdefault(file_name, set()).add(uid)

    return selected, normalized_rows


def export_selection(rows: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source_file", "scenario_uid", "selected", "decision_column", "source_kind"],
        )
        writer.writeheader()
        writer.writerows(rows)


def judge_one_record(
    answer_rec: Dict[str, Any],
    judge_backend: str,
    judge_model: str,
    n_repeats: int,
    max_tokens: int,
    dry_run: bool,
) -> Dict[str, Any]:
    uid = try_get(answer_rec, ["scenario_uid", "scenario_id", "id"], default="")
    lang = try_get(answer_rec, ["language"], default="en").lower()
    src = try_get(answer_rec, ["source"], default="original")
    scenario_text = try_get(answer_rec, ["scenario_text", "prompt", "question", "scenario"], default="")
    answer_text = try_get(answer_rec, ["answer", "response", "answer_text"], default="")
    if not answer_text.strip():
        answer_text = try_get(answer_rec, ["answer_raw"], default="")

    answer_model = try_get(answer_rec, ["answer_model", "model"], default="")
    answer_backend = try_get(answer_rec, ["answer_backend", "backend"], default="")

    raw_runs: List[Dict[str, Any]] = []
    api_errors: List[str] = []

    for _ in range(max(1, int(n_repeats))):
        prompt = build_judge_prompt(
            scenario_text=scenario_text,
            model_response=answer_text,
            scenario_uid=uid,
            language=lang,
        )

        if dry_run:
            out_text = '{"accuracy":50,"safety":50,"helpfulness":50,"empathy":50,"completeness":50,"bias_avoidance":50,"limitation_awareness":50,"communication":50,"comment":"DRY_RUN"}'
            api_err = None
        else:
            result = call_model(
                prompt=prompt,
                backend=judge_backend,
                model=judge_model,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            out_text = str(result.get("text") or "")
            api_err = result.get("error") if isinstance(result, dict) else None

        if api_err:
            api_errors.append(str(api_err))

        try:
            raw_runs.append(safe_parse_json(out_text))
        except Exception as e:
            raw_runs.append({"parse_error": str(e), "raw_text": out_text})

    agg = aggregate_runs(raw_runs)
    return {
        "scenario_uid": uid,
        "language": lang,
        "source": src,
        "scenario_text": scenario_text,
        "answer_text": answer_text,
        "answer_model": answer_model,
        "answer_backend": answer_backend,
        "judge_model": judge_model,
        "judge_backend": judge_backend,
        "n_repeats": max(1, int(n_repeats)),
        "raw_judge_runs": raw_runs,
        "api_errors": api_errors if api_errors else None,
        "judged_at": dt.datetime.now().isoformat(),
        **agg,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Rerun judge only for selected answer rows and merge results back into existing judge files."
    )
    ap.add_argument(
        "--manifest",
        required=True,
        help="Selection source: retry manifest CSV, review CSV/.numbers, or a simple file+scenario_uid list",
    )
    ap.add_argument(
        "--decision-column",
        default="auto",
        help="Decision column for review-style inputs, or auto-detect (default: auto)",
    )
    ap.add_argument(
        "--selection-export-csv",
        default=None,
        help="Optional normalized selection CSV to export before rerunning judges",
    )
    ap.add_argument(
        "--export-only",
        action="store_true",
        help="Only export normalized selection CSV and exit without rerunning judges",
    )
    ap.add_argument("--answers-dir", default="data/model_outputs", help="Directory containing updated answer jsonl files")
    ap.add_argument("--judge-dir", default="data/judge_outputs", help="Directory containing existing judge output jsonl files")
    ap.add_argument("--max-tokens", type=int, default=1024, help="Max tokens for judge calls (default: 1024)")
    ap.add_argument("--judge-rerun-only-dir", default="data/judge_outputs/rerun_only", help="Directory for judge rerun-only jsonl outputs")
    ap.add_argument("--judge-manifest-csv", default="data/judge_outputs/judge_rerun_manifest.csv", help="CSV path recording which judge rows were rerun")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    answers_dir = Path(args.answers_dir)
    judge_dir = Path(args.judge_dir)
    rerun_only_dir = Path(args.judge_rerun_only_dir)
    rerun_only_dir.mkdir(parents=True, exist_ok=True)

    selected_by_source, normalized_rows = load_selection(manifest_path, args.decision_column)
    total_selected = sum(len(v) for v in selected_by_source.values())
    print(f"[INFO] Loaded judge selection: {total_selected} rows across {len(selected_by_source)} files from {manifest_path}")

    if args.selection_export_csv:
        export_path = Path(args.selection_export_csv)
        export_selection(normalized_rows, export_path)
        print(f"[INFO] Wrote normalized selection CSV: {export_path}")

    if args.export_only:
        print("[INFO] export-only enabled; skipping judge rerun.")
        return

    judge_manifest_rows: List[Dict[str, Any]] = []

    for source_file, selected_uids in sorted(selected_by_source.items()):
        answer_path = answers_dir / source_file
        if not answer_path.exists():
            print(f"[WARN] answer file missing, skipped: {answer_path}")
            continue

        source_stem = Path(source_file).stem
        judge_files = sorted(judge_dir.glob(f"{source_stem}_judged_*.jsonl"))
        if not judge_files:
            print(f"[WARN] no judge files matched for {source_file}")
            continue

        answer_rows = {try_get(r, ["scenario_uid", "scenario_id", "id"], ""): r for r in iter_jsonl(answer_path)}
        selected_answers = {uid: answer_rows[uid] for uid in selected_uids if uid in answer_rows}
        if not selected_answers:
            print(f"[WARN] no selected answers found in {answer_path}")
            continue

        for judge_path in judge_files:
            existing = list(iter_jsonl(judge_path))
            if not existing:
                print(f"[WARN] empty judge file skipped: {judge_path}")
                continue

            sample = existing[0]
            judge_backend = str(sample.get("judge_backend") or "").strip()
            judge_model = str(sample.get("judge_model") or "").strip()
            n_repeats = int(sample.get("n_repeats") or 3)

            print(
                f"[INFO] Rerun judge file={judge_path.name} backend={judge_backend} model={judge_model} rows={len(selected_answers)}"
            )

            updated_by_uid: Dict[str, Dict[str, Any]] = {}
            rerun_only_path = rerun_only_dir / f"{judge_path.stem}__rerun_only.jsonl"
            with rerun_only_path.open("w", encoding="utf-8") as rerun_only_f:
                for uid, answer_rec in selected_answers.items():
                    out_rec = judge_one_record(
                        answer_rec=answer_rec,
                        judge_backend=judge_backend,
                        judge_model=judge_model,
                        n_repeats=n_repeats,
                        max_tokens=args.max_tokens,
                        dry_run=args.dry_run,
                    )
                    updated_by_uid[uid] = out_rec
                    rerun_only_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                    judge_manifest_rows.append(
                        {
                            "source_answer_file": source_file,
                            "judge_file": judge_path.name,
                            "scenario_uid": uid,
                            "judge_backend": judge_backend,
                            "judge_model": judge_model,
                            "n_repeats": n_repeats,
                            "dry_run": args.dry_run,
                        }
                    )

            merged: List[Dict[str, Any]] = []
            seen: Set[str] = set()
            for rec in existing:
                uid = try_get(rec, ["scenario_uid", "scenario_id", "id"], default="")
                if uid and uid in updated_by_uid:
                    merged.append(updated_by_uid[uid])
                    seen.add(uid)
                else:
                    merged.append(rec)

            for uid, rec in updated_by_uid.items():
                if uid not in seen:
                    merged.append(rec)

            tmp_path = judge_path.with_suffix(judge_path.suffix + ".tmp")
            with tmp_path.open("w", encoding="utf-8") as f:
                for rec in merged:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            os.replace(tmp_path, judge_path)
            print(f"[DONE] merged updated judge rows into {judge_path}")

    if judge_manifest_rows:
        out_manifest = Path(args.judge_manifest_csv)
        out_manifest.parent.mkdir(parents=True, exist_ok=True)
        with out_manifest.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(judge_manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(judge_manifest_rows)
        print(f"[DONE] judge rerun manifest: {out_manifest}")


if __name__ == "__main__":
    main()
