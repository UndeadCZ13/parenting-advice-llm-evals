#!/usr/bin/env python3
# scripts/retry_bad_generations.py

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_caller import call_model
from src.run_generation import (
    build_final_only_prompt,
    strip_reasoning,
    suspected_truncation,
    is_reasoning_model,
)
from src.config import (
    MAX_TOKENS_DEFAULT,
    MAX_TOKENS_REASONING_MODEL,
    RETRY_TOKEN_MULTIPLIER,
    STRIP_REASONING,
)


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


def is_empty_text(x: Any) -> bool:
    if x is None:
        return True
    if not isinstance(x, str):
        x = str(x)
    return len(x.strip()) == 0


def get_uid(rec: Dict[str, Any]) -> str:
    return pick_str(rec, ["scenario_uid", "scenario_id", "id"], default="")


def get_lang(rec: Dict[str, Any]) -> str:
    # prefer explicit language in record; fallback from filename handled outside
    return pick_str(rec, ["language", "lang"], default="en").lower()


def load_scenarios_map(scenarios_path: Path) -> Dict[str, Dict[str, Any]]:
    mp: Dict[str, Dict[str, Any]] = {}
    for s in iter_jsonl(scenarios_path):
        uid = get_uid(s)
        if uid:
            mp[uid] = s
    return mp


def has_prompt_text(rec: Optional[Dict[str, Any]]) -> bool:
    if not rec:
        return False
    return bool(pick_str(rec, ["scenario_text", "prompt", "question", "scenario"], default=""))


def normalize_yes_no(v: Any) -> str:
    return str(v or "").strip().lower()


def is_retry_decision(v: Any) -> bool:
    return normalize_yes_no(v) in {"retry", "rerun", "1", "true", "yes", "y"}


def iter_review_rows(review_list_path: Path) -> Iterable[Dict[str, Any]]:
    suffix = review_list_path.suffix.lower()
    if suffix == ".csv":
        with review_list_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield row
        return

    if suffix == ".numbers":
        try:
            from numbers_parser import Document
        except Exception as e:
            raise RuntimeError(
                "Reading .numbers review lists requires the 'numbers-parser' package."
            ) from e

        doc = Document(str(review_list_path))
        for sheet in doc.sheets:
            for table in sheet.tables:
                rows = list(table.rows(values_only=True))
                if not rows:
                    continue
                header = [str(x) if x is not None else "" for x in rows[0]]
                for raw_row in rows[1:]:
                    row: Dict[str, Any] = {}
                    for i, col in enumerate(header):
                        row[col] = raw_row[i] if i < len(raw_row) else None
                    yield row
                return
        raise ValueError(f"No table rows found in {review_list_path}")

    raise ValueError(f"Unsupported review list format: {review_list_path}")


def resolve_decision_column(review_list_path: Path, requested: str) -> str:
    if requested and requested != "auto":
        return requested

    candidates: Sequence[str] = ("manual_action", "retry")
    for row in iter_review_rows(review_list_path):
        keys = set(row.keys())
        for cand in candidates:
            if cand in keys:
                return cand
        break
    raise ValueError(f"Could not auto-detect decision column in {review_list_path}")


def load_retry_selection(review_list_path: Path, decision_column: str) -> Dict[str, Set[str]]:
    selected: Dict[str, Set[str]] = {}
    resolved = resolve_decision_column(review_list_path, decision_column)
    for row in iter_review_rows(review_list_path):
        if not is_retry_decision(row.get(resolved)):
            continue
        file_name = str(row.get("file") or "").strip()
        uid = str(row.get("scenario_uid") or "").strip()
        if not file_name or not uid:
            continue
        selected.setdefault(file_name, set()).add(uid)
    return selected


def should_retry(rec: Dict[str, Any], mode: str) -> bool:
    ans = rec.get("answer")
    empty = is_empty_text(ans)
    trunc = pick_bool(rec, ["suspected_truncation"], default=False)

    if mode == "empty":
        return empty
    if mode == "trunc":
        return trunc
    # both (default)
    return bool(empty or trunc)


def should_retry_rerun_attempt(final_text: str, finish_reason: Any, api_error: Any) -> bool:
    if not is_empty_text(api_error):
        return True
    if is_empty_text(final_text):
        return True
    if isinstance(finish_reason, str) and finish_reason.strip().lower() == "length":
        return True
    return False



def get_base_max_tokens(model_name: str) -> int:
    reasoning = is_reasoning_model(model_name)
    return MAX_TOKENS_REASONING_MODEL if reasoning else MAX_TOKENS_DEFAULT


def regenerate_one_pass(
    scenario_rec: Dict[str, Any],
    backend: str,
    model_name: str,
    max_tokens: int,
    max_retries_per_call: int = 15,
) -> Tuple[Dict[str, Any], int]:
    """
    Do exactly one outer rerun pass for one item.
    The underlying API call may still retry up to `max_retries_per_call`.
    Returns: (update_fields, next_max_tokens)
    """
    lang = get_lang(scenario_rec)
    user_prompt = pick_str(scenario_rec, ["scenario_text", "prompt", "question", "scenario"], default="")
    if not user_prompt:
        return ({
            "answer_raw": "",
            "answer": "",
            "suspected_truncation": True,
            "api_error": "scenario_text missing",
            "retry_success": False,
            "retry_final_max_tokens": max_tokens,
        }, max_tokens)

    try:
        prompt = build_final_only_prompt(user_prompt, lang, model_name)  # new
    except TypeError:
        prompt = build_final_only_prompt(user_prompt, lang)  # old

    resp = call_model(
        prompt=prompt,
        backend=backend,
        model=model_name,
        max_tokens=max_tokens,
        max_retries=max_retries_per_call,
    )

    raw = resp.get("text", "") or ""
    finish_reason = resp.get("finish_reason")
    api_error = resp.get("error")

    cleaned = strip_reasoning(raw) if STRIP_REASONING else {
        "final": raw,
        "stripped": False,
        "removed_chars": 0,
    }
    final = cleaned["final"]
    retry_needed = should_retry_rerun_attempt(final, finish_reason, api_error)
    next_max_tokens = max_tokens
    if isinstance(finish_reason, str) and finish_reason.strip().lower() == "length":
        next_max_tokens = int(max_tokens * RETRY_TOKEN_MULTIPLIER)

    if not retry_needed:
        return ({
            "answer_raw": raw,
            "answer": final,
            "finish_reason": finish_reason,
            "api_error": api_error,
            "reasoning_stripped": cleaned.get("stripped", False),
            "removed_reasoning_chars": cleaned.get("removed_chars", 0),
            "suspected_truncation": suspected_truncation(final, finish_reason),
            "retry_success": True,
            "retry_final_max_tokens": max_tokens,
        }, next_max_tokens)

    return ({
        "answer_raw": raw if "raw" in locals() else "",
        "answer": final if "final" in locals() else "",
        "finish_reason": finish_reason if "finish_reason" in locals() else None,
        "api_error": api_error if "api_error" in locals() else "retry_exhausted",
        "reasoning_stripped": cleaned.get("stripped", False) if "cleaned" in locals() else False,
        "removed_reasoning_chars": cleaned.get("removed_chars", 0) if "cleaned" in locals() else 0,
        "suspected_truncation": True,
        "retry_success": False,
        "retry_final_max_tokens": max_tokens,
    }, next_max_tokens)


def process_file(
    in_path: Path,
    scenarios_map: Dict[str, Dict[str, Any]],
    attempts: int,
    max_retries_per_call: int,
    out_path: Path,
    output_label: Optional[str] = None,
    dry_run: bool = False,
    mode: str = "both",
    selected_uids: Optional[Set[str]] = None,
    mode_label: Optional[str] = None,
) -> Tuple[int, int, int, List[Dict[str, Any]]]:
    """
    Retry selected records in round-robin passes.
    One item gets one outer pass, then if still failing it is deferred to the queue tail.
    Returns: (n_total, n_bad, n_fixed, rerun_manifest_rows)
    """
    records = list(iter_jsonl(in_path))
    n_total = len(records)

    selected_uid_set = selected_uids or set()

    bad_indices: List[int] = []
    for i, rec in enumerate(records):
        uid = get_uid(rec)
        if not uid:
            continue
        if selected_uids is not None:
            if uid in selected_uid_set:
                bad_indices.append(i)
            continue
        if should_retry(rec, mode):
            bad_indices.append(i)

    n_bad = len(bad_indices)
    if n_bad == 0:
        if selected_uids is not None:
            print(f"[SKIP] {in_path.name}: no records selected from review list. No output file generated.")
        else:
            print(f"[SKIP] {in_path.name}: no records to retry (mode={mode}). No output file generated.")
        return n_total, 0, 0, []

    n_fixed = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    retry_mode_value = mode_label or mode
    rerun_rows: List[Dict[str, Any]] = []
    updated_records = list(records)

    queue: List[Dict[str, Any]] = []
    for idx in bad_indices:
        rec = records[idx]
        uid = get_uid(rec)
        backend = pick_str(rec, ["backend", "answer_backend"], default="").lower()
        if backend == "local":
            backend = "ollama"
        model_name = pick_str(rec, ["model", "answer_model"], default="")
        scenario_rec = rec
        if not has_prompt_text(scenario_rec):
            scenario_rec = scenarios_map.get(uid) or rec
        queue.append(
            {
                "idx": idx,
                "uid": uid,
                "backend": backend,
                "model_name": model_name,
                "scenario_rec": scenario_rec,
                "attempt_count": 0,
                "max_tokens": get_base_max_tokens(model_name),
                "deferred_count": 0,
            }
        )

    while queue:
        state = queue.pop(0)
        idx = int(state["idx"])
        uid = str(state["uid"])
        rec = records[idx]
        state["attempt_count"] = int(state["attempt_count"]) + 1

        print(
            f"[RETRY] file={in_path.name} uid={uid} backend={state['backend']} model={state['model_name']} "
            f"pass={state['attempt_count']}/{attempts}"
        )

        if dry_run:
            upd = {"retry_success": False, "api_error": "DRY_RUN", "retry_final_max_tokens": state["max_tokens"]}
            next_max_tokens = int(state["max_tokens"])
        else:
            upd, next_max_tokens = regenerate_one_pass(
                scenario_rec=state["scenario_rec"],
                backend=str(state["backend"]),
                model_name=str(state["model_name"]),
                max_tokens=int(state["max_tokens"]),
                max_retries_per_call=max_retries_per_call,
            )

        if upd.get("retry_success"):
            n_fixed += 1
            total_attempts = int(state["attempt_count"])
            deferred_rounds = int(state["deferred_count"])
        else:
            state["max_tokens"] = next_max_tokens
            if int(state["attempt_count"]) < attempts:
                state["deferred_count"] = int(state["deferred_count"]) + 1
                print(
                    f"[DEFER] file={in_path.name} uid={uid} reason={upd.get('api_error') or upd.get('finish_reason') or 'retry_needed'} "
                    f"next_pass={int(state['attempt_count']) + 1}/{attempts}"
                )
                queue.append(state)
                continue
            total_attempts = int(state["attempt_count"])
            deferred_rounds = int(state["deferred_count"])

        new_rec = {
            **rec,
            **upd,
            "retried": True,
            "retry_mode": retry_mode_value,
            "retry_attempts": total_attempts,
            "retry_deferred_rounds": deferred_rounds,
        }
        updated_records[idx] = new_rec
        rerun_rows.append(
            {
                "source_file": in_path.name,
                "output_file": output_label or out_path.name,
                "scenario_uid": uid,
                "backend": state["backend"],
                "model": state["model_name"],
                "language": get_lang(rec),
                "retry_mode": retry_mode_value,
                "dry_run": dry_run,
                "retry_success": bool(upd.get("retry_success")),
                "retry_attempts": total_attempts,
                "retry_deferred_rounds": deferred_rounds,
                "old_finish_reason": rec.get("finish_reason"),
                "new_finish_reason": new_rec.get("finish_reason", rec.get("finish_reason")),
                "old_answer_len": 0 if rec.get("answer") is None else len(str(rec.get("answer"))),
                "new_answer_len": 0 if new_rec.get("answer") is None else len(str(new_rec.get("answer"))),
                "api_error": upd.get("api_error"),
            }
        )

    with out_path.open("w", encoding="utf-8") as f:
        for rec in updated_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return n_total, n_bad, n_fixed, rerun_rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Retry empty/trunc generations in model_outputs using scenario_uid.")
    ap.add_argument("--scenarios", default=None, help="Optional scenario jsonl file; original answer records are preferred when they already contain scenario_text")
    ap.add_argument("--inputs", default="data/model_outputs", help="Input file or directory (default: data/model_outputs)")
    ap.add_argument("--pattern", default="*.jsonl", help="Glob pattern if inputs is a directory (default: *.jsonl)")
    ap.add_argument("--attempts", type=int, default=15, help="Max attempts per bad scenario (default: 15)")
    ap.add_argument("--max-retries-per-call", type=int, default=15, help="Internal retry per API call (default: 15)")
    ap.add_argument("--write-mode", choices=["inplace", "copy"], default="inplace", help="Write updated rows back to the original file, or to a copied file (default: inplace)")
    ap.add_argument("--out-dir", default=None, help="Output directory when --write-mode=copy (default: alongside input files)")
    ap.add_argument("--suffix", default="_retried", help="Suffix for copied output filename or manifest naming (default: _retried)")
    ap.add_argument("--rerun-only-dir", default=None, help="Directory to write rerun-only jsonl files (default: data/model_outputs/rerun_only)")
    ap.add_argument("--manifest-csv", default=None, help="CSV path to record exactly which rows were rerun (default: data/model_outputs/retry_manifest<suffix>.csv)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--retry-mode",choices=["empty", "trunc", "both"],default="both",help="Which records to retry: empty only, trunc only, or both (default: both)",)
    ap.add_argument("--review-list", default=None, help="CSV from check_generation_completeness.py; only rows marked for retry will be rerun")
    ap.add_argument("--decision-column", default="auto", help="Column in --review-list that marks rows to retry, or auto-detect from .csv/.numbers (default: auto)")

    args = ap.parse_args()

    scenarios_map: Dict[str, Dict[str, Any]] = {}
    if args.scenarios:
        scenarios_path = Path(args.scenarios)
        if not scenarios_path.exists():
            raise FileNotFoundError(scenarios_path)
        scenarios_map = load_scenarios_map(scenarios_path)
        print(f"[INFO] Loaded scenarios: {len(scenarios_map)} from {scenarios_path}")
    else:
        print("[INFO] No external scenarios file provided; will use scenario_text stored in answer records.")

    inp = Path(args.inputs)
    files: List[Path] = []
    selected_by_file: Optional[Dict[str, Set[str]]] = None
    mode_label = args.retry_mode

    if args.review_list:
        review_list_path = Path(args.review_list)
        if not review_list_path.exists():
            raise FileNotFoundError(review_list_path)
        resolved_decision_column = resolve_decision_column(review_list_path, args.decision_column)
        selected_by_file = load_retry_selection(review_list_path, resolved_decision_column)
        chosen = sum(len(v) for v in selected_by_file.values())
        print(f"[INFO] Loaded retry selection: {chosen} rows across {len(selected_by_file)} files from {review_list_path} using column={resolved_decision_column}")
        mode_label = f"manual_list:{review_list_path.name}"

    if inp.is_file():
        files = [inp]
    else:
        files = sorted(inp.glob(args.pattern))
    if not files:
        print(f"[WARN] No input files matched: {inp} pattern={args.pattern}")
        return

    if selected_by_file is not None:
        files = [fp for fp in files if fp.name in selected_by_file]
        if not files:
            print(f"[WARN] No input files matched entries in review list: {inp}")
            return

    total_bad = 0
    total_fixed = 0
    total_records = 0
    manifest_rows: List[Dict[str, Any]] = []

    for fp in files:
        if args.write_mode == "copy":
            out_dir = Path(args.out_dir) if args.out_dir else fp.parent
            out_path = out_dir / f"{fp.stem}{args.suffix}.jsonl"
        else:
            out_dir = fp.parent
            out_path = fp.with_suffix(fp.suffix + ".tmp")

        rerun_only_base = Path(args.rerun_only_dir) if args.rerun_only_dir else (ROOT / "data" / "model_outputs" / "rerun_only")
        rerun_only_path = rerun_only_base / f"{fp.stem}{args.suffix}__rerun_only.jsonl"

        n_total, n_bad, n_fixed, rerun_rows = process_file(
            in_path=fp,
            scenarios_map=scenarios_map,
            attempts=args.attempts,
            max_retries_per_call=args.max_retries_per_call,
            out_path=out_path,
            output_label=fp.name if args.write_mode == "inplace" else out_path.name,
            dry_run=args.dry_run,
            mode=args.retry_mode,
            selected_uids=None if selected_by_file is None else selected_by_file.get(fp.name, set()),
            mode_label=mode_label,
        )

        total_records += n_total
        total_bad += n_bad
        total_fixed += n_fixed
        manifest_rows.extend(rerun_rows)

        if rerun_rows:
            rerun_only_base.mkdir(parents=True, exist_ok=True)
            selected_ids = {row["scenario_uid"] for row in rerun_rows}
            with rerun_only_path.open("w", encoding="utf-8") as f:
                for rec in iter_jsonl(out_path):
                    if get_uid(rec) in selected_ids:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if args.write_mode == "inplace" and out_path.exists():
            os.replace(out_path, fp)

        target_path = fp if args.write_mode == "inplace" else out_path
        print(f"[DONE] {fp.name}: total={n_total} bad={n_bad} fixed={n_fixed} -> {target_path}")

    default_manifest_dir = (Path(args.out_dir) if (args.out_dir and args.write_mode == "copy") else (ROOT / "data" / "model_outputs"))
    manifest_path = Path(args.manifest_csv) if args.manifest_csv else (default_manifest_dir / f"retry_manifest{args.suffix}.csv")
    if manifest_rows:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)
        print(f"[SUMMARY] manifest={manifest_path}")

    print("============================================================")
    print(f"[SUMMARY] files={len(files)} total_records={total_records} bad={total_bad} fixed={total_fixed} mode={mode_label}")
    if total_bad > 0:
        print(f"[SUMMARY] fix_rate={total_fixed/total_bad:.2%}")


if __name__ == "__main__":
    main()
