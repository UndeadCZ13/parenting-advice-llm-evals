#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.judges.judge_prompts import RUBRIC_KEYS, build_judge_prompt
from src.model_caller import call_openai_chat
from src.run_generation import build_final_only_prompt, strip_reasoning, suspected_truncation
from src.run_judging import aggregate_runs, safe_parse_json, try_get
from src.run_style_judging import build_prompt as build_style_prompt, safe_parse_dims


SCENARIO_FILES = {
    "en": ROOT / "data" / "scenarios" / "views" / "en" / "parentbench_v0_en.jsonl",
    "zh": ROOT / "data" / "scenarios" / "views" / "zh" / "parentbench_v0_zh.jsonl",
}

MODEL_OUTPUTS_DIR = ROOT / "data" / "model_outputs"
JUDGE_OUTPUTS_DIR = ROOT / "data" / "judge_outputs"
STYLE_OUTPUTS_DIR = JUDGE_OUTPUTS_DIR / "style_2d"


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


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return list(iter_jsonl(path))


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


def load_selection(path: Path) -> Dict[str, Set[str]]:
    df_rows: Dict[str, Set[str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            selected = str(row.get("selected") or "").strip().lower()
            if selected not in {"true", "1", "yes", "y"}:
                continue
            file_name = str(row.get("source_file") or row.get("file") or "").strip()
            uid = str(row.get("scenario_uid") or "").strip()
            if not file_name or not uid:
                continue
            df_rows.setdefault(file_name, set()).add(uid)
    return df_rows


def load_scenarios() -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for lang, path in SCENARIO_FILES.items():
        for row in iter_jsonl(path):
            uid = str(row.get("scenario_uid") or "").strip()
            if uid:
                out[(lang, uid)] = row
    return out


def get_lang_from_file_name(file_name: str) -> str:
    name = file_name.lower()
    if name.startswith("zh_") or "_zh" in name:
        return "zh"
    return "en"


def strip_terminal_noise(text: str) -> str:
    s = text or ""
    s = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", s)
    s = re.sub(r"\x1b\][^\x07]*(?:\x07|\x1b\\)", "", s)
    s = s.replace("\r", "")
    s = re.sub(r"[\u2800-\u28ff]", "", s)
    s = re.sub(r"[?]2026[hl]|[?]25[hl]|\[K|\[1G|\[2K", "", s)
    s = re.sub(r"^\s*Thinking\.\.\.\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\.\.\.\s*done thinking\.\s*", "", s, flags=re.IGNORECASE)
    return s.strip()


def run_ollama_cli(prompt: str, model: str, json_mode: bool = False, timeout_sec: int = 900) -> Dict[str, Any]:
    cmd = [
        "ollama",
        "run",
        model,
        "--hidethinking",
        "--think",
        "false",
        "--nowordwrap",
    ]
    if json_mode:
        cmd.extend(["--format", "json"])
    env = os.environ.copy()
    env["OLLAMA_NOHISTORY"] = "1"
    try:
        res = subprocess.run(
            cmd,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {"text": "", "finish_reason": None, "error": f"timeout_after_{timeout_sec}s"}

    raw = strip_terminal_noise((res.stdout or "") + ("\n" + res.stderr if res.stderr else ""))
    if res.returncode != 0:
        return {"text": raw, "finish_reason": None, "error": f"returncode={res.returncode}"}
    return {"text": raw, "finish_reason": "stop", "error": None}


def call_generation_model(
    backend: str,
    model: str,
    prompt: str,
    max_tokens: int,
    ollama_timeout_sec: int,
) -> Dict[str, Any]:
    if backend == "ollama":
        return run_ollama_cli(prompt=prompt, model=model, json_mode=False, timeout_sec=ollama_timeout_sec)
    if backend == "openai":
        return call_openai_chat(prompt=prompt, model=model, max_tokens=max_tokens, max_retries=5, reasoning_effort="low")
    raise ValueError(f"Unsupported generation backend: {backend}")


def call_judge_model(
    backend: str,
    model: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 1024,
    json_mode: bool = True,
) -> Dict[str, Any]:
    full_prompt = prompt if not system_prompt else f"{system_prompt}\n\n{prompt}"
    if backend == "ollama":
        return run_ollama_cli(prompt=full_prompt, model=model, json_mode=json_mode, timeout_sec=900)
    if backend == "openai":
        return call_openai_chat(
            prompt=prompt,
            model=model,
            system_prompt=system_prompt,
            temperature=0.0,
            max_tokens=max_tokens,
            max_retries=5,
            reasoning_effort="low",
        )
    raise ValueError(f"Unsupported judge backend: {backend}")


def build_generation_prompt(scenario_text: str, lang: str, model_name: str) -> str:
    try:
        return build_final_only_prompt(scenario_text, lang, model_name)
    except TypeError:
        return build_final_only_prompt(scenario_text, lang)


def regenerate_answer(
    record: Dict[str, Any],
    scenario_rec: Dict[str, Any],
    attempts: int = 4,
    ollama_timeout_sec: int = 600,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    backend = str(record.get("backend") or "").strip().lower()
    model = str(record.get("model") or "").strip()
    lang = str(record.get("language") or "").strip().lower()
    scenario_text = str(scenario_rec.get("scenario_text") or "").strip()
    if backend == "local":
        backend = "ollama"
    if not scenario_text:
        return None, "missing_scenario_text"

    base_max_tokens = 4096 if backend == "openai" and model.startswith("gpt-5") else 2048
    max_tokens = base_max_tokens
    last_err = "unknown"

    for attempt in range(1, attempts + 1):
        prompt = build_generation_prompt(scenario_text, lang, model)
        result = call_generation_model(
            backend=backend,
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            ollama_timeout_sec=ollama_timeout_sec,
        )
        raw = str(result.get("text") or "")
        finish_reason = result.get("finish_reason")
        api_error = result.get("error")

        cleaned = strip_reasoning(raw)
        final = cleaned["final"]
        if final.strip() and not api_error and str(finish_reason or "").strip().lower() != "length":
            updated = {
                **record,
                **scenario_rec,
                "answer_raw": raw,
                "answer": final,
                "reasoning_stripped": cleaned.get("stripped", False),
                "removed_reasoning_chars": cleaned.get("removed_chars", 0),
                "suspected_truncation": suspected_truncation(final, finish_reason),
                "retry_used": True,
                "finish_reason": finish_reason or "stop",
                "api_error": None,
            }
            return updated, None

        last_err = str(api_error or finish_reason or "empty_answer")
        if str(finish_reason or "").strip().lower() == "length":
            max_tokens *= 2

    return None, last_err


def rerun_rubric_record(answer_rec: Dict[str, Any], judge_backend: str, judge_model: str, n_repeats: int, max_tokens: int = 1024) -> Dict[str, Any]:
    uid = try_get(answer_rec, ["scenario_uid", "scenario_id", "id"], default="")
    lang = try_get(answer_rec, ["language"], default="en").lower()
    src = try_get(answer_rec, ["source"], default="original")
    scenario_text = try_get(answer_rec, ["scenario_text"], default="")
    answer_text = try_get(answer_rec, ["answer"], default="")
    answer_model = try_get(answer_rec, ["model", "answer_model"], default="")
    answer_backend = try_get(answer_rec, ["backend", "answer_backend"], default="")

    raw_runs: List[Dict[str, Any]] = []
    api_errors: List[str] = []
    for _ in range(max(1, n_repeats)):
        prompt = build_judge_prompt(
            scenario_text=scenario_text,
            model_response=answer_text,
            scenario_uid=uid,
            language=lang,
        )
        result = call_judge_model(
            backend=judge_backend,
            model=judge_model,
            prompt=prompt,
            system_prompt=None,
            max_tokens=max_tokens,
            json_mode=True,
        )
        out_text = str(result.get("text") or "")
        api_err = result.get("error")
        if api_err:
            api_errors.append(str(api_err))
        try:
            raw_runs.append(safe_parse_json(out_text))
        except Exception as e:
            retry_result = call_judge_model(
                backend=judge_backend,
                model=judge_model,
                prompt=prompt + "\n\nReturn ONLY one JSON object.",
                system_prompt=None,
                max_tokens=max_tokens,
                json_mode=True,
            )
            retry_text = str(retry_result.get("text") or "")
            retry_err = retry_result.get("error")
            if retry_err:
                api_errors.append(str(retry_err))
            try:
                raw_runs.append(safe_parse_json(retry_text))
            except Exception:
                raw_runs.append({"parse_error": str(e), "raw_text": out_text[:500]})

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
        "n_repeats": max(1, n_repeats),
        "raw_judge_runs": raw_runs,
        "api_errors": api_errors if api_errors else None,
        "judged_at": dt.datetime.now().isoformat(),
        **agg,
    }


def rerun_style_record(
    answer_rec: Dict[str, Any],
    judge_backend: str,
    judge_model: str,
    source_file: str,
    max_tokens: int = 256,
    parse_retries: int = 1,
) -> Dict[str, Any]:
    uid = try_get(answer_rec, ["scenario_uid", "scenario_id", "id"], default="")
    lang = try_get(answer_rec, ["language"], default="en").lower()
    scenario_text = try_get(answer_rec, ["scenario_text"], default="")
    answer_text = try_get(answer_rec, ["answer"], default="")
    model_name = Path(source_file).stem or try_get(answer_rec, ["model"], default="")

    if not answer_text.strip():
        return {
            "scenario_uid": uid,
            "language": lang,
            "model_name": model_name,
            "judge_model": judge_model,
            "judge_backend": judge_backend,
            "responsiveness": 0.0,
            "demandingness": 0.0,
            "error": "empty_answer",
        }

    system_prompt, user_prompt = build_style_prompt(lang, scenario_text, answer_text)
    last_error = None
    parsed = None
    for attempt in range(parse_retries + 1):
        prompt = user_prompt if attempt == 0 else user_prompt + "\n\nREMINDER: Output STRICT JSON only."
        system = system_prompt if attempt == 0 else "Output ONLY valid JSON. No markdown. No explanations."
        result = call_judge_model(
            backend=judge_backend,
            model=judge_model,
            prompt=prompt,
            system_prompt=system,
            max_tokens=max_tokens,
            json_mode=True,
        )
        text = str(result.get("text") or "")
        last_error = result.get("error")
        parsed = safe_parse_dims(text)
        if parsed is not None:
            break

    if parsed is None:
        return {
            "scenario_uid": uid,
            "language": lang,
            "model_name": model_name,
            "judge_model": judge_model,
            "judge_backend": judge_backend,
            "error": last_error or "parse_failed",
        }

    return {
        "scenario_uid": uid,
        "language": lang,
        "model_name": model_name,
        "judge_model": judge_model,
        "judge_backend": judge_backend,
        **parsed,
    }


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Repair selected answers and incrementally rerun rubric/style judges.")
    ap.add_argument("--selection", required=True, help="CSV with source_file, scenario_uid, selected=True")
    ap.add_argument("--report-dir", default="results/analysis/repair_run", help="Directory for repair manifests")
    ap.add_argument("--only-file", action="append", default=[], help="Limit processing to one or more source_file values")
    ap.add_argument("--gen-attempts-openai", type=int, default=3)
    ap.add_argument("--gen-attempts-ollama", type=int, default=2)
    ap.add_argument("--ollama-gen-timeout-sec", type=int, default=600)
    args = ap.parse_args()

    selection_path = Path(args.selection)
    if not selection_path.exists():
        raise FileNotFoundError(selection_path)

    report_dir = ROOT / args.report_dir if not str(args.report_dir).startswith("/") else Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    selected_by_file = load_selection(selection_path)
    if args.only_file:
        keep = set(args.only_file)
        selected_by_file = {k: v for k, v in selected_by_file.items() if k in keep}
    scenarios = load_scenarios()
    print(f"[INFO] selected files={len(selected_by_file)} rows={sum(len(v) for v in selected_by_file.values())}")

    answer_manifest: List[Dict[str, Any]] = []
    successful_answer_updates: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for source_file, selected_uids in sorted(selected_by_file.items()):
        answer_path = MODEL_OUTPUTS_DIR / source_file
        rows = read_jsonl(answer_path)
        rows_by_uid = {try_get(r, ["scenario_uid", "scenario_id", "id"], default=""): r for r in rows}
        updated_by_uid: Dict[str, Dict[str, Any]] = {}

        print(f"[ANS] file={source_file} selected={len(selected_uids)}")
        for uid in sorted(selected_uids):
            print(f"[ANS] uid={uid} source={source_file}")
            rec = rows_by_uid.get(uid)
            if rec is None:
                answer_manifest.append({"source_file": source_file, "scenario_uid": uid, "status": "missing_answer_row"})
                continue
            lang = str(rec.get("language") or get_lang_from_file_name(source_file)).lower()
            scenario_rec = scenarios.get((lang, uid))
            if scenario_rec is None:
                answer_manifest.append({"source_file": source_file, "scenario_uid": uid, "status": "missing_scenario"})
                continue

            backend = str(rec.get("backend") or "").strip().lower()
            attempts = args.gen_attempts_ollama if backend in {"ollama", "local"} else args.gen_attempts_openai
            updated, err = regenerate_answer(
                rec,
                scenario_rec,
                attempts=attempts,
                ollama_timeout_sec=args.ollama_gen_timeout_sec,
            )
            if updated is None:
                print(f"[ANS] failed uid={uid} source={source_file} error={err}")
                answer_manifest.append({"source_file": source_file, "scenario_uid": uid, "status": "failed", "error": err})
                continue

            updated_by_uid[uid] = updated
            print(f"[ANS] updated uid={uid} source={source_file}")
            answer_manifest.append({"source_file": source_file, "scenario_uid": uid, "status": "updated", "error": ""})

        if updated_by_uid:
            merged = merge_rows_by_uid(rows, updated_by_uid)
            write_jsonl(answer_path, merged)
            successful_answer_updates[source_file] = updated_by_uid
            print(f"[ANS] updated {source_file}: {len(updated_by_uid)} rows")

    with (report_dir / "answer_repair_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["source_file", "scenario_uid", "status", "error"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(answer_manifest)

    rubric_manifest: List[Dict[str, Any]] = []
    style_manifest: List[Dict[str, Any]] = []

    for source_file, updated_by_uid in sorted(successful_answer_updates.items()):
        source_stem = Path(source_file).stem
        rubric_files = sorted(JUDGE_OUTPUTS_DIR.glob(f"{source_stem}_judged_*.jsonl"))
        style_files = sorted(STYLE_OUTPUTS_DIR.glob(f"{source_stem}__dims__judge=*.jsonl"))

        for judge_path in rubric_files:
            existing = read_jsonl(judge_path)
            if not existing:
                continue
            sample = existing[0]
            judge_backend = str(sample.get("judge_backend") or "").strip()
            judge_model = str(sample.get("judge_model") or "").strip()
            n_repeats = int(sample.get("n_repeats") or 3)
            print(f"[RUBRIC] file={judge_path.name} rows={len(updated_by_uid)} judge={judge_backend}:{judge_model}")

            rerun_rows = {
                uid: rerun_rubric_record(ans, judge_backend=judge_backend, judge_model=judge_model, n_repeats=n_repeats)
                for uid, ans in updated_by_uid.items()
            }
            merged = merge_rows_by_uid(existing, rerun_rows)
            write_jsonl(judge_path, merged)

            rerun_only_path = report_dir / "rubric_rerun_only" / judge_path.name
            rerun_only_path.parent.mkdir(parents=True, exist_ok=True)
            write_jsonl(rerun_only_path, list(rerun_rows.values()))
            for uid in rerun_rows:
                rubric_manifest.append({"source_file": source_file, "judge_file": judge_path.name, "scenario_uid": uid})

        for style_path in style_files:
            existing = read_jsonl(style_path)
            if not existing:
                continue
            sample = existing[0]
            judge_backend = str(sample.get("judge_backend") or "").strip()
            judge_model = str(sample.get("judge_model") or "").strip()
            print(f"[STYLE] file={style_path.name} rows={len(updated_by_uid)} judge={judge_backend}:{judge_model}")

            rerun_rows = {
                uid: rerun_style_record(ans, judge_backend=judge_backend, judge_model=judge_model, source_file=source_file)
                for uid, ans in updated_by_uid.items()
            }
            merged = merge_rows_by_uid(existing, rerun_rows)
            write_jsonl(style_path, merged)

            rerun_only_path = report_dir / "style_rerun_only" / style_path.name
            rerun_only_path.parent.mkdir(parents=True, exist_ok=True)
            write_jsonl(rerun_only_path, list(rerun_rows.values()))
            for uid in rerun_rows:
                style_manifest.append({"source_file": source_file, "style_file": style_path.name, "scenario_uid": uid})

    with (report_dir / "rubric_repair_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["source_file", "judge_file", "scenario_uid"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rubric_manifest)

    with (report_dir / "style_repair_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["source_file", "style_file", "scenario_uid"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(style_manifest)

    updated_answers = sum(1 for r in answer_manifest if r["status"] == "updated")
    failed_answers = sum(1 for r in answer_manifest if r["status"] != "updated")
    print(f"[DONE] updated_answers={updated_answers} non_updated={failed_answers} rubric_updates={len(rubric_manifest)} style_updates={len(style_manifest)}")


if __name__ == "__main__":
    main()
