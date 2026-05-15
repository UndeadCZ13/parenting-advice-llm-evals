#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-scenario export for human annotations.

For each scenario_uid in SCENARIO_UIDS:
- Create folder: results/human_annotation_pack/<scenario_uid>/
- Write:
  1) answers.md  : scenario text (Question/Description/Tags) + all models' EN answers
  2) judge_gpt52.csv : gpt-5.2 judge scores+comment for all answer_model on this scenario
     including:
       - overall_mean (8 rubrics mean)
       - 8 rubrics scores
       - cognitive_constructive_competence (accuracy+completeness+communication+helpfulness mean)
       - alignment_social_responsibility (empathy+bias_avoidance+limitation_awareness+safety mean)
       - comment
"""

from __future__ import annotations

import os
import re
import json
import glob
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# =========================
# 0) EDIT: SCENARIO LIST
# =========================
SCENARIO_UIDS: List[str] = [
     "pb_v0_0068",
     "pb_v0_0026",
     "pb_v0_0045",
     "pb_v0_0020",
     "pb_v0_0019",
     "pb_v0_0041",
     "pb_v0_0019",
     "pb_v0_0013",

]


# =========================
# 1) PATH SETUP (matches your project)
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # .../parentbench-llm-evals

MODEL_OUTPUT_DIR = PROJECT_ROOT / "data" / "model_outputs"
SCENARIO_XLSX_PATH = PROJECT_ROOT / "data" / "scenarios" / "v0_ParentBench_Scenarios.xlsx"
MERGED_JUDGE_CSV_PATH = PROJECT_ROOT / "results" / "merged" / "all_judge_scores.csv"

OUTPUT_ROOT = PROJECT_ROOT / "results" / "human_annotation_pack"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# =========================
# 2) CONSTANTS
# =========================
RUBRICS = [
    "accuracy",
    "completeness",
    "communication",
    "helpfulness",
    "empathy",
    "bias_avoidance",
    "limitation_awareness",
    "safety",
]
CONSTRUCTIVE_RUBRICS = ["accuracy", "completeness", "communication", "helpfulness"]
RESPONSIBLE_RUBRICS = ["empathy", "bias_avoidance", "limitation_awareness", "safety"]

# Common schema guesses for JSONL
SCENARIO_KEY_CANDIDATES = ["scenario_uid", "uid", "scenario_id", "scenario", "id"]
ANSWER_KEY_CANDIDATES = [
    "answer",
    "final",
    "final_answer",
    "response",
    "output",
    "output_text",
    "completion",
    "text",
    "content",
    "model_answer",
]

# Nested key guesses
NESTED_OBJ_KEYS = ["result", "generation", "output", "response_obj", "data", "sample", "item"]


# =========================
# 3) HELPERS
# =========================
def die(msg: str) -> None:
    raise SystemExit(f"[ERROR] {msg}")


def pb_uid_to_row_index(uid: str) -> int:
    """
    pb_v0_0068 -> 68
    """
    m = re.match(r"^pb_v0_(\d{4})$", uid.strip())
    if not m:
        die(f"Invalid scenario uid format: {uid} (expected like pb_v0_0068)")
    return int(m.group(1))


def extract_field(obj: Dict[str, Any], candidates: List[str]) -> Optional[Any]:
    for k in candidates:
        if k in obj and obj[k] not in (None, ""):
            return obj[k]
    return None


def normalize_answer_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, dict):
        # common chat format: {"role": "...", "content": "..."}
        if "content" in x and isinstance(x["content"], str):
            return x["content"].strip()
        return json.dumps(x, ensure_ascii=False, indent=2)
    if isinstance(x, list):
        # common chat list format
        parts = []
        for it in x:
            if isinstance(it, dict) and "content" in it:
                parts.append(str(it["content"]))
            else:
                parts.append(str(it))
        return "\n".join(parts).strip()
    return str(x).strip()


def infer_answer_model_from_path(path: Path) -> str:
    """
    Try to infer model name from the JSONL path.
    Preferred: parent folder name or filename stem.
    """
    stem = path.stem  # e.g. en_openai_gpt-5.2
    # Heuristic: strip leading "en_"
    if stem.startswith("en_"):
        stem = stem[3:]
    # If you store by model folder, prefer that
    parent = path.parent.name
    # Pick whichever looks more "model-like"
    # (many projects use folders like openai_gpt-5.2 or gpt-5.2)
    if parent and parent not in ("model_outputs", "outputs", "results", "data"):
        # but avoid generic folders
        if any(ch.isdigit() for ch in parent) or "-" in parent or ":" in parent:
            return parent
    return stem


# =========================
# 4) LOAD SCENARIOS FROM XLSX
# =========================
def load_scenarios(xlsx_path: Path) -> pd.DataFrame:
    if not xlsx_path.exists():
        die(f"Scenario xlsx not found: {xlsx_path}")
    df = pd.read_excel(xlsx_path)

    needed = {"Question", "Description", "Tags"}
    if not needed.issubset(set(df.columns)):
        die(f"Scenario xlsx missing expected columns {needed}. Found: {list(df.columns)}")
    return df


def get_scenario_text_block(scen_df: pd.DataFrame, scenario_uid: str) -> str:
    idx = pb_uid_to_row_index(scenario_uid)
    if idx < 0 or idx >= len(scen_df):
        die(f"Scenario uid {scenario_uid} -> row {idx} out of range (0..{len(scen_df)-1})")

    row = scen_df.iloc[idx]
    q = str(row["Question"]).strip()
    d = str(row["Description"]).strip()
    t = str(row["Tags"]).strip()

    return (
        f"**Scenario UID:** `{scenario_uid}`\n\n"
        f"**Question:**\n{q}\n\n"
        f"**Description / Context:**\n{d}\n\n"
        f"**Tags:**\n{t}\n"
    )


# =========================
# 5) LOAD MODEL ANSWERS (EN JSONL)
# =========================
def find_en_answer_jsonls(root_dir: Path) -> List[Path]:
    pattern = str(root_dir / "**" / "en_*.jsonl")
    return sorted(Path(p) for p in glob.glob(pattern, recursive=True))


def parse_answers_from_jsonl(jsonl_path: Path) -> Dict[str, str]:
    """
    Returns: scenario_uid -> answer_text
    """
    answers: Dict[str, str] = {}

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            uid = extract_field(obj, SCENARIO_KEY_CANDIDATES)

            # sometimes nested scenario object
            if not uid and "scenario" in obj and isinstance(obj["scenario"], dict):
                uid = extract_field(obj["scenario"], SCENARIO_KEY_CANDIDATES)

            # sometimes scenario_uid is nested deeper
            if not uid:
                for nk in NESTED_OBJ_KEYS:
                    if nk in obj and isinstance(obj[nk], dict):
                        uid = extract_field(obj[nk], SCENARIO_KEY_CANDIDATES)
                        if uid:
                            break

            if not uid or not isinstance(uid, str):
                continue

            ans = extract_field(obj, ANSWER_KEY_CANDIDATES)

            if ans is None:
                # nested answer
                for nk in NESTED_OBJ_KEYS:
                    if nk in obj and isinstance(obj[nk], dict):
                        ans = extract_field(obj[nk], ANSWER_KEY_CANDIDATES)
                        if ans is not None:
                            break

            answers[uid] = normalize_answer_text(ans)

    return answers


def load_all_model_answers(model_output_dir: Path) -> Dict[str, Dict[str, str]]:
    """
    answers_by_model[model_name][scenario_uid] = answer_text
    """
    if not model_output_dir.exists():
        die(f"MODEL_OUTPUT_DIR not found: {model_output_dir}")

    files = find_en_answer_jsonls(model_output_dir)
    if not files:
        die(f"No English answer JSONLs found under {model_output_dir} (pattern **/en_*.jsonl).")

    answers_by_model: Dict[str, Dict[str, str]] = {}
    for fp in files:
        model_name = infer_answer_model_from_path(fp)
        model_answers = parse_answers_from_jsonl(fp)
        if model_answers:
            # If a model appears multiple times (e.g. multiple runs), later files may override.
            # You can change to merge if needed.
            if model_name not in answers_by_model:
                answers_by_model[model_name] = {}
            answers_by_model[model_name].update(model_answers)

    if not answers_by_model:
        die("Failed to parse any answers from en_*.jsonl files. Check JSONL schema.")

    return answers_by_model


# =========================
# 6) LOAD GPT-5.2 JUDGE SCORES + DERIVED METRICS
# =========================
def load_gpt52_judge_scores(merged_csv: Path) -> pd.DataFrame:
    if not merged_csv.exists():
        die(f"Merged judge csv not found: {merged_csv}")

    df = pd.read_csv(merged_csv)

    required_cols = {"scenario_uid", "language", "answer_model", "judge_model", "comment"}
    if not required_cols.issubset(set(df.columns)):
        die(f"Merged judge csv missing required columns {required_cols}. Found: {list(df.columns)}")

    # keep only EN + gpt-5.2 judge
    df = df[(df["language"] == "en") & (df["judge_model"] == "gpt-5.2")].copy()

    # ensure rubric columns exist
    for r in RUBRICS:
        if r not in df.columns:
            die(f"Merged judge csv missing rubric column: {r}")

    # derived metrics
    df["overall_mean"] = df[RUBRICS].mean(axis=1)
    df["cognitive_constructive_competence"] = df[CONSTRUCTIVE_RUBRICS].mean(axis=1)
    df["alignment_social_responsibility"] = df[RESPONSIBLE_RUBRICS].mean(axis=1)

    return df


# =========================
# 7) EXPORT PER SCENARIO
# =========================
def write_answers_md(
    scen_df: pd.DataFrame,
    scenario_uid: str,
    answers_by_model: Dict[str, Dict[str, str]],
    out_file: Path,
) -> None:
    model_names = sorted(answers_by_model.keys())

    lines: List[str] = []
    lines.append(f"# Scenario {scenario_uid} — Text & Model Answers (EN)\n")
    lines.append(get_scenario_text_block(scen_df, scenario_uid))
    lines.append("\n---\n")
    lines.append("## Model Answers\n")

    for model in model_names:
        ans = answers_by_model.get(model, {}).get(scenario_uid, "")
        if not ans:
            ans = "*[MISSING ANSWER]*"
        lines.append(f"\n### {model}\n")
        lines.append(ans)
        lines.append("\n")

    out_file.write_text("\n".join(lines), encoding="utf-8")


def write_judge_csv(
    judge_df: pd.DataFrame,
    scenario_uid: str,
    out_file: Path,
) -> None:
    sub = judge_df[judge_df["scenario_uid"] == scenario_uid].copy()
    if sub.empty:
        # still write an empty file with headers
        cols = [
            "scenario_uid",
            "answer_model",
            "overall_mean",
            *RUBRICS,
            "cognitive_constructive_competence",
            "alignment_social_responsibility",
            "comment",
        ]
        pd.DataFrame(columns=cols).to_csv(out_file, index=False, encoding="utf-8-sig")
        return

    # Requested columns (+ keep backend/meta if present)
    cols = [
        "scenario_uid",
        "answer_model",
        "answer_backend",
        "judge_model",
        "judge_backend",
        "n_repeats",
        "overall_mean",
        *RUBRICS,
        "cognitive_constructive_competence",
        "alignment_social_responsibility",
        "comment",
    ]
    cols = [c for c in cols if c in sub.columns]

    sub = sub[cols].sort_values(["answer_model"], ascending=[True])
    sub.to_csv(out_file, index=False, encoding="utf-8-sig")


def main() -> None:
    if not SCENARIO_UIDS:
        die("SCENARIO_UIDS is empty. Put scenario ids like ['pb_v0_0068', ...] in the list at the top.")

    # Load inputs once
    scen_df = load_scenarios(SCENARIO_XLSX_PATH)
    answers_by_model = load_all_model_answers(MODEL_OUTPUT_DIR)
    judge_df = load_gpt52_judge_scores(MERGED_JUDGE_CSV_PATH)

    # Export each scenario
    for uid in SCENARIO_UIDS:
        scenario_dir = OUTPUT_ROOT / uid
        scenario_dir.mkdir(parents=True, exist_ok=True)

        answers_md = scenario_dir / "answers.md"
        judge_csv = scenario_dir / "judge_gpt52.csv"

        write_answers_md(scen_df, uid, answers_by_model, answers_md)
        write_judge_csv(judge_df, uid, judge_csv)

        print(f"[OK] {uid}")
        print(f"  - {answers_md}")
        print(f"  - {judge_csv}")

    # Optional: quick missing answers report
    models = sorted(answers_by_model.keys())
    missing_pairs = []
    for uid in SCENARIO_UIDS:
        for m in models:
            if not answers_by_model.get(m, {}).get(uid, ""):
                missing_pairs.append((uid, m))
    if missing_pairs:
        print(f"\n[WARN] Missing answers for {len(missing_pairs)} (scenario, model) pairs. First 15:")
        for uid, m in missing_pairs[:15]:
            print("  ", uid, m)

    print(f"\n[DONE] Output root: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
