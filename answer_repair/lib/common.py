from __future__ import annotations

import csv
import json
import re
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "answer_repair" / "workspace"
SCENARIO_FILES = {
    "en": ROOT / "data" / "scenarios" / "views" / "en" / "parentbench_v0_en.jsonl",
    "zh": ROOT / "data" / "scenarios" / "views" / "zh" / "parentbench_v0_zh.jsonl",
}

TRUE_STRINGS = {"true", "1", "yes", "y", "retry", "rerun", "repair"}

try:
    from src.run_generation import build_final_only_prompt, strip_reasoning
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Could not import src.run_generation helpers") from exc


@dataclass
class ReviewDecision:
    source_file: str
    scenario_uid: str
    manual_action: str
    manual_notes: str


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
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


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_toml(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def normalize_cell(value: Any) -> str:
    return str(value or "").strip()


def iter_table_rows(path: Path) -> Iterator[Dict[str, Any]]:
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
        except Exception as exc:
            raise RuntimeError(
                "Reading .numbers files requires the 'numbers-parser' package."
            ) from exc

        doc = Document(str(path))
        for sheet in doc.sheets:
            for table in sheet.tables:
                rows = list(table.rows(values_only=True))
                if not rows:
                    continue
                header = [normalize_cell(x) for x in rows[0]]
                for raw_row in rows[1:]:
                    row: Dict[str, Any] = {}
                    for idx, col in enumerate(header):
                        row[col] = raw_row[idx] if idx < len(raw_row) else None
                    yield row
                return
        raise ValueError(f"No table rows found in {path}")

    raise ValueError(f"Unsupported table format: {path}")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def normalize_language(raw_lang: Any, file_name: str = "", rules: Optional[Dict[str, Any]] = None) -> str:
    s = str(raw_lang or "").strip().lower()
    if rules:
        for key, cfg in rules.get("languages", {}).items():
            aliases = [str(x).strip().lower() for x in cfg.get("aliases", [])]
            if s == key or s in aliases:
                return key
    if s.startswith("zh"):
        return "zh"
    if s.startswith("en"):
        return "en"
    fname = file_name.lower()
    if fname.startswith("zh_") or "_zh" in fname:
        return "zh"
    return "en"


def get_lang_rules(rules: Dict[str, Any], lang: str) -> Dict[str, Any]:
    lang_key = normalize_language(lang, rules=rules)
    langs = rules.get("languages", {})
    if lang_key in langs:
        return langs[lang_key]
    if "en" in langs:
        return langs["en"]
    raise KeyError(f"No language rules found for lang={lang}")


def pick_str(rec: Dict[str, Any], keys: Sequence[str], default: str = "") -> str:
    for key in keys:
        val = rec.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return default


def is_empty_text(value: Any) -> bool:
    if value is None:
        return True
    return not str(value).strip()


def strip_trailing_noise(text: str, trim_right_chars: str) -> str:
    s = (text or "").rstrip()
    if not s:
        return ""
    s = s.rstrip(trim_right_chars)
    # remove one trailing bracketed clause if it appears to be a formatting tail
    s = re.sub(r"\([^()]*\)$", "", s).rstrip(trim_right_chars)
    s = re.sub(r"（[^（）]*）$", "", s).rstrip(trim_right_chars)
    return s


def last_visible_char(text: str) -> str:
    s = text.rstrip()
    return s[-1] if s else ""


def ends_with_emoji(text: str) -> bool:
    s = text.rstrip()
    if not s:
        return False
    for ch in reversed(s):
        if ch.isspace():
            continue
        code = ord(ch)
        return (
            0x1F300 <= code <= 0x1FAFF
            or 0x2600 <= code <= 0x27BF
            or 0x1F1E6 <= code <= 0x1F1FF
        )
    return False


def get_last_fragment(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return ""
    parts = re.split(r"[。.!?！？\n\r]+", s)
    frag = parts[-1].strip() if parts else s
    return frag


def fragment_word_count(text: str) -> int:
    return len([w for w in re.split(r"\s+", text.strip()) if w])


def has_obvious_cut_marker(text: str, rules: Dict[str, Any]) -> bool:
    s = (text or "").rstrip()
    if not s:
        return False
    markers = [str(x) for x in rules.get("global", {}).get("obvious_cut_markers", [])]
    return any(s.endswith(marker) for marker in markers)


def dangling_list_or_heading(text: str) -> bool:
    s = (text or "").rstrip()
    if not s:
        return False
    lines = [ln.rstrip() for ln in s.splitlines() if ln.strip()]
    if not lines:
        return False
    last = lines[-1].strip()
    if re.fullmatch(r"(?:[-*•]|\d+[.)])", last):
        return True
    if re.fullmatch(r"(?:[-*•]|\d+[.)])\s+", last):
        return True
    if re.fullmatch(r".{0,80}:$", last):
        return True
    return False


def build_prompt_exact(scenario_text: str, lang: str) -> str:
    try:
        return build_final_only_prompt(scenario_text, lang)
    except TypeError:
        return build_final_only_prompt(scenario_text, lang, "")


def normalize_manual_reply(raw_answer: str, final_answer: Optional[str] = None) -> Tuple[str, str, bool, int]:
    raw = str(raw_answer or "")
    if final_answer is not None and str(final_answer).strip():
        final = str(final_answer).strip()
        return raw, final, False, 0
    cleaned = strip_reasoning(raw)
    return raw, cleaned.get("final", "").strip(), bool(cleaned.get("stripped", False)), int(cleaned.get("removed_chars", 0))


def load_scenarios() -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for lang, path in SCENARIO_FILES.items():
        if not path.exists():
            continue
        for row in iter_jsonl(path):
            uid = str(row.get("scenario_uid") or "").strip()
            if uid:
                out[(lang, uid)] = row
    return out


def review_selection_values(rules: Dict[str, Any]) -> set[str]:
    values = rules.get("global", {}).get("selection_true_values", [])
    if values:
        return {str(v).strip().lower() for v in values}
    return set(TRUE_STRINGS)


def is_selected_value(value: Any, true_values: set[str]) -> bool:
    if isinstance(value, (int, float)):
        try:
            return float(value) == 1.0
        except Exception:
            pass
    s = str(value or "").strip().lower()
    if s in true_values:
        return True
    try:
        return float(s) == 1.0
    except Exception:
        return False


def load_review_decisions(path: Path) -> List[ReviewDecision]:
    out: List[ReviewDecision] = []
    for row in iter_table_rows(path):
        out.append(
            ReviewDecision(
                source_file=str(row.get("source_file") or row.get("file") or row.get("answer_file") or "").strip(),
                scenario_uid=str(row.get("scenario_uid") or "").strip(),
                manual_action=str(row.get("manual_action") or "").strip(),
                manual_notes=str(row.get("manual_notes") or "").strip(),
            )
        )
    return out


def selected_review_rows(path: Path, rules: Dict[str, Any]) -> Dict[str, set[str]]:
    true_values = review_selection_values(rules)
    selected: Dict[str, set[str]] = {}
    for row in load_review_decisions(path):
        if is_selected_value(row.manual_action, true_values) and row.source_file and row.scenario_uid:
            selected.setdefault(row.source_file, set()).add(row.scenario_uid)
    return selected


def preview_text(text: Any, limit: int) -> str:
    s = str(text or "").replace("\r", " ").replace("\n", " ").strip()
    s = re.sub(r"\s+", " ", s)
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 3)] + "..."


def answer_tail(text: Any, limit: int) -> str:
    s = str(text or "").replace("\r", "").strip()
    if len(s) <= limit:
        return s
    return s[-limit:]


def analyze_text_for_truncation(rec: Dict[str, Any], source_file: str, rules: Dict[str, Any]) -> Dict[str, Any]:
    lang = normalize_language(rec.get("language") or rec.get("lang"), file_name=source_file, rules=rules)
    lang_rules = get_lang_rules(rules, lang)
    global_rules = rules.get("global", {})

    answer_field = str(global_rules.get("answer_field", "answer"))
    answer = str(rec.get(answer_field) or rec.get("answer") or "")
    finish_reason = str(rec.get("finish_reason") or "").strip().lower()
    api_error = str(rec.get("api_error") or "").strip()
    legacy_flag = bool(rec.get("suspected_truncation", False))

    total_chars = len(answer.strip())
    trimmed_tail = strip_trailing_noise(answer, str(global_rules.get("trim_right_chars", "")))
    last_frag = get_last_fragment(trimmed_tail)
    nonterminal_end = False
    if trimmed_tail:
        sentence_end_chars = str(lang_rules.get("sentence_end_chars", ""))
        last = last_visible_char(trimmed_tail)
        nonterminal_end = bool(
            last
            and last not in sentence_end_chars
            and not (bool(global_rules.get("allow_emoji_terminal", True)) and ends_with_emoji(trimmed_tail))
            and total_chars >= int(lang_rules.get("min_chars_for_terminal_check", 20))
        )

    suspicious_terminal = bool(trimmed_tail and last_visible_char(trimmed_tail) in str(lang_rules.get("suspicious_terminal_chars", "")))
    marker_suffix = has_obvious_cut_marker(trimmed_tail, rules)
    dangling_terminal = dangling_list_or_heading(answer)
    hard_short = total_chars > 0 and total_chars <= int(lang_rules.get("always_candidate_chars_le", 0))
    soft_short = total_chars > 0 and total_chars <= int(lang_rules.get("soft_review_chars_le", 0))

    short_fragment = False
    if lang == "zh":
        short_fragment = 0 < len(last_frag) <= int(lang_rules.get("max_last_fragment_chars", 0))
    else:
        short_fragment = 0 < fragment_word_count(last_frag) <= int(lang_rules.get("max_last_fragment_words", 0))

    hard_reasons: List[str] = []
    soft_reasons: List[str] = []
    if is_empty_text(answer):
        hard_reasons.append("empty_answer")
    if api_error and bool(global_rules.get("candidate_on_api_error", True)):
        hard_reasons.append("api_error_present")
    if finish_reason == "length" and bool(global_rules.get("candidate_on_finish_reason_length", True)):
        hard_reasons.append("finish_reason_length")
    if hard_short:
        hard_reasons.append(f"chars_le_{lang_rules.get('always_candidate_chars_le')}")

    if nonterminal_end:
        soft_reasons.append("nonterminal_end")
    if suspicious_terminal:
        soft_reasons.append("suspicious_terminal_char")
    if marker_suffix:
        soft_reasons.append("obvious_cut_marker")
    if dangling_terminal:
        soft_reasons.append("dangling_list_or_heading")
    if short_fragment:
        soft_reasons.append("short_last_fragment")
    if soft_short:
        soft_reasons.append(f"chars_le_{lang_rules.get('soft_review_chars_le')}")
    if legacy_flag and bool(global_rules.get("include_legacy_truncation_flag_as_signal", True)):
        soft_reasons.append("legacy_truncation_flag")

    candidate = False
    if hard_reasons:
        candidate = True
    elif marker_suffix or suspicious_terminal or dangling_terminal:
        candidate = True
    elif nonterminal_end and short_fragment:
        candidate = True
    elif nonterminal_end and soft_short:
        candidate = True
    elif legacy_flag and not bool(global_rules.get("legacy_flag_requires_extra_signal", True)):
        candidate = True
    elif legacy_flag and (nonterminal_end or short_fragment or soft_short):
        candidate = True

    return {
        "language": lang,
        "char_count": total_chars,
        "answer_preview": preview_text(answer, int(global_rules.get("preview_chars", 240))),
        "answer_tail": answer_tail(answer, int(global_rules.get("tail_chars", 220))),
        "last_fragment": last_frag,
        "finish_reason": finish_reason,
        "api_error": api_error,
        "legacy_truncation_flag": legacy_flag,
        "nonterminal_end": nonterminal_end,
        "suspicious_terminal": suspicious_terminal,
        "marker_suffix": marker_suffix,
        "dangling_terminal": dangling_terminal,
        "short_fragment": short_fragment,
        "hard_short": hard_short,
        "soft_short": soft_short,
        "candidate": candidate,
        "hard_reasons": "|".join(hard_reasons),
        "soft_reasons": "|".join(soft_reasons),
        "all_reasons": "|".join(hard_reasons + soft_reasons),
    }
