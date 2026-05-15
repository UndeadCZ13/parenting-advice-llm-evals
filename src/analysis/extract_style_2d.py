from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


STYLE_KEYS = ["authoritative", "authoritarian", "permissive", "neglectful"]
DIM_KEYS = ["responsiveness", "demandingness"]


CANONICAL_MODEL_MAP = {
    "en_openai_gpt-5.2": "gpt-5.2",
    "zh_openai_gpt-5.2": "gpt-5.2",
    "en_openai_gpt-5-nano": "gpt-5-nano",
    "zh_openai_gpt-5-nano": "gpt-5-nano",
    "en_openai_gpt-4o-mini": "gpt-4o-mini",
    "zh_openai_gpt-4o-mini": "gpt-4o-mini",
    "en_ollama_gpt_oss": "gpt-oss:20b-cloud",
    "zh_ollama_gpt_oss": "gpt-oss:20b-cloud",
    "en_ollama_deepseek_v3": "deepseek-v3.1:671b-cloud",
    "zh_ollama_deepseek_v3": "deepseek-v3.1:671b-cloud",
    "en_ollama_deepseek_r1": "deepseek-r1:latest",
    "zh_ollama_deepseek_r1": "deepseek-r1:latest",
    "en_groq_qwen3_32b": "qwen/qwen3-32b",
    "zh_groq_qwen3_32b": "qwen/qwen3-32b",
    "en_ollama_qwen3_8b": "qwen3:8b",
    "zh_ollama_qwen3_8b": "qwen3:8b",
    "en_ollama_minimax_m2": "minimax-m2:cloud",
    "zh_ollama_minimax_m2": "minimax-m2:cloud",
    "en_ollama_ministral3_14b": "ministral-3:14b-cloud",
    "zh_ollama_ministral3_14b": "ministral-3:14b-cloud",
    "en_groq_llama33_70b": "llama-3.3-70b-versatile",
    "zh_groq_llama33_70b": "llama-3.3-70b-versatile",
    "en_groq_llama31_8b": "llama-3.1-8b-instant",
    "zh_groq_llama31_8b": "llama-3.1-8b-instant",
    "en_ollama_glm46": "glm-4.6:cloud",
    "zh_ollama_glm46": "glm-4.6:cloud",
    "en_ollama_glm4_9b": "glm4:9b",
    "zh_ollama_glm4_9b": "glm4:9b",
    "en_ollama_kimi_k2": "kimi-k2-thinking:cloud",
    "zh_ollama_kimi_k2": "kimi-k2-thinking:cloud",
}


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def normalize_lang(x: Any) -> str:
    s = str(x or "").lower().strip()
    if s in ("zh-cn", "zh-hans", "chinese", "cn"):
        return "zh"
    if s in ("en-us", "english"):
        return "en"
    if s in ("zh", "en"):
        return s
    return s or "en"


def canonical_model_name(x: Any) -> str:
    s = str(x or "").strip()
    return CANONICAL_MODEL_MAP.get(s, s)


def load_jsonl_files(files: List[Path]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                rec = json.loads(s)

                r = rec.get("responsiveness", None)
                d = rec.get("demandingness", None)

                rows.append({
                    "scenario_uid": rec.get("scenario_uid"),
                    "language": normalize_lang(rec.get("language", "en")),
                    "model_name": rec.get("model_name"),
                    "canonical_model": canonical_model_name(rec.get("model_name")),
                    "judge_model": rec.get("judge_model"),
                    "judge_backend": rec.get("judge_backend"),
                    "responsiveness": r,
                    "demandingness": d,
                    "error": rec.get("error"),
                    "source_file": fp.name,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # numeric + clip
    for c in DIM_KEYS:
        df[c] = pd.to_numeric(df[c], errors="coerce").clip(0, 1)

    return df


def compute_style_probs_product(r: float, d: float) -> Dict[str, float]:
    """
    Base mapping (your idea), sums to 1:
      authoritative = R*D
      permissive    = R*(1-D)
      authoritarian = (1-R)*D
      neglectful    = (1-R)*(1-D)
    """
    r = float(r); d = float(d)
    out = {
        "authoritative": r * d,
        "permissive": r * (1 - d),
        "authoritarian": (1 - r) * d,
        "neglectful": (1 - r) * (1 - d),
    }
    s = sum(out.values())
    if s > 0:
        out = {k: out[k] / s for k in out}
    return out


def sharpen_probs(probs: Dict[str, float], gamma: float) -> Dict[str, float]:
    """
    Improve separability: p' = p^gamma / sum(p^gamma)
    gamma=1 keeps original.
    """
    g = float(gamma)
    if g <= 0:
        g = 1.0
    xs = np.array([max(probs.get(k, 0.0), 0.0) for k in STYLE_KEYS], dtype=float)
    if xs.sum() <= 0:
        xs = np.ones_like(xs) / len(xs)
    xs = np.power(xs, g)
    xs = xs / xs.sum()
    return {k: float(xs[i]) for i, k in enumerate(STYLE_KEYS)}


def dominant_style(probs: Dict[str, float]) -> Tuple[str, float]:
    k = max(STYLE_KEYS, key=lambda x: probs.get(x, -1))
    return k, float(probs.get(k, 0.0))


def add_style_columns(
    df_valid: pd.DataFrame,
    gamma: float,
    conf_threshold: float,
    center_eps: float,
) -> pd.DataFrame:
    """
    Adds:
      - authoritative/authoritarian/permissive/neglectful probabilities (after optional sharpening)
      - dominant_style, confidence, uncertain
    """
    recs = []
    for _, row in df_valid.iterrows():
        r = float(row["responsiveness"])
        d = float(row["demandingness"])

        base = compute_style_probs_product(r, d)
        probs = sharpen_probs(base, gamma=gamma)

        dom, conf = dominant_style(probs)
        near_center = (abs(r - 0.5) < center_eps) and (abs(d - 0.5) < center_eps)
        uncertain = (conf < conf_threshold) or near_center

        out = {k: round(probs[k], 6) for k in STYLE_KEYS}
        out.update({
            "dominant_style": dom,
            "confidence": round(conf, 6),
            "uncertain": bool(uncertain),
        })
        recs.append(out)

    add = pd.DataFrame(recs)
    return pd.concat([df_valid.reset_index(drop=True), add], axis=1)


def compute_model_summary(df_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Model-level summary table.
    """
    group_key = "canonical_model" if "canonical_model" in df_rows.columns else "model_name"
    g = df_rows.groupby([group_key, "judge_model", "language"], dropna=False)

    mean_cols = ["responsiveness", "demandingness", "confidence"] + STYLE_KEYS
    mean_df = g[mean_cols].mean(numeric_only=True).reset_index()

    meta = g.agg(
        count=("scenario_uid", "count"),
        uncertain_rate=("uncertain", "mean"),
    ).reset_index()

    # dominant distribution
    dom = (
        df_rows.groupby([group_key, "judge_model", "language"])["dominant_style"]
        .value_counts(normalize=True)
        .rename("dominant_share")
        .reset_index()
    )
    piv = dom.pivot_table(
        index=[group_key, "judge_model", "language"],
        columns="dominant_style",
        values="dominant_share",
        fill_value=0.0
    ).reset_index()

    out = mean_df.merge(meta, on=[group_key, "judge_model", "language"], how="left")
    out = out.merge(piv, on=[group_key, "judge_model", "language"], how="left")

    # ensure all style columns exist
    for k in STYLE_KEYS:
        if k not in out.columns:
            out[k] = 0.0
        if f"{k}" not in out.columns:
            out[k] = out.get(k, 0.0)

    if group_key != "model_name":
        out = out.rename(columns={group_key: "model_name"})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="data/judge_outputs/style_2d", help="style_2d jsonl dir")
    ap.add_argument("--include-subdirs", action="store_true")
    ap.add_argument("--language", default="all", choices=["all", "en", "zh"])
    ap.add_argument("--judge-model", default=None, help="optional filter judge_model")
    ap.add_argument("--gamma", type=float, default=1, help="sharpening; 1.0=no sharpen")
    ap.add_argument("--conf-threshold", type=float, default=0.35)
    ap.add_argument("--center-eps", type=float, default=0.07)
    ap.add_argument("--out-csv", default="results/analysis/style_2d_tables/style_2d_rows.csv")
    ap.add_argument("--out-model-csv", default="results/analysis/style_2d_tables/style_2d_model_summary.csv")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    files = sorted(in_dir.rglob("*.jsonl") if args.include_subdirs else in_dir.glob("*.jsonl"))
    if not files:
        raise ValueError(f"No jsonl files under {in_dir}")

    df = load_jsonl_files(files)
    if df.empty:
        raise ValueError("No records loaded.")

    if args.judge_model:
        df = df[df["judge_model"].astype(str) == str(args.judge_model)].copy()

    if args.language != "all":
        df = df[df["language"] == args.language].copy()

    # keep only valid numeric rows
    df_valid = df.dropna(subset=DIM_KEYS).copy()

    # derive style columns
    df_rows = add_style_columns(
        df_valid,
        gamma=args.gamma,
        conf_threshold=args.conf_threshold,
        center_eps=args.center_eps,
    )

    out_csv = Path(args.out_csv)
    out_model_csv = Path(args.out_model_csv)
    safe_mkdir(out_csv.parent)
    safe_mkdir(out_model_csv.parent)

    df_rows.to_csv(out_csv, index=False, encoding="utf-8")
    model_sum = compute_model_summary(df_rows)
    model_sum.to_csv(out_model_csv, index=False, encoding="utf-8")

    print(f"[OK] rows csv -> → {out_csv}")
    print(f"[OK] model csv  → {out_model_csv}")
    print(f"[INFO] rows={len(df_rows)}  models={model_sum['model_name'].nunique()}")


if __name__ == "__main__":
    main()
