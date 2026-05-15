# src/analysis/build_networks.py
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd


DEFAULT_RUBRICS: List[str] = [
    "accuracy",
    "safety",
    "helpfulness",
    "empathy",
    "completeness",
    "bias_avoidance",
    "limitation_awareness",
    "communication",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Model & Scenario Similarity Networks for Gephi (language-aware)."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Merged score CSV path (e.g., results/merged/all_judge_scores.csv)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/analysis",
        help="Output base dir (default results/analysis). Will write to <out_dir>/<language>/networks/",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="all",
        choices=["all", "en", "zh"],
        help="Filter by language column if present.",
    )
    parser.add_argument(
        "--k_neighbors_model",
        type=int,
        default=5,
        help="Top-K neighbors per node in Model network (default 5).",
    )
    parser.add_argument(
        "--k_neighbors_scenario",
        type=int,
        default=10,
        help="Top-K neighbors per node in Scenario network (default 10).",
    )
    parser.add_argument(
        "--min_corr",
        type=float,
        default=None,
        help="Min correlation threshold for Model Network edges (positive only).",
    )
    parser.add_argument(
        "--min_sim",
        type=float,
        default=None,
        help="Min similarity threshold for Scenario Network edges (positive only).",
    )
    parser.add_argument(
        "--no_overall_in_scenario",
        action="store_true",
        help="Build Scenario Network without overall_mean as feature.",
    )
    parser.add_argument(
        "--judge",
        type=str,
        default=None,
        help="Optional: only use rows from a specific judge_model.",
    )
    return parser.parse_args()


def detect_scenario_col(df: pd.DataFrame) -> str:
    if "scenario_uid" in df.columns:
        return "scenario_uid"
    if "scenario_id" in df.columns:
        return "scenario_id"
    raise ValueError("No scenario_uid/scenario_id column found.")


def detect_model_col(df: pd.DataFrame) -> str:
    if "answer_model" in df.columns:
        return "answer_model"
    if "model" in df.columns:
        return "model"
    raise ValueError("No answer_model/model column found.")


def ensure_overall_mean(df: pd.DataFrame, rubrics: List[str]) -> pd.DataFrame:
    if "overall_mean" not in df.columns:
        avail = [r for r in rubrics if r in df.columns]
        if not avail:
            raise ValueError("No rubric columns found to compute overall_mean.")
        df["overall_mean"] = df[avail].mean(axis=1)
    return df


def filter_language(df: pd.DataFrame, language: str) -> pd.DataFrame:
    lang = (language or "all").lower().strip()
    if lang == "all":
        return df
    if "language" not in df.columns:
        print("[WARN] No 'language' column; skip language filtering.")
        return df
    return df[df["language"].astype(str).str.lower() == lang].copy()


def to_corr_edges(mat: pd.DataFrame, k_neighbors: int, min_corr: Optional[float]) -> pd.DataFrame:
    """
    mat: index=node, columns=features
    """
    X = mat.values.astype(float)
    # correlation across rows
    C = np.corrcoef(X)
    nodes = mat.index.astype(str).tolist()

    edges = []
    for i, a in enumerate(nodes):
        sims = []
        for j, b in enumerate(nodes):
            if i == j:
                continue
            v = float(C[i, j])
            if np.isnan(v):
                continue
            if v <= 0:
                continue
            sims.append((b, v))
        sims.sort(key=lambda x: x[1], reverse=True)
        kept = sims[: max(0, int(k_neighbors))]
        for b, v in kept:
            if min_corr is not None and v < min_corr:
                continue
            edges.append((a, b, v))

    return pd.DataFrame(edges, columns=["source", "target", "weight"])


def to_cosine_edges(mat: pd.DataFrame, k_neighbors: int, min_sim: Optional[float]) -> pd.DataFrame:
    X = mat.values.astype(float)
    # cosine similarity
    norm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    Xn = X / norm
    S = Xn @ Xn.T
    nodes = mat.index.astype(str).tolist()

    edges = []
    for i, a in enumerate(nodes):
        sims = []
        for j, b in enumerate(nodes):
            if i == j:
                continue
            v = float(S[i, j])
            if np.isnan(v):
                continue
            if v <= 0:
                continue
            sims.append((b, v))
        sims.sort(key=lambda x: x[1], reverse=True)
        kept = sims[: max(0, int(k_neighbors))]
        for b, v in kept:
            if min_sim is not None and v < min_sim:
                continue
            edges.append((a, b, v))

    return pd.DataFrame(edges, columns=["source", "target", "weight"])


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    df = pd.read_csv(in_path)

    # optional judge filter
    if args.judge and "judge_model" in df.columns:
        df = df[df["judge_model"].astype(str) == args.judge].copy()

    df = filter_language(df, args.language)

    scenario_col = detect_scenario_col(df)
    model_col = detect_model_col(df)

    rubrics = [r for r in DEFAULT_RUBRICS if r in df.columns]
    df = ensure_overall_mean(df, rubrics)

    out_root = Path(args.out_dir) / args.language / "networks"
    os.makedirs(out_root, exist_ok=True)

    # -------------------------
    # Model network: model x scenario features
    # -------------------------
    # Use pivot: rows=models, cols=scenarios, values=overall_mean
    model_mat = df.pivot_table(index=model_col, columns=scenario_col, values="overall_mean", aggfunc="mean")
    model_mat = model_mat.fillna(model_mat.mean(axis=0))

    model_nodes = pd.DataFrame({ "id": model_mat.index.astype(str).tolist() })
    model_edges = to_corr_edges(model_mat, k_neighbors=args.k_neighbors_model, min_corr=args.min_corr)

    model_nodes.to_csv(out_root / "model_nodes.csv", index=False, encoding="utf-8")
    model_edges.to_csv(out_root / "model_edges.csv", index=False, encoding="utf-8")

    # -------------------------
    # Scenario network: scenario x model features (optionally include overall_mean)
    # -------------------------
    scen_mat = df.pivot_table(index=scenario_col, columns=model_col, values="overall_mean", aggfunc="mean")
    scen_mat = scen_mat.fillna(scen_mat.mean(axis=0))

    if not args.no_overall_in_scenario:
        scen_mat["overall_mean_feat"] = scen_mat.mean(axis=1)

    scen_nodes = pd.DataFrame({ "id": scen_mat.index.astype(str).tolist() })
    scen_edges = to_cosine_edges(scen_mat, k_neighbors=args.k_neighbors_scenario, min_sim=args.min_sim)

    scen_nodes.to_csv(out_root / "scenario_nodes.csv", index=False, encoding="utf-8")
    scen_edges.to_csv(out_root / "scenario_edges.csv", index=False, encoding="utf-8")

    print(f"[DONE] networks -> {out_root.resolve()}")


if __name__ == "__main__":
    main()
