# src/analysis/scoring_utils.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional
import statistics

from src.judges.judge_prompts import RUBRIC_KEYS


@dataclass
class ScoreAggregate:
    avg_scores: Dict[str, Optional[float]]
    std_scores: Dict[str, Optional[float]]
    comment: Optional[str]


def aggregate_runs(raw_runs: List[Dict[str, Any]], rubric_keys: Iterable[str] = RUBRIC_KEYS) -> ScoreAggregate:
    avg: Dict[str, Optional[float]] = {}
    std: Dict[str, Optional[float]] = {}

    for k in rubric_keys:
        vals: List[float] = []
        for r in raw_runs:
            v = r.get(k)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        if not vals:
            avg[k] = None
            std[k] = None
        else:
            avg[k] = sum(vals) / len(vals)
            std[k] = statistics.pstdev(vals) if len(vals) > 1 else 0.0

    comment: Optional[str] = None
    for r in raw_runs:
        c = r.get("comment")
        if isinstance(c, str) and c.strip():
            comment = c.strip()
            break

    return ScoreAggregate(avg_scores=avg, std_scores=std, comment=comment)
