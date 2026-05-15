# scripts/build_scenario_views.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--en", default="data/scenarios/views/en/parentbench_v0.jsonl")
    ap.add_argument("--zh", default="data/scenarios/views/zh/parentbench_v0.jsonl")
    ap.add_argument("--out", default="data/scenarios/views/en_zh/parentbench_v0.jsonl")
    args = ap.parse_args()

    en_rows = read_jsonl(Path(args.en))
    zh_rows = read_jsonl(Path(args.zh))

    # keep both, rely on (scenario_uid, language) for uniqueness
    out_rows = en_rows + zh_rows
    write_jsonl(Path(args.out), out_rows)

    print(f"✅ built en_zh view: {len(en_rows)} + {len(zh_rows)} = {len(out_rows)} -> {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
