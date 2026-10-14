#!/usr/bin/env python3
"""Convert official benchmark annotations to JSONL without changing their fields."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        import pyarrow.parquet as pq
    except ImportError:
        parser.error("Install the conversion dependency: python -m pip install pyarrow")

    rows = pq.read_table(args.input).to_pylist()
    if not rows:
        parser.error("The input contains no annotations")
    required = {"question", "question_type", "ground_truth", "scene_id"}
    if not required.issubset(rows[0]):
        parser.error(f"Missing annotation fields: {sorted(required - rows[0].keys())}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"Wrote {len(rows)} annotations to {args.output}")


if __name__ == "__main__":
    main()
