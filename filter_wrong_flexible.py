#!/usr/bin/env python3
"""
filter_wrong_flexible.py

Read a JSON-Lines evaluation file produced by lm-evaluation-harness and
**print to STDOUT** every line that

  1. has `"filter": "flexible-extract"` and
  2. was scored as incorrect (exact_match == 0.0, or is_correct == False).

Usage
-----
    python filter_wrong_flexible.py path/to/samples_file.jsonl \
        > wrong_flexible.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Select wrong answers with flexible-extract filter."
    )
    p.add_argument(
        "jsonl",
        type=Path,
        help="Path to the *.jsonl file written by the harness (samples_*.jsonl)",
    )
    return p.parse_args()


def is_wrong(obj: dict) -> bool:
    # print(obj.keys())
    """True if the record is marked wrong by any known field."""
    # Newer harness versions use `exact_match`; very old ones use `is_correct`.
    if "exact_match" in obj:
        return float(obj["exact_match"]) == 0.0
    if "is_correct" in obj:        # fallback for older runs
        return obj["is_correct"] is False
    if "pass_at_1" in obj:
        # print(f"Warning: 'pass_at_1' field found")
        return float(obj["pass_at_1"]) == 0.0
    # If neither field exists, assume unknown → not counted.
    return False


def main() -> None:
    args = parse_args()

    try:
        with args.jsonl.open("r", encoding="utf-8") as fh:
            for line in fh:
                obj = json.loads(line)
                if (
                    obj.get("filter") == "flexible-extract"
                    and is_wrong(obj)
                    # is_wrong(obj)
                ):
                    # Emit the original line verbatim (keeps hashing fields intact)
                    sys.stdout.write(line)
    except FileNotFoundError:
        sys.stderr.write(f"File not found: {args.jsonl}\n")
        sys.exit(1)
    except json.JSONDecodeError as e:
        sys.stderr.write(f"JSON parse error on line {e.lineno}: {e.msg}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
