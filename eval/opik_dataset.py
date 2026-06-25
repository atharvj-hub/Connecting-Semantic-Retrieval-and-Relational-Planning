"""
opik_dataset.py  —  M2: promote the local gold.jsonl test set into Opik.

v1 kept the gold set as a file you remembered to run `run_eval.py` against. In
v2 it becomes a first-class *dataset* living inside the Opik platform, so the
test questions, their expected behaviour, and (later) experiment runs against
them all sit in one place a non-engineer can browse.

This is idempotent: Opik dedups dataset items by content, so re-running after you
add questions to gold.jsonl just inserts the new ones.

Run:
    python eval/opik_dataset.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import config                      # noqa: F401  (loads .env -> OPIK_* vars)

import opik

GOLD = Path(__file__).resolve().parent / "gold.jsonl"
DATASET_NAME = "text2sql-gold"


def main():
    items = [json.loads(l) for l in GOLD.read_text(encoding="utf-8").splitlines()
             if l.strip()]
    client = opik.Opik()
    ds = client.get_or_create_dataset(
        name=DATASET_NAME,
        description="Text-to-SQL gold question set (capability + robustness tiers).")
    # Each dataset item: the question is the input; everything else is the
    # expected/reference metadata the grader compares against.
    # NB: 'id' is reserved by Opik (must be a UUID), so the gold id is 'gold_id'.
    ds.insert([{
        "gold_id": g["id"],
        "question": g["question"],
        "category": g["category"],
        "expected": g["expected"],
        "tables": g.get("tables", []),
        "reference_sql": g.get("reference_sql"),
        "tags": g.get("tags", []),
    } for g in items])
    print(f"Upserted {len(items)} items into Opik dataset '{DATASET_NAME}'.")
    print(f"View at: http://localhost:5173  (project/dataset: {DATASET_NAME})")


if __name__ == "__main__":
    main()
