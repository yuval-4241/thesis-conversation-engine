"""
#7 — Human-in-the-Loop Review Queue.

Any item that is low-confidence (below CONFIDENCE_ROUTE_THRESHOLD) or
flagged by the No-Masking guardrail gets written here instead of being
auto-finalized. This is also where your Cohen's/Fleiss' kappa dataset
comes from over time: once a human labels these, compare human_label vs.
evaluator's original category.

Backed by a flat JSONL file for now (#7 as a standalone project would
add Slack/Notion routing + escalation on top of this same schema).
"""

import json
import os
import time
import config


def enqueue(item: dict, reason: str):
    os.makedirs(config.DATA_DIR, exist_ok=True)
    record = {
        "queued_at": time.time(),
        "reason": reason,           # "low_confidence" | "guardrail_flag" | both
        "human_label": None,        # filled in later by a reviewer
        **item,
    }
    with open(config.REVIEW_QUEUE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_pending() -> list:
    if not os.path.exists(config.REVIEW_QUEUE_PATH):
        return []
    with open(config.REVIEW_QUEUE_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
