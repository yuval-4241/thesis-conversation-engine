"""
Day 3 batch runner — runs every persona against every session_1 question
through graph.run_conversation() (6 personas x 6 questions = 36 conversations)
and writes this batch's records + a summary to results/.

data/run_log.jsonl still gets one line per successful conversation as
before (graph.py's finalize() node is unchanged) — this is the cumulative
dataset across all sessions. results/ holds just this batch's own output,
so it's easy to inspect or hand to the advisor demo without wading through
the cumulative log.
"""

import json
import os
import time
from datetime import datetime

import config
from graph import run_conversation
from prompts.persona_prompts import PERSONAS
from question_bank import QuestionBank

SESSION_NUMBER = 1


def main():
    questions = QuestionBank().session(SESSION_NUMBER)
    persona_keys = list(PERSONAS.keys())
    total = len(persona_keys) * len(questions)

    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(config.RESULTS_DIR, f"run_demo_{batch_id}.jsonl")
    summary_path = os.path.join(config.RESULTS_DIR, f"run_demo_{batch_id}_summary.json")

    succeeded = 0
    errored = 0
    routed_to_counts = {}
    category_counts = {}

    with open(results_path, "w", encoding="utf-8") as results_file:
        i = 0
        for persona_key in persona_keys:
            for q in questions:
                i += 1
                question_text = q["text"]
                try:
                    final_state = run_conversation(persona_key, question_text)
                    record = {**final_state, "status": "ok"}
                    succeeded += 1
                    routed_to = final_state.get("routed_to", "unknown")
                    routed_to_counts[routed_to] = routed_to_counts.get(routed_to, 0) + 1
                    category = final_state.get("evaluation", {}).get("category", "unknown")
                    category_counts[category] = category_counts.get(category, 0) + 1
                    print(
                        f"[{i}/{total}] persona={persona_key} "
                        f"category={category} routed_to={routed_to}"
                    )
                except Exception as e:
                    record = {
                        "timestamp": time.time(),
                        "persona": persona_key,
                        "question": question_text,
                        "status": "error",
                        "error": str(e),
                    }
                    errored += 1
                    print(f"[{i}/{total}] persona={persona_key} ERROR: {e}")

                results_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "batch_id": batch_id,
        "session": SESSION_NUMBER,
        "total": total,
        "succeeded": succeeded,
        "errored": errored,
        "routed_to": routed_to_counts,
        "category_breakdown": category_counts,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Batch summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nRecords: {results_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
