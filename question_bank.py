"""
Question Bank — loads the 30-question set (6 per session x 5 sessions),
organized by category and difficulty, matching the pipeline spec.
Same pattern as response_bank.py: swap the seed file for your reviewed
question set without touching this module.
"""

import json
import os
import config

QUESTION_BANK_PATH = os.path.join(config.DATA_DIR, "question_bank_seed.json")


class QuestionBank:
    def __init__(self, path: str = QUESTION_BANK_PATH):
        with open(path, "r", encoding="utf-8") as f:
            self._data = json.load(f)
        self._data.pop("_note", None)

    def session(self, session_number: int) -> list:
        key = f"session_{session_number}"
        if key not in self._data:
            raise KeyError(f"No questions found for {key}")
        return self._data[key]

    def all_questions(self) -> list:
        """Flat list of every question across all sessions, each tagged with its session."""
        flat = []
        for session_key, items in self._data.items():
            session_num = int(session_key.split("_")[1])
            for item in items:
                flat.append({**item, "session": session_num})
        return flat

    def by_category(self, category: str) -> list:
        return [q for q in self.all_questions() if q["category"] == category]
