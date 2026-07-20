"""
#6 — Response Bank.

Loads the VA-matched robot response database and retrieves a response for
a given (valence, arousal) cell. Currently backed by a placeholder seed
file — swap data/response_bank_seed.json for your real 90-entry Hebrew
database (10 responses per VA cell) without changing this module's API.
"""

import json
import random
import config


class ResponseBank:
    def __init__(self, path: str = config.RESPONSE_BANK_PATH):
        with open(path, "r", encoding="utf-8") as f:
            self._data = json.load(f)
        self._data.pop("_note", None)

    def get(self, valence: str, arousal: str) -> str:
        key = config.va_cell_key(valence, arousal)
        options = self._data.get(key)
        if not options:
            raise KeyError(f"No responses found for VA cell '{key}'")
        return random.choice(options)

    def coverage_report(self) -> dict:
        """Sanity check: how many entries exist per cell (flag gaps vs. target of 10)."""
        return {cell: len(v) for cell, v in self._data.items()}
