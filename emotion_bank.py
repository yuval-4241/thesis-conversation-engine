"""
Emotion Bank — coordinated facial-reaction lookup for the Furhat bridge.

Companion to response_bank.py: picks the robot's facial emotion+intensity
from the SAME (valence, arousal) pair response_bank.py already uses for
the spoken reply, so the two channels can never disagree. See
docs/superpowers/specs/2026-08-11-robot-reaction-bridge-design.md for the
full design and why an independent second judgment (e.g. a separate LLM
call picking the emotion on its own) was rejected.

Prototype status: not wired into graph.py. Feeds furhat-emotion-study's
RemoteControl bridge (a separate Kotlin repo) once/if this graduates out
of prototype scope.
"""

import json
import config


class EmotionBank:
    def __init__(self, path: str = config.EMOTION_BANK_PATH):
        with open(path, "r", encoding="utf-8") as f:
            self._data = json.load(f)
        self._data.pop("_note", None)

    def get(self, valence: str, arousal: str) -> dict:
        key = config.va_cell_key(valence, arousal)
        reaction = self._data.get(key)
        if not reaction:
            raise KeyError(f"No emotion reaction found for VA cell '{key}'")
        return reaction
