"""
Emotion Classifier — direct Ekman-emotion classification for the Furhat
face+voice reaction, independent of evaluator.py.

One LLM call, sees only (question, answer_text) -- not the evaluator's
rationale or category -- so this stays a genuinely separate read on the
answer's emotional content rather than a restatement of the evaluator's
own judgment. Picks freely among all 7 of Ekman's core emotions, no
register restriction (unlike the earlier VA-keyed emotion_bank.py, which
excluded ANGER/CONTEMPT/DISGUST as "too hostile" -- that restriction was
a property of the old mechanism, deliberately dropped here). See
docs/superpowers/specs/2026-08-16-emotion-first-classification-design.md.

response_bank.py still needs a (valence, arousal) pair to pick spoken
text, so this module also derives one from the classification result --
valence from a fixed emotion lookup, arousal from intensity -- rather
than reusing evaluator.py's own valence/arousal. Face (the classified
emotion) and text (the derived-VA response) both trace back to this one
classify_emotion() call, preserving the "one source of truth" coherence
guarantee from the original VA-first design.
"""

import json
import re
import config
import llm_client

CORE_EMOTIONS = ["ANGER", "CONTEMPT", "DISGUST", "FEAR", "HAPPINESS", "SADNESS", "SURPRISE"]

# Grounded in Schneider et al. (2025)'s circumplex quadrants: HAPPINESS in
# the positive/high-arousal quadrant, the rest in the negative quadrants,
# SURPRISE conventionally placed near-neutral valence at high arousal.
EMOTION_TO_VALENCE = {
    "HAPPINESS": "high",
    "SURPRISE": "medium",
    "ANGER": "low",
    "CONTEMPT": "low",
    "DISGUST": "low",
    "FEAR": "low",
    "SADNESS": "low",
}

# Arousal is derived from intensity, NOT fixed per emotion -- fixing both
# per emotion would make most of response_bank.py's 9 VA cells permanently
# unreachable (e.g. HAPPINESS always landing on high/high). This keeps all
# 9 cells reachable via at least one emotion at some intensity.
INTENSITY_TO_AROUSAL = {1: "low", 2: "medium", 3: "high"}

# Fail-closed default when the LLM returns something unparseable/invalid --
# same "fail closed" philosophy as emotion_bank.pick_emotion_llm()'s
# candidates[0] fallback. SURPRISE/2 is the mid-valence, mid-intensity
# cell: a parsing failure implies neither a positive nor a negative read.
_FALLBACK = {"emotion": "SURPRISE", "intensity": 2}

_SYSTEM_PROMPT = (
    "You are classifying the emotional content of a job interview answer, "
    "independent of how good or bad the answer is.\n\n"
    "First, find the specific word or phrase in the answer that most "
    "signals an emotion -- not the overall tone of a confident or "
    "professional-sounding answer, just what's actually there. Generic "
    "confident or professional phrasing on its own is not a signal of "
    "happiness -- do not default to HAPPINESS just because an answer "
    "sounds competent or matter-of-fact.\n\n"
    "Then pick exactly one of Ekman's core emotions that best fits: "
    "ANGER, CONTEMPT, DISGUST, FEAR, HAPPINESS, SADNESS, SURPRISE. "
    "Also rate its intensity from 1 (mild) to 3 (strong).\n\n"
    "Respond with ONLY a JSON object of the exact shape: "
    '{"cue": "<the word or phrase, or \\"none\\" if truly nothing '
    'stands out>", "emotion": "<ONE_OF_THE_7_WORDS_ABOVE>", '
    '"intensity": <1, 2, or 3>}'
)


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def classify_emotion(question: str, answer_text: str) -> dict:
    user_message = f"Interview question: {question}\n\nCandidate answer: {answer_text}"

    raw = llm_client.call_llm(
        system=_SYSTEM_PROMPT,
        user_message=user_message,
        max_tokens=80,
        temperature=0.8,  # variety over reproducibility -- see spec
    )
    raw = _strip_fences(raw)

    try:
        data = json.loads(raw)
        emotion = data["emotion"]
        intensity = int(data["intensity"])
        if emotion not in EMOTION_TO_VALENCE or intensity not in INTENSITY_TO_AROUSAL:
            raise ValueError(f"Unexpected emotion/intensity: {data!r}")
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return dict(_FALLBACK)

    return {"emotion": emotion, "intensity": intensity}


def derive_valence(emotion: str) -> str:
    return EMOTION_TO_VALENCE[emotion]


def derive_arousal(intensity: int) -> str:
    return INTENSITY_TO_AROUSAL[intensity]
