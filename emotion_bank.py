"""
Emotion Bank — coordinated facial-reaction lookup for the Furhat bridge.

Companion to response_bank.py: picks the robot's facial emotion+intensity
from the SAME (valence, arousal) pair response_bank.py already uses for
the spoken reply, so the two channels can never disagree on overall tone.

Each VA cell offers a small set of pre-approved, valence-consistent
candidate emotions rather than one fixed emotion, where the data supports
it. Restricted to Ekman's core 7 (Anger, Contempt, Disgust, Fear,
Happiness, Sadness, Surprise) -- no LOVE/SHAME/INTEREST. The core 7 skew
heavily negative, so only low valence (SADNESS/FEAR) actually has 2 valid
candidates; high (HAPPINESS) and medium (SURPRISE) each only have one
genuine option. Which candidate gets used when there's more than one:

- config.EMOTION_LLM_SELECTION_ENABLED=False (default): candidates[0],
  deterministic, no LLM call. Free, matches every existing test/batch run.
- =True: an LLM (pick_emotion_llm) picks between the pre-approved options
  based on the actual answer content -- automated variety without the
  face ever landing outside the valence family the text already implies.
  Cells with only one candidate never call the LLM, flag or not --
  nothing to choose between.

An earlier design let an LLM choose completely freely across all 10
emotions, independent of valence/arousal -- rejected, since that risks a
face that contradicts the spoken reply's tone, which actively undermines
this tool's purpose for autistic users practicing to read consistent
social signals. The candidate-list constraint is what makes automated
selection safe to reintroduce. See
docs/superpowers/specs/2026-08-11-robot-reaction-bridge-design.md.
"""

import json
import config
import llm_client


class EmotionBank:
    def __init__(self, path: str = config.EMOTION_BANK_PATH):
        with open(path, "r", encoding="utf-8") as f:
            self._data = json.load(f)
        self._data.pop("_note", None)

    def get(
        self,
        valence: str,
        arousal: str,
        question: str = None,
        answer_text: str = None,
        rationale: str = None,
    ) -> dict:
        key = config.va_cell_key(valence, arousal)
        cell = self._data.get(key)
        if not cell:
            raise KeyError(f"No emotion reaction found for VA cell '{key}'")

        candidates = cell["candidates"]
        if config.EMOTION_LLM_SELECTION_ENABLED and len(candidates) > 1:
            emotion = pick_emotion_llm(
                candidates, question=question, answer_text=answer_text, rationale=rationale
            )
        else:
            emotion = candidates[0]  # nothing to choose between otherwise

        return {"emotion": emotion, "intensity": cell["intensity"]}


def pick_emotion_llm(candidates: list, question: str, answer_text: str, rationale: str) -> str:
    """Picks between exactly two pre-approved, already valence-consistent
    emotions based on the answer's actual content. Not a free choice across
    all emotions -- both options were already vetted to fit the same
    overall tone as the spoken reply, so this only refines which one fits
    better, never picks something that contradicts the text."""
    system = (
        "You are choosing which of two possible robot facial reactions best "
        "fits a job interview answer, based on its content. Both options are "
        "already appropriate in overall tone -- you are only picking the more "
        "specific fit, not judging the answer's quality.\n\n"
        f"Options: {candidates[0]} or {candidates[1]}\n\n"
        "Respond with ONLY one of those two exact words, nothing else."
    )
    user_message = (
        f"Interview question: {question}\n\n"
        f"Candidate answer: {answer_text}\n\n"
        f"Evaluator's rationale: {rationale}"
    )

    raw = llm_client.call_llm(
        system=system,
        user_message=user_message,
        max_tokens=10,
        temperature=0.3,
    ).strip()

    if raw in candidates:
        return raw
    return candidates[0]  # fail closed on any unexpected response
