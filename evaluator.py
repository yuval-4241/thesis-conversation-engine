"""
#2 — LLM Evaluator.

Classifies a participant answer as GOOD/NEUTRAL/BAD based on CONTENT
PRESENCE ONLY (not delivery style), and assigns a valence/arousal target
representing the interviewer's interpretive emotional stance — NOT a
deterministic quality readout. Emotion is decoupled from quality by design
(see thesis note: decoupling is essential to preserve perspective-taking
measurement).

Outputs structured JSON with a rationale field for inter-rater audit
(Cohen's/Fleiss' kappa against human labels).
"""

import json
import re
from dataclasses import dataclass
import llm_client
from prompts.evaluator_prompt import EVALUATOR_SYSTEM_PROMPT


@dataclass
class Evaluation:
    category: str          # GOOD / NEUTRAL / BAD
    valence: str            # low / medium / high
    arousal: str             # low / medium / high
    confidence: float        # 0-1, model's self-reported confidence
    rationale: str           # free text — audited by the No-Masking guardrail


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def evaluate(question: str, answer_text: str) -> Evaluation:
    raw = llm_client.call_llm(
        system=EVALUATOR_SYSTEM_PROMPT,
        user_message=f"Interview question: {question}\n\nCandidate answer: {answer_text}",
        max_tokens=400,
        temperature=0.3,  # lower temp — this is a judgment task, not creative
    )
    raw = _strip_fences(raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Evaluator returned non-JSON output: {raw!r}") from e

    return Evaluation(
        category=data["category"],
        valence=data["valence"],
        arousal=data["arousal"],
        confidence=float(data["confidence"]),
        rationale=data["rationale"],
    )
