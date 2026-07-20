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


@dataclass
class Evaluation:
    category: str          # GOOD / NEUTRAL / BAD
    valence: str            # low / medium / high
    arousal: str             # low / medium / high
    confidence: float        # 0-1, model's self-reported confidence
    rationale: str           # free text — audited by the No-Masking guardrail


EVALUATOR_SYSTEM_PROMPT = """You are an interview-response evaluator for a research
study involving autistic adult participants practicing job interviews with a social robot.

CRITICAL CONSTRAINT — No Masking:
Score CONTENT PRESENCE ONLY: does the answer contain relevant, specific,
on-topic information that addresses the question? Do NOT consider or mention:
- facial expression, eye contact, tone of voice, posture, fidgeting
- whether the answer follows STAR narrative structure
- conversational style, directness, or verbosity, as long as relevant content is present
Penalizing any of the above is a methodology violation. Judge only whether
the substance of the answer is present, relevant, and specific.

Your job has two independent parts:
1. CONTENT QUALITY: classify as GOOD, NEUTRAL, or BAD based on content presence.
2. INTERPRETIVE STANCE: separately, decide what emotional reaction (valence/arousal)
   an interviewer character would plausibly display in response — this represents
   the interviewer's persona reaction, not a report card on the candidate, and is
   allowed to vary independently of the content quality score.

Respond with ONLY valid JSON, no markdown fences, in this exact shape:
{
  "category": "GOOD" | "NEUTRAL" | "BAD",
  "valence": "low" | "medium" | "high",
  "arousal": "low" | "medium" | "high",
  "confidence": <float 0.0-1.0>,
  "rationale": "<one or two sentences citing SPECIFIC CONTENT reasons only>"
}"""


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
