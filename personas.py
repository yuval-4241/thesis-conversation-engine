"""
#8 — Synthetic Participant Simulator.

Generates interview answers in different communication styles so the
evaluator (#2) can be stress-tested before any real participant is involved.
Doubles as a No-Masking bias check: if the evaluator systematically scores
non-linear / minimal / blunt personas worse than 'concise_confident' at
equal information content, that's evidence of style bias to fix in #2/#5.
"""

from dataclasses import dataclass
import llm_client
from prompts.persona_prompts import PERSONAS, PERSONA_SYSTEM_TEMPLATE


@dataclass
class SyntheticAnswer:
    persona: str
    question: str
    answer_text: str


def generate_synthetic_answer(persona_key: str, question: str) -> SyntheticAnswer:
    if persona_key not in PERSONAS:
        raise ValueError(f"Unknown persona: {persona_key}")

    system = PERSONA_SYSTEM_TEMPLATE.format(
        persona_description=PERSONAS[persona_key]
    )
    text = llm_client.call_llm(
        system=system,
        user_message=f"Interview question: {question}",
        max_tokens=400,
    ).strip()

    return SyntheticAnswer(persona=persona_key, question=question, answer_text=text)
