"""
#8 — Synthetic Participant Simulator.

Generates interview answers in different communication styles so the
evaluator (#2) can be stress-tested before any real participant is involved.
Doubles as a No-Masking bias check: if the evaluator systematically scores
non-linear / minimal / blunt personas worse than 'concise_confident' at
equal information content, that's evidence of style bias to fix in #2/#5.
"""

from dataclasses import dataclass
import config
import llm_client


@dataclass
class SyntheticAnswer:
    persona: str
    question: str
    answer_text: str


PERSONA_SYSTEM_TEMPLATE = """You are simulating a job interview candidate for a research study.
Persona style: {persona_description}

Answer the interview question in-character. Respond ONLY with the candidate's
spoken answer — no meta-commentary, no labels. Keep it realistic in length
for a spoken interview answer (2-6 sentences unless the persona is minimal)."""


def generate_synthetic_answer(persona_key: str, question: str) -> SyntheticAnswer:
    if persona_key not in config.PERSONAS:
        raise ValueError(f"Unknown persona: {persona_key}")

    system = PERSONA_SYSTEM_TEMPLATE.format(
        persona_description=config.PERSONAS[persona_key]
    )
    text = llm_client.call_llm(
        system=system,
        user_message=f"Interview question: {question}",
        max_tokens=400,
    ).strip()

    return SyntheticAnswer(persona=persona_key, question=question, answer_text=text)
