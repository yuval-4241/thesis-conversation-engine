"""
Central config for the Conversation Engine v1.
Combines pieces #8 (synthetic personas), #2 (evaluator), #5 (no-masking guardrail),
#6 (response bank), #7 (human review queue) into one runnable loop.
"""

import os

# ── LLM settings ────────────────────────────────────────────────────────────
# LLM_PROVIDER selects which backend llm_client.get_client() talks to.
# "lab" (default) = lab GPU server. "openai" = OpenAI's API directly —
# used when the lab server is unreachable/busy.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "lab")

LAB_LLM_BASE_URL = "http://100.110.96.82:8000/v1"
LAB_LLM_TOKEN = os.environ.get("LAB_LLM_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Model choice: start fast/cheap while building, move up once quality matters.
# For Hebrew response generation later, consider dictalm3-12b / dictalm3-24b.
if LLM_PROVIDER == "openai":
    MODEL_FAST = "gpt-4o-mini"
    MODEL_DEFAULT = "gpt-4o-mini"
    MODEL_HEBREW = "gpt-4o-mini"
else:
    MODEL_FAST = "llama3.1-8b"      # quick loop while building/debugging
    MODEL_DEFAULT = "gpt-oss-20b"   # good speed/quality tradeoff — use this by default
    MODEL_HEBREW = "dictalm3-12b"   # for Hebrew-heavy tasks later (response bank, personas)

MODEL = MODEL_DEFAULT

# ── VA Matrix (3x3) — must match your thesis's response bank structure ────
VALENCE_LEVELS = ["low", "medium", "high"]
AROUSAL_LEVELS = ["low", "medium", "high"]

def va_cells():
    """All 9 (valence, arousal) combinations, e.g. ('high','medium')."""
    return [(v, a) for v in VALENCE_LEVELS for a in AROUSAL_LEVELS]

def va_cell_key(valence: str, arousal: str) -> str:
    return f"{valence}_{arousal}"

# ── Evaluator categories ───────────────────────────────────────────────────
QUALITY_CATEGORIES = ["GOOD", "NEUTRAL", "BAD"]

# ── Confidence routing threshold ───────────────────────────────────────────
# Below this, item goes to the human review queue (#7) regardless of guardrail.
CONFIDENCE_ROUTE_THRESHOLD = 0.70

# ── No-Masking guardrail (#5) ──────────────────────────────────────────────
# Phrase/pattern families the guardrail scans the evaluator's OWN rationale for.
# These represent the evaluator smuggling in neurotypical-norm judgments
# (posture, eye contact, STAR conformity, tone-policing) instead of judging
# content/informational presence only. Kept at pattern level, not exhaustive —
# extend this list as you observe real evaluator drift.
NO_MASKING_FLAG_PATTERNS = [
    "eye contact",
    "didn't smile",
    "flat affect",
    "monotone",
    "awkward",
    "lacked enthusiasm",
    "should have structured",
    "not in STAR format",
    "too blunt",
    "too direct",
    "poor body language",
    "didn't elaborate enough",  # borderline — content-length is ok to flag,
                                  # but phrase it around content, not style
    "sounded robotic",
    "seemed nervous",
    "fidget",
]

# ── Persona library for synthetic participants (#8) ─────────────────────────
# Each persona deliberately diverges from neurotypical narrative norms so the
# evaluator gets stress-tested against exactly the population it must NOT
# penalize for style.
PERSONAS = {
    "direct_minimal": (
        "Answers in short, literal, factual sentences. Does not elaborate "
        "beyond what was asked. No narrative framing, no small talk."
    ),
    "non_linear_detailed": (
        "Answers with a lot of relevant detail but not in chronological or "
        "STAR order — jumps between context, outcome, and reasoning."
    ),
    "info_dense_technical": (
        "Gives precise, technical, information-dense answers with domain "
        "vocabulary; low emotional language."
    ),
    "conversational_tangential": (
        "Answers thoroughly but includes tangents loosely related to the "
        "question before returning to the point."
    ),
    "concise_confident": (
        "Gives a clear, well-organized, moderately detailed answer — a "
        "'control' persona representing a strong content-quality answer."
    ),
    "vague_low_effort": (
        "Gives a short, evasive, low-information answer — a 'control' "
        "persona representing a genuinely weak content-quality answer."
    ),
}

# ── File locations ──────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESPONSE_BANK_PATH = os.path.join(DATA_DIR, "response_bank_seed.json")
REVIEW_QUEUE_PATH = os.path.join(DATA_DIR, "review_queue.jsonl")
RUN_LOG_PATH = os.path.join(DATA_DIR, "run_log.jsonl")
