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

# ── File locations ──────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESPONSE_BANK_PATH = os.path.join(DATA_DIR, "response_bank_seed.json")
EMOTION_BANK_PATH = os.path.join(DATA_DIR, "emotion_bank_seed.json")
REVIEW_QUEUE_PATH = os.path.join(DATA_DIR, "review_queue.jsonl")
RUN_LOG_PATH = os.path.join(DATA_DIR, "run_log.jsonl")

# Batch-run outputs (e.g. run_demo.py) — separate from the cumulative
# data/run_log.jsonl, so each batch's records/summary are easy to inspect
# on their own.
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# ── Robot reaction bridge (furhat-emotion-study, opt-in) ───────────────────
# Off by default — most runs (pytest, run_demo.py, ab_test.py) never attempt
# this call. Only set ROBOT_REACTION_ENABLED=true when the Furhat skill is
# actually running and you want a live/simulated robot reaction alongside
# the spoken reply. See docs/superpowers/specs/2026-08-11-robot-reaction-bridge-design.md.
ROBOT_REACTION_ENABLED = os.environ.get("ROBOT_REACTION_ENABLED", "false").lower() == "true"
ROBOT_BRIDGE_URL = "http://localhost:8765/react"
ROBOT_BRIDGE_TIMEOUT = 2.0  # seconds — never let a dead robot connection stall the pipeline
