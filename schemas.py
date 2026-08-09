"""
Pydantic schemas that validate structured LLM output before it's trusted
downstream. See docs/unified_architecture.md (Task 4) for the design this
implements — this file is the "Schema Guard" that closes the gap where a
malformed-but-parseable evaluator response (e.g. "category": "good"
lowercase, or confidence > 1.0) could previously propagate silently.

Category/VALevel are hardcoded to match config.py's QUALITY_CATEGORIES /
VALENCE_LEVELS / AROUSAL_LEVELS exactly — Pydantic Literal types need
static values, so these can't just read the config lists at runtime.
tests/test_schemas.py checks the two stay in sync.
"""

from typing import Literal

from pydantic import BaseModel, Field

Category = Literal["GOOD", "NEUTRAL", "BAD"]
VALevel = Literal["low", "medium", "high"]


class EvaluatorOutputSchema(BaseModel):
    category: Category
    valence: VALevel
    arousal: VALevel
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)
