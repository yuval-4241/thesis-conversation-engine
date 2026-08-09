"""
#5 — No-Masking Guardrail.

Scans the evaluator's own rationale text for language that smuggles in
neurotypical-norm judgments (delivery style, STAR conformity, eye contact,
tone) instead of judging content presence only. This is the seam where
masking pressure would first appear in the pipeline, so it sits directly
after the evaluator and before any response/response-bank action is taken.

Deliberately rule-based + pattern-level (not a second LLM call) so it's
fast, auditable, and its behavior is fully inspectable — you can hand the
pattern list to your ethics board.
"""

from dataclasses import dataclass, field
import config


@dataclass
class GuardrailResult:
    passed: bool
    matched_patterns: list = field(default_factory=list)
    style_leakage_score: int = 0

    def __post_init__(self):
        # Always derived from matched_patterns — never independently settable,
        # so the two fields can't drift out of sync with each other.
        self.style_leakage_score = len(self.matched_patterns)

    def as_dict(self):
        return {
            "passed": self.passed,
            "matched_patterns": self.matched_patterns,
            "style_leakage_score": self.style_leakage_score,
        }


def check_rationale(rationale_text: str) -> GuardrailResult:
    lowered = rationale_text.lower()
    matches = [p for p in config.NO_MASKING_FLAG_PATTERNS if p.lower() in lowered]
    return GuardrailResult(passed=(len(matches) == 0), matched_patterns=matches)
