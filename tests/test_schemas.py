import typing

import pytest
from pydantic import ValidationError

import config
from schemas import EvaluatorOutputSchema


def test_valid_evaluation_passes():
    data = {
        "category": "GOOD",
        "valence": "medium",
        "arousal": "medium",
        "confidence": 0.9,
        "rationale": "Answer is specific and on-topic.",
    }
    result = EvaluatorOutputSchema(**data)
    assert result.category == "GOOD"
    assert result.confidence == 0.9


def test_invalid_category_rejected():
    data = {
        "category": "good",  # lowercase — not one of GOOD/NEUTRAL/BAD
        "valence": "medium",
        "arousal": "medium",
        "confidence": 0.9,
        "rationale": "Answer is specific and on-topic.",
    }
    with pytest.raises(ValidationError):
        EvaluatorOutputSchema(**data)


def test_confidence_above_one_rejected():
    data = {
        "category": "GOOD",
        "valence": "medium",
        "arousal": "medium",
        "confidence": 1.5,  # out of [0.0, 1.0]
        "rationale": "Answer is specific and on-topic.",
    }
    with pytest.raises(ValidationError):
        EvaluatorOutputSchema(**data)


def test_confidence_below_zero_rejected():
    data = {
        "category": "GOOD",
        "valence": "medium",
        "arousal": "medium",
        "confidence": -0.1,
        "rationale": "Answer is specific and on-topic.",
    }
    with pytest.raises(ValidationError):
        EvaluatorOutputSchema(**data)


def test_empty_rationale_rejected():
    data = {
        "category": "GOOD",
        "valence": "medium",
        "arousal": "medium",
        "confidence": 0.9,
        "rationale": "",
    }
    with pytest.raises(ValidationError):
        EvaluatorOutputSchema(**data)


def test_invalid_valence_rejected():
    data = {
        "category": "GOOD",
        "valence": "extreme",  # not one of low/medium/high
        "arousal": "medium",
        "confidence": 0.9,
        "rationale": "Answer is specific and on-topic.",
    }
    with pytest.raises(ValidationError):
        EvaluatorOutputSchema(**data)


def test_category_choices_match_config():
    """Drift guard: schemas.py hardcodes the same values as config.py's
    QUALITY_CATEGORIES (Pydantic Literal types need static values). If
    someone updates config.py without updating schemas.py, this test
    catches the mismatch instead of it silently going stale."""
    args = typing.get_args(EvaluatorOutputSchema.model_fields["category"].annotation)
    assert set(args) == set(config.QUALITY_CATEGORIES)


def test_valence_choices_match_config():
    args = typing.get_args(EvaluatorOutputSchema.model_fields["valence"].annotation)
    assert set(args) == set(config.VALENCE_LEVELS)


def test_arousal_choices_match_config():
    args = typing.get_args(EvaluatorOutputSchema.model_fields["arousal"].annotation)
    assert set(args) == set(config.AROUSAL_LEVELS)
