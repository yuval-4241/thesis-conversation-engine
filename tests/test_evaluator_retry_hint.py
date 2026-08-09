import json

import evaluator


def _fake_valid_json():
    return json.dumps(
        {
            "category": "GOOD",
            "valence": "medium",
            "arousal": "medium",
            "confidence": 0.9,
            "rationale": "Specific and relevant.",
        }
    )


def test_retry_hint_appended_to_user_message_when_given(monkeypatch):
    captured = {}

    def fake_call_llm(system, user_message, max_tokens=500, temperature=0.7, model=None):
        captured["user_message"] = user_message
        return _fake_valid_json()

    monkeypatch.setattr(evaluator.llm_client, "call_llm", fake_call_llm)

    evaluator.evaluate(
        "Tell me about yourself.",
        "I have five years of experience.",
        retry_hint="confidence must be <= 1.0",
    )

    assert "confidence must be <= 1.0" in captured["user_message"]


def test_no_retry_hint_by_default_leaves_message_unchanged(monkeypatch):
    captured = {}

    def fake_call_llm(system, user_message, max_tokens=500, temperature=0.7, model=None):
        captured["user_message"] = user_message
        return _fake_valid_json()

    monkeypatch.setattr(evaluator.llm_client, "call_llm", fake_call_llm)

    evaluator.evaluate("Tell me about yourself.", "I have five years of experience.")

    assert captured["user_message"] == (
        "Interview question: Tell me about yourself.\n\n"
        "Candidate answer: I have five years of experience."
    )
