import json

import graph


def test_select_response_includes_disabled_robot_reaction_by_default(monkeypatch):
    monkeypatch.setattr(graph.config, "ROBOT_REACTION_ENABLED", False)
    monkeypatch.setattr(
        graph.emotion_classifier, "classify_emotion",
        lambda **k: {"emotion": "SURPRISE", "intensity": 2}
    )

    state = {
        "question": "Q",
        "answer_text": "A",
        "evaluation": {"category": "GOOD", "valence": "medium", "arousal": "medium", "confidence": 0.9},
    }
    result = graph.select_response(state)

    assert result["robot_reaction"] == {"sent": False, "reason": "disabled"}
    assert result["routed_to"] == "finalized"
    assert result["final_response"]  # still picks a real text reply

    # emotion_reaction is exactly what classify_emotion returned -- no VA
    # lookup involved on this side anymore.
    assert result["emotion_reaction"] == {"emotion": "SURPRISE", "intensity": 2}


def test_select_response_sends_reaction_when_enabled(monkeypatch):
    monkeypatch.setattr(graph.config, "ROBOT_REACTION_ENABLED", True)
    monkeypatch.setattr(
        graph.emotion_classifier, "classify_emotion",
        lambda **k: {"emotion": "HAPPINESS", "intensity": 3}
    )

    captured = {}

    def fake_send_reaction(emotion, intensity, text=None):
        captured["emotion"] = emotion
        captured["intensity"] = intensity
        captured["text"] = text
        return {"sent": True, "reason": None}

    monkeypatch.setattr(graph.robot_bridge, "send_reaction", fake_send_reaction)

    state = {
        "question": "Q",
        "answer_text": "A",
        "evaluation": {"category": "GOOD", "valence": "high", "arousal": "high", "confidence": 0.9},
    }
    result = graph.select_response(state)

    assert captured["emotion"] == "HAPPINESS"
    assert captured["intensity"] == 3
    assert captured["text"] == result["final_response"]  # same text response_bank picked
    assert result["robot_reaction"] == {"sent": True, "reason": None}
    assert result["emotion_reaction"] == {"emotion": "HAPPINESS", "intensity": 3}


def test_select_response_derives_va_from_classified_emotion_not_evaluator(monkeypatch):
    """The spoken reply must come from the classified emotion's derived
    (valence, arousal), not from evaluator's own valence/arousal -- the two
    signals are allowed to diverge by design (see spec section 4)."""
    monkeypatch.setattr(graph.config, "ROBOT_REACTION_ENABLED", False)
    monkeypatch.setattr(
        graph.emotion_classifier, "classify_emotion",
        lambda **k: {"emotion": "SADNESS", "intensity": 1}  # derives to low/low
    )

    captured = {}

    def fake_get(valence, arousal):
        captured["va"] = (valence, arousal)
        return "some reply"

    monkeypatch.setattr(graph._response_bank, "get", fake_get)

    # Evaluator's own valence/arousal is deliberately the OPPOSITE of what
    # SADNESS/1 derives to, to prove select_response() ignores it.
    state = {
        "question": "Q",
        "answer_text": "A",
        "evaluation": {"category": "GOOD", "valence": "high", "arousal": "high", "confidence": 0.9},
    }
    result = graph.select_response(state)

    assert captured["va"] == ("low", "low")
    assert result["final_response"] == "some reply"


def test_select_response_does_not_mutate_evaluators_own_valence_arousal(monkeypatch):
    monkeypatch.setattr(graph.config, "ROBOT_REACTION_ENABLED", False)
    monkeypatch.setattr(
        graph.emotion_classifier, "classify_emotion",
        lambda **k: {"emotion": "FEAR", "intensity": 3}
    )

    evaluation = {"category": "GOOD", "valence": "high", "arousal": "high", "confidence": 0.9}
    state = {"question": "Q", "answer_text": "A", "evaluation": dict(evaluation)}
    graph.select_response(state)

    # select_response() derives its own VA pair from the classified emotion
    # for response_bank/robot_bridge -- it must never touch the evaluator's
    # own valence/arousal, which guardrail.check_consistency() and
    # run_log.jsonl still depend on independently.
    assert state["evaluation"] == evaluation


def test_finalize_logs_robot_reaction_field(monkeypatch, tmp_path):
    log_path = tmp_path / "run_log.jsonl"
    monkeypatch.setattr(graph.config, "RUN_LOG_PATH", str(log_path))
    monkeypatch.setattr(graph.config, "DATA_DIR", str(tmp_path))

    state = {
        "question": "Q",
        "persona": "concise_confident",
        "answer_text": "A",
        "evaluation": {"category": "GOOD", "valence": "medium", "arousal": "medium", "confidence": 0.9},
        "guardrail_result": {"passed": True, "matched_patterns": [], "style_leakage_score": 0},
        "consistency": {"consistent": True, "note": None},
        "final_response": "some reply",
        "routed_to": "finalized",
        "robot_reaction": {"sent": False, "reason": "disabled"},
        "emotion_reaction": {"emotion": "SURPRISE", "intensity": 2},
    }
    graph.finalize(state)

    with open(log_path) as f:
        record = json.loads(f.readline())
    assert record["robot_reaction"] == {"sent": False, "reason": "disabled"}
    assert record["emotion_reaction"] == {"emotion": "SURPRISE", "intensity": 2}


def test_finalize_logs_null_robot_reaction_on_review_queue_path(monkeypatch, tmp_path):
    log_path = tmp_path / "run_log.jsonl"
    monkeypatch.setattr(graph.config, "RUN_LOG_PATH", str(log_path))
    monkeypatch.setattr(graph.config, "DATA_DIR", str(tmp_path))

    # route_to_review never calls select_response, so robot_reaction is
    # never set on this path -- same absent-key pattern as guardrail_result
    # on the schema_invalid path.
    state = {
        "question": "Q",
        "persona": "concise_confident",
        "answer_text": "A",
        "evaluation": {"category": "NEUTRAL", "valence": "medium", "arousal": "medium", "confidence": 0.5},
        "guardrail_result": {"passed": True, "matched_patterns": [], "style_leakage_score": 0},
        "consistency": {"consistent": True, "note": None},
        "final_response": None,
        "routed_to": "review_queue",
    }
    graph.finalize(state)

    with open(log_path) as f:
        record = json.loads(f.readline())
    assert record["robot_reaction"] is None
    assert record["emotion_reaction"] is None
