import json

import graph


def test_select_response_includes_disabled_robot_reaction_by_default(monkeypatch):
    monkeypatch.setattr(graph.config, "ROBOT_REACTION_ENABLED", False)

    state = {
        "evaluation": {"category": "GOOD", "valence": "medium", "arousal": "medium", "confidence": 0.9},
    }
    result = graph.select_response(state)

    assert result["robot_reaction"] == {"sent": False, "reason": "disabled"}
    assert result["routed_to"] == "finalized"
    assert result["final_response"]  # still picks a real text reply


def test_select_response_sends_reaction_when_enabled(monkeypatch):
    monkeypatch.setattr(graph.config, "ROBOT_REACTION_ENABLED", True)

    captured = {}

    def fake_send_reaction(emotion, intensity, text=None):
        captured["emotion"] = emotion
        captured["intensity"] = intensity
        captured["text"] = text
        return {"sent": True, "reason": None}

    monkeypatch.setattr(graph.robot_bridge, "send_reaction", fake_send_reaction)

    state = {
        "evaluation": {"category": "GOOD", "valence": "high", "arousal": "high", "confidence": 0.9},
    }
    result = graph.select_response(state)

    # high/high maps to HAPPINESS @ 3 in data/emotion_bank_seed.json
    assert captured["emotion"] == "HAPPINESS"
    assert captured["intensity"] == 3
    assert captured["text"] == result["final_response"]  # same text response_bank picked
    assert result["robot_reaction"] == {"sent": True, "reason": None}


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
    }
    graph.finalize(state)

    with open(log_path) as f:
        record = json.loads(f.readline())
    assert record["robot_reaction"] == {"sent": False, "reason": "disabled"}


def test_finalize_logs_null_robot_reaction_on_review_queue_path(monkeypatch, tmp_path):
    log_path = tmp_path / "run_log.jsonl"
    monkeypatch.setattr(graph.config, "RUN_LOG_PATH", str(log_path))
    monkeypatch.setattr(graph.config, "DATA_DIR", str(tmp_path))

    # route_to_review never calls select_response, so robot_reaction is
    # never set on this path — same absent-key pattern as guardrail_result
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
