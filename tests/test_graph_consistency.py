import graph


def test_consistency_node_flags_bad_high_valence():
    state = {
        "evaluation": {"category": "BAD", "valence": "high", "arousal": "medium", "confidence": 0.9},
    }
    result = graph.consistency_node(state)

    assert result["consistency"]["consistent"] is False
    assert result["consistency"]["note"] == "BAD category paired with high valence"


def test_consistency_node_passes_good_high_valence():
    state = {
        "evaluation": {"category": "GOOD", "valence": "high", "arousal": "medium", "confidence": 0.9},
    }
    result = graph.consistency_node(state)

    assert result["consistency"]["consistent"] is True
    assert result["consistency"]["note"] is None


def test_graph_includes_consistency_check_node():
    compiled = graph._build_graph()
    node_names = set(compiled.get_graph().nodes.keys())
    assert "consistency_check" in node_names


def test_consistency_check_runs_after_guardrail_check_before_review_routing():
    compiled = graph._build_graph()
    edges = {(e.source, e.target) for e in compiled.get_graph().edges}
    assert ("guardrail_check", "consistency_check") in edges


def test_reason_uses_mild_tier_below_severity_threshold():
    state = {
        "evaluation": {"confidence": 0.9},
        "guardrail_result": {"passed": False, "matched_patterns": ["eye contact"], "style_leakage_score": 1},
    }
    assert graph._build_review_reason(state) == "guardrail_flag"


def test_reason_uses_severe_tier_at_threshold():
    state = {
        "evaluation": {"confidence": 0.9},
        "guardrail_result": {"passed": False, "matched_patterns": ["eye contact", "monotone"], "style_leakage_score": 2},
    }
    assert graph._build_review_reason(state) == "guardrail_flag_severe"


def test_reason_combines_low_confidence_and_severity_tier():
    state = {
        "evaluation": {"confidence": 0.5},
        "guardrail_result": {"passed": False, "matched_patterns": ["eye contact", "monotone"], "style_leakage_score": 2},
    }
    assert graph._build_review_reason(state) == "low_confidence+guardrail_flag_severe"


def test_reason_escalates_with_consistency_flag_when_already_flagged():
    state = {
        "evaluation": {"confidence": 0.9},
        "guardrail_result": {"passed": False, "matched_patterns": ["eye contact"], "style_leakage_score": 1},
        "consistency": {"consistent": False, "note": "BAD category paired with high valence"},
    }
    assert graph._build_review_reason(state) == "guardrail_flag+consistency_flag"


def test_reason_combines_all_three_tags():
    state = {
        "evaluation": {"confidence": 0.5},
        "guardrail_result": {"passed": False, "matched_patterns": ["eye contact", "monotone"], "style_leakage_score": 2},
        "consistency": {"consistent": False, "note": "BAD category paired with high valence"},
    }
    assert graph._build_review_reason(state) == "low_confidence+guardrail_flag_severe+consistency_flag"


def test_reason_does_not_escalate_when_nothing_else_flagged():
    """Consistency alone must never produce a reason on its own — it can
    only escalate a route that was already happening for another cause."""
    state = {
        "evaluation": {"confidence": 0.9},
        "guardrail_result": {"passed": True, "matched_patterns": [], "style_leakage_score": 0},
        "consistency": {"consistent": False, "note": "BAD category paired with high valence"},
    }
    assert graph._build_review_reason(state) == ""


def test_route_decision_ignores_consistency_when_otherwise_clean(monkeypatch):
    """route_decision() must not be affected by an inconsistent-but-otherwise-clean
    evaluation — it still only looks at confidence and guardrail passed."""
    state = {
        "evaluation": {"category": "BAD", "valence": "high", "arousal": "medium", "confidence": 0.9},
        "guardrail_result": {"passed": True, "matched_patterns": [], "style_leakage_score": 0},
        "consistency": {"consistent": False, "note": "BAD category paired with high valence"},
    }
    assert graph.route_decision(state) == "select_response"


def test_finalize_logs_consistency_field(monkeypatch, tmp_path):
    log_path = tmp_path / "run_log.jsonl"
    monkeypatch.setattr(graph.config, "RUN_LOG_PATH", str(log_path))
    monkeypatch.setattr(graph.config, "DATA_DIR", str(tmp_path))

    state = {
        "question": "Q",
        "persona": "concise_confident",
        "answer_text": "A",
        "evaluation": {"category": "BAD", "valence": "high", "arousal": "medium", "confidence": 0.9},
        "guardrail_result": {"passed": True, "matched_patterns": [], "style_leakage_score": 0},
        "consistency": {"consistent": False, "note": "BAD category paired with high valence"},
        "final_response": "some reply",
        "routed_to": "finalized",
    }
    graph.finalize(state)

    import json
    with open(log_path) as f:
        record = json.loads(f.readline())
    assert record["consistency"] == {"consistent": False, "note": "BAD category paired with high valence"}


def test_finalize_logs_null_consistency_on_schema_invalid_path(monkeypatch, tmp_path):
    log_path = tmp_path / "run_log.jsonl"
    monkeypatch.setattr(graph.config, "RUN_LOG_PATH", str(log_path))
    monkeypatch.setattr(graph.config, "DATA_DIR", str(tmp_path))

    # On the schema_invalid path, guardrail_check and consistency_check never
    # run, so LangGraph never sets these channels — the keys are *absent*
    # from state entirely, not present with value None.
    state = {
        "question": "Q",
        "persona": "concise_confident",
        "answer_text": "A",
        "evaluation": {"category": "good", "valence": "medium", "arousal": "medium", "confidence": 0.9},
        "final_response": None,
        "routed_to": "review_queue",
    }
    graph.finalize(state)

    import json
    with open(log_path) as f:
        record = json.loads(f.readline())
    assert record["guardrail_result"] is None
    assert record["consistency"] is None
