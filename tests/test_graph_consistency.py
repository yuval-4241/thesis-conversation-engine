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
