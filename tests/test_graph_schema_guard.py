import graph
from evaluator import Evaluation


def _good_evaluation():
    return Evaluation(
        category="GOOD",
        valence="medium",
        arousal="medium",
        confidence=0.9,
        rationale="Specific and relevant content.",
    )


def _bad_schema_evaluation():
    return Evaluation(
        category="good",  # invalid: not one of GOOD/NEUTRAL/BAD
        valence="medium",
        arousal="medium",
        confidence=0.9,
        rationale="Specific and relevant content.",
    )


def test_evaluate_node_passes_through_when_schema_valid(monkeypatch):
    monkeypatch.setattr(graph.evaluator, "evaluate", lambda *a, **k: _good_evaluation())

    state = {"question": "Q", "answer_text": "A"}
    result = graph.evaluate_node(state)

    assert result["schema_invalid"] is False
    assert result["evaluation"]["category"] == "GOOD"


def test_evaluate_node_retries_once_then_succeeds(monkeypatch):
    calls = []

    def fake_evaluate(question, answer_text, prompt_variant="A", retry_hint=None):
        calls.append(retry_hint)
        if retry_hint is None:
            return _bad_schema_evaluation()
        return _good_evaluation()

    monkeypatch.setattr(graph.evaluator, "evaluate", fake_evaluate)

    state = {"question": "Q", "answer_text": "A"}
    result = graph.evaluate_node(state)

    assert len(calls) == 2
    assert calls[0] is None
    assert calls[1] is not None
    assert result["schema_invalid"] is False
    assert result["evaluation"]["category"] == "GOOD"


def test_evaluate_node_flags_schema_invalid_after_two_failures(monkeypatch):
    monkeypatch.setattr(graph.evaluator, "evaluate", lambda *a, **k: _bad_schema_evaluation())

    state = {"question": "Q", "answer_text": "A"}
    result = graph.evaluate_node(state)

    assert result["schema_invalid"] is True


def test_route_after_evaluate_skips_guardrail_when_schema_invalid():
    assert graph._route_after_evaluate({"schema_invalid": True}) == "route_to_review"


def test_route_after_evaluate_goes_to_guardrail_when_schema_valid():
    assert graph._route_after_evaluate({"schema_invalid": False}) == "guardrail_check"


def test_route_to_review_uses_schema_invalid_reason(monkeypatch):
    enqueued = {}

    def fake_enqueue(item, reason):
        enqueued["item"] = item
        enqueued["reason"] = reason

    monkeypatch.setattr(graph.review_queue, "enqueue", fake_enqueue)

    state = {
        "schema_invalid": True,
        "question": "Q",
        "persona": "concise_confident",
        "answer_text": "A",
        "evaluation": {"category": "good"},
    }
    result = graph.route_to_review(state)

    assert enqueued["reason"] == "schema_invalid"
    assert result["routed_to"] == "review_queue"


def test_route_to_review_backward_compatible_without_schema_invalid_key(monkeypatch):
    """Existing low_confidence/guardrail_flag paths never set schema_invalid
    at all — state.get() must default to falsy without raising KeyError."""
    enqueued = {}

    def fake_enqueue(item, reason):
        enqueued["reason"] = reason

    monkeypatch.setattr(graph.review_queue, "enqueue", fake_enqueue)

    state = {
        "question": "Q",
        "persona": "vague_low_effort",
        "answer_text": "A",
        "evaluation": {"category": "NEUTRAL", "confidence": 0.5},
        "guardrail_result": {"passed": True, "matched_patterns": [], "style_leakage_score": 0},
    }
    result = graph.route_to_review(state)

    assert enqueued["reason"] == "low_confidence"
    assert result["routed_to"] == "review_queue"
