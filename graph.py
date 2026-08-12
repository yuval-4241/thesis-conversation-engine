"""
LangGraph wiring for the Conversation Engine pipeline (Day 1 step 4):

    generate_answer -> evaluate -> guardrail_check -> ROUTE:
        flagged OR low confidence -> route_to_review -> finalize
        else                      -> select_response -> finalize

Every run appends one line to data/run_log.jsonl regardless of routing.
Wires together personas.py, evaluator.py, guardrail.py, response_bank.py,
emotion_bank.py, robot_bridge.py, and review_queue.py without modifying
their internals.
"""

import json
import os
import time
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

import config
import personas
import evaluator
import guardrail
import review_queue
import robot_bridge
from pydantic import ValidationError
from response_bank import ResponseBank
from emotion_bank import EmotionBank
from schemas import EvaluatorOutputSchema


class ConversationState(TypedDict, total=False):
    question: str
    persona: str
    answer_text: str
    evaluation: dict          # category, valence, arousal, confidence, rationale
    schema_invalid: bool      # True if evaluation failed EvaluatorOutputSchema twice
    guardrail_result: dict    # passed, matched_patterns, style_leakage_score
    consistency: dict         # consistent, note — from guardrail.check_consistency()
    final_response: Optional[str]
    robot_reaction: dict      # sent, reason — from robot_bridge.send_reaction()
    routed_to: str            # "review_queue" or "finalized"


_response_bank = ResponseBank()
_emotion_bank = EmotionBank()


def generate_answer(state: ConversationState) -> dict:
    ans = personas.generate_synthetic_answer(state["persona"], state["question"])
    return {"answer_text": ans.answer_text}


def _evaluation_to_dict(ev) -> dict:
    return {
        "category": ev.category,
        "valence": ev.valence,
        "arousal": ev.arousal,
        "confidence": ev.confidence,
        "rationale": ev.rationale,
    }


def evaluate_node(state: ConversationState) -> dict:
    ev = evaluator.evaluate(state["question"], state["answer_text"])
    evaluation_dict = _evaluation_to_dict(ev)

    try:
        EvaluatorOutputSchema(**evaluation_dict)
        return {"evaluation": evaluation_dict, "schema_invalid": False}
    except ValidationError as e:
        retry_ev = evaluator.evaluate(
            state["question"], state["answer_text"], retry_hint=str(e)
        )
        retry_dict = _evaluation_to_dict(retry_ev)
        try:
            EvaluatorOutputSchema(**retry_dict)
            return {"evaluation": retry_dict, "schema_invalid": False}
        except ValidationError:
            return {"evaluation": retry_dict, "schema_invalid": True}


def guardrail_node(state: ConversationState) -> dict:
    g = guardrail.check_rationale(state["evaluation"]["rationale"])
    return {"guardrail_result": g.as_dict()}


def consistency_node(state: ConversationState) -> dict:
    ev = state["evaluation"]
    result = guardrail.check_consistency(ev["category"], ev["valence"])
    return {"consistency": result.as_dict()}


def _is_flagged(state: ConversationState) -> bool:
    return not state["guardrail_result"]["passed"]


def _is_low_confidence(state: ConversationState) -> bool:
    return state["evaluation"]["confidence"] < config.CONFIDENCE_ROUTE_THRESHOLD


def _is_schema_invalid(state: ConversationState) -> bool:
    return state.get("schema_invalid", False)


def _route_after_evaluate(state: ConversationState) -> str:
    if _is_schema_invalid(state):
        return "route_to_review"
    return "guardrail_check"


def route_decision(state: ConversationState) -> str:
    if _is_flagged(state) or _is_low_confidence(state):
        return "route_to_review"
    return "select_response"


def route_to_review(state: ConversationState) -> dict:
    if state.get("schema_invalid"):
        reason = "schema_invalid"
    else:
        reason = _build_review_reason(state)

    review_queue.enqueue(
        {
            "question": state["question"],
            "persona": state["persona"],
            "answer_text": state["answer_text"],
            "evaluation": state["evaluation"],
            "guardrail_result": state.get("guardrail_result"),
        },
        reason,
    )
    return {"final_response": None, "routed_to": "review_queue"}


def _build_review_reason(state: ConversationState) -> str:
    tags = []
    if _is_low_confidence(state):
        tags.append("low_confidence")
    if _is_flagged(state):
        severity = state["guardrail_result"].get("style_leakage_score", 0)
        tags.append("guardrail_flag_severe" if severity >= 2 else "guardrail_flag")

    consistency = state.get("consistency")
    if tags and consistency and not consistency.get("consistent", True):
        tags.append("consistency_flag")

    return "+".join(tags)


def select_response(state: ConversationState) -> dict:
    ev = state["evaluation"]
    response = _response_bank.get(ev["valence"], ev["arousal"])

    reaction = _emotion_bank.get(ev["valence"], ev["arousal"])
    robot_reaction = robot_bridge.send_reaction(reaction["emotion"], reaction["intensity"], text=response)

    return {
        "final_response": response,
        "robot_reaction": robot_reaction,
        "routed_to": "finalized",
    }


def finalize(state: ConversationState) -> dict:
    os.makedirs(config.DATA_DIR, exist_ok=True)
    record = {
        "timestamp": time.time(),
        "question": state["question"],
        "persona": state["persona"],
        "answer_text": state["answer_text"],
        "evaluation": state["evaluation"],
        "guardrail_result": state.get("guardrail_result"),
        "consistency": state.get("consistency"),
        "final_response": state.get("final_response"),
        "robot_reaction": state.get("robot_reaction"),
        "routed_to": state["routed_to"],
    }
    with open(config.RUN_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {}


def _build_graph():
    graph = StateGraph(ConversationState)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("guardrail_check", guardrail_node)
    graph.add_node("consistency_check", consistency_node)
    graph.add_node("route_to_review", route_to_review)
    graph.add_node("select_response", select_response)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("generate_answer")
    graph.add_edge("generate_answer", "evaluate")
    graph.add_conditional_edges(
        "evaluate",
        _route_after_evaluate,
        {
            "route_to_review": "route_to_review",
            "guardrail_check": "guardrail_check",
        },
    )
    graph.add_edge("guardrail_check", "consistency_check")
    graph.add_conditional_edges(
        "consistency_check",
        route_decision,
        {
            "route_to_review": "route_to_review",
            "select_response": "select_response",
        },
    )
    graph.add_edge("route_to_review", "finalize")
    graph.add_edge("select_response", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


_compiled_graph = _build_graph()


def run_conversation(persona_key: str, question: str) -> dict:
    initial_state: ConversationState = {"question": question, "persona": persona_key}
    result = _compiled_graph.invoke(initial_state)
    return dict(result)


if __name__ == "__main__":
    from question_bank import QuestionBank

    bank = QuestionBank()
    real_question = bank.session(1)[0]["text"]

    final_state = run_conversation("concise_confident", real_question)
    print(json.dumps(final_state, ensure_ascii=False, indent=2))
