"""
LangGraph wiring for the Conversation Engine pipeline (Day 1 step 4):

    generate_answer -> evaluate -> guardrail_check -> ROUTE:
        flagged OR low confidence -> route_to_review -> finalize
        else                      -> select_response -> finalize

Every run appends one line to data/run_log.jsonl regardless of routing.
Does not modify personas.py, evaluator.py, guardrail.py, response_bank.py,
review_queue.py, config.py, or llm_client.py — only wires them together.
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
from response_bank import ResponseBank


class ConversationState(TypedDict, total=False):
    question: str
    persona: str
    answer_text: str
    evaluation: dict          # category, valence, arousal, confidence, rationale
    guardrail_result: dict    # passed, matched_patterns
    final_response: Optional[str]
    routed_to: str            # "review_queue" or "finalized"


_response_bank = ResponseBank()


def generate_answer(state: ConversationState) -> dict:
    ans = personas.generate_synthetic_answer(state["persona"], state["question"])
    return {"answer_text": ans.answer_text}


def evaluate_node(state: ConversationState) -> dict:
    ev = evaluator.evaluate(state["question"], state["answer_text"])
    return {
        "evaluation": {
            "category": ev.category,
            "valence": ev.valence,
            "arousal": ev.arousal,
            "confidence": ev.confidence,
            "rationale": ev.rationale,
        }
    }


def guardrail_node(state: ConversationState) -> dict:
    g = guardrail.check_rationale(state["evaluation"]["rationale"])
    return {"guardrail_result": g.as_dict()}


def _is_flagged(state: ConversationState) -> bool:
    return not state["guardrail_result"]["passed"]


def _is_low_confidence(state: ConversationState) -> bool:
    return state["evaluation"]["confidence"] < config.CONFIDENCE_ROUTE_THRESHOLD


def route_decision(state: ConversationState) -> str:
    if _is_flagged(state) or _is_low_confidence(state):
        return "route_to_review"
    return "select_response"


def route_to_review(state: ConversationState) -> dict:
    flagged = _is_flagged(state)
    low_confidence = _is_low_confidence(state)
    if flagged and low_confidence:
        reason = "low_confidence+guardrail_flag"
    elif flagged:
        reason = "guardrail_flag"
    else:
        reason = "low_confidence"

    review_queue.enqueue(
        {
            "question": state["question"],
            "persona": state["persona"],
            "answer_text": state["answer_text"],
            "evaluation": state["evaluation"],
            "guardrail_result": state["guardrail_result"],
        },
        reason,
    )
    return {"final_response": None, "routed_to": "review_queue"}


def select_response(state: ConversationState) -> dict:
    ev = state["evaluation"]
    response = _response_bank.get(ev["valence"], ev["arousal"])
    return {"final_response": response, "routed_to": "finalized"}


def finalize(state: ConversationState) -> dict:
    os.makedirs(config.DATA_DIR, exist_ok=True)
    record = {
        "timestamp": time.time(),
        "question": state["question"],
        "persona": state["persona"],
        "answer_text": state["answer_text"],
        "evaluation": state["evaluation"],
        "guardrail_result": state["guardrail_result"],
        "final_response": state.get("final_response"),
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
    graph.add_node("route_to_review", route_to_review)
    graph.add_node("select_response", select_response)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("generate_answer")
    graph.add_edge("generate_answer", "evaluate")
    graph.add_edge("evaluate", "guardrail_check")
    graph.add_conditional_edges(
        "guardrail_check",
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
