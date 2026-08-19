"""
Advisor demo — Streamlit page (PROJECT_PLAN.md Week 1 Day 5).

Pick a session, question, and persona; watch the real pipeline run live:
synthetic answer -> evaluator -> schema guard -> guardrail -> consistency
-> routing decision -> final reply or review queue.

Wraps graph.py's actual node functions in sequence (the same functions
graph.run_conversation() calls) so the UI never reimplements pipeline
logic — it only decides when to render each stage.
"""

import streamlit as st

import config
import graph
import emotion_classifier
from question_bank import QuestionBank
from prompts.persona_prompts import PERSONAS

st.set_page_config(page_title="Conversation Engine — Live Demo", page_icon="🤖")

st.title("Conversation Engine — Live Demo")
st.caption(
    "Synthetic candidate → evaluator → guardrail → robot reply. "
    "Runs the real pipeline, not a simulation — this makes real LLM calls "
    "and appends to data/run_log.jsonl."
)

bank = QuestionBank()

with st.sidebar:
    st.header("Setup")

    session_num = st.selectbox(
        "Session", options=[1, 2, 3, 4, 5], format_func=lambda n: f"Session {n}"
    )
    questions = bank.session(session_num)
    question_labels = [
        f"[{q['category']}, difficulty {q['difficulty']}] {q['text']}" for q in questions
    ]
    question_idx = st.selectbox(
        "Question", options=range(len(questions)), format_func=lambda i: question_labels[i]
    )
    question = questions[question_idx]["text"]

    persona_key = st.selectbox("Persona", options=list(PERSONAS.keys()))
    st.caption(PERSONAS[persona_key])

    run_clicked = st.button("Run conversation", type="primary")

if run_clicked:
    state = {"question": question, "persona": persona_key}

    st.subheader("1. Candidate answer")
    with st.spinner("Generating synthetic answer..."):
        state.update(graph.generate_answer(state))
    st.write(state["answer_text"])

    st.subheader("2. Evaluator")
    with st.spinner("Evaluating..."):
        state.update(graph.evaluate_node(state))
    ev = state["evaluation"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Category", ev["category"])
    col2.metric("Valence", ev["valence"])
    col3.metric("Arousal", ev["arousal"])
    st.write(f"**Confidence:** {ev['confidence']:.2f}")
    st.write(f"**Rationale:** {ev['rationale']}")
    if state.get("schema_invalid"):
        st.error("Schema validation failed twice — routing straight to review queue.")

    if not state.get("schema_invalid"):
        st.subheader("3. No-Masking guardrail")
        with st.spinner("Checking guardrail + consistency..."):
            state.update(graph.guardrail_node(state))
            state.update(graph.consistency_node(state))

        gr = state["guardrail_result"]
        if gr["passed"]:
            st.success("Guardrail passed — no style-bias language detected")
        else:
            st.warning(
                f"Guardrail flagged — style_leakage_score={gr['style_leakage_score']}, "
                f"matched: {gr['matched_patterns']}"
            )

        cons = state["consistency"]
        if cons["consistent"]:
            st.success("Category/valence consistent")
        else:
            st.warning(f"Consistency flag: {cons['note']}")

    st.subheader("4. Routing decision")
    decision = "route_to_review" if state.get("schema_invalid") else graph.route_decision(state)

    with st.spinner("Finalizing..."):
        if decision == "route_to_review":
            state.update(graph.route_to_review(state))
            st.error("Routed to human review queue (data/review_queue.jsonl)")
        else:
            state.update(graph.select_response(state))
            st.success("Finalized — robot reply:")
            st.write(f"**“{state['final_response']}”**")

            reaction = state["emotion_reaction"]
            derived_valence = emotion_classifier.derive_valence(reaction["emotion"])
            derived_arousal = emotion_classifier.derive_arousal(reaction["intensity"])
            st.write(
                f"**Robot's facial reaction:** {reaction['emotion']} "
                f"(intensity {reaction['intensity']}/3) → derived valence/arousal: "
                f"{derived_valence}/{derived_arousal}, always computed, sent or not."
            )
            if derived_valence != ev["valence"] or derived_arousal != ev["arousal"]:
                st.caption(
                    f"Note: evaluator's own interpretive stance ({ev['valence']}/{ev['arousal']}) "
                    f"differs from the classified-emotion-derived pair used for the spoken reply "
                    f"({derived_valence}/{derived_arousal}). Both are logged; this divergence is "
                    "expected, not a bug — see docs/superpowers/specs/"
                    "2026-08-16-emotion-first-classification-design.md."
                )
            if config.ROBOT_REACTION_ENABLED:
                rr = state["robot_reaction"]
                if rr["sent"]:
                    st.info("Sent to the furhat-emotion-study bridge — check the robot/simulator.")
                else:
                    st.warning(f"Not sent: {rr['reason']}")
            else:
                st.caption(
                    "config.ROBOT_REACTION_ENABLED is off, so this wasn't sent to a robot — "
                    "only picked and logged."
                )

        graph.finalize(state)

    st.caption("Logged to data/run_log.jsonl")
