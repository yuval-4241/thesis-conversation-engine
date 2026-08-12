"""
Robot Reaction Bridge — sends emotion_bank's (emotion, intensity) pick,
plus response_bank's spoken reply text, to furhat-emotion-study's
RemoteControl HTTP endpoint (a separate Kotlin repo), so the
physical/virtual robot shows a matching facial expression while actually
speaking the reply, both from the same (valence, arousal) source.

Opt-in via config.ROBOT_REACTION_ENABLED (default False) — most runs
(pytest, run_demo.py, ab_test.py) never attempt this call, so a Furhat
skill not being reachable never affects them. Never raises: a disconnected
robot must never break the evaluation pipeline for a candidate.

See docs/superpowers/specs/2026-08-11-robot-reaction-bridge-design.md.
"""

import json
import requests
import config


def send_reaction(emotion: str, intensity: int, text: str = None) -> dict:
    if not config.ROBOT_REACTION_ENABLED:
        return {"sent": False, "reason": "disabled"}

    payload = {"emotion": emotion, "intensity": intensity}
    if text:
        payload["text"] = text

    # ensure_ascii=False: send raw UTF-8 (e.g. Hebrew) instead of \uXXXX
    # escapes -- RemoteControl.kt's hand-rolled regex parser expects the
    # literal characters, not JSON unicode escape sequences.
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    try:
        resp = requests.post(
            config.ROBOT_BRIDGE_URL,
            data=body,
            headers={"Content-Type": "application/json; charset=utf-8"},
            timeout=config.ROBOT_BRIDGE_TIMEOUT,
        )
        resp.raise_for_status()
        return {"sent": True, "reason": None}
    except requests.exceptions.RequestException as e:
        print(f"[robot_bridge] Failed to send reaction: {e}")
        return {"sent": False, "reason": str(e)}
