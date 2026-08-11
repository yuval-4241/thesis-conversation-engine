# Coordinated Face + Voice Reaction — Design (Prototype)

**Date:** 2026-08-11
**Status:** Approved, prototype scope (not wired into the graded text-only
pipeline — see Scope below).
**Spans two repos:** `conversation_engine` (Python) and `furhat-emotion-study`
(Kotlin, a separate Furhat Skill project at
`/Users/yuvalzohar/Documents/GitHub/furhat-emotion-study`).

## Context

`furhat-emotion-study` already has real, working facial-gesture code:
`EmotionLibrary.getEmotion(type: String, intensity: Int): Gesture`
(`gestures/EmotionsLibrary.kt`) builds a Furhat `Gesture` from ARKit/AU
blend-shape parameters for any of `EmotionType` (HAPPINESS, FEAR, SADNESS,
ANGER, SURPRISE, DISGUST, CONTEMPT, LOVE, SHAME, INTEREST) at intensity
1-3, with hand-tuned `Microexpression` overlays. Today it's only triggered
by a hardcoded test flow (`Test_AU.kt`) — nothing external can call it.

Separately, `conversation_engine`'s `response_bank.py` already picks a
spoken reply from `(valence, arousal)` via a 3×3 grid lookup
(`config.va_cell_key()` / `data/response_bank_seed.json`).

## Rejected approach, and why

The initial proposal added a second, independent LLM call that would pick
`reaction_emotion`/`reaction_intensity` directly (inspired by "Empathic
Prompting," arXiv:2510.20743, which conditions LLM response tone on a
structured `<emotion, intensity, valence, arousal>` tuple). **Rejected**
because two independently-computed channels (text reply from one source,
face from another) can disagree — e.g. a calm spoken reply paired with an
angry face. For a tool whose purpose is helping autistic adults practice
reading consistent social/emotional signals, an incoherent robot signal
doesn't just look bad, it actively undermines the thing the tool teaches.

## Chosen approach: one source of truth, two expressions of it

Both the spoken reply and the facial reaction are derived from the **same**
`(valence, arousal)` pair the evaluator already produces — never two
independent judgments. This guarantees agreement by construction, not by
hoping two LLM calls happen to align.

```
evaluator.evaluate() → (valence, arousal)
        │
        ├──→ response_bank.get(valence, arousal)  → spoken text   (existing)
        └──→ emotion_bank.get(valence, arousal)    → (emotion, intensity)  (NEW)
                        │
                        ▼
              HTTP POST → furhat-emotion-study's local bridge
                        │
                        ▼
        EmotionLibrary.getEmotion(emotion, intensity) → furhat.gesture(...)
```

## Components

**1. `data/emotion_bank_seed.json`** (new, `conversation_engine`) — same
shape and loading convention as `data/response_bank_seed.json`: 9 keys via
`config.va_cell_key()`, each mapping to `{"emotion": <EmotionType string>,
"intensity": 1-3}`. Hand-authored and reviewable, not AI-generated at
runtime — same "everything pre-written, nothing improvised live"
philosophy the response bank and guardrail already follow. Placeholder
values for now (deferred: real hand-authored values are a separate task,
same status as the response bank's own placeholder note).

**2. `emotion_bank.py`** (new, `conversation_engine`) — mirrors
`response_bank.py` exactly:
```python
class EmotionBank:
    def __init__(self, path: str = config.EMOTION_BANK_PATH):
        ...
    def get(self, valence: str, arousal: str) -> dict:
        # returns {"emotion": ..., "intensity": ...}
        # raises KeyError on missing cell, same as ResponseBank.get()
```
No LLM calls. Not wired into `graph.py` in this pass — that's future work
once/if this prototype graduates out of prototype status (per the scope
decision below).

**3. `RemoteControl.kt`** (new, `furhat-emotion-study`,
`src/main/kotlin/furhatos/app/emotionskill/remote/`) — the bridge. A
minimal HTTP server using `com.sun.net.httpserver.HttpServer` (JDK
built-in, no new Gradle dependency), bound to `localhost` only, one
endpoint:
```
POST /react
Body: {"emotion": "ANGER", "intensity": 2}
```
Parses the body (hand-rolled minimal JSON parsing — the two fields are
simple enough not to justify adding a JSON library dependency), calls
`EmotionLibrary.getEmotion(emotion, intensity)`, then
`furhat.gesture(gesture, async = true)`. Runs on a background thread,
started from `EmotionskillSkill.start()` so it's live whenever the skill
is running.

**4. Python-side sender** — a small function (in a new prototype script,
not wired into `graph.py`) that does
`requests.post("http://localhost:8765/react", json={"emotion": ..., "intensity": ...})`.

## Scope: prototype only

Per explicit decision: this stays separate from the graded, text-only
pipeline for now. `emotion_bank.py` is not called from `graph.py` in this
pass. `evaluator.py`'s methodology-locked `EVALUATOR_SYSTEM_PROMPT_A`/`_B`
are not touched — nothing here required editing them, since the design
dropped the second-LLM-call idea. If/when this graduates to real
integration, that's a separate, explicitly-flagged decision (touches
`CLAUDE.md`'s stated project phase).

## Testing

- `emotion_bank.py`: real unit tests, same style as
  `tests/test_response_bank.py` if one exists, or matching
  `tests/test_guardrail.py`'s conventions — no LLM involved, fully
  testable.
- `RemoteControl.kt`: **not** JUnit-style. Following this repo's existing
  convention for anything touching live external processes (see
  `OBSController.kt` / `src/test/kotlin/OBSSmokeTest.kt`, run via its own
  Gradle `JavaExec` task, manually verified against console output) — a
  new `RemoteControlSmokeTest.kt` under `src/test/kotlin/`, its own Gradle
  task, sends a real HTTP request to a running instance and the human
  watches the robot/simulator react. This cannot be meaningfully
  automated without a running Furhat simulator, which isn't available in
  this environment.

## Known open risk (accepted, not solved here)

Whether the official Furhat Remote API could someday replace this custom
bridge is unresolved — `EmotionLibrary.getEmotion()` builds gestures
dynamically at runtime rather than from pre-registered named gestures, and
compatibility with the official API's by-name triggering hasn't been
verified. Not blocking for a custom-bridge prototype; worth revisiting if
this becomes real integration.
