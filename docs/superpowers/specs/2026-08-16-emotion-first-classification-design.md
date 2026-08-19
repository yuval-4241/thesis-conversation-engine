# Emotion-First Classification — Design

**Date:** 2026-08-16
**Status:** Approved, replaces the `EMOTION_LLM_SELECTION_ENABLED` +
candidate-list mechanism from `emotion_bank.py` (2026-08-16, same day —
that mechanism shipped and was immediately superseded once this design
was worked through).

## Context

Today's flow (as of commit `9dde17f`): `evaluator.evaluate()` produces
`(category, valence, arousal, rationale)` in one LLM call, governed by the
methodology-locked `EVALUATOR_SYSTEM_PROMPT_A`/`_B`. `response_bank.py`
and `emotion_bank.py` both key off that same `(valence, arousal)` pair —
the "one source of truth" guarantee that keeps the spoken reply and the
robot's face coherent. `emotion_bank.py`'s candidate-list mechanism
(9 VA cells, each offering 1-2 pre-approved emotions) picks *which*
emotion within the valence family the VA pair implies.

Two academic sources surfaced this session prompted a reconsideration:

- **Tseng et al. (2014)**, *Using the Circumplex Model of Affect to Study
  Valence and Arousal Ratings of Emotional Faces by Children and Adults
  with Autism Spectrum Disorders*, JADD 44(6) — autistic participants
  showed a "constricted range" rating emotional faces on VA, relevant to
  how intensity should be calibrated for this population.
- **Schneider et al. (2025)**, *Datasets for Valence and Arousal
  Inference: A Survey*, arXiv:2510.00738 — Figure 2 shows Russell's
  circumplex split into 4 quadrants (Excited/Happy/Pleased;
  Annoying/Angry/Nervous; Sad/Bored/Sleepy; Relaxed/Peaceful/Calm),
  showing that arousal should sometimes change *which* emotion fits, not
  just how strongly it's expressed.

Working through where quadrant-awareness could actually apply surfaced
that only the low-valence family (`SADNESS`/`FEAR`) had more than one
candidate to begin with — everything else was already a single fixed
choice. This led to reconsidering the direction of the whole pipeline:
rather than narrowing a VA-derived candidate list, classify the emotion
directly from content, then derive VA from that classification for
`response_bank.py`'s benefit.

## Chosen approach

**1. New module, `emotion_classifier.py`** — one LLM call, independent of
`evaluator.py` (no shared code, no shared prompt, sees only `question` and
`answer_text`, not the evaluator's rationale or category):

```python
def classify_emotion(question: str, answer_text: str) -> dict:
    # Returns: {"emotion": "FEAR", "intensity": 2}
```

Picks freely among all 7 of Ekman's core emotions (Anger, Contempt,
Disgust, Fear, Happiness, Sadness, Surprise) — no register restriction.
This is an explicit reversal of an earlier design choice in
`emotion_bank_seed.json` (2026-08-16, same day) that excluded
Anger/Contempt/Disgust as "too hostile for a supportive robot." Noted
here so a future reader doesn't read that as an oversight: it was
deliberately dropped when the mechanism changed, per direct instruction.

**2. Fixed emotion→valence lookup** (data, not prompt-driven — reviewable
the same way `emotion_bank_seed.json`'s mapping already is):

| Emotion | Valence |
|---|---|
| HAPPINESS | high |
| SURPRISE | medium |
| ANGER, FEAR, DISGUST, CONTEMPT, SADNESS | low |

Grounded in Schneider et al.'s quadrants: HAPPINESS in the
positive/high-arousal quadrant, SADNESS/ANGER/FEAR in the negative
quadrants, SURPRISE conventionally placed near-neutral valence at high
arousal in circumplex literature generally (not itself one of
Schneider et al.'s four labeled quadrant terms, but consistent with
where standard circumplex placements put it).

**Arousal is deliberately *not* fixed per emotion.** An earlier draft of
this design fixed both valence and arousal per emotion (e.g. HAPPINESS
always high/high) — rejected once traced through: it would make most of
`response_bank.py`'s 9 cells permanently unreachable (e.g. `high_low`,
`high_medium` never selected if HAPPINESS only ever produces `(high,
high)`). Instead, **arousal is derived from `intensity`** using the exact
mapping already established (1→low, 2→medium, 3→high) — the same
`intensity` the LLM call returns for the face gesture does double duty as
the arousal source for text selection. This restores full 9-cell
reachability: every emotion can appear at any of the 3 intensities, so
every valence row of `response_bank.py` stays reachable via at least one
emotion.

**3. Data flow:**

```
classify_emotion(question, answer_text) -> {emotion, intensity}
        |
        +--> derived_valence = LOOKUP[emotion]
        |    derived_arousal = INTENSITY_TO_AROUSAL[intensity]
        |         `--> response_bank.get(derived_valence, derived_arousal) -> spoken text
        `--> {emotion, intensity} sent directly to robot_bridge.send_reaction() -> face
```

Face and text still trace back to the exact same `classify_emotion()`
call — the coherence guarantee from the original 2026-08-11 design is
preserved, just via emotion-first rather than VA-first data flow.

**4. `evaluator.py` is completely untouched.** No prompt edits, no
methodology-locked content modified. Its `valence`/`arousal` fields keep
existing for their current three purposes — `EvaluatorOutputSchema`
validation, `guardrail.check_consistency()` (category vs. valence), and
being logged in `run_log.jsonl` — they just stop being what determines
the spoken reply or the robot's face. **Two valence signals now
genuinely coexist and can diverge** (the evaluator's own "interpretive
stance" vs. the classified-emotion-derived valence) — this is an accepted
consequence, not a bug to reconcile. Both get logged, so the divergence
rate itself becomes a measurable, analyzable quantity later (does the
evaluator's own stance agree with the independently classified emotion?).

**5. Replaces, not adds to, `EMOTION_LLM_SELECTION_ENABLED`.** That flag
and `emotion_bank.py`'s candidate-list mechanism (shipped same day,
commit `9dde17f`) are superseded — running two different LLM-driven
emotion-selection systems side by side would add complexity without a
real benefit, and this mechanism is strictly more expressive (7 emotions
x 3 intensities = 21 possible outcomes, vs. 1-2 pre-approved candidates
per VA cell).

## What happens to `emotion_bank.py`

Its `EmotionBank.get(valence, arousal, ...)` interface and
`data/emotion_bank_seed.json` become unused by the new flow (which
computes `derived_valence`/`derived_arousal` inline from the
classification, not via a VA-keyed lookup). Left in place, not deleted,
in this spec — removal is an implementation-time decision, not a design
one; whoever executes the plan should confirm whether to delete the now-
dead code path or keep it dormant (e.g. behind a feature flag for
comparison) before committing to deletion.

## Testing

- `emotion_classifier.py`: mock `llm_client.call_llm`, verify parsing of
  `{emotion, intensity}` from the LLM response, verify fail-safe behavior
  on a malformed/unexpected response (same "fail closed" philosophy as
  `emotion_bank.pick_emotion_llm()`).
- Emotion→valence lookup: a table-driven test asserting every one of the
  7 core emotions maps to a value in `config.VALENCE_LEVELS`.
- `graph.py` wiring: `select_response()` calls `classify_emotion()`,
  derives valence/arousal, calls `response_bank.get()` with the derived
  pair, and passes `{emotion, intensity}` to `robot_bridge.send_reaction()`
  — verify with the same monkeypatch-based node-function tests already
  used throughout `tests/test_graph_robot_reaction.py`.
- Regression: confirm `evaluator.py`'s own valence/arousal still get
  logged unchanged, and `guardrail.check_consistency()` still runs against
  the evaluator's own valence, not the derived one.
