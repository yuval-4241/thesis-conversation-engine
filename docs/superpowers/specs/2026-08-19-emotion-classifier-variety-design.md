# Emotion Classifier Variety — Design

**Date:** 2026-08-19
**Status:** Approved

## Context

`emotion_classifier.py` (shipped earlier this session, see
`docs/superpowers/specs/2026-08-16-emotion-first-classification-design.md`)
makes one LLM call per candidate answer, picking freely among Ekman's core
7 emotions plus an intensity 1-3, at `temperature=0.3`.

Running the live demo surfaced a strong skew toward HAPPINESS. Isolating
just this session's `classify_emotion()`-era log entries (filtering out
older `INTEREST` entries from the pre-core-7 `emotion_bank.py` mechanism
this replaced): 8 HAPPINESS, 2 SADNESS out of 10.

A direct test confirmed the classifier isn't stuck — it correctly picks
ANGER/FEAR when the answer text has explicit emotional language:

| Answer | Result |
|---|---|
| "I completely failed the project, my manager was furious" | ANGER, 3 |
| "I was terrified we would miss the deadline" | FEAR, 3 |
| "I am very organized and detail-oriented" | HAPPINESS, 2 |
| "My coworker took credit for my work, I was extremely angry" | ANGER, 3 |
| "I have 5 years of experience in QA testing" | HAPPINESS, 1 |

It defaults to HAPPINESS specifically on bland, factual, low-signal
answers — which is most of what `personas.py`'s synthetic candidates
generate. With no explicit emotional language to anchor on, the model
reads generic confident/professional phrasing as mildly positive.

**Goal (explicit, from direct instruction this session):** push toward
more visible emotion variety for the demo/UX, even at the cost of
sometimes moving genuinely neutral content away from its most "obvious"
reading. This is a UX-tuning decision, not an accuracy-first one — noted
explicitly so a future reader doesn't mistake the chosen approach for a
claim that the resulting classifications are more "correct."

Scope check: this only touches `emotion_classifier.py`'s prompt and one
`llm_client.call_llm()` kwarg. It does not touch `evaluator.py` or any
methodology-locked content (`config.NO_MASKING_FLAG_PATTERNS`,
`PERSONAS`, `EVALUATOR_SYSTEM_PROMPT_A`/`_B`), and it isn't in scope for
the No-Masking guardrail, which only scans the evaluator's own rationale
text. Confirmed with the user before starting: normal engineering
judgment call, no flagging needed.

## Academic grounding

**Duong et al. (2025)**, *CHEER-Ekman: Fine-grained Embodied Emotion
Classification*, ACL 2025 (arXiv:2506.01047) — studies LLM prompting
strategies for classification into Ekman's six basic emotions, the same
taxonomy (plus Contempt) used here. Two findings apply directly:

1. **Chain-of-thought (CoT) prompting significantly improves accuracy** on
   this exact task — a CoT-prompted 8B model nearly matched a 70B model's
   performance (within 7 F1 points), and CoT specifically helped where the
   model was previously anchoring on the wrong signal.
2. **Their error analysis documents almost exactly our failure mode**: the
   model "predicted Joy... despite strong physiological cues that more
   closely reflect Surprise," because it "prioritize[d] surface-level
   celebratory language over conflicting embodied cues" (e.g. a nearby
   sentence mentioning someone smiling and waving pulled an unrelated
   clause toward Joy). This is the same shape as our classifier reading
   generically confident/professional phrasing as HAPPINESS.
3. Simplified, plain-language prompts outperformed technical/clinical
   wording by up to 29.5 F1 points in their experiments — our current
   prompt is already plain, so this mainly confirms not to make it more
   clinical while making the other changes below.

They also test best-worst scaling (BWS) — presenting the LLM with tuples
of several candidate sentences per comparison round, which outperformed
zero-shot by ~20 points. Not applicable here: our use case classifies one
answer at a time in a live interview flow, with no batch of candidate
sentences to compare against. Noted and explicitly not adopted.

## Chosen approach

Two independent, additive changes to `emotion_classifier.py`:

**1. Chain-of-thought step, discarded after parsing.** The LLM's JSON
response gains one extra field, `"cue"` — the specific word or phrase in
the answer driving its judgment — that must be produced *before*
`"emotion"` in the JSON object (field order in the schema description
nudges the model to reason before naming, without a separate call).
`classify_emotion()`'s return shape stays exactly `{"emotion", "intensity"}`
— `"cue"` is parsed and then dropped, purely a reasoning scratchpad. This
keeps `graph.py`, `robot_bridge.py`, `demo_app.py`, and every existing
test's expectations on `classify_emotion()`'s output untouched.

**2. One explicit anti-default instruction**, directly informed by the
CHEER-Ekman error analysis: tell the model not to read generic
confident/professional phrasing as automatically positive, and to prefer
SURPRISE over HAPPINESS when an answer is simply informationally dense
without any actual emotional language — SURPRISE is the more neutral
"noteworthy, not necessarily positive" reading for factual content with
no emotional signal, whereas HAPPINESS/SADNESS both assert an emotional
direction the text doesn't actually contain. This isn't inventing a new
default so much as picking a less presumptuous one when there's truly no
signal either way.

**3. Temperature `0.3` → `0.8`.** Mechanical variety lever, independent of
the prompt changes. Non-deterministic: the same answer can classify
differently across runs. Explicitly acceptable per the stated goal
(UX variety over reproducibility) — noting this so it's not later mistaken
for a regression if two identical answers get different reactions.

**New system prompt (replaces `_SYSTEM_PROMPT`):**

```python
_SYSTEM_PROMPT = (
    "You are classifying the emotional content of a job interview answer, "
    "independent of how good or bad the answer is.\n\n"
    "First, find the specific word or phrase in the answer that most "
    "signals an emotion -- not the overall tone of a confident or "
    "professional-sounding answer, just what's actually there. Generic "
    "confident or professional phrasing on its own is not a signal of "
    "happiness -- if an answer is simply factual with no real emotional "
    "language, prefer SURPRISE (noteworthy, not positive or negative) "
    "over defaulting to HAPPINESS.\n\n"
    "Then pick exactly one of Ekman's core emotions that best fits: "
    "ANGER, CONTEMPT, DISGUST, FEAR, HAPPINESS, SADNESS, SURPRISE. "
    "Also rate its intensity from 1 (mild) to 3 (strong).\n\n"
    "Respond with ONLY a JSON object of the exact shape: "
    '{"cue": "<the word or phrase, or \\"none\\" if truly nothing '
    'stands out>", "emotion": "<ONE_OF_THE_7_WORDS_ABOVE>", '
    '"intensity": <1, 2, or 3>}'
)
```

`classify_emotion()`'s parsing gains a `data.get("cue")` read (for the CoT
effect) but doesn't validate or return it — only `emotion`/`intensity`
still go through the existing `EMOTION_TO_VALENCE`/`INTENSITY_TO_AROUSAL`
validation and fail-closed fallback. `"cue"` is optional and never
required: read with `.get()`, not `[...]`, so a response missing it is
still well-formed as long as `emotion`/`intensity` are present and valid.
This is deliberate, not an oversight — it's what keeps every existing
test's two-key mocked responses (no `"cue"`) passing unchanged (see "What
stays unchanged" below), and the CoT effect only depends on the *prompt*
asking the model to produce `"cue"` before `"emotion"` in a real response,
not on this code enforcing its presence.

`max_tokens` needs to grow slightly to fit the `"cue"` text before
`"emotion"` — `50` → `80`.

## What stays unchanged

- `classify_emotion(question, answer_text) -> {"emotion": str, "intensity": int}`
  — same signature, same return shape.
- `derive_valence()`, `derive_arousal()`, `EMOTION_TO_VALENCE`,
  `INTENSITY_TO_AROUSAL`, `CORE_EMOTIONS`, `_FALLBACK` — untouched.
- `graph.py`'s `select_response()` wiring — untouched, it only calls
  `classify_emotion()`/`derive_valence()`/`derive_arousal()` by name.
- `evaluator.py` and all methodology-locked content — untouched.
- Every existing test in `tests/test_emotion_classifier.py` and
  `tests/test_graph_robot_reaction.py` that mocks
  `emotion_classifier.llm_client.call_llm` with a two-key JSON response
  (no `"cue"`) — still valid, since `"cue"` is read with `.get("cue")`,
  never required. Only real (non-test) LLM responses are expected to
  include it, per the new prompt.

## Testing

- New test: mock a response including `"cue"`, verify `classify_emotion()`
  still returns only `{"emotion", "intensity"}` (no `"cue"` leaks into the
  return value).
- New test: mock a response *without* `"cue"` (old two-key shape), verify
  it still parses successfully — confirms backward compatibility with
  every existing test in `tests/test_emotion_classifier.py`.
- Existing fail-closed tests (non-JSON, unknown emotion, out-of-range
  intensity, missing `emotion`/`intensity` keys) — re-run unchanged, must
  still pass.
- Manual/live check (not a pytest test): re-run the same 5-answer sample
  from this session's investigation through the live LLM after the prompt
  change, to sanity-check the anti-default instruction actually reduces
  the HAPPINESS skew on the 2 bland/factual answers, without asserting an
  exact expected emotion (LLM output, not deterministic, especially at
  `temperature=0.8`).
