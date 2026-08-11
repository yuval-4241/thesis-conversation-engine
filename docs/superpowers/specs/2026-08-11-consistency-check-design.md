# Consistency Check + Guardrail Severity Routing — Design

**Date:** 2026-08-11
**Status:** Approved, pending implementation.

## Context

`docs/unified_architecture.md` (Task 2) identified two gaps that are already
scoped but not built:

1. `evaluator.evaluate()` generates `category` and `valence`/`arousal` in one
   LLM call, deliberately decoupled — nothing catches an internally odd pair
   like `BAD` + `valence="high"`.
2. `guardrail.py`'s `GuardrailResult.style_leakage_score` (match count) is
   computed but unused — routing only looks at the binary `passed` field.

This spec closes both gaps in one pass, since they both change the same
`route_to_review()` reason-building logic.

## 1. Consistency check (`guardrail.py`)

New function, same file as `check_rationale()`, same dataclass style:

```python
@dataclass
class ConsistencyResult:
    consistent: bool
    note: str | None = None

    def as_dict(self):
        return {"consistent": self.consistent, "note": self.note}

_INCONSISTENT_PAIRS = {
    ("BAD", "high"), ("GOOD", "low"),
    ("NEUTRAL", "high"), ("NEUTRAL", "low"),
}

def check_consistency(category: str, valence: str) -> ConsistencyResult:
    if (category, valence) in _INCONSISTENT_PAIRS:
        return ConsistencyResult(False, f"{category} category paired with {valence} valence")
    return ConsistencyResult(True)
```

**Rule (chosen: "B — include NEUTRAL extremes"):** flag the two opposite-sign
extremes (`BAD`+high, `GOOD`+low) plus `NEUTRAL` paired with either extreme.
`medium` valence never flags, regardless of category. Arousal is not checked
— it's independent of quality by design and out of scope for this check.

This lives in `guardrail.py` rather than a new module or inline in
`graph.py`, despite checking structured fields rather than rationale text,
because it's still "audit the evaluator's output for a specific failure
mode" — the same job the rest of the file does.

## 2. Graph wiring (`graph.py`)

New node `consistency_check`, added after `guardrail_check`:

```
generate_answer -> evaluate -> [schema guard] -> guardrail_check
                                                        -> consistency_check
                                                        -> route_decision
```

- Calls `guardrail.check_consistency(evaluation["category"], evaluation["valence"])`.
- Stores the result under `state["consistency"]`.
- Only runs on the schema-valid path (same reachability as `guardrail_check`
  today) — if `schema_invalid` is `True`, this node never runs and
  `state["consistency"]` stays absent.

`route_decision()` (the boolean review-vs-finalize gate) is **not**
changed — it still only looks at `flagged` and `low_confidence`. Consistency
never triggers a route on its own; it only escalates the reason string of a
route that was already happening. This was an explicit choice (see
decisions below), not an oversight.

## 3. `route_to_review()` reason-building — refactor

The current 3-branch `if/elif/else` doesn't scale to the new dimensions
(severity tier × consistency flag), so it becomes a tag list:

```python
tags = []
if low_confidence:
    tags.append("low_confidence")
if flagged:
    tags.append("guardrail_flag_severe" if style_leakage_score >= 2 else "guardrail_flag")
if tags and consistency and not consistency["consistent"]:
    tags.append("consistency_flag")
reason = "+".join(tags)
```

- **Severity threshold:** `style_leakage_score >= 2` → `guardrail_flag_severe`
  instead of `guardrail_flag`. One matched phrase can be borderline; two or
  more is a stronger signal the rationale is genuinely leaning on style
  language.
- The `if tags` guard is what makes `consistency_flag` purely an escalator —
  it can never appear as the sole reason. If an item is high-confidence and
  guardrail-passed but consistency-flagged, it does **not** route to review;
  the consistency result is still recorded (next section), just not acted on.
- `schema_invalid` remains its own exclusive early return, unchanged —
  `consistency_check` never runs on that path anyway.

Existing reason value `"low_confidence+guardrail_flag"` is superseded by the
tag-join output, which now also covers `"low_confidence+guardrail_flag_severe"`,
`"guardrail_flag+consistency_flag"`, `"low_confidence+guardrail_flag_severe+consistency_flag"`,
and so on. No `Literal` type currently enforces the reason value in
`review_queue.py` (the `ReviewQueueItemSchema` in `unified_architecture.md`
was a design sketch, never implemented), so this doesn't require a schema
migration.

## 4. Logging (`finalize()`)

Add `"consistency": state.get("consistency")` to every `run_log.jsonl`
record, unconditionally — same "log everything regardless of routing"
convention `finalize()` already follows for `evaluation` and
`guardrail_result`. On the `schema_invalid` path this is logged as `null`.

This is what makes the "only escalate" choice safe to defend later: even
though a silent mismatch (high-confidence, guardrail-passed, but
category/VA inconsistent) doesn't force human review today, its rate is
still measurable from the log.

## 5. Tests

- `tests/test_consistency.py` (new): truth table covering all 4 flagged
  pairs (`BAD`/high, `GOOD`/low, `NEUTRAL`/high, `NEUTRAL`/low) plus a
  sample of consistent pairs (`GOOD`/high, `BAD`/low, anything/medium).
- Extend the graph routing tests (alongside `tests/test_graph_schema_guard.py`)
  to cover: severity-tier reason strings, consistency-flag escalation of an
  existing route, and confirm a silent (non-escalating) consistency mismatch
  still appears in the logged record.

## Explicit decisions (for future reference, not just this file)

- Consistency check never triggers routing on its own — only escalates an
  existing low-confidence/guardrail route. Deliberate deviation from
  `unified_architecture.md`'s original sketch, which routed every
  inconsistent pair unconditionally.
- Severity tiering only affects the *reason string* on an already-flagged
  item; it does not lower the trigger threshold for routing (any single
  guardrail match already routes today, unchanged).
