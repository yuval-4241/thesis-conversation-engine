# Consistency Check + Guardrail Severity Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a category/valence consistency check to the evaluation pipeline, and use it plus the existing `style_leakage_score` to make review-queue routing reasons more informative — without changing which items get routed to review today.

**Architecture:** One new pure function (`guardrail.check_consistency`) evaluated as a new LangGraph node (`consistency_check`) between `guardrail_check` and the existing routing decision. The existing binary route-vs-finalize gate (`route_decision`) is untouched; only the *reason string* built inside `route_to_review()` changes, from a hardcoded if/elif chain to a composable tag list. The consistency result is always written to `data/run_log.jsonl` via `finalize()`, whether or not it affected routing.

**Tech Stack:** Python 3.10+, `dataclasses` (matches existing `GuardrailResult` style — no Pydantic here, this is deterministic internal logic, not untrusted LLM output), LangGraph `StateGraph`, `pytest` + `monkeypatch` (matches existing test style in `tests/test_graph_schema_guard.py`).

## Global Constraints

- Design source of truth: `docs/superpowers/specs/2026-08-11-consistency-check-design.md`. Follow it exactly; these constraints are copied from it, not reinterpreted.
- Inconsistent pairs (rule B — category × valence only, arousal not checked): `("BAD", "high")`, `("GOOD", "low")`, `("NEUTRAL", "high")`, `("NEUTRAL", "low")`. `medium` valence never flags, regardless of category.
- Severity threshold: `style_leakage_score >= 2` on an already-flagged item → reason tag `guardrail_flag_severe` instead of `guardrail_flag`. Below 2, unchanged `guardrail_flag`.
- The consistency flag is a pure **escalator**: it can only ever be appended to a reason that already has at least one tag (`low_confidence` and/or a `guardrail_flag*` tag). It must never appear as the sole reason, and must never by itself cause `route_decision()` to send an item to review.
- `route_decision()` (the boolean review-vs-finalize gate) is **not modified** — it keeps checking only `_is_flagged()` and `_is_low_confidence()`.
- `schema_invalid` stays an exclusive early return in `route_to_review()` — unchanged, and `consistency_check` never runs on that path (same reachability as today's `guardrail_check`).
- Every `run_log.jsonl` record gets a `"consistency"` key, unconditionally — `None`/`null` on the `schema_invalid` path where the node never ran.
- New code goes in `guardrail.py` (the `check_consistency` function + `ConsistencyResult` dataclass), not a new module, not inline in `graph.py`.

---

## File Structure

- **Modify:** `guardrail.py` — add `ConsistencyResult` dataclass and `check_consistency()` function, alongside the existing `GuardrailResult`/`check_rationale()`.
- **Create:** `tests/test_consistency.py` — truth-table tests for `check_consistency()`.
- **Modify:** `graph.py` — add `consistency` to `ConversationState`, add `consistency_node()`, wire it into `_build_graph()`, refactor `route_to_review()`'s reason-building into `_build_review_reason()`, add `"consistency"` to the `finalize()` log record.
- **Create:** `tests/test_graph_consistency.py` — graph-level tests for the new node, the severity tiers, the escalation behavior, and the log-even-when-silent behavior. Kept separate from `tests/test_graph_schema_guard.py` (which stays focused on schema-guard behavior only, per that file's existing single responsibility).

---

### Task 1: `check_consistency()` in `guardrail.py`

**Files:**
- Modify: `guardrail.py`
- Test: `tests/test_consistency.py` (new)

**Interfaces:**
- Consumes: nothing new — plain strings.
- Produces: `ConsistencyResult` dataclass with fields `consistent: bool`, `note: str | None = None`, and method `as_dict() -> dict` returning `{"consistent": ..., "note": ...}`. Function `check_consistency(category: str, valence: str) -> ConsistencyResult`. These are what Task 2 and Task 3 call.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_consistency.py`:

```python
from guardrail import ConsistencyResult, check_consistency


def test_bad_category_with_high_valence_is_inconsistent():
    result = check_consistency("BAD", "high")
    assert result.consistent is False
    assert result.note == "BAD category paired with high valence"


def test_good_category_with_low_valence_is_inconsistent():
    result = check_consistency("GOOD", "low")
    assert result.consistent is False
    assert result.note == "GOOD category paired with low valence"


def test_neutral_category_with_high_valence_is_inconsistent():
    result = check_consistency("NEUTRAL", "high")
    assert result.consistent is False


def test_neutral_category_with_low_valence_is_inconsistent():
    result = check_consistency("NEUTRAL", "low")
    assert result.consistent is False


def test_good_category_with_high_valence_is_consistent():
    result = check_consistency("GOOD", "high")
    assert result.consistent is True
    assert result.note is None


def test_bad_category_with_low_valence_is_consistent():
    result = check_consistency("BAD", "low")
    assert result.consistent is True


def test_medium_valence_never_flags_regardless_of_category():
    for category in ("GOOD", "NEUTRAL", "BAD"):
        result = check_consistency(category, "medium")
        assert result.consistent is True, f"{category}+medium should be consistent"


def test_as_dict_shape():
    result = check_consistency("BAD", "high")
    d = result.as_dict()
    assert d == {"consistent": False, "note": "BAD category paired with high valence"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_consistency.py -v`
Expected: FAIL with `ImportError: cannot import name 'ConsistencyResult' from 'guardrail'` (or `check_consistency`).

- [ ] **Step 3: Implement in `guardrail.py`**

Add this after the existing `GuardrailResult` class and before `check_rationale()`:

```python
@dataclass
class ConsistencyResult:
    consistent: bool
    note: str = None

    def as_dict(self):
        return {"consistent": self.consistent, "note": self.note}


_INCONSISTENT_PAIRS = {
    ("BAD", "high"),
    ("GOOD", "low"),
    ("NEUTRAL", "high"),
    ("NEUTRAL", "low"),
}


def check_consistency(category: str, valence: str) -> ConsistencyResult:
    if (category, valence) in _INCONSISTENT_PAIRS:
        return ConsistencyResult(
            consistent=False,
            note=f"{category} category paired with {valence} valence",
        )
    return ConsistencyResult(consistent=True)
```

Note: `guardrail.py` already has `from dataclasses import dataclass, field` at the top — no new import needed for `dataclass`. `note: str = None` (not `str | None = None`) matches this file's existing style, which doesn't use `Optional`/union syntax elsewhere.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_consistency.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add guardrail.py tests/test_consistency.py
git commit -m "$(cat <<'EOF'
Add check_consistency() to guardrail.py

Flags four category/valence combinations (BAD+high, GOOD+low,
NEUTRAL+high, NEUTRAL+low) where the evaluator's VA "interpretive
stance" looks like it might be leaking in as a disguised quality
score. Pure function, not yet wired into graph.py.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Wire `consistency_check` into `graph.py`

**Files:**
- Modify: `graph.py:30-38` (`ConversationState`), `graph.py:78-80` (add node function after `guardrail_node`), `graph.py:156-187` (`_build_graph()`)
- Test: `tests/test_graph_consistency.py` (new)

**Interfaces:**
- Consumes: `guardrail.check_consistency(category: str, valence: str) -> ConsistencyResult` from Task 1.
- Produces: `consistency_node(state: ConversationState) -> dict` returning `{"consistency": {"consistent": bool, "note": str|None}}`. `ConversationState` gains a `consistency: dict` field. Task 3 reads `state.get("consistency")`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_graph_consistency.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_graph_consistency.py -v`
Expected: FAIL — `AttributeError: module 'graph' has no attribute 'consistency_node'` on the first two, node/edge assertions fail on the last two (node not in graph yet).

- [ ] **Step 3: Implement in `graph.py`**

In the `ConversationState` TypedDict (currently `graph.py:30-38`), add one field — the block becomes:

```python
class ConversationState(TypedDict, total=False):
    question: str
    persona: str
    answer_text: str
    evaluation: dict          # category, valence, arousal, confidence, rationale
    schema_invalid: bool      # True if evaluation failed EvaluatorOutputSchema twice
    guardrail_result: dict    # passed, matched_patterns, style_leakage_score
    consistency: dict         # consistent, note — from guardrail.check_consistency()
    final_response: Optional[str]
    routed_to: str            # "review_queue" or "finalized"
```

Immediately after `guardrail_node()` (currently `graph.py:78-80`), add:

```python
def consistency_node(state: ConversationState) -> dict:
    ev = state["evaluation"]
    result = guardrail.check_consistency(ev["category"], ev["valence"])
    return {"consistency": result.as_dict()}
```

In `_build_graph()` (currently `graph.py:156-187`), register the node and reroute the edges. Before:

```python
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("guardrail_check", guardrail_node)
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
    graph.add_conditional_edges(
        "guardrail_check",
        route_decision,
        {
            "route_to_review": "route_to_review",
            "select_response": "select_response",
        },
    )
```

After:

```python
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
```

(The rest of `_build_graph()` — the `route_to_review`/`select_response`/`finalize` edges and the final `return graph.compile()` — is unchanged.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_graph_consistency.py -v`
Expected: 4 passed.

Then run the full suite to confirm nothing existing broke:

Run: `pytest tests/ -v`
Expected: all passing (31 tests: the prior 23 + 8 new from Task 1 — this task adds no new count since its 4 tests were counted already... actually: 23 existing + 8 (Task 1) + 4 (Task 2) = 35 passed).

- [ ] **Step 5: Commit**

```bash
git add graph.py tests/test_graph_consistency.py
git commit -m "$(cat <<'EOF'
Wire consistency_check as a new graph node between guardrail and routing

Runs guardrail.check_consistency() on every schema-valid evaluation and
stores the result in state["consistency"]. Does not yet affect routing
or logging — route_decision() and finalize() are unchanged in this
commit, wired up in the next one.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Severity-tiered, consistency-escalated routing reasons + logging

**Files:**
- Modify: `graph.py:107-130` (`route_to_review()`, split out `_build_review_reason()`), `graph.py:139-153` (`finalize()`)
- Test: `tests/test_graph_consistency.py` (extend from Task 2)

**Interfaces:**
- Consumes: `state["consistency"]` (dict, from Task 2's `consistency_node`), `state["guardrail_result"]["style_leakage_score"]` (int, already existed before this plan), `_is_flagged(state)` and `_is_low_confidence(state)` (already exist in `graph.py`).
- Produces: `_build_review_reason(state: ConversationState) -> str`, called only from `route_to_review()`. `finalize()`'s log record gains a `"consistency"` key.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_graph_consistency.py`:

```python
def test_reason_uses_mild_tier_below_severity_threshold():
    state = {
        "evaluation": {"confidence": 0.9},
        "guardrail_result": {"passed": False, "matched_patterns": ["eye contact"], "style_leakage_score": 1},
    }
    assert graph._build_review_reason(state) == "guardrail_flag"


def test_reason_uses_severe_tier_at_threshold():
    state = {
        "evaluation": {"confidence": 0.9},
        "guardrail_result": {"passed": False, "matched_patterns": ["eye contact", "monotone"], "style_leakage_score": 2},
    }
    assert graph._build_review_reason(state) == "guardrail_flag_severe"


def test_reason_combines_low_confidence_and_severity_tier():
    state = {
        "evaluation": {"confidence": 0.5},
        "guardrail_result": {"passed": False, "matched_patterns": ["eye contact", "monotone"], "style_leakage_score": 2},
    }
    assert graph._build_review_reason(state) == "low_confidence+guardrail_flag_severe"


def test_reason_escalates_with_consistency_flag_when_already_flagged():
    state = {
        "evaluation": {"confidence": 0.9},
        "guardrail_result": {"passed": False, "matched_patterns": ["eye contact"], "style_leakage_score": 1},
        "consistency": {"consistent": False, "note": "BAD category paired with high valence"},
    }
    assert graph._build_review_reason(state) == "guardrail_flag+consistency_flag"


def test_reason_does_not_escalate_when_nothing_else_flagged():
    """Consistency alone must never produce a reason on its own — it can
    only escalate a route that was already happening for another cause."""
    state = {
        "evaluation": {"confidence": 0.9},
        "guardrail_result": {"passed": True, "matched_patterns": [], "style_leakage_score": 0},
        "consistency": {"consistent": False, "note": "BAD category paired with high valence"},
    }
    assert graph._build_review_reason(state) == ""


def test_route_decision_ignores_consistency_when_otherwise_clean(monkeypatch):
    """route_decision() must not be affected by an inconsistent-but-otherwise-clean
    evaluation — it still only looks at confidence and guardrail passed."""
    state = {
        "evaluation": {"category": "BAD", "valence": "high", "arousal": "medium", "confidence": 0.9},
        "guardrail_result": {"passed": True, "matched_patterns": [], "style_leakage_score": 0},
        "consistency": {"consistent": False, "note": "BAD category paired with high valence"},
    }
    assert graph.route_decision(state) == "select_response"


def test_finalize_logs_consistency_field(monkeypatch, tmp_path):
    log_path = tmp_path / "run_log.jsonl"
    monkeypatch.setattr(graph.config, "RUN_LOG_PATH", str(log_path))
    monkeypatch.setattr(graph.config, "DATA_DIR", str(tmp_path))

    state = {
        "question": "Q",
        "persona": "concise_confident",
        "answer_text": "A",
        "evaluation": {"category": "BAD", "valence": "high", "arousal": "medium", "confidence": 0.9},
        "guardrail_result": {"passed": True, "matched_patterns": [], "style_leakage_score": 0},
        "consistency": {"consistent": False, "note": "BAD category paired with high valence"},
        "final_response": "some reply",
        "routed_to": "finalized",
    }
    graph.finalize(state)

    import json
    with open(log_path) as f:
        record = json.loads(f.readline())
    assert record["consistency"] == {"consistent": False, "note": "BAD category paired with high valence"}


def test_finalize_logs_null_consistency_on_schema_invalid_path(monkeypatch, tmp_path):
    log_path = tmp_path / "run_log.jsonl"
    monkeypatch.setattr(graph.config, "RUN_LOG_PATH", str(log_path))
    monkeypatch.setattr(graph.config, "DATA_DIR", str(tmp_path))

    state = {
        "question": "Q",
        "persona": "concise_confident",
        "answer_text": "A",
        "evaluation": {"category": "good", "valence": "medium", "arousal": "medium", "confidence": 0.9},
        "guardrail_result": None,
        "final_response": None,
        "routed_to": "review_queue",
    }
    graph.finalize(state)

    import json
    with open(log_path) as f:
        record = json.loads(f.readline())
    assert record["consistency"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_graph_consistency.py -v`
Expected: FAIL — `AttributeError: module 'graph' has no attribute '_build_review_reason'` on the reason tests; the log-field tests fail with `KeyError: 'consistency'`.

- [ ] **Step 3: Implement in `graph.py`**

Replace `route_to_review()` (currently `graph.py:107-130`):

```python
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
```

In `finalize()` (currently `graph.py:139-153`), add one key to the record dict:

```python
def finalize(state: ConversationState) -> dict:
    os.makedirs(config.DATA_DIR, exist_ok=True)
    record = {
        "timestamp": time.time(),
        "question": state["question"],
        "persona": state["persona"],
        "answer_text": state["answer_text"],
        "evaluation": state["evaluation"],
        "guardrail_result": state["guardrail_result"],
        "consistency": state.get("consistency"),
        "final_response": state.get("final_response"),
        "routed_to": state["routed_to"],
    }
    with open(config.RUN_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_graph_consistency.py -v`
Expected: 11 passed (4 from Task 2 + 7 new).

Then run the full suite:

Run: `pytest tests/ -v`
Expected: all passing (23 original + 8 Task 1 + 11 Task 2/3 = 42 passed). Pay particular attention to `tests/test_graph_schema_guard.py::test_route_to_review_backward_compatible_without_schema_invalid_key` and `test_route_to_review_uses_schema_invalid_reason` — these exercise `route_to_review()` directly and must still pass unchanged, since `_build_review_reason()` preserves the exact same output strings for the low_confidence/guardrail_flag-only cases those tests cover.

- [ ] **Step 5: Manual end-to-end check**

Run: `python graph.py`
Expected: prints final state as JSON, includes a `"consistency"` field (top-level in the printed dict only if you also add it to the returned state — note `run_conversation()` returns `dict(result)`, and `result` is the graph's final state, which already includes `state["consistency"]` from Task 2's node since LangGraph merges node return dicts into state). Confirm the printed JSON has a `consistency` key with `consistent`/`note`, and check `data/run_log.jsonl`'s last line has a non-missing `"consistency"` key.

- [ ] **Step 6: Commit**

```bash
git add graph.py tests/test_graph_consistency.py
git commit -m "$(cat <<'EOF'
Tier guardrail severity and escalate consistency flags in review reasons

route_to_review()'s reason string now comes from a tag list instead of
a fixed if/elif chain: guardrail_flag splits into guardrail_flag /
guardrail_flag_severe at style_leakage_score >= 2, and a category/VA
consistency mismatch appends consistency_flag — but only when the item
was already routing to review for another reason. route_decision()
itself is unchanged: consistency never triggers a route on its own.
finalize() now logs the consistency result on every run, so silent
(non-escalating) mismatches are still countable later.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Post-Plan Verification

- [ ] Full suite green: `pytest tests/ -v` shows 42 passed, 0 failed.
- [ ] `python graph.py` runs end-to-end without error and its printed JSON includes `consistency`.
- [ ] Re-read `docs/superpowers/specs/2026-08-11-consistency-check-design.md` section by section and confirm each numbered section (1–5) has a corresponding completed task above. (It does: spec §1 → Task 1, §2 → Task 2, §3 → Task 3, §4 → Task 3, §5 → tests folded into each task.)
