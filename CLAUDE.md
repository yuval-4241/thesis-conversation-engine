# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Conversation Engine v1 — synthetic-tested evaluator pipeline for a Furhat
social robot job-interview training system (thesis project). The **No-Masking
constraint is a hard methodology requirement**: the evaluator must judge
answer *content* only, never delivery style (tone, eye contact, STAR
structure, directness) — this is what `guardrail.py` and
`config.NO_MASKING_FLAG_PATTERNS` exist to enforce and audit. Every design
decision gets checked against this constraint, including test scaffolding
like `personas.py` and the guardrail's own pattern list.

Robot integration (Furhat) is now a working prototype, not wired up by
default: `graph.py`'s `select_response()` optionally sends a coordinated
facial-emotion + spoken-reply reaction to a separate Furhat Skill project
(`furhat-emotion-study`, a different repo, Kotlin/JVM) over a local HTTP
bridge. Gated by `config.ROBOT_REACTION_ENABLED` (env-driven, **default
False**) — `pytest`, `run_demo.py`, and `ab_test.py` never attempt the call
unless it's explicitly turned on. See
`docs/superpowers/specs/2026-08-11-robot-reaction-bridge-design.md`.

## Stack

- Python 3.10+
- LLM access via an OpenAI-compatible client (`llm_client.py`) — either the
  lab GPU server or OpenAI directly, switched by `LLM_PROVIDER`
- Orchestration: LangGraph (`graph.py`)
- Data analysis: pandas + scipy (`ab_test.py`)
- Demo: Streamlit (not built yet)

## Commands

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export LAB_LLM_TOKEN="your_token"        # or set LLM_PROVIDER=openai + OPENAI_API_KEY

# Run one conversation through the full graph (prints final state as JSON)
python graph.py

# Batch runner: every persona x every session_1 question, writes results/ + appends to data/run_log.jsonl
python run_demo.py

# A/B test evaluator prompt variants A vs B (scipy chi-square test), writes data/ab_test_results.{csv,json}
python ab_test.py --n 20

# Smoke-test a single module standalone, e.g.:
python -c "import evaluator; print(evaluator.evaluate('Tell me about yourself', 'I have 5 years in QA.'))"
```

```bash
# Run the automated test suite (67 tests: schemas, guardrail, consistency,
# evaluator, graph routing, emotion classifier, robot-reaction bridge)
pytest tests/ -v
```

Beyond the automated suite, end-to-end verification is also done by running
the modules/scripts above and inspecting output (e.g. `python graph.py` for
a real, non-mocked full-pipeline pass).

## Architecture

Five independently-testable pieces wired into one LangGraph state machine
(`graph.py`):

```
persona answer -> evaluator -> schema guard (retry once on failure)
                                    |
                        still invalid after retry?
                    yes -> review queue
                    no  -> guardrail check
                                    |
                low confidence OR flagged?
                    yes -> review queue
                    no  -> response bank -> final robot reply
```

- **`personas.py`** (#8 synthetic participant) — generates fake candidate
  answers in different communication styles via `generate_synthetic_answer()`.
  Doubles as a bias probe: if the evaluator scores non-linear/minimal/blunt
  personas worse than `concise_confident` at equal information content,
  that's evidence of style bias to fix upstream.
- **`evaluator.py`** (#2) — `evaluate()` makes one LLM call that returns
  category (GOOD/NEUTRAL/BAD, content-presence-only) *and* an independent
  valence/arousal "interpretive stance" + rationale, in a single JSON
  response. Category and VA are deliberately decoupled — VA is not a quality
  readout, so an internally "inconsistent" pair (e.g. `BAD` + high valence)
  is allowed by design, not a bug. Takes an optional `retry_hint` string,
  used by `graph.py` to re-prompt with the Pydantic validation error when
  the first response fails the schema guard. Its `valence`/`arousal` still
  feed `EvaluatorOutputSchema` validation, `guardrail.check_consistency()`,
  and `run_log.jsonl` — since `emotion_classifier.py` was introduced, they
  no longer determine the robot's spoken reply or facial reaction, which
  are driven by an independently classified emotion instead. The two
  valence signals can diverge; both are logged, and that divergence is
  intentional, not a bug.
- **`schemas.py`** — `EvaluatorOutputSchema` (Pydantic) validates the
  evaluator's JSON before it's trusted downstream (category/VA enum values,
  `confidence` in [0,1], non-empty rationale). `Category`/`VALevel` are
  hardcoded `Literal`s mirroring `config.py`'s category/VA lists —
  `tests/test_schemas.py` checks the two don't drift apart.
- **`guardrail.py`** (#5 No-Masking guardrail) — `check_rationale()` scans
  the evaluator's own rationale text (not the candidate's answer) for
  style-bias language from `config.NO_MASKING_FLAG_PATTERNS`. Deliberately
  rule-based, not a second LLM call, so it stays fast and fully auditable.
  `GuardrailResult.style_leakage_score` is derived (`len(matched_patterns)`),
  never set independently, so it can't drift out of sync with the match list.
  `check_consistency(category, valence)` flags four suspicious pairs (e.g.
  `BAD`+high valence) where the VA "reaction" looks like it might be a
  disguised quality score — non-blocking on its own, it only escalates the
  review-queue reason when something else already triggered a route there.
- **`response_bank.py`** (#6) — `ResponseBank.get(valence, arousal)` returns
  a random VA-matched robot reply from the 3x3 VA matrix
  (`config.va_cells()`). Backed by `data/response_bank_seed.json`
  (placeholder Hebrew data, 9 cells) — swap the seed file without touching
  the module's API.
- **`emotion_classifier.py`** — `classify_emotion(question, answer_text)`,
  one LLM call, independent of `evaluator.py` (no shared prompt, no shared
  code — sees only the question and answer text). Picks freely among all
  7 of Ekman's core emotions (Anger, Contempt, Disgust, Fear, Happiness,
  Sadness, Surprise) plus an intensity 1-3. `derive_valence()` and
  `derive_arousal()` then turn that pick into a `(valence, arousal)` pair
  for `response_bank.py`'s spoken reply — valence from a fixed
  emotion→valence table, arousal from intensity — so the robot's face
  (the classified emotion) and spoken reply (the derived-VA text) both
  trace back to the same classification call and can never contradict
  each other. This is "emotion-first": the reverse of the earlier
  VA-first `emotion_bank.py` mechanism it replaces. See
  `docs/superpowers/specs/2026-08-16-emotion-first-classification-design.md`.
- **`robot_bridge.py`** — `send_reaction(emotion, intensity, text)` POSTs
  to `furhat-emotion-study`'s local HTTP bridge (a separate Kotlin repo,
  `localhost:8765/react` by default) so the robot shows the matching face
  while speaking the reply. Opt-in via `config.ROBOT_REACTION_ENABLED`
  (default `False`); never raises — a disconnected/absent robot must never
  break the evaluation pipeline for a candidate.
- **`question_bank.py`** — loads the 30-question interview set
  (`data/question_bank_seed.json`), organized into 5 sessions of 6 questions
  each, increasing in difficulty.
- **`review_queue.py`** (#7 HITL) — `enqueue()` writes to
  `data/review_queue.jsonl` when an item is low-confidence
  (`confidence < config.CONFIDENCE_ROUTE_THRESHOLD`) or guardrail-flagged.
  This becomes the golden-set / Cohen's-kappa dataset once a human fills in
  `human_label`.
- **`graph.py`** — wires all of the above into a LangGraph `StateGraph`
  (`ConversationState` TypedDict). Routing logic (`route_decision`,
  `_route_after_evaluate`) lives here, not in the individual modules. After
  `evaluate`, the output is checked against `EvaluatorOutputSchema`; on
  failure the evaluator is called once more with `retry_hint` set, and if
  that retry also fails validation the item is routed straight to the
  review queue (`reason="schema_invalid"`) without ever reaching the
  guardrail. Every run appends one record to `data/run_log.jsonl`
  regardless of routing outcome, via the `finalize` node — this is the
  cumulative dataset for later kappa/bias analysis.
- **`run_demo.py`** — batch driver: every persona x every session-1
  question through `graph.run_conversation()`, writing a per-batch JSONL +
  summary to `results/` (separate from the cumulative `data/run_log.jsonl`).
- **`ab_test.py`** — compares `EVALUATOR_SYSTEM_PROMPT_A` vs `_B`
  (`prompts/evaluator_prompt.py`) by scoring the *same* generated answers
  under both variants, then runs `scipy.stats.chi2_contingency` on the
  (variant x category) `pandas.crosstab` to test whether prompt wording
  shifts the category distribution.
- **`llm_client.py`** — the only file that imports `openai` directly. Every
  other module calls `llm_client.call_llm()`; swapping providers/backends
  only ever touches this file.
- **`config.py`** — central settings: model names, VA matrix helpers,
  `CONFIDENCE_ROUTE_THRESHOLD`, file paths, and `NO_MASKING_FLAG_PATTERNS`.
- **`prompts/`** — all prompt/persona text (`persona_prompts.py`,
  `evaluator_prompt.py`). Prompts do not live inline in modules or in
  `config.py`.

**Data flow contract:** dataclasses (`SyntheticAnswer`, `Evaluation`,
`GuardrailResult`) are produced by each module standalone, then converted to
plain dicts as they enter `ConversationState` in `graph.py` — the graph
nodes never pass dataclass instances between each other, only dicts.

**Methodology-locked content** (do not edit without flagging — see rules
below): `config.NO_MASKING_FLAG_PATTERNS`, `PERSONAS`
(`prompts/persona_prompts.py`), and `EVALUATOR_SYSTEM_PROMPT_A`/`_B`
(`prompts/evaluator_prompt.py`). These encode thesis-methodology decisions,
not implementation details.

See `docs/unified_architecture.md` for a design-level gap analysis against
the broader academic pipeline this repo implements a slice of. The schema
guard (Task 4 in that doc) is now implemented (`schemas.py` + `graph.py`);
other analyzed gaps (e.g. category/VA consistency checks) are not — check
the doc's status table before assuming something described there exists.

## Rules for every session

- **Use `config.MODEL_FAST` while building/debugging** (`llama3.1-8b` on the
  lab server, `gpt-4o-mini` under `LLM_PROVIDER=openai`). Fast feedback
  matters more than quality during iteration. Switch to `MODEL_DEFAULT`
  only for real test passes.
- **Never use 70B+ models.** Too slow, blocks the shared GPU.
- **Never commit `.env` or hardcode `LAB_LLM_TOKEN`/`OPENAI_API_KEY`.**
- **Don't modify `config.NO_MASKING_FLAG_PATTERNS`, `PERSONAS`, or
  `EVALUATOR_SYSTEM_PROMPT_A`/`_B` without flagging it first** — these are
  thesis-methodology decisions, not code decisions.
- Current day-by-day task list and progress live in `PROJECT_PLAN.md`
  (week-by-week plan) and `SESSION_LOG.md` (running log, "resume from here"
  notes) — check those for what's already built vs. up next, rather than
  relying on any task list baked into this file.

## If something doesn't work

Report in this exact format — do not guess at the root cause, list options:

**Possible causes:**
- (bullet each plausible cause)

**Possible fixes:**
- (bullet each fix option, tag with effort: low/medium/high)

## End of session

Stop and report:
- what ran successfully
- what's in `data/run_log.jsonl` (row count, any errors)
- exact next step for next session

## Learning and Interview-Preparation Mode

I am using this project both to build a working solution and to prepare
for technical job interviews. One job requirement I'm preparing for
specifically needs: Python, Pandas, NumPy, SciPy.

Involve me actively in development rather than completing everything
without my participation.

### How to work with me

1. Before an important technical or implementation decision, explain the
   available options in simple English and ask me to choose.

2. When using Python, Pandas, NumPy, SciPy, or another important library:
   - Explicitly say which library is being used.
   - Explain why it fits this task.
   - Explain the main functions/classes/data structures involved.
   - Show the relevant code and walk through how it works.
   - Point out details useful in a technical interview.
   - Mention reasonable alternatives and why we did/didn't choose them.

3. Give me small learning tasks during the project:
   - Ask me to predict what a piece of code will do.
   - Ask me to choose between two implementation approaches.
   - Ask me to explain a function in my own words.
   - Direct me to inspect a specific file/function/output/doc section.
   - Ask me to implement a small, manageable part before doing it yourself.

4. Don't make this overwhelming:
   - One focused question at a time.
   - Simple, clear English.
   - Break complex implementations into small steps.
   - Separate essential knowledge from optional advanced details.

5. After implementing an important feature, give a short learning summary:
   - What we implemented.
   - Which Python tools/libraries we used, and why.
   - What I should understand for an interview.
   - One possible interview question about this implementation.

6. Don't ask my input on trivial choices (formatting, obvious variable
   names, routine boilerplate). Ask about decisions with educational or
   architectural value.

Balance active learning with steady progress toward a correct, clean,
tested, maintainable implementation.
