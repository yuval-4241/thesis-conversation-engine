# CLAUDE.md — Project Instructions for Claude Code

## Project
Conversation Engine v1 — synthetic-tested evaluator pipeline for a Furhat
social robot interview-training system (thesis project, No-Masking constraint
is a hard requirement, see guardrail.py).

## Stack
- Python 3.10+
- LLM: lab GPU server, OpenAI-compatible API (see llm_client.py)
- Token: `LAB_LLM_TOKEN` env var (never hardcode, never commit)
- Orchestration: LangGraph
- Demo: Streamlit

## Files already built (do not redesign, extend only)
- `config.py` — VA matrix, no-masking patterns, model names
- `prompts/` — all prompt/persona text (persona_prompts.py, evaluator_prompt.py);
  prompts no longer live in config.py or inline in modules
- `llm_client.py` — provider abstraction, ALL llm calls go through `call_llm()`
- `personas.py` — synthetic candidate generator (#8)
- `evaluator.py` — GOOD/NEUTRAL/BAD + VA classifier (#2)
- `guardrail.py` — scans evaluator rationale for masking-coded language (#5)
- `response_bank.py` — loads/queries VA-matched robot responses (#6)
- `question_bank.py` — loads/queries session question set
- `review_queue.py` — logs low-confidence/flagged items for human review (#7)
- `data/response_bank_seed.json`, `data/question_bank_seed.json` — placeholder data

## TODAY — Day 1 tasks, in order

1. **Verify connection to lab server.**
   - Confirm Tailscale is connected (`ping 100.110.96.82`)
   - Set `LAB_LLM_TOKEN` env var
   - Run a raw test call using `gpt-oss-20b` per the lab guide's quick-start
   - If this fails, STOP and report — don't touch other code until this works

2. **Install deps:** `pip install -r requirements.txt`

3. **Smoke-test each existing module individually, in this order:**
   - `llm_client.call_llm(...)` with a trivial prompt
   - `personas.generate_synthetic_answer("concise_confident", "Tell me about yourself")`
   - `evaluator.evaluate(question, answer_text)` on the output above
   - `guardrail.check_rationale(evaluation.rationale)`
   - `response_bank.ResponseBank().get(evaluation.valence, evaluation.arousal)`
   - Print every result. Do not move to graph.py until all five work standalone.

4. **Build `graph.py`** — LangGraph state machine wiring the above into one flow:
   ```
   persona_answer -> evaluate -> guardrail_check -> 
     [flagged OR confidence < config.CONFIDENCE_ROUTE_THRESHOLD]
        -> review_queue.enqueue(...)
     [else]
        -> response_bank.get(...) -> finalize
   ```
   Log every stage's output to `data/run_log.jsonl` regardless of routing.

5. **Run ONE full manual pass** through the graph with a real question from
   `question_bank.py` and one persona. Confirm output end-to-end before
   batching.

## Rules for this session

- **Use `MODEL_FAST` (`llama3.1-8b`) while building/debugging.** Fast feedback
  loop matters more than quality right now. Switch to `MODEL_DEFAULT` only
  for the final Day 1 test pass.
- **Never use 70B+ models today.** Too slow, blocks the shared GPU.
- **Never commit `.env` or hardcode the token.** Check `.gitignore` covers it.
- **Don't modify `config.py`'s NO_MASKING_FLAG_PATTERNS or PERSONAS** without
  flagging it — these are thesis-methodology decisions, not code decisions.

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
- exact next step for Day 2
