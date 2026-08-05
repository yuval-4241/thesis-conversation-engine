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
