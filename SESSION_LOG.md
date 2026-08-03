# Session Log

Running audit of decisions and progress made while working on the Conversation
Engine, session by session. Complements `CLAUDE.md` (which holds the standing
plan/rules) — this file records what actually happened and why.

---

## 2026-07-20 — Day 1

**Git setup**
- Discovered the local git repo was rooted at the home directory (`~/.git`)
  instead of the project folder — only `README.md` was ever tracked, so no
  data was lost. Removed `~/.git`, re-initialized git inside
  `conversation_engine/`, merged with the existing GitHub history, and pushed.
  Repo is now correctly scoped to the project folder only.
- Remote: `github.com/yuval-4241/thesis-conversation-engine`, branch `main`.

**Environment**
- Created `venv/` (Python 3.14.6, Homebrew) and installed
  `requirements.txt` (openai, langgraph, streamlit + deps).

**LLM provider — deviation from plan**
- Lab GPU server (`100.110.96.82`) was busy, so Day 1 step 1 (verify lab
  connection) was skipped for today rather than blocking on it.
- Added an `LLM_PROVIDER` switch (`config.py` / `llm_client.py`) so
  `call_llm()` can target either the lab server (`"lab"`, default) or OpenAI
  directly (`"openai"`). No other module changed — matches the abstraction
  `llm_client.py` was already designed for.
- Today's session runs with `LLM_PROVIDER=openai`, `OPENAI_API_KEY` set in
  `.env` (gitignored, never committed). Model for fast/default/Hebrew all set
  to `gpt-4o-mini` while on this provider.
- Smoke test: `llm_client.call_llm()` round-tripped successfully against
  OpenAI (`pong` test).
- **Next-session action:** when the lab server is free again, switch back by
  unsetting `LLM_PROVIDER` (or setting it to `"lab"`) and doing the real Day 1
  step 1 lab verification per `CLAUDE.md` before relying on lab-hosted models
  (`llama3.1-8b`, `gpt-oss-20b`, `dictalm3-12b`) again — those model names
  don't exist on OpenAI's API.

**Status:** Day 1 steps 1 (lab check) substituted with an OpenAI path; step 2
(deps) done. Steps 3–5 (module smoke tests, `graph.py`, full manual pass) not
started yet.

---

## 2026-07-21 — continued

**Docs**
- Added `PROJECT_PLAN.md` (user-authored — full week 1/2 plan, working
  rules: short/bullets, explain job-skill relevance, end-of-session paper
  summaries). Flagged to user that its "Current status" checklist (mock
  mode, all 5 modules tested) doesn't match reality yet — left as-is
  pending user decision on whether to update it or fold its rules into
  `CLAUDE.md`. Not resolved yet.

**Uncommitted changes (as of this entry):**
- `config.py`, `llm_client.py` — the `LLM_PROVIDER` OpenAI-switch edits from
  2026-07-20, not yet committed.
- `PROJECT_PLAN.md`, `SESSION_LOG.md` — new, untracked.
- Nothing pushed to GitHub since the initial merge commit (`fb7843f`).

**RESUME FROM HERE (next session/restart):**
1. Decide whether to commit the pending changes above (user was asked,
   hadn't answered before restart).
2. Run Day 1 step 3 — smoke-test each module standalone, in order:
   `personas.generate_synthetic_answer` → `evaluator.evaluate` →
   `guardrail.check_rationale` → `response_bank.get`. None of these have
   been run yet this project.
3. Still using `LLM_PROVIDER=openai` (`.env`) since lab server
   (`100.110.96.82`) hasn't been re-checked. Switch back to lab when it's
   free, per the 2026-07-20 note above.
4. After smoke tests pass: build `graph.py`, then one full manual pass,
   per `CLAUDE.md` steps 4–5.
