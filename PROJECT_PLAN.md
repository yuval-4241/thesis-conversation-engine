# PROJECT_PLAN.md — Conversation Engine v1

## Overall view

Building the first working version of Phase 2–4 of the thesis pipeline:
**User Answer → Evaluate → Guardrail → Robot Response**, testable end-to-end
on synthetic data before any real participant is involved.

Five pieces, one pipeline:
- **#8 Synthetic Participant** — fake candidate answers, different styles, for testing without real people
- **#2 Evaluator** — judges answer content (GOOD/NEUTRAL/BAD) + picks robot's VA reaction, with rationale
- **#5 No-Masking Guardrail** — scans evaluator's rationale for style-bias language (STAR, eye contact, tone, etc.)
- **#6 Response Bank** — VA-matched robot reply text (9 cells, currently placeholder Hebrew data)
- **#7 Human Review Queue** — low-confidence or flagged items go here instead of auto-shipping

```
persona answer -> evaluator -> guardrail check
                                    |
                low confidence OR flagged?
                    yes -> review queue
                    no  -> response bank -> final robot reply
```

Everything logs to `data/run_log.jsonl` regardless of routing — this becomes
the dataset for kappa/bias analysis in week 2.

## Stack

- Python 3.10+, GitHub, VS Code, Claude Code
- LLM: `llm_client.py` abstraction — swappable backend, currently **mock mode**
  (no real API needed yet; lab server connection pending)
- Orchestration: LangGraph
- Demo: Streamlit
- Furhat robot integration: not yet — this phase is text-only, robot wiring comes later

## Current status

- [x] Day 1 modules built: config, llm_client, personas, evaluator, guardrail, response_bank, question_bank, review_queue
- [x] Mock mode working — all 5 modules tested standalone, chained correctly
- [ ] Lab server connection — blocked, using mock mode meanwhile
- [ ] `graph.py` — LangGraph wiring, not built yet
- [ ] `run_demo.py` — batch runner, not built yet
- [ ] `demo_app.py` — Streamlit demo, not built yet

---

## Week 1 — Build + first advisor demo

| Day | Task | Job-skill flag |
|---|---|---|
| 1 | Repo + env setup. Provider abstraction (`llm_client.py`). Smoke-test all 5 modules standalone. | ⭐ API integration / mocking |
| 2 | Build `graph.py` (LangGraph): full flow wired, with routing to review queue. One manual test run. | ⭐⭐ orchestration |
| 3 | Build `run_demo.py`. Run 30–50 synthetic conversations, log everything. | data pipeline |
| 4 | Read guardrail catches from day 3. Tighten flag list. Build coverage report. | QA/eval skill |
| 5 | **Advisor demo.** Streamlit page: pick question + persona → see answer → evaluator → guardrail → robot response, live. | ⭐ demo/comms skill |

**Advisor demo goal:** prove the architecture runs end-to-end and the
No-Masking guardrail visibly fires — not yet proving the evaluator is accurate.
Say that limitation out loud in the demo.

## Week 2 — Make it defensible

| Day | Task | Job-skill flag |
|---|---|---|
| 6 | Hand-label 20–30 logged answers yourself (golden set). | — |
| 7 | Compute Cohen's kappa (evaluator vs. your labels). | stats/eval skill |
| 8 | Add human-review UI to Streamlit (click to label queued items). | ⭐ HITL system |
| 9 | Bias check: GOOD/NEUTRAL/BAD distribution across personas at matched difficulty. | ⭐ bias/fairness eval |
| 10 | One-page write-up: diagram, guardrail list, kappa number, bias result. Reusable in thesis methods section. | — |

**Week 2 goal:** move from "it runs" to "it's measured" — a first
reliability number and a documented bias check, not just a demo.

---

## How we work together (rules for every session)

- **Keep it short.** Bullets over paragraphs. Not English-native, don't read long text.
- **Every technical thing:** explain what the library/tool is, what it does,
  how it plugs into this project, and flag if it's a job-interview-relevant
  skill (if yes, spend more time on it).
- **If something breaks:** never guess the cause. Give bulleted list of
  possible causes, then bulleted list of fixes, each tagged with effort
  (low/medium/high).
- **End of each work session:** 2 paper summaries, for audit against
  literature (Mishra et al. 2023, Marino et al. 2019, Kumazaki RCT,
  InterViewR, InterviewAI).
- **No Masking is a hard constraint**, not a preference — every design
  decision gets checked against it, including test scaffolding like
  personas and guardrail patterns.
