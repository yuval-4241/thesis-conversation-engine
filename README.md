# Conversation Engine v1

Synthetic-tested evaluator pipeline for a Furhat social robot job-interview
training system. Combines: synthetic participant simulation, LLM response
evaluation, a No-Masking guardrail, a VA-matched response bank, and a
human review queue — into one runnable pipeline.

See `CLAUDE.md` for build instructions and current status.

## Setup

```bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
export LAB_LLM_TOKEN="your_token"   # get from Tomer
```

## Status
Day 1 in progress — see CLAUDE.md for task list.
