"""
ab_test.py — A/B test: EVALUATOR_SYSTEM_PROMPT_A vs _B (prompts/evaluator_prompt.py).

Generates N synthetic answers (persona x question_bank session 1), then
scores EACH answer under BOTH prompt variants — same answer_text for A and
B, so any difference in the GOOD/NEUTRAL/BAD distribution is attributable
to the prompt wording, not to persona-generation variance.

Uses a chi-square test of independence (scipy.stats.chi2_contingency) on
the (prompt_variant x category) contingency table (pandas.crosstab) to
check whether variant B's stricter No-Masking self-check instruction
shifts the evaluator's category distribution.
"""

import argparse
import itertools
import json
import os

import pandas as pd
from scipy.stats import chi2_contingency

import config
import personas
import evaluator
from prompts.persona_prompts import PERSONAS
from question_bank import QuestionBank

SESSION_NUMBER = 1
DEFAULT_N = 20

CSV_PATH = os.path.join(config.DATA_DIR, "ab_test_results.csv")
JSON_PATH = os.path.join(config.DATA_DIR, "ab_test_results.json")


def _build_answer_pool(n: int) -> list:
    """N (persona, question) combos from session 1, cycling if n > 36."""
    questions = QuestionBank().session(SESSION_NUMBER)
    persona_keys = list(PERSONAS.keys())
    combos = list(itertools.product(persona_keys, questions))

    pool = []
    for i in range(n):
        persona_key, q = combos[i % len(combos)]
        ans = personas.generate_synthetic_answer(persona_key, q["text"])
        pool.append(
            {
                "answer_id": i,
                "persona": persona_key,
                "question": q["text"],
                "answer_text": ans.answer_text,
            }
        )
        print(f"[{i + 1}/{n}] generated answer: persona={persona_key}")
    return pool


def run_ab_test(n: int = DEFAULT_N) -> pd.DataFrame:
    pool = _build_answer_pool(n)

    rows = []
    for item in pool:
        for variant in ("A", "B"):
            ev = evaluator.evaluate(
                item["question"], item["answer_text"], prompt_variant=variant
            )
            rows.append(
                {
                    "answer_id": item["answer_id"],
                    "prompt_variant": variant,
                    "category": ev.category,
                    "valence": ev.valence,
                    "arousal": ev.arousal,
                    "confidence": ev.confidence,
                }
            )
            print(
                f"answer_id={item['answer_id']} variant={variant} "
                f"category={ev.category} confidence={ev.confidence}"
            )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="A/B test evaluator prompt variants A vs B."
    )
    parser.add_argument(
        "--n", type=int, default=DEFAULT_N, help="Number of synthetic answers (default 20)."
    )
    args = parser.parse_args()

    df = run_ab_test(args.n)

    print("\n=== Contingency table (prompt_variant x category) ===")
    crosstab = pd.crosstab(df["prompt_variant"], df["category"])
    print(crosstab)

    chi2, p, dof, expected = chi2_contingency(crosstab)
    chi2, p, dof = float(chi2), float(p), int(dof)

    print("\n=== Chi-square test of independence ===")
    print(f"chi2 = {chi2:.4f}")
    print(f"p-value = {p:.4f}")
    print(f"dof = {dof}")
    print("Expected frequencies (if the prompt made no difference):")
    print(pd.DataFrame(expected, index=crosstab.index, columns=crosstab.columns))

    significant = p < 0.05
    verdict = (
        "Significant difference (p<0.05)" if significant else "No significant difference detected"
    )
    print(f"\nVerdict: {verdict}")

    small_sample_warning = bool((crosstab.values < 5).any())
    if small_sample_warning:
        print(
            "Caveat: at least one cell has count < 5 — chi-square p-value may "
            "be unreliable at this sample size."
        )

    os.makedirs(config.DATA_DIR, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)

    results = {
        "n": args.n,
        "chi2": chi2,
        "p": p,
        "dof": dof,
        "significant": significant,
        "small_sample_warning": small_sample_warning,
    }
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {CSV_PATH}")
    print(f"Saved: {JSON_PATH}")


if __name__ == "__main__":
    main()
