"""
eval/judge.py
-------------
Step 12: LLM judge for draft quality evaluation.

Scores each golden-set example's draft on a 1-5 scale across:
  1. issue_understanding   — Does the reply show it understood the customer's issue?
  2. correctness           — Is the reply factually grounded (no hallucinated facts)?
  3. brand_voice           — Does it match the brand voice card?
  4. actionability         — Does it offer a concrete next step?
  5. escalation_appropriate — Is the escalation decision (escalate/not) appropriate?

Uses model "openai/gpt-oss-120b" via Groq (separate quota from drafting models).
Falls back to OPENAI_API_KEY / OpenAI if PREFER_OPENAI=1 env var is set.

NOTE: Using a different model family (GPT-lineage) from the Llama drafter/classifier
reduces same-family scoring bias. See decision_log.md entry [12].
"""

import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from tqdm import tqdm

from src.llm_client import call_llm_json, GROQ_BASE_URL

JUDGE_MODEL   = "openai/gpt-oss-120b"
FALLBACK_MODEL = "llama-3.1-8b-instant"   # if judge model unavailable

RUBRIC = """
Score the following customer-support REPLY on each dimension from 1 (very poor) to 5 (excellent):

1. issue_understanding: Does the reply correctly identify and acknowledge the customer's issue?
2. correctness: Does the reply avoid asserting facts not in the customer message or precedents?
3. brand_voice: Does the reply sound empathetic, clear, professional, and action-oriented?
4. actionability: Does the reply offer a concrete next step the customer can take?
5. escalation_appropriate: Is the decision to escalate (or not) appropriate for this situation?

Respond ONLY with JSON:
{
  "issue_understanding": <1-5>,
  "correctness": <1-5>,
  "brand_voice": <1-5>,
  "actionability": <1-5>,
  "escalation_appropriate": <1-5>,
  "overall_comment": "<one sentence>"
}
"""


def judge_example(
    customer_text: str,
    draft_reply: str,
    escalate: bool,
    gold_escalate: bool | None = None,
    *,
    bypass_cache: bool = False,
) -> dict:
    messages = [
        {"role": "system", "content": RUBRIC},
        {
            "role": "user",
            "content": (
                f"CUSTOMER MESSAGE:\n{customer_text[:500]}\n\n"
                f"DRAFT REPLY:\n{draft_reply[:500]}\n\n"
                f"ESCALATED: {escalate}"
                + (f"\nGOLD ESCALATE: {gold_escalate}" if gold_escalate is not None else "")
            ),
        },
    ]

    try:
        result = call_llm_json(
            JUDGE_MODEL, messages,
            temperature=0.1,
            max_tokens=200,
            bypass_cache=bypass_cache,
        )
    except Exception:
        # Fallback to cheaper model if judge unavailable
        result = call_llm_json(
            FALLBACK_MODEL, messages,
            temperature=0.1,
            max_tokens=200,
            bypass_cache=bypass_cache,
        )
        result["_used_fallback"] = True

    dims = ["issue_understanding", "correctness", "brand_voice",
            "actionability", "escalation_appropriate"]
    out  = {}
    for d in dims:
        raw = result.get(d, 3)
        try:
            out[f"judge_{d}"] = max(1, min(5, int(raw)))
        except (ValueError, TypeError):
            out[f"judge_{d}"] = 3
    out["judge_comment"]    = str(result.get("overall_comment", ""))
    out["judge_used_fallback"] = bool(result.get("_used_fallback", False))
    return out


def run_judge(
    golden_csv: str,
    out_csv: str,
    *,
    bypass_cache: bool = False,
) -> None:
    df = pd.read_csv(golden_csv)
    records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Judging"):
        text     = str(row.get("inbound_text", ""))
        draft    = str(row.get("draft_reply",  ""))
        escalate = str(row.get("pred_escalate", "false")).lower() in {"true", "1"}
        gold_esc = None
        if pd.notna(row.get("gold_escalate")) and str(row.get("gold_escalate")) != "":
            gold_esc = str(row["gold_escalate"]).lower() in {"true", "1"}

        scores = judge_example(text, draft, escalate, gold_esc,
                               bypass_cache=bypass_cache)
        records.append(scores)

    scores_df = pd.DataFrame(records)
    out_df    = pd.concat([df.reset_index(drop=True), scores_df], axis=1)
    out_df.to_csv(out_csv, index=False)
    print(f"[judge] Saved judged results to {out_csv}")

    dim_cols = [c for c in scores_df.columns if c.startswith("judge_") and
                not c.endswith(("comment", "fallback"))]
    print("\n=== Judge Score Averages ===")
    for col in dim_cols:
        print(f"  {col}: {scores_df[col].mean():.2f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden",  default="data/golden/golden_set.csv")
    ap.add_argument("--out",     default="data/golden/judged.csv")
    ap.add_argument("--bypass-cache", action="store_true")
    args = ap.parse_args()
    run_judge(args.golden, args.out, bypass_cache=args.bypass_cache)
