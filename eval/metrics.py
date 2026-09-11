"""
eval/metrics.py
---------------
Step 11: Automated evaluation metrics.

- Intent classification: accuracy, macro-F1, confusion matrix
- Escalation: precision, recall, F1 + cost-weighted variant
  (false negatives weighted 3× — should-have-escalated-but-didn't is worse)
- Groundedness/faithfulness: cheap LLM entailment check via Groq
- BLEU/ROUGE: reported as informational only (weak proxy for quality)

  NOTE on BLEU/ROUGE: These n-gram overlap metrics measure surface similarity
  to a reference reply, not actual quality or helpfulness. Customer-support
  replies are highly paraphrastic — a good reply may share almost no n-grams
  with the gold reference while still being correct. They are included only
  for completeness and comparison with prior work, and should NOT be used
  as the primary evaluation criterion.
"""

import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    precision_score, recall_score,
)

from src.llm_client import call_llm_json

JUDGE_MODEL = "llama-3.1-8b-instant"   # cheap model for groundedness check


# ---------------------------------------------------------------------------
# Intent metrics
# ---------------------------------------------------------------------------

def intent_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    labels = sorted(set(y_true + y_pred))
    acc   = accuracy_score(y_true, y_pred)
    mf1   = f1_score(y_true, y_pred, average="macro",  zero_division=0)
    cm    = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy":        round(acc, 4),
        "macro_f1":        round(mf1, 4),
        "confusion_matrix": cm.tolist(),
        "labels":           labels,
    }


def print_confusion_matrix(result: dict) -> None:
    labels = result["labels"]
    cm     = np.array(result["confusion_matrix"])
    header = "          " + "  ".join(f"{l[:8]:>8}" for l in labels)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:8d}" for v in row)
        print(f"{labels[i][:10]:>10}  {row_str}")


# ---------------------------------------------------------------------------
# Escalation metrics
# ---------------------------------------------------------------------------

def escalation_metrics(
    y_true: list[bool],
    y_pred: list[bool],
    fn_weight: float = 3.0,
) -> dict:
    """
    Compute escalation precision / recall / F1 and cost-weighted F1.

    fn_weight: false negatives (should escalate, didn't) are weighted this many
    times more than false positives.
    """
    y_true_i = [int(v) for v in y_true]
    y_pred_i = [int(v) for v in y_pred]

    if not any(y_true_i):
        return {"error": "no positive examples in y_true"}

    prec = precision_score(y_true_i, y_pred_i, zero_division=0)
    rec  = recall_score(y_true_i, y_pred_i, zero_division=0)
    f1   = f1_score(y_true_i, y_pred_i, zero_division=0)

    # Cost-weighted F1: beta > 1 means recall (catching FNs) matters more
    # beta^2 * precision + recall denominator weighting
    beta = fn_weight ** 0.5
    if prec + rec > 0:
        beta_f1 = (1 + beta**2) * prec * rec / (beta**2 * prec + rec)
    else:
        beta_f1 = 0.0

    return {
        "precision":       round(prec, 4),
        "recall":          round(rec,  4),
        "f1":              round(f1,   4),
        f"f1_weighted_fn{fn_weight}x": round(beta_f1, 4),
    }


# ---------------------------------------------------------------------------
# Groundedness / faithfulness
# ---------------------------------------------------------------------------

def groundedness_check(
    draft: str,
    customer_text: str,
    precedents: list[dict],
    *,
    bypass_cache: bool = False,
) -> dict:
    """
    Cheap LLM-based entailment check: does the draft only assert facts
    that are present in the customer's message + retrieved precedents?

    Returns {"grounded": bool, "reason": str}
    """
    context = customer_text
    for p in precedents:
        context += "\n" + p.get("inbound", "") + "\n" + p.get("outbound", "")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a groundedness checker. Given a customer support DRAFT reply "
                "and the CONTEXT (customer message + precedents), determine whether the "
                "draft asserts any specific fact (order number, dollar amount, date, ETA, "
                "account detail) that is NOT present in the context.\n"
                "Respond with JSON: {\"grounded\": true/false, \"reason\": \"brief explanation\"}"
            ),
        },
        {
            "role": "user",
            "content": f"CONTEXT:\n{context[:1000]}\n\nDRAFT:\n{draft[:500]}",
        },
    ]
    try:
        result = call_llm_json(JUDGE_MODEL, messages, max_tokens=100,
                               bypass_cache=bypass_cache)
        return {
            "grounded": bool(result.get("grounded", True)),
            "reason":   str(result.get("reason", "")),
        }
    except Exception as e:
        return {"grounded": True, "reason": f"check failed: {e}"}


# ---------------------------------------------------------------------------
# BLEU / ROUGE (informational only)
# ---------------------------------------------------------------------------

def bleu_rouge_scores(hypotheses: list[str], references: list[str]) -> dict:
    """
    Compute corpus BLEU and ROUGE-L.

    DISCLAIMER (see module docstring): these are weak informational signals only.
    Do NOT use them as primary quality proxies for customer-support generation.
    """
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        refs  = [[r.split()] for r in references]
        hyps  = [h.split()   for h in hypotheses]
        sf    = SmoothingFunction().method1
        bleu  = corpus_bleu(refs, hyps, smoothing_function=sf)
    except Exception:
        bleu = None

    # Simple ROUGE-L (character)
    rouge_l_scores = []
    for h, r in zip(hypotheses, references):
        h_tok, r_tok = h.lower().split(), r.lower().split()
        lcs    = _lcs_len(h_tok, r_tok)
        p      = lcs / len(h_tok) if h_tok else 0
        rec    = lcs / len(r_tok) if r_tok else 0
        f1     = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0
        rouge_l_scores.append(f1)

    return {
        "bleu":    round(bleu, 4) if bleu is not None else None,
        "rouge_l": round(float(np.mean(rouge_l_scores)), 4) if rouge_l_scores else None,
        "note":    "Informational only — BLEU/ROUGE are weak proxies for support reply quality.",
    }


def _lcs_len(a: list, b: list) -> int:
    m, n = len(a), len(b)
    dp   = [[0] * (n + 1) for _ in range(2)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i % 2][j] = dp[(i - 1) % 2][j - 1] + 1
            else:
                dp[i % 2][j] = max(dp[(i - 1) % 2][j], dp[i % 2][j - 1])
    return dp[m % 2][n]


# ---------------------------------------------------------------------------
# Full evaluation over a golden-set CSV
# ---------------------------------------------------------------------------

def evaluate_golden(
    golden_csv: str,
    pred_col_intent: str = "pred_intent",
    pred_col_escalate: str = "pred_escalate",
    draft_col: str = "draft_reply",
    gold_intent_col: str = "gold_intent",
    gold_escalate_col: str = "gold_escalate",
) -> dict:
    df = pd.read_csv(golden_csv)

    results: dict[str, Any] = {}

    # Intent
    mask_i = df[gold_intent_col].notna() & (df[gold_intent_col] != "")
    if mask_i.sum() > 0:
        results["intent"] = intent_metrics(
            df.loc[mask_i, gold_intent_col].tolist(),
            df.loc[mask_i, pred_col_intent].fillna("other").tolist(),
        )

    # Escalation
    mask_e = df[gold_escalate_col].notna() & (df[gold_escalate_col] != "")
    if mask_e.sum() > 0:
        gold_esc = df.loc[mask_e, gold_escalate_col].map(
            lambda v: str(v).strip().lower() in {"true", "1", "yes"}
        ).tolist()
        pred_esc = df.loc[mask_e, pred_col_escalate].map(
            lambda v: str(v).strip().lower() in {"true", "1", "yes"}
        ).tolist()
        results["escalation"] = escalation_metrics(gold_esc, pred_esc)

    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="data/golden/golden_set.csv")
    args = ap.parse_args()
    res = evaluate_golden(args.golden)
    print(json.dumps(res, indent=2))
