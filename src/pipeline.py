"""
src/pipeline.py
---------------
Step 8: Orchestration — chains classify → retrieve → draft → escalate.

handle_message(text, thread_history) ->
  {intent, confidence, draft_reply, escalate, escalate_reason}
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classify  import classify
from src.retrieval import query as retrieve
from src.draft     import draft_reply
from src.escalate  import escalate


def handle_message(
    text: str,
    thread_history: list[dict] | None = None,
    *,
    k_retrieve: int = 3,
    bypass_cache: bool = False,
) -> dict:
    """
    Full pipeline: classify → retrieve → draft → escalate.

    Parameters
    ----------
    text            : The customer's latest message (PII already redacted).
    thread_history  : List of prior turns [{"role": "user"|"assistant", "text": str}, ...]
    k_retrieve      : Number of precedent pairs to retrieve.
    bypass_cache    : Skip LLM cache (for eval-quick).

    Returns
    -------
    {
        "intent":          str,
        "confidence":      float,
        "draft_reply":     str,
        "escalate":        bool,
        "escalate_reason": str,
    }
    """
    if thread_history is None:
        thread_history = []

    # Step 1: Classify
    clf = classify(text, bypass_cache=bypass_cache)
    intent     = clf["intent"]
    confidence = clf["confidence"]

    # Step 2: Retrieve relevant precedents
    precedents = retrieve(text, k=k_retrieve, intent_filter=intent)

    # Step 3: Draft reply
    draft = draft_reply(
        text,
        prior_turns=thread_history,
        retrieved_precedents=precedents,
        intent=intent,
        bypass_cache=bypass_cache,
    )

    # Step 4: Escalation check
    esc = escalate(
        classifier_output=clf,
        draft=draft,
        prior_turns=thread_history,
        customer_text=text,
        precedents=precedents,
    )

    return {
        "intent":          intent,
        "confidence":      confidence,
        "draft_reply":     draft,
        "escalate":        esc["escalate"],
        "escalate_reason": esc["reason"],
    }


if __name__ == "__main__":
    import json, argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?",
                    default="My account was charged twice for the same order!")
    args = ap.parse_args()

    result = handle_message(args.text)
    print(json.dumps(result, indent=2))
