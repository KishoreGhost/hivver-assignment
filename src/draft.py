"""
src/draft.py
------------
Step 6: Reply drafter using Groq llama-3.3-70b-versatile.

Drafts a reply given:
  - customer message
  - prior conversation turns
  - top-k retrieved resolved historical (inbound, outbound) pairs

HARD CONSTRAINT enforced in prompt: never state a specific fact
(order number, dollar amount, date/ETA) not present in the customer's
message or the retrieved precedent.

Brand voice card: fill in BRAND_VOICE below after reviewing ~30 sample
outbound replies printed by src/ingest.py.

TODO: After the user provides 3-4 brand voice adjectives + do/don't examples,
update BRAND_VOICE dict below and regenerate the system prompt.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_client import call_llm
from src.retrieval   import query as retrieve

MODEL = "llama-3.3-70b-versatile"

# ---------------------------------------------------------------------------
# TODO: Fill this in after reviewing sample outbound replies.
# The user will supply 3-4 adjectives and do/don't examples.
# ---------------------------------------------------------------------------
BRAND_VOICE = {
    "adjectives": ["empathetic", "professional", "courteous", "solution-oriented"],
    "dos": [
        "If the customer expresses frustration or distress, proactively acknowledge it with professional empathy and an apology (e.g., 'This is certainly not the experience we want you to have; we are here to help get this sorted out').",
        "Tailor the troubleshooting steps directly to the customer's specific device, OS version, or feature concern.",
        "Provide clear, actionable next steps or invite the user to Direct Message (DM) with details to troubleshoot safely.",
        "Maintain a supportive, calm, and reassuring tone at all times.",
    ],
    "donts": [
        "HARD CONSTRAINT: NEVER invent or assume facts, dates, dollar amounts, order numbers, or ETAs not in the message or precedents.",
        "Never dismiss, argue with, or ignore a frustrated customer's feelings.",
        "Never use canned robotic boilerplate or repeat the customer's text verbatim.",
    ],
}
# ---------------------------------------------------------------------------


def _brand_voice_card() -> str:
    adj  = ", ".join(BRAND_VOICE["adjectives"])
    dos  = "\n".join(f"  - DO: {d}"   for d in BRAND_VOICE["dos"])
    dnts = "\n".join(f"  - DON'T: {d}" for d in BRAND_VOICE["donts"])
    return f"Brand voice: {adj}\n{dos}\n{dnts}"


def _format_precedents(precedents: list[dict]) -> str:
    if not precedents:
        return "No precedents found."
    lines = []
    for i, p in enumerate(precedents, 1):
        lines.append(f"[Precedent {i}]")
        lines.append(f"  Customer: {p['inbound'][:200]}")
        lines.append(f"  Agent:    {p['outbound'][:200]}")
    return "\n".join(lines)


def _format_prior_turns(prior_turns: list[dict]) -> str:
    if not prior_turns:
        return ""
    lines = []
    for t in prior_turns:
        role = "Customer" if t.get("role") == "user" else "Agent"
        lines.append(f"{role}: {t.get('text','')[:200]}")
    return "\n".join(lines)


SYSTEM_PROMPT_TEMPLATE = """\
You are a customer support agent.

{voice_card}

HARD CONSTRAINT: You must NEVER state any specific fact — order number, dollar
amount, date, ETA, account detail — that is NOT explicitly present in the
customer's message or the precedent examples below. If you don't have the fact,
acknowledge the issue and direct the customer to the appropriate channel.

Historical resolved precedents (use these for tone and approach, not as facts):
{precedents}
"""


def draft_reply(
    customer_text: str,
    prior_turns: list[dict] | None = None,
    retrieved_precedents: list[dict] | None = None,
    *,
    k_retrieve: int = 3,
    intent: str | None = None,
    bypass_cache: bool = False,
) -> str:
    """
    Draft a support reply.

    If retrieved_precedents is not supplied, runs a live retrieval query.
    """
    if retrieved_precedents is None:
        retrieved_precedents = retrieve(customer_text, k=k_retrieve,
                                        intent_filter=intent)

    system = SYSTEM_PROMPT_TEMPLATE.format(
        voice_card=_brand_voice_card(),
        precedents=_format_precedents(retrieved_precedents),
    )

    messages = [{"role": "system", "content": system}]

    # Add prior turns
    prior_str = _format_prior_turns(prior_turns or [])
    if prior_str:
        messages.append({
            "role": "user",
            "content": f"Previous conversation:\n{prior_str}",
        })
        messages.append({
            "role": "assistant",
            "content": "Understood, I'll take the conversation history into account.",
        })

    messages.append({"role": "user", "content": customer_text})

    reply = call_llm(
        MODEL, messages,
        temperature=0.4,
        max_tokens=300,
        bypass_cache=bypass_cache,
    )
    return reply.strip()


def print_sample_outbound(threads_parquet: str = "data/sample/brand_threads.parquet",
                           n: int = 30) -> None:
    """Print n sample outbound replies to help author the brand voice card."""
    import pandas as pd
    df = pd.read_parquet(threads_parquet)
    sample = df["outbound_text"].dropna().sample(n=min(n, len(df)), random_state=42)
    print(f"\n=== Sample outbound replies (n={len(sample)}) ===")
    for i, text in enumerate(sample, 1):
        print(f"\n[{i}] {text[:300]}")
    print("\n=== End of sample ===")
    print("Review the above and provide 3-4 brand voice adjectives + do/don't examples.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-outbound", action="store_true",
                    help="Print sample outbound replies for brand voice card")
    ap.add_argument("--text", type=str,
                    default="My order hasn't arrived and it's been 2 weeks!")
    args = ap.parse_args()

    if args.sample_outbound:
        print_sample_outbound()
    else:
        reply = draft_reply(args.text)
        print(f"\nDraft reply:\n{reply}")
