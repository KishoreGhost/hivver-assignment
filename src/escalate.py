"""
src/escalate.py
---------------
Step 7: Rule-based escalation gate.

Returns {escalate: bool, reason: str} given:
  - classifier output (intent + confidence)
  - draft reply
  - thread state (prior_turns)
  - input facts (customer_text + precedents)

POLICY NOTE: There is no ground-truth escalation label in the Twitter
customer-support dataset. This escalation policy is entirely authored by us
based on reasonable customer-service heuristics — it is NOT learned from
real escalation events. See decision_log.md entry [7] for details.

Escalation conditions (configurable thresholds):
  1. Classifier confidence below CONF_THRESHOLD.
  2. Intent is in HIGH_RISK_INTENTS (user-supplied after step 3).
  3. Thread already has >= UNRESOLVED_TURN_THRESHOLD unresolved turns.
  4. Draft asserts a fact not grounded in customer_text + precedents
     (groundedness check, shared with metrics.py).
"""

import re
from typing import Any

# ---------------------------------------------------------------------------
# Configurable thresholds
# ---------------------------------------------------------------------------
CONF_THRESHOLD          = 0.55   # escalate if confidence < this
UNRESOLVED_TURN_THRESHOLD = 3    # escalate if >= this many unresolved turns

# ---------------------------------------------------------------------------
# TODO: After step 3 cluster naming, fill in the intents that are high-risk.
# The user will supply this list after reviewing the taxonomy.
# High-risk intents are those that may require legal, financial, or safety attention.
# ---------------------------------------------------------------------------
HIGH_RISK_INTENTS: set[str] = {
    "device_os_glitch",
    "software_update_ios",
    "account_icloud_store",
    "macos_hardware_support",
    "battery_power_drain",
}
# ---------------------------------------------------------------------------

# Patterns that indicate specific facts were asserted
_FACT_PATTERNS = [
    re.compile(r"\$[\d,]+"),                                # dollar amounts
    re.compile(r"\b\d{4,}\b"),                             # long numbers (order IDs)
    re.compile(r"\d{1,2}/\d{1,2}(?:/\d{2,4})?"),          # dates: MM/DD or MM/DD/YYYY
    re.compile(r"\b\d+ (?:days?|hours?|weeks?)\b"),        # ETAs
]


def _extract_asserted_facts(text: str) -> list[str]:
    """Extract specific facts mentioned in a text (numbers, dates, amounts)."""
    found = []
    for pat in _FACT_PATTERNS:
        found.extend(pat.findall(text))
    return found


def _is_grounded(draft: str, customer_text: str, precedents: list[dict]) -> bool:
    """
    Check whether every specific fact in the draft is grounded in the
    customer_text or the retrieved precedents.

    Implementation: extract fact-like tokens from the draft, then check
    if each appears verbatim in the allowed context.
    """
    draft_facts = _extract_asserted_facts(draft)
    if not draft_facts:
        return True

    allowed_text = customer_text
    for p in precedents:
        allowed_text += " " + p.get("inbound", "") + " " + p.get("outbound", "")

    for fact in draft_facts:
        if fact not in allowed_text:
            return False
    return True


def escalate(
    classifier_output: dict,
    draft: str,
    prior_turns: list[dict],
    customer_text: str = "",
    precedents: list[dict] | None = None,
    *,
    conf_threshold: float = CONF_THRESHOLD,
    unresolved_threshold: int = UNRESOLVED_TURN_THRESHOLD,
    high_risk_intents: set[str] | None = None,
) -> dict:
    """
    Evaluate whether this interaction should be escalated.

    Returns {"escalate": bool, "reason": str}
    """
    if precedents is None:
        precedents = []
    if high_risk_intents is None:
        high_risk_intents = HIGH_RISK_INTENTS

    intent     = classifier_output.get("intent", "other")
    confidence = float(classifier_output.get("confidence", 0.0))

    # --- Rule 1: Low classifier confidence ---
    if confidence < conf_threshold:
        return {
            "escalate": True,
            "reason":   f"Classifier confidence {confidence:.2f} < threshold {conf_threshold:.2f}",
        }

    # --- Rule 2: High-risk intent ---
    if intent in high_risk_intents:
        return {
            "escalate": True,
            "reason":   f"Intent '{intent}' is tagged as high-risk",
        }

    # --- Rule 3: Long unresolved thread ---
    # Count turns where role=user (customer) without closure signals
    n_unresolved = sum(
        1 for t in prior_turns
        if t.get("role") == "user"
        and not any(kw in t.get("text", "").lower()
                    for kw in ("thank", "thanks", "resolved", "sorted", "fixed"))
    )
    if n_unresolved >= unresolved_threshold:
        return {
            "escalate": True,
            "reason":   f"Thread has {n_unresolved} unresolved customer turns (>= {unresolved_threshold})",
        }

    # --- Rule 4: Draft not grounded ---
    if not _is_grounded(draft, customer_text, precedents):
        return {
            "escalate": True,
            "reason":   "Draft asserts a specific fact not present in customer message or precedents",
        }

    return {"escalate": False, "reason": "All escalation rules passed"}


if __name__ == "__main__":
    # Quick smoke test
    result = escalate(
        classifier_output={"intent": "order_status", "confidence": 0.3},
        draft="Your order will arrive on 12/25.",
        prior_turns=[],
        customer_text="Where is my package?",
        precedents=[],
    )
    print(result)
