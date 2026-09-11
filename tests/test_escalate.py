"""
tests/test_escalate.py
----------------------
Tests for all branches of the escalation gate (src/escalate.py).

Each test drives one or more escalation conditions and verifies
the correct {escalate, reason} output.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.escalate import escalate, _is_grounded


CLF_HIGH_CONF = {"intent": "order_status", "confidence": 0.9}
CLF_LOW_CONF  = {"intent": "order_status", "confidence": 0.3}
DRAFT_CLEAN   = "We're looking into your order and will update you shortly."
DRAFT_FACT    = "Your order #99999 will arrive on 12/25 and cost $50."
PRIOR_CLEAN   = []
PRIOR_LONG    = [
    {"role": "user",      "text": "This is broken"},
    {"role": "assistant", "text": "Let us check"},
    {"role": "user",      "text": "Still broken"},
    {"role": "assistant", "text": "We are looking"},
    {"role": "user",      "text": "STILL broken"},  # 3rd unresolved
]


class TestEscalateRule1_LowConfidence:
    def test_low_confidence_triggers(self):
        result = escalate(CLF_LOW_CONF, DRAFT_CLEAN, PRIOR_CLEAN)
        assert result["escalate"] is True
        assert "confidence" in result["reason"]

    def test_high_confidence_passes(self):
        result = escalate(CLF_HIGH_CONF, DRAFT_CLEAN, PRIOR_CLEAN)
        assert result["escalate"] is False

    def test_custom_threshold(self):
        clf = {"intent": "order_status", "confidence": 0.6}
        # Should not escalate with default 0.55 threshold
        r1 = escalate(clf, DRAFT_CLEAN, PRIOR_CLEAN, conf_threshold=0.55)
        assert r1["escalate"] is False
        # Should escalate with higher threshold
        r2 = escalate(clf, DRAFT_CLEAN, PRIOR_CLEAN, conf_threshold=0.7)
        assert r2["escalate"] is True


class TestEscalateRule2_HighRisk:
    def test_high_risk_intent_triggers(self):
        clf = {"intent": "fraud_report", "confidence": 0.95}
        result = escalate(clf, DRAFT_CLEAN, PRIOR_CLEAN,
                          high_risk_intents={"fraud_report"})
        assert result["escalate"] is True
        assert "high-risk" in result["reason"]

    def test_non_high_risk_passes(self):
        result = escalate(CLF_HIGH_CONF, DRAFT_CLEAN, PRIOR_CLEAN,
                          high_risk_intents={"fraud_report"})
        assert result["escalate"] is False


class TestEscalateRule3_LongThread:
    def test_long_unresolved_thread_triggers(self):
        result = escalate(CLF_HIGH_CONF, DRAFT_CLEAN, PRIOR_LONG,
                          unresolved_threshold=3)
        assert result["escalate"] is True
        assert "unresolved" in result["reason"]

    def test_short_thread_passes(self):
        prior = [{"role": "user", "text": "Thanks!"}]
        result = escalate(CLF_HIGH_CONF, DRAFT_CLEAN, prior,
                          unresolved_threshold=3)
        assert result["escalate"] is False

    def test_resolved_thread_passes(self):
        prior = [
            {"role": "user", "text": "My order is late"},
            {"role": "assistant", "text": "We'll check"},
            {"role": "user", "text": "Thanks, it arrived"},   # closure
        ]
        result = escalate(CLF_HIGH_CONF, DRAFT_CLEAN, prior,
                          unresolved_threshold=3)
        assert result["escalate"] is False


class TestEscalateRule4_Groundedness:
    def test_hallucinated_fact_triggers(self):
        result = escalate(
            CLF_HIGH_CONF, DRAFT_FACT, PRIOR_CLEAN,
            customer_text="Where is my order?",
            precedents=[],
        )
        assert result["escalate"] is True
        assert "fact" in result["reason"].lower()

    def test_grounded_draft_passes(self):
        result = escalate(
            CLF_HIGH_CONF, DRAFT_CLEAN, PRIOR_CLEAN,
            customer_text="Where is my order?",
            precedents=[],
        )
        assert result["escalate"] is False

    def test_fact_in_customer_message_passes(self):
        # Fact appears in customer text → grounded
        cust_text = "I was charged $50 for order #99999"
        draft = "Your charge of $50 for order #99999 is being investigated."
        result = escalate(
            CLF_HIGH_CONF, draft, PRIOR_CLEAN,
            customer_text=cust_text,
            precedents=[],
        )
        assert result["escalate"] is False


class TestGroundednessHelper:
    def test_no_facts_always_grounded(self):
        assert _is_grounded("We'll look into it.", "help me", []) is True

    def test_fact_in_precedent_grounded(self):
        prec = [{"inbound": "order #12345", "outbound": "found it"}]
        assert _is_grounded("Your order #12345 is on the way.", "help", prec) is True

    def test_fact_not_in_context_ungrounded(self):
        assert _is_grounded("Order arrives 12/25.", "help me", []) is False
