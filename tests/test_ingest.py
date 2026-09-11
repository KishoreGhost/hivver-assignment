"""
tests/test_ingest.py
--------------------
Tests for thread reconstruction (src/ingest.py).

Uses hand-built fixture rows to verify:
  - inbound/outbound pairing
  - prior_turn chain walking (up to 4)
  - PII redaction (@handle → [USER])
  - empty in_response_to_tweet_id handled gracefully
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest

from src.ingest import build_threads, redact


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, dtype=str)
    df["inbound"] = df["inbound"].map(
        lambda v: str(v).strip().lower() in {"true", "1", "yes"}
    )
    df["tweet_id"] = df["tweet_id"].str.strip()
    df["text"] = df["text"].fillna("")
    return df


BRAND_ID = "brand_co"

SIMPLE_ROWS = [
    # A simple 2-turn thread: customer asks (inbound), brand replies (outbound)
    {"tweet_id": "1", "author_id": "customer_a", "inbound": "true",
     "text": "Hello @brand_co my order is late!", "created_at": "2023-01-01",
     "response_tweet_id": "", "in_response_to_tweet_id": ""},
    {"tweet_id": "2", "author_id": BRAND_ID, "inbound": "false",
     "text": "Hi @customer_a, sorry to hear that!", "created_at": "2023-01-01",
     "response_tweet_id": "", "in_response_to_tweet_id": "1"},
]

MULTI_TURN_ROWS = [
    # 4-turn thread: 2 customer messages, 2 brand replies
    {"tweet_id": "10", "author_id": "cust_b",  "inbound": "true",
     "text": "I have a problem", "created_at": "2023-01-02",
     "response_tweet_id": "", "in_response_to_tweet_id": ""},
    {"tweet_id": "11", "author_id": BRAND_ID,  "inbound": "false",
     "text": "What is your problem?", "created_at": "2023-01-02",
     "response_tweet_id": "", "in_response_to_tweet_id": "10"},
    {"tweet_id": "12", "author_id": "cust_b",  "inbound": "true",
     "text": "My order is wrong", "created_at": "2023-01-02",
     "response_tweet_id": "", "in_response_to_tweet_id": "11"},
    {"tweet_id": "13", "author_id": BRAND_ID,  "inbound": "false",
     "text": "We will fix it!", "created_at": "2023-01-02",
     "response_tweet_id": "", "in_response_to_tweet_id": "12"},
]

NO_REPLY_ROWS = [
    # Inbound message with no outbound reply
    {"tweet_id": "20", "author_id": "cust_c", "inbound": "true",
     "text": "Anyone there?", "created_at": "2023-01-03",
     "response_tweet_id": "", "in_response_to_tweet_id": ""},
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRedact:
    def test_handle_replaced(self):
        assert redact("Hi @alice and @bob") == "Hi [USER] and [USER]"

    def test_no_handle(self):
        assert redact("plain text") == "plain text"

    def test_empty_string(self):
        assert redact("") == ""


class TestBuildThreads:
    def test_simple_pair(self):
        df = make_df(SIMPLE_ROWS)
        records = build_threads(df, BRAND_ID)
        assert len(records) == 1
        rec = records[0]
        assert rec["inbound_id"]  == "1"
        assert rec["outbound_id"] == "2"
        # PII should be redacted
        assert "@customer_a" not in rec["outbound_text"]
        assert "[USER]" in rec["outbound_text"]

    def test_inbound_text_redacted(self):
        df = make_df(SIMPLE_ROWS)
        records = build_threads(df, BRAND_ID)
        # "@brand_co" in inbound text should be redacted
        assert "@brand_co" not in records[0]["inbound_text"]

    def test_multi_turn_prior(self):
        df = make_df(MULTI_TURN_ROWS)
        records = build_threads(df, BRAND_ID)
        # Two brand outbound tweets (11, 13)
        assert len(records) == 2

        # Record for tweet 13 (last brand reply) should have prior_turns
        r13 = next(r for r in records if r["outbound_id"] == "13")
        # prior_turns should include at least turn from tweet 10 and/or 11
        assert len(r13["prior_turns"]) > 0

    def test_prior_turns_capped_at_4(self):
        # Build a long 10-turn thread and verify cap
        rows = []
        for i in range(10):
            role = "true" if i % 2 == 0 else "false"
            author = "cust_d" if i % 2 == 0 else BRAND_ID
            rows.append({
                "tweet_id": str(100 + i),
                "author_id": author,
                "inbound": role,
                "text": f"Turn {i}",
                "created_at": "2023-01-04",
                "response_tweet_id": "",
                "in_response_to_tweet_id": str(100 + i - 1) if i > 0 else "",
            })
        df  = make_df(rows)
        recs = build_threads(df, BRAND_ID)
        for r in recs:
            assert len(r["prior_turns"]) <= 4, \
                f"prior_turns exceeded 4: {len(r['prior_turns'])}"

    def test_no_outbound_reply(self):
        df = make_df(NO_REPLY_ROWS)
        records = build_threads(df, BRAND_ID)
        assert records == []

    def test_wrong_brand_ignored(self):
        df = make_df(SIMPLE_ROWS)
        records = build_threads(df, "other_brand")
        assert records == []
