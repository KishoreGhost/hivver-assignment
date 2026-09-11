"""
src/ingest.py
-------------
Step 2: Thread reconstruction for the chosen brand.

Joins inbound and outbound tweets into records:
  (inbound_text, outbound_text, prior_turns: list[dict])

prior_turns is capped at the last 4 turns before the current inbound.

Also writes a deterministic subsample to data/sample/brand_threads.parquet
(seed=42, at most MAX_THREADS threads).

Usage:
  python src/ingest.py --csv data/twcs.csv --brand <author_id> [--max-threads 20000]
"""

import argparse, re, sys
from pathlib import Path
from typing import Optional

import pandas as pd

SEED = 42
MAX_THREADS = 20_000
PII_RE = re.compile(r"@\w+")   # redact Twitter handles inline


def redact(text: str) -> str:
    """Replace @handles with [USER] to protect PII before storage."""
    return PII_RE.sub("[USER]", str(text or ""))


def load_df(csv_path: str) -> pd.DataFrame:
    use_cols = ["tweet_id", "author_id", "inbound", "text",
                "in_response_to_tweet_id", "created_at"]
    df = pd.read_csv(
        csv_path,
        usecols=use_cols,
        dtype={c: str for c in use_cols},
        low_memory=False,
    )
    df["inbound"] = df["inbound"].str.strip().str.lower().isin({"true", "1", "yes"})
    df["tweet_id"] = df["tweet_id"].str.strip()
    df["text"] = df["text"].fillna("")
    df["in_response_to_tweet_id"] = df["in_response_to_tweet_id"].fillna("")
    return df


def build_threads(df: pd.DataFrame, brand_id: str) -> list[dict]:
    """
    Reconstruct conversation threads for a single brand.

    Vectorized: uses merge/explode instead of Python row loops.
    Returns one record per (inbound, outbound) pair.
    """
    # Index all rows by tweet_id for fast lookup
    by_id: dict[str, dict] = df.set_index("tweet_id").to_dict("index")

    # Filter brand outbound tweets
    brand_out = df[(~df["inbound"]) & (df["author_id"] == brand_id)].copy()
    brand_out["resp_id"] = brand_out["in_response_to_tweet_id"].str.split(",")
    brand_out = brand_out.explode("resp_id")
    brand_out["resp_id"] = brand_out["resp_id"].str.strip()
    brand_out = brand_out[brand_out["resp_id"] != ""]

    # Join to inbound tweets
    inbound_df = df[df["inbound"]][["tweet_id", "text", "in_response_to_tweet_id"]].copy()
    inbound_df = inbound_df.rename(columns={
        "tweet_id": "in_id",
        "text":     "in_text",
        "in_response_to_tweet_id": "in_prev_id",
    })

    joined = brand_out.merge(inbound_df, left_on="resp_id", right_on="in_id", how="inner")

    records: list[dict] = []
    for _, row in joined.iterrows():
        in_id   = str(row["in_id"])
        out_id  = str(row["tweet_id"])
        in_text = redact(str(row["in_text"]))
        out_text = redact(str(row["text"]))

        # Walk prior turns (up to 4) from the inbound's predecessor
        prior: list[dict] = []
        cursor_id = str(row.get("in_prev_id") or "").strip()
        for _ in range(4):
            if not cursor_id or cursor_id not in by_id:
                break
            prev = by_id[cursor_id]
            prior.insert(0, {
                "role": "user" if prev.get("inbound") else "assistant",
                "text": redact(str(prev.get("text", ""))),
            })
            cursor_id = str(prev.get("in_response_to_tweet_id") or "").strip()

        records.append({
            "inbound_id":    in_id,
            "outbound_id":   out_id,
            "inbound_text":  in_text,
            "outbound_text": out_text,
            "prior_turns":   prior,
            "created_at":    str(row.get("created_at", "")),
        })

    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",         default="data/twcs.csv")
    ap.add_argument("--brand",       required=True, help="Brand author_id")
    ap.add_argument("--max-threads", type=int, default=MAX_THREADS)
    ap.add_argument("--out",         default="data/sample/brand_threads.parquet")
    args = ap.parse_args()

    print(f"[ingest] Loading {args.csv} ...")
    df = load_df(args.csv)
    print(f"[ingest] Building threads for brand={args.brand} ...")
    records = build_threads(df, args.brand)

    print(f"[ingest] Found {len(records):,} (inbound, outbound) pairs")

    # Deterministic subsample
    if len(records) > args.max_threads:
        import random
        rng = random.Random(SEED)
        records = rng.sample(records, args.max_threads)
        print(f"[ingest] Subsampled to {len(records):,} threads (seed={SEED})")

    out_df = pd.DataFrame(records)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(str(out_path), index=False)
    print(f"[ingest] Saved to {out_path}")
    print(out_df.head(3).to_string())


if __name__ == "__main__":
    main()
