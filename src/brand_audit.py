"""
src/brand_audit.py
------------------
Step 1: Rank brands in the Twitter customer-support dataset.

Metrics per brand:
  - tweet_volume   : total inbound messages
  - reply_rate     : fraction of inbound that have a same-thread brand reply
  - lang_purity    : fraction of sampled tweets detected as English (langdetect,
                     sample ~500 per brand for speed)

Outputs a ranked table and writes report/decision_log.md with the brand choice.

Usage:
  python src/brand_audit.py --csv data/twcs.csv --top 20
"""

import argparse, random, re, sys, warnings
from pathlib import Path

import pandas as pd
from langdetect import detect, LangDetectException

SEED = 42
random.seed(SEED)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HANDLE_RE = re.compile(r"@\w+")

def _detect_lang(text: str) -> str:
    try:
        return detect(str(text))
    except LangDetectException:
        return "unknown"


def _lang_purity(texts: list[str], n: int = 500) -> float:
    """Return fraction of sampled texts detected as English."""
    sample = random.sample(texts, min(n, len(texts)))
    english = sum(1 for t in sample if _detect_lang(t) == "en")
    return english / len(sample) if sample else 0.0


def _extract_brand(author_id: str, inbound: bool) -> str:
    """Brand identifier is the author_id on outbound (brand) tweets."""
    return str(author_id)


def load_and_verify(csv_path: str) -> pd.DataFrame:
    """
    Fast CSV load: read only the 4 columns needed for audit.
    Uses native dtypes (avoids dtype=str overhead on 3M rows).
    """
    # Peek at header first
    header_df = pd.read_csv(csv_path, nrows=0)
    cols = list(header_df.columns)
    print(f"[brand_audit] Columns: {cols}")

    REQUIRED = {
        "tweet_id", "author_id", "inbound",
        "text", "in_response_to_tweet_id",
    }
    missing = REQUIRED - set(cols)
    if missing:
        sys.exit(f"[brand_audit] ERROR: missing columns: {missing}")

    # Read only the columns we need with explicit dtypes — much faster
    use_cols = ["tweet_id", "author_id", "inbound", "text",
                "in_response_to_tweet_id"]
    df = pd.read_csv(
        csv_path,
        usecols=use_cols,
        dtype={
            "tweet_id":                str,
            "author_id":               str,
            "inbound":                 str,
            "text":                    str,
            "in_response_to_tweet_id": str,
        },
        low_memory=False,
    )
    print(f"[brand_audit] Loaded {len(df):,} rows")

    # Normalise boolean
    df["inbound"] = df["inbound"].str.strip().str.lower().isin(
        {"true", "1", "yes"}
    )
    df["tweet_id"] = df["tweet_id"].str.strip()
    df["text"]     = df["text"].fillna("")
    df["in_response_to_tweet_id"] = df["in_response_to_tweet_id"].fillna("")
    return df


# ---------------------------------------------------------------------------
# Main audit logic
# ---------------------------------------------------------------------------

def run_audit(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Vectorized brand audit — no Python row loops over the full dataset."""
    outbound = df[~df["inbound"]].copy()
    inbound  = df[ df["inbound"]].copy()

    # Build mapping: inbound_tweet_id → brand author_id (vectorized explode)
    out = outbound[["author_id", "in_response_to_tweet_id"]].copy()
    out["in_response_to_tweet_id"] = out["in_response_to_tweet_id"].fillna("")
    # Explode comma-separated inbound IDs (rare but possible)
    out = out.assign(
        resp_id=out["in_response_to_tweet_id"].str.split(",")
    ).explode("resp_id")
    out["resp_id"] = out["resp_id"].str.strip()
    out = out[out["resp_id"] != ""]

    # Join: for each inbound tweet, which brand replied?
    merged = inbound[["tweet_id", "text", "author_id"]].merge(
        out[["resp_id", "author_id"]].rename(
            columns={"author_id": "brand_id", "resp_id": "tweet_id"}
        ),
        on="tweet_id", how="left",
    )

    # Per-brand volume and reply rate
    merged["replied"] = merged["brand_id"].notna()
    # Fill brand_id for "no reply" rows with a dummy so we can still count volume
    merged["brand_id"] = merged["brand_id"].fillna("__no_reply__")

    brand_stats = (
        merged.groupby("brand_id")
        .agg(volume=("tweet_id", "count"), replied=("replied", "sum"))
        .reset_index()
    )
    brand_stats = brand_stats[
        (brand_stats["brand_id"] != "__no_reply__") &
        (brand_stats["volume"] >= 200)
    ]
    brand_stats["reply_rate"] = brand_stats["replied"] / brand_stats["volume"]

    # Language purity: sample up to 500 texts per brand
    text_map = merged.groupby("brand_id")["text"].apply(list).to_dict()

    lang_purities = {}
    for brand in brand_stats["brand_id"]:
        texts = text_map.get(brand, [])
        lang_purities[brand] = _lang_purity(texts)
    brand_stats["lang_purity"] = brand_stats["brand_id"].map(lang_purities).round(3)
    brand_stats["reply_rate"]  = brand_stats["reply_rate"].round(3)
    brand_stats["score"]       = (
        brand_stats["volume"] * brand_stats["reply_rate"] * brand_stats["lang_purity"]
    ).round(1)

    result = (
        brand_stats.rename(columns={"brand_id": "brand"})
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
        .head(top_n)
    )
    return result


def print_table(df: pd.DataFrame) -> None:
    print("\n=== Brand Ranking ===")
    print(df.to_string(index=True))
    print()


def choose_brand(ranked: pd.DataFrame, lang_threshold: float = 0.85) -> str:
    """Pick highest-score brand with lang_purity >= threshold."""
    for _, row in ranked.iterrows():
        if row["lang_purity"] >= lang_threshold:
            return str(row["brand"])
    # fallback: just pick the top
    return str(ranked.iloc[0]["brand"])


def write_decision_log(ranked: pd.DataFrame, chosen: str) -> None:
    log_dir = Path("report")
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "decision_log.md"

    top5 = ranked.head(5).to_markdown(index=False)
    entry = f"""
## [1] Brand Choice

**Dataset**: thoughtvector/customer-support-on-twitter  
**Ranking metric**: `score = volume * reply_rate * lang_purity`  
Rewards high-volume brands that actually respond AND do so mostly in English.

Top-5 candidates:

{top5}

**Chosen brand author_id**: `{chosen}`  
**Reasoning**: Highest composite score with lang_purity >= 0.85, meaning the
overwhelming majority of its tweets are English — minimising noise from
multilingual clusters in step 3.

---
"""
    if path.exists():
        existing = path.read_text()
    else:
        existing = "# Decision Log\n"

    if "## [1] Brand Choice" not in existing:
        path.write_text(existing + entry)
    print(f"[brand_audit] Decision log written to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",  default="data/twcs.csv", help="Path to raw CSV")
    ap.add_argument("--top",  type=int, default=20,    help="Show top N brands")
    args = ap.parse_args()

    df     = load_and_verify(args.csv)
    ranked = run_audit(df, top_n=args.top)
    print_table(ranked)

    chosen = choose_brand(ranked)
    print(f"[brand_audit] Chosen brand: {chosen}")
    write_decision_log(ranked, chosen)


if __name__ == "__main__":
    main()
