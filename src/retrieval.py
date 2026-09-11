"""
src/retrieval.py
----------------
Step 5: Resolved-pair retrieval index.

Heuristic for "resolved" pairs:
  A pair is considered resolved if the customer sent NO follow-up, OR
  the next customer message contains closure keywords (thank, thanks, resolved,
  solved, fixed, great, awesome, appreciate, perfect, sorted).
  A secondary cheap pass: if the follow-up is ambiguous, skip it.
  Documented explicitly here: this is a keyword heuristic, not a learned model.

Builds a cosine-similarity index over embeddings of resolved inbound messages.
Query returns top-k (inbound, outbound) pairs, filtered by intent when possible.
"""

import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"
SEED        = 42

CLOSURE_KEYWORDS = {
    "thank", "thanks", "thx", "ty", "resolved", "solve", "solved",
    "fixed", "fix", "great", "awesome", "appreciate", "perfect",
    "sorted", "wonderful", "brilliant", "cheers", "helpful",
}

INDEX_PATH  = Path("data/sample/retrieval_index.npz")
PAIRS_PATH  = Path("data/sample/resolved_pairs.parquet")


# ---------------------------------------------------------------------------
# Heuristic resolution detection
# ---------------------------------------------------------------------------

def _has_closure_token(text: str) -> bool:
    tokens = set(text.lower().split())
    return bool(tokens & CLOSURE_KEYWORDS)


def _is_continued_complaint(text: str) -> bool:
    """Heuristic: text looks like a continuation complaint if short + no closure."""
    words = text.lower().split()
    if len(words) > 3 and not _has_closure_token(text):
        return True
    return False


def find_resolved_pairs(threads_parquet: str) -> pd.DataFrame:
    """
    From the thread records, identify resolved pairs.

    A pair is "resolved" if:
      1. No follow-up inbound message exists in the data (thread ended), OR
      2. The follow-up message contains closure keywords.

    DESIGN NOTE: We have no ground-truth "resolved" label in this dataset.
    The heuristic above is authored by us and documented in decision_log.md.
    """
    df = pd.read_parquet(threads_parquet)
    print(f"[retrieval] Total pairs: {len(df):,}")

    resolved_mask = []
    for _, row in df.iterrows():
        prior_val = row.get("prior_turns")
        if prior_val is None:
            prior = []
        elif isinstance(prior_val, (list, np.ndarray)):
            prior = list(prior_val)
        elif isinstance(prior_val, str):
            try:
                prior = json.loads(prior_val)
            except Exception:
                prior = []
        else:
            prior = []

        if len(prior) == 0:
            # Single-turn, assume resolved (no follow-up recorded)
            resolved_mask.append(True)
        elif len(prior) >= 2:
            last_user = next(
                (t["text"] for t in reversed(prior) if isinstance(t, dict) and t.get("role") == "user"),
                ""
            )
            resolved_mask.append(_has_closure_token(last_user))
        else:
            resolved_mask.append(True)

    df["resolved"] = resolved_mask
    resolved = df[df["resolved"]].copy()

    # Map intent from clusters.parquet if available
    clusters_path = Path("data/sample/clusters.parquet")
    if clusters_path.exists():
        try:
            from src.taxonomy import INTENT_NAMES
            cdf = pd.read_parquet(str(clusters_path))
            id_to_intent = {}
            for _, crow in cdf.iterrows():
                cid = crow.get("cluster", -1)
                id_to_intent[str(crow["inbound_id"])] = INTENT_NAMES.get(cid, "other")
            resolved["intent"] = resolved["inbound_id"].astype(str).map(id_to_intent).fillna("other")
        except Exception as e:
            print(f"[retrieval] Warning: failed to map intents: {e}")
            resolved["intent"] = "other"
    else:
        resolved["intent"] = "other"

    print(f"[retrieval] Resolved pairs: {len(resolved):,}")
    return resolved


# ---------------------------------------------------------------------------
# Build index
# ---------------------------------------------------------------------------

def build_index(resolved: pd.DataFrame, model_name: str = EMBED_MODEL, max_pairs: int = 5000) -> None:
    # Subsample if large to keep index fast and lightweight
    if len(resolved) > max_pairs:
        resolved = resolved.sample(n=max_pairs, random_state=SEED).copy().reset_index(drop=True)
        print(f"[retrieval] Sampled {max_pairs:,} resolved pairs for retrieval index")
    else:
        resolved = resolved.reset_index(drop=True)

    texts = resolved["inbound_text"].fillna("").tolist()
    model = SentenceTransformer(model_name)
    print(f"[retrieval] Embedding {len(texts):,} resolved inbound messages ...")
    vecs  = model.encode(texts, batch_size=64, show_progress_bar=True,
                         convert_to_numpy=True, normalize_embeddings=True)

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(INDEX_PATH), embeddings=vecs)
    resolved.to_parquet(str(PAIRS_PATH), index=False)
    print(f"[retrieval] Saved index to {INDEX_PATH} and pairs to {PAIRS_PATH}")


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

_cache_vecs: np.ndarray | None = None
_cache_pairs: pd.DataFrame | None = None
_embed_model: SentenceTransformer | None = None


def _load_resources():
    global _cache_vecs, _cache_pairs, _embed_model
    if _cache_vecs is None:
        data        = np.load(str(INDEX_PATH))
        _cache_vecs = data["embeddings"]
        _cache_pairs = pd.read_parquet(str(PAIRS_PATH))
        try:
            _embed_model = SentenceTransformer(EMBED_MODEL, local_files_only=True)
        except Exception:
            _embed_model = SentenceTransformer(EMBED_MODEL)


def query(
    text: str,
    k: int = 5,
    intent_filter: str | None = None,
) -> list[dict]:
    """
    Return top-k resolved (inbound, outbound) pairs similar to `text`.

    If intent_filter is given, prefer pairs from the same intent cluster.
    Falls back to global top-k if filtered set is too small.
    """
    _load_resources()

    q_vec = _embed_model.encode([text], normalize_embeddings=True,
                                convert_to_numpy=True)[0]
    sims  = _cache_vecs @ q_vec   # cosine similarity (vecs are normalized)

    pairs_df = _cache_pairs.copy()
    pairs_df["_sim"] = sims

    if intent_filter and "intent" in pairs_df.columns:
        filtered = pairs_df[pairs_df["intent"] == intent_filter]
        if len(filtered) >= k:
            pairs_df = filtered

    top = pairs_df.nlargest(k, "_sim")
    results = []
    for _, row in top.iterrows():
        results.append({
            "inbound":  row["inbound_text"],
            "outbound": row["outbound_text"],
            "sim":      round(float(row["_sim"]), 3),
        })
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--parquet", default="data/sample/brand_threads.parquet")
    ap.add_argument("--query",   type=str, default=None)
    ap.add_argument("--k",       type=int, default=3)
    args = ap.parse_args()

    if args.build:
        resolved = find_resolved_pairs(args.parquet)
        build_index(resolved)

    if args.query:
        results = query(args.query, k=args.k)
        for r in results:
            print(f"sim={r['sim']:.3f}  in: {r['inbound'][:80]}")
            print(f"          out: {r['outbound'][:80]}\n")
