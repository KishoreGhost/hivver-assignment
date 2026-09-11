"""
src/taxonomy.py
---------------
Step 3: Intent taxonomy discovery via embedding + clustering.

1. Loads inbound texts from data/sample/brand_threads.parquet.
2. Embeds up to 10,000 texts using a local sentence-transformers model.
3. Clusters with k-means (tries k in [6..12]) and optionally HDBSCAN.
4. Prints top TF-IDF terms and 5 example texts per cluster.
5. Saves cluster labels + embeddings to data/sample/clusters.parquet.

After reviewing the printout, fill in INTENT_NAMES below (the TODO spot).

Usage:
  python src/taxonomy.py [--parquet data/sample/brand_threads.parquet] [--k 9]
"""

import argparse, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

SEED       = 42
EMBED_MODEL = "all-MiniLM-L6-v2"   # local; ~90 MB download on first run
MAX_TEXTS   = 10_000
SAMPLE_OUT  = Path("data/sample")

# ---------------------------------------------------------------------------
# TODO: After reviewing cluster printout, fill in human-readable names here.
# Keys are cluster indices (ints), values are intent strings.
# Add a "catch-all" entry with key -1 (used for HDBSCAN noise or low-conf).
# ---------------------------------------------------------------------------
INTENT_NAMES: dict[int, str] = {
    -1: "other",
    0:  "non_english_inquiry",
    1:  "keyboard_text_glitch",
    2:  "connectivity_network_issue",
    3:  "conversation_followup_status",
    4:  "device_os_glitch",
    5:  "media_audio_services",
    6:  "software_update_ios",
    7:  "account_icloud_store",
    8:  "app_compatibility_settings",
    9:  "macos_hardware_support",
    10: "battery_power_drain",
}
# ---------------------------------------------------------------------------


def load_data(parquet: str, max_texts: int = MAX_TEXTS) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_parquet(parquet)
    df = df.dropna(subset=["inbound_text"]).copy()
    if len(df) > max_texts:
        df = df.sample(n=max_texts, random_state=SEED).copy().reset_index(drop=True)
        print(f"[taxonomy] Subsampled to {max_texts:,} texts (seed={SEED})")
    else:
        df = df.reset_index(drop=True)
    texts = df["inbound_text"].tolist()
    print(f"[taxonomy] Loaded {len(texts):,} inbound texts")
    return df, texts


def embed(texts: list[str], model_name: str = EMBED_MODEL) -> np.ndarray:
    print(f"[taxonomy] Embedding with {model_name} ...")
    model = SentenceTransformer(model_name)
    # Encode in batches with progress
    vecs = model.encode(texts, batch_size=64, show_progress_bar=True,
                        convert_to_numpy=True, normalize_embeddings=True)
    return vecs


def find_best_k(vecs: np.ndarray, k_range: range) -> int:
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        labels = km.fit_predict(vecs)
        sil = silhouette_score(vecs, labels, sample_size=2000, random_state=SEED)
        scores[k] = sil
        print(f"  k={k}  silhouette={sil:.4f}")
    best_k = max(scores, key=scores.__getitem__)
    print(f"[taxonomy] Best k by silhouette: {best_k}")
    return best_k


def cluster(vecs: np.ndarray, k: int) -> np.ndarray:
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    labels = km.fit_predict(vecs)
    return labels


def top_tfidf(texts: list[str], labels: np.ndarray, cluster_id: int,
              top_n: int = 15) -> list[str]:
    cluster_texts = [t for t, l in zip(texts, labels) if l == cluster_id]
    if not cluster_texts:
        return []
    vec = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
    X   = vec.fit_transform(cluster_texts)
    mean_tfidf = np.asarray(X.mean(axis=0)).flatten()
    top_idx    = mean_tfidf.argsort()[::-1][:top_n]
    terms      = vec.get_feature_names_out()
    return [terms[i] for i in top_idx]


def print_clusters(texts: list[str], labels: np.ndarray, k: int) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("CLUSTER TAXONOMY PRINTOUT — assign names in INTENT_NAMES dict")
    lines.append("=" * 70)
    for cid in range(k):
        members = [t for t, l in zip(texts, labels) if l == cid]
        terms   = top_tfidf(texts, labels, cid)
        lines.append(f"\n--- Cluster {cid} ({len(members):,} texts) ---")
        lines.append(f"  Top terms: {', '.join(terms[:10])}")
        lines.append("  Examples:")
        for ex in members[:5]:
            # Clean non-ascii for terminal safe display
            clean_ex = ex[:120].encode('ascii', 'replace').decode('ascii')
            lines.append(f"    * {clean_ex}")
    lines.append("=" * 70)
    output = "\n".join(lines)
    print(output)
    return output


def save_clusters(df: pd.DataFrame, labels: np.ndarray, vecs: np.ndarray) -> None:
    df_out = df.copy()
    df_out["cluster"] = labels
    # Store embeddings as JSON string to keep parquet simple
    df_out["embedding"] = [json.dumps(v.tolist()) for v in vecs]
    SAMPLE_OUT.mkdir(parents=True, exist_ok=True)
    out = SAMPLE_OUT / "clusters.parquet"
    df_out.to_parquet(str(out), index=False)
    print(f"[taxonomy] Saved clusters to {out}")


def print_intent_config(k: int) -> None:
    """Print a skeleton INTENT_NAMES dict for the user to fill in."""
    print("\n# ---- Paste this into taxonomy.py INTENT_NAMES after naming ----")
    print("INTENT_NAMES: dict[int, str] = {")
    print("    -1: 'other',  # catch-all")
    for i in range(k):
        print(f"    {i}: '',  # TODO: name this cluster")
    print("}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/sample/brand_threads.parquet")
    ap.add_argument("--k",  type=int, default=0,
                    help="Force k; 0=auto-select from silhouette")
    ap.add_argument("--k-min", type=int, default=6)
    ap.add_argument("--k-max", type=int, default=12)
    ap.add_argument("--max-texts", type=int, default=MAX_TEXTS)
    args = ap.parse_args()

    df, texts = load_data(args.parquet, max_texts=args.max_texts)
    vecs = embed(texts)

    if args.k > 0:
        best_k = args.k
    else:
        best_k = find_best_k(vecs, range(args.k_min, args.k_max + 1))

    labels = cluster(vecs, best_k)
    summary_text = print_clusters(texts, labels, best_k)
    
    # Save printout to file
    Path("report").mkdir(parents=True, exist_ok=True)
    with open("report/taxonomy_clusters.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)
    
    save_clusters(df, labels, vecs)
    print_intent_config(best_k)


if __name__ == "__main__":
    main()
