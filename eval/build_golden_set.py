"""
eval/build_golden_set.py
------------------------
Step 9: Build a stratified golden evaluation set.

Draws 150-250 examples stratified by:
  - intent (from cluster labels)
  - turn count (single-turn vs multi-turn)
  - sentiment bucket (positive / neutral / negative via VADER)

Output: data/golden/golden_set.csv with blank label columns for human annotation.
See data/golden/README.md for sampling documentation.

IMPORTANT: Do NOT auto-generate gold_intent, gold_escalate, gold_escalate_reason,
or reply_checklist — these are left blank for human labeling.
"""

import argparse, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

SEED       = 42
TARGET_N   = 200   # target total examples

rng = random.Random(SEED)


def sentiment_bucket(text: str, analyzer: SentimentIntensityAnalyzer) -> str:
    score = analyzer.polarity_scores(str(text))["compound"]
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    return "neutral"


def build_golden_set(clusters_parquet: str, out_csv: str, target: int = TARGET_N) -> None:
    df = pd.read_parquet(clusters_parquet)

    # Require cluster column from taxonomy step
    if "cluster" not in df.columns:
        raise ValueError("Run taxonomy.py first to add cluster labels.")

    analyzer = SentimentIntensityAnalyzer()

    # --- Feature engineering ---
    df["sentiment"] = df["inbound_text"].apply(
        lambda t: sentiment_bucket(t, analyzer)
    )
    import numpy as np

    def _is_multi_turn(p):
        if p is None:
            return False
        if isinstance(p, str):
            try:
                p = json.loads(p)
            except Exception:
                return False
        if isinstance(p, (list, np.ndarray)):
            return len(p) > 0
        return False

    df["turn_type"] = df["prior_turns"].apply(
        lambda p: "multi_turn" if _is_multi_turn(p) else "single_turn"
    )
    df["cluster_str"] = df["cluster"].astype(str)

    # --- Stratified sampling ---
    # Group by (cluster_str, turn_type, sentiment)
    strata = df.groupby(["cluster_str", "turn_type", "sentiment"])
    n_strata = len(strata)
    per_stratum = max(1, target // n_strata)

    samples = []
    for _, group in strata:
        n = min(per_stratum, len(group))
        samples.append(group.sample(n=n, random_state=SEED))

    golden = pd.concat(samples).drop_duplicates(subset=["inbound_id"]).reset_index(drop=True)

    # Trim or pad to target
    if len(golden) > target:
        golden = golden.sample(n=target, random_state=SEED).reset_index(drop=True)

    print(f"[golden_set] Sampled {len(golden)} examples across {n_strata} strata")

    # --- Output columns ---
    # Pipeline outputs (will be filled by run_eval.py, left blank here)
    out = pd.DataFrame({
        "example_id":          range(len(golden)),
        "inbound_text":        golden["inbound_text"].values,
        "prior_turns_json":    golden["prior_turns"].apply(
                                   lambda p: json.dumps(list(p) if isinstance(p, np.ndarray) else (p if p is not None else [])) if not isinstance(p, str) else p
                               ).values,
        "cluster":             golden["cluster"].values,
        "sentiment":           golden["sentiment"].values,
        "turn_type":           golden["turn_type"].values,
        # --- Human-labeling columns (leave blank) ---
        "gold_intent":         "",
        "gold_escalate":       "",
        "gold_escalate_reason": "",
        "reply_checklist":     "",
        # --- Pipeline outputs (filled by eval run) ---
        "pred_intent":         "",
        "pred_confidence":     "",
        "draft_reply":         "",
        "pred_escalate":       "",
        "pred_escalate_reason": "",
    })

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(str(out_path), index=False)
    print(f"[golden_set] Saved to {out_path}")

    # Write README
    readme = Path(out_path.parent / "README.md")
    readme.write_text(f"""# Golden Evaluation Set

## How the sample was drawn

- **Source**: `data/sample/clusters.parquet` (brand threads with cluster labels from taxonomy step)
- **Seed**: {SEED} (deterministic)
- **Target size**: {target} examples
- **Stratification variables**:
  - `cluster` (intent cluster from k-means, step 3)
  - `turn_type`: single_turn (no prior turns) vs. multi_turn
  - `sentiment`: positive / neutral / negative (VADER compound score thresholds ±0.05)
- **Per-stratum quota**: `max(1, {target} // n_strata)` examples each
- **Deduplication**: by `inbound_id`

## Column descriptions

| Column | Description |
|--------|-------------|
| `example_id` | Sequential row ID |
| `inbound_text` | Customer message (PII redacted: @handles → [USER]) |
| `prior_turns_json` | JSON list of prior conversation turns |
| `cluster` | K-means cluster id from taxonomy step |
| `sentiment` | VADER sentiment bucket |
| `turn_type` | single_turn or multi_turn |
| `gold_intent` | **Human label**: correct intent name |
| `gold_escalate` | **Human label**: True/False |
| `gold_escalate_reason` | **Human label**: brief reason for escalation decision |
| `reply_checklist` | **Human label**: what a good reply must/must-not contain |
| `pred_*` | Filled by eval/run_eval.py |

## Important notes

- Labels (`gold_*`) are left blank for human annotation — do NOT auto-generate them.
- The sample is NOT random over the full dataset: it over-represents rare intents/sentiments
  for coverage. This means accuracy computed on this set is NOT representative of live traffic.
  Noted in report/REPORT.md under "What is misleading about my headline number".
""", encoding="utf-8")
    print(f"[golden_set] README written to {readme}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/sample/clusters.parquet")
    ap.add_argument("--out",     default="data/golden/golden_set.csv")
    ap.add_argument("--n",       type=int, default=TARGET_N)
    args = ap.parse_args()
    build_golden_set(args.parquet, args.out, target=args.n)
