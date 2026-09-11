"""
eval/run_eval_quick.py
----------------------
Step 14b: Quick evaluation (make eval-quick).

Runs the same pipeline on a small random subset (~30-40 examples)
with cache BYPASSED, so a reviewer can verify the pipeline genuinely
reproduces numbers without redoing the full judge pass live.

Target: completes in < 15 minutes.
Budget: ~35 classify + 35 draft + 35 judge calls (well within daily limits).

Usage:
  python eval/run_eval_quick.py --golden data/golden/golden_set.csv --n 35
"""

import argparse, json, random, sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline  import handle_message
from eval.metrics  import evaluate_golden
from eval.judge    import run_judge

SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="data/golden/golden_set.csv")
    ap.add_argument("--n",      type=int, default=35)
    ap.add_argument("--out",    default="data/golden/quick_eval.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.golden)
    rng = random.Random(SEED)
    idx = rng.sample(range(len(df)), min(args.n, len(df)))
    subset = df.iloc[idx].copy().reset_index(drop=True)

    print(f"[eval-quick] Running on {len(subset)} examples (cache bypassed) ...")
    rows = []
    for _, row in tqdm(subset.iterrows(), total=len(subset)):
        text  = str(row["inbound_text"])
        prior = json.loads(str(row.get("prior_turns_json") or "[]"))
        result = handle_message(text, prior, bypass_cache=True)   # bypass cache!
        rows.append(result)

    pred_df = pd.DataFrame(rows)
    subset["pred_intent"]          = pred_df["intent"].values
    subset["pred_confidence"]      = pred_df["confidence"].values
    subset["draft_reply"]          = pred_df["draft_reply"].values
    subset["pred_escalate"]        = pred_df["escalate"].values
    subset["pred_escalate_reason"] = pred_df["escalate_reason"].values

    subset.to_csv(args.out, index=False)
    print(f"[eval-quick] Predictions saved to {args.out}")

    # Judge
    judged_out = args.out.replace(".csv", "_judged.csv")
    run_judge(args.out, judged_out, bypass_cache=True)

    # Metrics
    metrics = evaluate_golden(args.out)
    print("\n=== Quick Eval Metrics ===")
    print(json.dumps(metrics, indent=2))

    committed = Path("report/metrics.json")
    if committed.exists():
        committed_m = json.loads(committed.read_text())
        print("\n=== Comparison to committed metrics ===")
        for key in ["intent", "escalation"]:
            c = committed_m.get("pipeline", {}).get(key, {})
            q = metrics.get(key, {})
            print(f"  [{key}]")
            for metric in ["accuracy", "macro_f1", "f1"]:
                cv = c.get(metric, "—")
                qv = q.get(metric, "—")
                print(f"    {metric}: committed={cv}  quick={qv}")


if __name__ == "__main__":
    main()
