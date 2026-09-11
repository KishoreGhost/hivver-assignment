"""
eval/run_eval.py
----------------
Step 14a: Full evaluation pipeline (make eval-full).

Runs: classify → draft → escalate → judge over the full golden set.
Computes metrics vs baselines. Writes report/metrics.json + summary tables.

NOTE: This burns ~200 classify + 200 draft + 200 judge API calls.
      Budget: ~200/14400 RPD for 8b (classify) + ~200/1000 RPD for 70b (draft)
      + ~200/1000 RPD for gpt-oss-120b (judge).
      Run once, commit output. Use eval-quick for reviewer verification.

Usage:
  python eval/run_eval.py --golden data/golden/golden_set.csv
"""

import argparse, json, sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline    import handle_message
from eval.metrics    import evaluate_golden, bleu_rouge_scores, groundedness_check
from eval.judge      import run_judge
from eval.baselines  import TrivialBaseline, SimpleBaseline


def run_pipeline_on_golden(golden_csv: str, *, bypass_cache: bool = False) -> str:
    """Run handle_message on every row, fill in pred_* columns."""
    df = pd.read_csv(golden_csv)
    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Pipeline"):
        text   = str(row["inbound_text"])
        prior  = json.loads(str(row.get("prior_turns_json") or "[]"))
        result = handle_message(text, prior, bypass_cache=bypass_cache)
        rows.append(result)

    pred_df = pd.DataFrame(rows)
    df["pred_intent"]          = pred_df["intent"].values
    df["pred_confidence"]      = pred_df["confidence"].values
    df["draft_reply"]          = pred_df["draft_reply"].values
    df["pred_escalate"]        = pred_df["escalate"].values
    df["pred_escalate_reason"] = pred_df["escalate_reason"].values

    out_path = golden_csv.replace(".csv", "_with_preds.csv")
    df.to_csv(out_path, index=False)
    print(f"[run_eval] Saved predictions to {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden",  default="data/golden/golden_set.csv")
    ap.add_argument("--threads", default="data/sample/brand_threads.parquet")
    args = ap.parse_args()

    pred_csv = run_pipeline_on_golden(args.golden)

    # Judge
    judged_csv = "data/golden/judged.csv"
    run_judge(pred_csv, judged_csv)

    # Metrics: pipeline
    print("\n=== Pipeline Metrics ===")
    metrics = evaluate_golden(pred_csv)
    print(json.dumps(metrics, indent=2))

    # Metrics: baselines
    golden = pd.read_csv(pred_csv)
    threads = pd.read_parquet(args.threads)
    labeled = golden[golden["gold_intent"].notna() & (golden["gold_intent"] != "")]

    triv = TrivialBaseline(); triv.fit(labeled)
    simp = SimpleBaseline();  simp.fit(labeled, threads)
    triv_preds = triv.predict_batch(golden)
    simp_preds = simp.predict_batch(golden)

    triv_preds.to_csv("data/golden/baseline_trivial.csv", index=False)
    simp_preds.to_csv("data/golden/baseline_simple.csv",  index=False)

    triv_metrics = evaluate_golden("data/golden/baseline_trivial.csv")
    simp_metrics = evaluate_golden("data/golden/baseline_simple.csv")

    # BLEU/ROUGE (informational)
    if labeled.shape[0] > 0 and "draft_reply" in golden.columns:
        bleu_r = bleu_rouge_scores(
            golden["draft_reply"].fillna("").tolist(),
            golden["outbound_text"].fillna("").tolist() if "outbound_text" in golden.columns
            else [""] * len(golden),
        )
    else:
        bleu_r = {}

    output = {
        "pipeline": metrics,
        "baseline_trivial": triv_metrics,
        "baseline_simple":  simp_metrics,
        "bleu_rouge_informational": bleu_r,
    }

    out_path = Path("report/metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n[run_eval] All metrics written to {out_path}")

    # Summary table
    print("\n=== Summary Table ===")
    print(f"{'System':<25} {'Acc':>6} {'MacroF1':>8} {'EscF1':>8}")
    for name, m in [("Pipeline", metrics),
                    ("Trivial Baseline", triv_metrics),
                    ("Simple Baseline", simp_metrics)]:
        acc = m.get("intent", {}).get("accuracy", "—")
        mf1 = m.get("intent", {}).get("macro_f1", "—")
        ef1 = m.get("escalation", {}).get("f1", "—")
        print(f"{name:<25} {str(acc):>6} {str(mf1):>8} {str(ef1):>8}")


if __name__ == "__main__":
    main()
