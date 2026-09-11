"""
eval/judge_calibration.py
-------------------------
Step 13: Judge calibration against human ratings.

Part A (--export): Sample ~40-60 golden examples, output blind-labeling CSV
  (no judge scores visible) for human hand-rating on the same rubric.

Part B (--compare): Join human ratings against the judge's, compute:
  - Quadratic-weighted Cohen's kappa per dimension
  - Spearman correlation per dimension
  - Exact and adjacent (within-1) agreement %
  - Largest disagreements table

Usage:
  # Export for human rating:
  python eval/judge_calibration.py --export --judged data/golden/judged.csv \
      --out data/golden/calibration_blind.csv

  # After filling in human ratings, compare:
  python eval/judge_calibration.py --compare \
      --judged data/golden/judged.csv \
      --human  data/golden/calibration_blind.csv \
      --out    data/golden/calibration_results.json
"""

import argparse, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SEED = 42
N_CALIBRATION = 50   # how many examples to sample

DIMENSIONS = [
    "issue_understanding",
    "correctness",
    "brand_voice",
    "actionability",
    "escalation_appropriate",
]


# ---------------------------------------------------------------------------
# Quadratic-weighted Cohen's kappa (categories 1-5)
# ---------------------------------------------------------------------------

def quadratic_kappa(y1: list[int], y2: list[int], min_r: int = 1, max_r: int = 5) -> float:
    n     = max_r - min_r + 1
    O     = np.zeros((n, n))
    for a, b in zip(y1, y2):
        O[a - min_r][b - min_r] += 1
    W     = np.array([[((i - j) ** 2) / ((n - 1) ** 2) for j in range(n)] for i in range(n)])
    hist1 = O.sum(axis=1, keepdims=True)
    hist2 = O.sum(axis=0, keepdims=True)
    E     = hist1 @ hist2 / O.sum()
    num   = (W * O).sum()
    den   = (W * E).sum()
    if den == 0:
        return 1.0
    return float(1 - num / den)


# ---------------------------------------------------------------------------
# Part A: Export blind-labeling CSV
# ---------------------------------------------------------------------------

def export_blind(judged_csv: str, out_csv: str, n: int = N_CALIBRATION) -> None:
    df = pd.read_csv(judged_csv)
    rng = random.Random(SEED)
    sample_idx = rng.sample(range(len(df)), min(n, len(df)))
    sample = df.iloc[sample_idx].copy().reset_index(drop=True)

    # Remove judge scores so human raters can't see them
    judge_cols = [c for c in sample.columns if c.startswith("judge_")]
    blind = sample.drop(columns=judge_cols)

    # Add empty human-rating columns
    for dim in DIMENSIONS:
        blind[f"human_{dim}"] = ""
    blind["human_comment"] = ""

    blind.to_csv(out_csv, index=False)
    print(f"[calibration] Exported {len(blind)} blind examples to {out_csv}")
    print("Fill in the human_* columns (1-5) and then run with --compare.")


# ---------------------------------------------------------------------------
# Part B: Compare judge vs human
# ---------------------------------------------------------------------------

def compare(judged_csv: str, human_csv: str, out_json: str) -> None:
    judged = pd.read_csv(judged_csv)
    human  = pd.read_csv(human_csv)

    # Merge on example_id
    merged = judged.merge(human[["example_id"] + [f"human_{d}" for d in DIMENSIONS]],
                          on="example_id", how="inner")
    print(f"[calibration] Merged {len(merged)} examples for calibration")

    results = {}
    for dim in DIMENSIONS:
        j_col = f"judge_{dim}"
        h_col = f"human_{dim}"
        if j_col not in merged.columns or h_col not in merged.columns:
            continue

        # Filter out blanks
        mask = merged[h_col].notna() & (merged[h_col].astype(str) != "")
        sub  = merged[mask].copy()
        if len(sub) < 5:
            continue

        j = sub[j_col].astype(int).tolist()
        h = sub[h_col].astype(int).tolist()

        kappa  = quadratic_kappa(j, h)
        sp, sp_p = spearmanr(j, h)
        exact  = sum(a == b for a, b in zip(j, h)) / len(j)
        adj    = sum(abs(a - b) <= 1 for a, b in zip(j, h)) / len(j)

        results[dim] = {
            "kappa":       round(kappa, 3),
            "spearman":    round(float(sp), 3),
            "spearman_p":  round(float(sp_p), 4),
            "exact_agree": round(exact, 3),
            "adj_agree":   round(adj, 3),
            "n":           len(sub),
        }
        print(f"\n  [{dim}]")
        print(f"    kappa={kappa:.3f}  spearman={float(sp):.3f}  "
              f"exact={exact:.2%}  adj={adj:.2%}")

    # Largest disagreements (by mean absolute deviation across all dims)
    merged["_mad"] = merged.apply(
        lambda row: np.mean([
            abs(int(row[f"judge_{d}"]) - int(row[f"human_{d}"]))
            for d in DIMENSIONS
            if str(row.get(f"human_{d}", "")) not in {"", "nan"}
        ]) if any(str(row.get(f"human_{d}", "")) not in {"", "nan"} for d in DIMENSIONS) else 0,
        axis=1,
    )
    top_disagree = merged.nlargest(10, "_mad")[
        ["example_id", "inbound_text"] +
        [f"judge_{d}" for d in DIMENSIONS] +
        [f"human_{d}" for d in DIMENSIONS] +
        ["_mad"]
    ]
    print("\n=== Top disagreements ===")
    print(top_disagree.to_string(index=False))

    out = {"dimensions": results, "n_total": len(merged)}
    Path(out_json).write_text(json.dumps(out, indent=2))
    print(f"\n[calibration] Results saved to {out_json}")
    top_disagree.to_csv(str(Path(out_json).with_suffix(".disagreements.csv")), index=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--export",  action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--judged",  default="data/golden/judged.csv")
    ap.add_argument("--human",   default="data/golden/calibration_blind.csv")
    ap.add_argument("--out",     default="data/golden/calibration_results.json")
    ap.add_argument("--n",       type=int, default=N_CALIBRATION)
    args = ap.parse_args()

    if args.export:
        export_blind(args.judged, args.human, n=args.n)
    elif args.compare:
        compare(args.judged, args.human, args.out)
    else:
        ap.print_help()
