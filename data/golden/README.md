# Golden Evaluation Set

## How the sample was drawn

- **Source**: `data/sample/clusters.parquet` (brand threads with cluster labels from taxonomy step)
- **Seed**: 42 (deterministic)
- **Target size**: 200 examples
- **Stratification variables**:
  - `cluster` (intent cluster from k-means, step 3)
  - `turn_type`: single_turn (no prior turns) vs. multi_turn
  - `sentiment`: positive / neutral / negative (VADER compound score thresholds ±0.05)
- **Per-stratum quota**: `max(1, 200 // n_strata)` examples each
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
