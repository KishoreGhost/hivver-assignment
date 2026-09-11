# AI Customer-Support Agent — Twitter Dataset

## Overview
An AI customer-support agent trained on the [thoughtvector/customer-support-on-twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) Kaggle dataset, focused on a single high-volume English-language brand.

The system chains:
1. **Intent classification** (Groq `llama-3.1-8b-instant`, JSON mode, few-shot)
2. **Resolved-pair retrieval** (sentence-transformers embeddings, cosine-sim index)
3. **Reply drafting** (Groq `llama-3.3-70b-versatile`, brand voice card + factual grounding constraint)
4. **Escalation gate** (rule-based: confidence, high-risk intent, thread length, groundedness)

---

## Setup

### 1. Prerequisites
- Python 3.12+
- Kaggle API credentials (`~/.kaggle/kaggle.json`)
- `GROQ_API_KEY` environment variable

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download dataset
```bash
make data
# or manually:
kaggle datasets download -d thoughtvector/customer-support-on-twitter -p data --unzip
```

### 4. Run brand audit (Step 1)
```bash
make audit
# Review output table, note chosen brand in report/decision_log.md
```

### 5. Build pipeline (Steps 2–8, in order)
```bash
make ingest BRAND=<chosen_brand_author_id>
make taxonomy                # prints cluster printout → fill INTENT_NAMES
make classify-setup          # after naming clusters
make retrieval
make draft-setup             # prints 30 sample outbound replies → supply brand voice card
```
**Stop here and provide:**
- Cluster names for `INTENT_NAMES` in `src/taxonomy.py`
- Brand voice card adjectives + do/don't in `src/draft.py`
- High-risk intent tags in `src/escalate.py`

### 6. Build eval artifacts (Steps 9–10)
```bash
make golden
make baselines
```

### 7. Reproduce results (reviewer quick run)
```bash
make eval-quick    # ~30-40 examples, cache bypassed, < 15 minutes
```

### 8. Full eval (once, commit output)
```bash
make eval-full     # full golden set → report/metrics.json
```

---

## Quota Estimate for a Full Run

| Model | Calls | Free-tier daily limit | % used |
|-------|------:|---------------------:|-------:|
| `llama-3.1-8b-instant` (classify + groundedness) | ~400 | 14,400 RPD | 2.8% |
| `llama-3.3-70b-versatile` (draft) | ~200 | 1,000 RPD | 20% |
| `openai/gpt-oss-120b` (judge) | ~200 | 1,000 RPD | 20% |

`eval-quick` (~35 examples): ~35 + 35 + 35 = 105 calls total.  
The disk cache (`cache/llm_cache.sqlite`) means repeated runs cost zero quota.

---

## Project Structure
```
hivver-assignment/
├── src/
│   ├── llm_client.py    # Shared cached + rate-limited LLM client
│   ├── brand_audit.py   # Step 1: brand ranking
│   ├── ingest.py        # Step 2: thread reconstruction
│   ├── taxonomy.py      # Step 3: embedding + clustering
│   ├── classify.py      # Step 4: intent classifier
│   ├── retrieval.py     # Step 5: resolved-pair retrieval index
│   ├── draft.py         # Step 6: reply drafter
│   ├── escalate.py      # Step 7: escalation gate
│   └── pipeline.py      # Step 8: orchestration
├── eval/
│   ├── build_golden_set.py
│   ├── baselines.py
│   ├── metrics.py
│   ├── judge.py
│   ├── judge_calibration.py
│   ├── run_eval.py        # make eval-full
│   └── run_eval_quick.py  # make eval-quick
├── data/
│   ├── twcs.csv           # raw dataset (gitignored)
│   ├── sample/            # subsampled working data
│   └── golden/            # golden eval set + labels
├── tests/
│   ├── test_ingest.py
│   ├── test_escalate.py
│   └── test_metrics.py
├── report/
│   ├── decision_log.md
│   ├── REPORT.md
│   └── metrics.json       # committed after eval-full
├── cache/                 # LLM response cache (gitignored)
├── Makefile
├── requirements.txt
└── README.md
```

---

## Where to Find Things
- **Headline results**: `report/metrics.json` + printed table from `make eval-full`
- **Decision rationale**: `report/decision_log.md`
- **Full analysis**: `report/REPORT.md`
- **Brand voice card**: `src/draft.py` → `BRAND_VOICE` dict
- **Cluster names / intent taxonomy**: `src/taxonomy.py` → `INTENT_NAMES` + `src/classify.py`
- **Escalation config**: `src/escalate.py` → thresholds + `HIGH_RISK_INTENTS`
