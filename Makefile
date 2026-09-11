PY ?= python3
KAGGLE ?= kaggle

.PHONY: all data audit ingest taxonomy classify-setup retrieval draft-setup \
        golden baselines eval-full eval-quick test clean

# -----------------------------------------------------------------------
# Data download
# -----------------------------------------------------------------------
data/twcs/twcs.csv:
	@echo "Downloading dataset from Kaggle..."
	"$(KAGGLE)" datasets download -d thoughtvector/customer-support-on-twitter -p data --unzip
	@echo "Note: CSV will be at data/twcs/twcs.csv"
	@echo "Dataset ready."

CSV := data/twcs/twcs.csv
data: $(CSV)

# -----------------------------------------------------------------------
# Step 1: Brand audit
# -----------------------------------------------------------------------
audit: $(CSV)
	"$(PY)" src/brand_audit.py --csv $(CSV)

# -----------------------------------------------------------------------
# Step 2: Thread reconstruction  (set BRAND= to override)
# -----------------------------------------------------------------------
BRAND ?= AppleSupport   # Chosen brand from brand_audit
ingest: $(CSV)
	"$(PY)" src/ingest.py --csv $(CSV) --brand $(BRAND)

# -----------------------------------------------------------------------
# Step 3: Taxonomy discovery
# -----------------------------------------------------------------------
taxonomy: data/sample/brand_threads.parquet
	"$(PY)" src/taxonomy.py --parquet data/sample/brand_threads.parquet

# -----------------------------------------------------------------------
# Step 4: Build few-shot examples (after naming clusters)
# -----------------------------------------------------------------------
classify-setup: data/sample/clusters.parquet
	"$(PY)" -c "from src.classify import build_few_shots; build_few_shots()"

# -----------------------------------------------------------------------
# Step 5: Build retrieval index
# -----------------------------------------------------------------------
retrieval: data/sample/brand_threads.parquet
	"$(PY)" src/retrieval.py --build --parquet data/sample/brand_threads.parquet

# -----------------------------------------------------------------------
# Step 6: Print sample outbound replies for brand voice card
# -----------------------------------------------------------------------
draft-setup: data/sample/brand_threads.parquet
	"$(PY)" src/draft.py --sample-outbound

# -----------------------------------------------------------------------
# Step 9: Build golden set
# -----------------------------------------------------------------------
golden: data/sample/clusters.parquet
	"$(PY)" eval/build_golden_set.py --parquet data/sample/clusters.parquet

# -----------------------------------------------------------------------
# Step 10: Baselines
# -----------------------------------------------------------------------
baselines: data/golden/golden_set.csv data/sample/brand_threads.parquet
	"$(PY)" eval/baselines.py

# -----------------------------------------------------------------------
# Step 14a: Full eval (run once, commit output)
# -----------------------------------------------------------------------
eval-full: data/golden/golden_set.csv
	"$(PY)" eval/run_eval.py --golden data/golden/golden_set.csv

# -----------------------------------------------------------------------
# Step 14b: Quick eval (reviewer verification, < 15 min)
# -----------------------------------------------------------------------
eval-quick: data/golden/golden_set.csv
	"$(PY)" eval/run_eval_quick.py --golden data/golden/golden_set.csv --n 35

# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------
test:
	"$(PY)" -m pytest tests/ -v

# -----------------------------------------------------------------------
# Full pipeline demo
# -----------------------------------------------------------------------
demo:
	"$(PY)" src/pipeline.py "My order has been stuck in transit for 10 days!"

# -----------------------------------------------------------------------
# Clean generated artifacts (keep committed data)
# -----------------------------------------------------------------------
clean:
	"$(PY)" -c "import pathlib; [p.unlink() for p in pathlib.Path('cache').glob('*.sqlite') if p.is_file()]"
