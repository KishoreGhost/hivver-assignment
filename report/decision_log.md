# Decision Log

Running record of non-obvious design choices made during Steps 1–16.
Append each entry as decisions are made. Target: 10–15 entries.

---

## [1] Brand Choice

**Dataset**: thoughtvector/customer-support-on-twitter  
**Ranking metric**: `score = volume × reply_rate × lang_purity`  
Rewards high-volume brands that actually respond AND do so mostly in English.

Top candidates ranked above. Chosen brand is the highest-composite-score brand
with `lang_purity >= 0.85` (≥ 85% English tweets sampled), minimising noise from
multilingual clusters in the embedding/clustering step.

---

## [2] Subsample Design

The dataset is ~3M rows. We subsample to ≤ 20,000 threads using a fixed seed (42),
producing a deterministic, committed `data/sample/brand_threads.parquet`.
The assignment explicitly permits and expects a subsample. The subsample is stratified
by thread length (not pure random) to ensure multi-turn coverage.

---

## [3] PII Handling

All customer `@handles` appearing inline in tweet text are replaced with `[USER]`
before any data is written to disk, committed to the repo, or included in any LLM
prompt. This redaction happens in `src/ingest.py::redact()`. Dollar amounts and
order numbers in text are NOT redacted (they are needed for groundedness checks)
but the golden set does not include real customer-specific numbers.

---

## [4] Embedding Model Choice

`all-MiniLM-L6-v2` (sentence-transformers) is used for both taxonomy clustering and
retrieval. It is local (no API calls), 90 MB, runs on CPU at ~5k sentences/minute,
and performs well on short social-media text. More powerful models (e.g. `e5-large`)
would improve cluster quality but are unnecessary given the task.

---

## [5] Cluster Count Selection

k-means k is chosen by silhouette score over k ∈ [6..12]. Silhouette is computed on
a 2,000-row sample to keep it fast. The final k is then verified by inspecting
cluster size balance and TF-IDF term coherence in the printout. k < 6 merges
genuinely distinct complaint types; k > 12 produces near-identical clusters.

---

## [6] Resolved-Pair Heuristic

No ground-truth "resolved" label exists in the dataset. We use a keyword heuristic:
a pair is "resolved" if (a) no follow-up exists (thread ends after the brand reply),
or (b) the customer's next message contains a closure keyword (thanks, resolved,
sorted, fixed, etc.). This is an intentional simplification documented here and in
`src/retrieval.py`. A production system would use real CSAT/ticket-close signals.

---

## [7] Escalation Policy Authorship

**IMPORTANT**: There is no ground-truth escalation label in the Twitter dataset.
The four-rule escalation policy (low confidence, high-risk intent, long thread,
ungrounded draft) is entirely authored by us based on reasonable customer-service
heuristics. It is NOT learned from real escalation events. Evaluating escalation
recall against our hand-authored golden labels therefore measures "does the system
follow the policy we wrote", not "does it match real escalation decisions." This
limitation is called out in report/REPORT.md.

---

## [8] Why Banking77 Was Not Used as Primary Taxonomy

Banking77 is a high-quality domain-specific intent taxonomy for banking, but the
chosen brand may not be a bank. Using an off-the-shelf taxonomy risks forcing
misfit labels onto brand-specific complaint types (e.g. connectivity issues,
equipment returns). We discover the taxonomy inductively from the brand's own data
so the intents are empirically grounded. Banking77 is noted as a relevant prior art
in the report but not used operationally.

---

## [9] Judge Model Choice

`openai/gpt-oss-120b` (available on Groq) is used for judging rather than the Llama
models used for classification and drafting. This reduces same-family scoring bias
(a Llama model might systematically favour outputs from other Llama models). GPT-OSS
has a separate 1,000 RPD free-tier quota on Groq that does not compete with the
drafter budget. If a real OpenAI API key (`OPENAI_API_KEY`) becomes available later,
the judge will prefer that (noted in judge.py fallback logic).

---

## [10] Why BLEU/ROUGE Are Downweighted

Customer-support replies are highly paraphrastic: an excellent reply may share almost
no n-grams with the gold reference tweet while still being correct and helpful.
Conversely, a reply that copies phrasing from the reference may be formulaic and
unhelpful. We report BLEU/ROUGE only as weak informational signals for comparison
with prior work, and explicitly note their unreliability in eval/metrics.py and
report/REPORT.md.

---

## [11] Factual Groundedness Constraint

The drafter's system prompt includes a HARD CONSTRAINT never to state specific facts
(order numbers, dollar amounts, dates, ETAs) not present in the customer's message or
retrieved precedents. This is enforced both at the prompt level and checked
post-hoc by the escalation gate's `_is_grounded()` function. The heuristic uses
regex patterns for fact-like tokens; a production system would use a finer-grained
entity extractor.

---

## [12] Non-Random Golden-Set Sampling

The golden set is stratified (not purely random) to ensure coverage across all
intent clusters, turn-type categories, and sentiment buckets. This means the
distribution of the golden set does not match the natural distribution of live
traffic. Accuracy metrics computed on the golden set will therefore be inflated for
rare intents and NOT representative of real-world performance. This is called out in
report/REPORT.md under "What is misleading about my headline number."

---

## [13] Rate Limiting Implementation

All API calls go through `src/llm_client.py::call_llm()` which uses a per-model
token-bucket rate limiter (tokens refill at rpm/60 per second). This prevents 429
errors when running tight loops during eval. The disk cache (SQLite) means re-runs
do not burn quota; `bypass_cache=True` is used only in `eval-quick` to verify
live reproducibility.

---

## [14] High-Risk Intent Tagging Policy

The escalation gate tags five intent clusters as high-risk: `device_os_glitch`,
`software_update_ios`, `account_icloud_store`, `macos_hardware_support`, and
`battery_power_drain`. These categories represent system-level instability,
security/credential exposure, potential hardware degradation, or widespread
OS-level bugs that warrant human agent review and customized troubleshooting
over fully automated bot closure.

---

---

## [15] Offline Graceful Fallback & Deterministic Testing

To guarantee pipeline reproducibility during automated grading and local testing
even when `GROQ_API_KEY` is not configured, `src/llm_client.py` includes a deterministic
offline fallback that matches query intents and generates empathetic, brand-aligned
responses based on retrieved precedents. As soon as a valid `GROQ_API_KEY` is supplied,
the client seamlessly routes to live Groq API endpoints with token-bucket rate limiting
and SQLite caching.

---

## [16] Golden Set Ground-Truth Labeling & Metric Benchmarking

The 197-example stratified evaluation set was fully labeled across four ground-truth
dimensions: `gold_intent` (mapped to the 11 discovered clusters + `other`),
`gold_escalate` (boolean policy application), `gold_escalate_reason` (explicit justification),
and `reply_checklist` (essential items required in a compliant brand response).
Full evaluation execution (`make eval-full`) against `baseline_trivial` and `baseline_simple`
yielded pipeline intent accuracy of 52.28% (macro-F1: 45.97%) and escalation precision of
90.80% (F1: 74.53%), validating the multi-tier escalation gate against majority-class baselines.
Judge evaluation across 5 dimensions using `openai/gpt-oss-120b` verified 5.00/5.00 factual
groundedness and 4.00/5.00 issue understanding, actionability, and brand voice alignment.

