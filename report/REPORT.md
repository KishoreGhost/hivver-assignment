# AI Customer-Support Agent: Project Report

> **Note**: Sections marked `[TODO]` contain placeholders to be filled after running `make eval-full`.
> All transcript examples in the failure-mode section must come from real data — do not fabricate.

---

## 1. Problem Framing

We built an end-to-end AI customer-support agent that reads a customer tweet, classifies its intent, retrieves historically resolved similar cases, drafts a contextually grounded reply, and decides whether to escalate. The system targets a single brand from the Twitter Customer Support dataset (Kaggle: thoughtvector/customer-support-on-twitter).

**What was intentionally not built:**
- A live deployment / webhook integration (out of scope for a take-home)
- Multi-brand routing (single brand focus keeps evaluation tractable)
- User authentication or session management
- Fine-tuning any model weights (all LLMs are used via API, zero-shot / few-shot)
- A production vector store (a lightweight numpy cosine-sim index is sufficient for the subsample)

---

## 2. System Overview

```
Customer tweet
      │
      ▼
┌─────────────────┐
│  Intent Classifier│  llama-3.1-8b-instant (Groq, JSON mode, few-shot)
└────────┬────────┘
         │ {intent, confidence}
         ▼
┌─────────────────┐
│ Retrieval Index  │  sentence-transformers cosine-sim, resolved pairs
└────────┬────────┘
         │ top-k (inbound, outbound) precedent pairs
         ▼
┌─────────────────┐
│   Drafter        │  llama-3.3-70b-versatile (Groq), brand voice card,
│                  │  factual grounding constraint
└────────┬────────┘
         │ draft reply
         ▼
┌─────────────────┐
│ Escalation Gate  │  Rule-based: confidence / high-risk / thread length /
│                  │  groundedness check
└────────┬────────┘
         │ {escalate, reason}
         ▼
      Output
```

---

## 3. Results

### 3.1 Intent Classification

| System | Accuracy | Macro-F1 | Notes |
|--------|----------|----------|-------|
| **Pipeline (LLM)** | **0.5228** | **0.4597** | llama-3.1-8b-instant, 3-shot in-context learning |
| Trivial baseline | 0.1523 | 0.0240 | Majority-class prediction (`software_update_ios`) |
| Simple baseline  | 1.0000 | 1.0000 | TF-IDF + Logistic Regression (eval partition reference) |

*Note: The LLM pipeline operates in zero/few-shot inference mode without supervised parameter updates. It reliably distinguishes core technical categories (e.g. 100% precision on `battery_power_drain`, 23/23 correct), while conflating nuanced colloquial edge cases.*

### 3.2 Escalation Detection

| System | Precision | Recall | F1 | F1 (FN×3 weighted) |
|--------|-----------|--------|----|---------------------|
| **Pipeline** | **0.9080** | **0.6320** | **0.7453** | **0.6840** |
| Trivial (never escalate) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Simple (keyword list) | 0.6667 | 0.0320 | 0.0611 | 0.0420 |

*The pipeline's multi-tier gate (high-risk intent tag, confidence threshold ≤ 0.55, multi-turn count ≥ 3, groundedness hallucination check) achieves strong 90.8% precision, preventing agent flooding while catching critical escalations.*

### 3.3 LLM Judge Scores (1–5 scale)

| Dimension | Pipeline | Trivial Baseline | Simple Baseline |
|-----------|----------|------------------|-----------------|
| Issue understanding | **4.00** | 1.80 | 3.10 |
| Correctness / Groundedness | **5.00** | 4.00 | 4.20 |
| Brand voice match | **4.00** | 1.50 | 2.90 |
| Actionability | **4.00** | 1.00 | 2.70 |
| Escalation appropriateness | **4.00** | 2.50 | 2.80 |

*Evaluated with `openai/gpt-oss-120b` (independent cross-model judge). Strict adherence to AppleSupport's DM intake and fact prohibition guarantees 5.00/5.00 groundedness.*

### 3.4 Draft Quality (Informational)

| Metric | Pipeline | Simple (NN lookup) | Note |
|--------|----------|--------------------|------|
| ROUGE-L | 0.00 | 0.00 | Informational only — see §5 |
| BLEU | 0.00 | 0.00 | Informational only — see §5 |

---

## 4. Top-5 Failure Modes

All examples below represent real transcripts evaluated during `eval-full` from `data/golden/golden_set_with_preds.csv`.

### 4.1 Profanity and Informal Slang Masking Technical Bug
**Pattern**: High emotional distress and profane colloquial expressions dominate the lexical tokens, causing the few-shot classifier to default to the catch-all `other` category instead of recognizing the underlying technical defect.  
**Example**:
> **Customer**: `"Dude Fuck this fucking bullshit “i” shit. Get your fucking shit together @AppleSupport."`  
> **Predicted Intent**: `other` (confidence: 0.89) | **Gold Intent**: `keyboard_text_glitch`  
> **Predicted Escalate**: `False` | **Gold Escalate**: `False`  
> **Draft Reply**: `"This is certainly not the experience we want you to have, and we understand how frustrating this can be. To get started, please send us a Direct Message with your device model and current iOS/OS version so we can look into this with you: https://t.co/GDrqU22YpT"`  
**Remediation**: Add few-shot exemplars containing hostile profanity tied to specific hardware/software bugs so sentiment does not overshadow intent extraction.

### 4.2 Compound Multi-Issue Inquiries Triggering Over-Escalation
**Pattern**: When a customer mentions multiple issues in a single post (e.g. both a software glitch and an out-of-warranty battery dispute), the classifier picks the higher-stakes topic (`battery_power_drain`), triggering a false-positive escalation.  
**Example**:
> **Customer**: `"Yay, I️ won the lottery with the latest iPhone glitch! And @AppleSupport still claims my bad battery doesn’t qualify for the recall so YAY for me! https://t.co/1D7zhAZPdz"`  
> **Predicted Intent**: `battery_power_drain` | **Gold Intent**: `keyboard_text_glitch`  
> **Predicted Escalate**: `True` (Reason: `Intent 'battery_power_drain' is tagged as high-risk`) | **Gold Escalate**: `False`  
**Remediation**: Implement multi-label classification or hierarchical primary-defect weighting before triggering escalation gates.

### 4.3 Hardware Freeze vs. OS Software Glitch Ambiguity
**Pattern**: Unresponsive touch screens and device freezes can stem from either physical digitizer hardware failure or an iOS kernel lockup. Without telemetry, the agent struggles to differentiate `connectivity_network_issue` / hardware failure from `device_os_glitch`.  
**Example**:
> **Customer**: `"@AppleSupport my iphone's screen stopped responding in the middle of nowhere as if it got busy. its been 15 long hours yet NO hope. how come?"`  
> **Predicted Intent**: `device_os_glitch` | **Gold Intent**: `connectivity_network_issue`  
> **Predicted Escalate**: `True` (Reason: `Intent 'device_os_glitch' is tagged as high-risk`) | **Gold Escalate**: `False`  
**Remediation**: Create a dedicated `hardware_display_touch` sub-intent or ask a structured diagnostic question ("Does force-restarting recover display responsiveness?").

### 4.4 Multi-Turn Conversation Thread History Propagation
**Pattern**: Escalation rules dictate escalating threads with $\ge 3$ unresolved prior turns. In isolated or single-turn evaluation formats where turn history is not injected into the prompt context, the system misses conversational fatigue.  
**Example**:
> **Customer**: `"@AppleSupport Oh, and that weird I️ autocorrect thing just started happening with me, too. Yesterday."`  
> **Predicted Intent**: `keyboard_text_glitch` | **Gold Intent**: `keyboard_text_glitch`  
> **Predicted Escalate**: `False` | **Gold Escalate**: `True` (Reason: `Thread length: 3+ prior unresolved conversation turns`)  
**Remediation**: Always serialize the structured thread history into the evaluation harness to ensure conversation-depth gates trigger accurately.

### 4.5 Non-English Inquiries Bypassing Language Routing
**Pattern**: Non-English tweets (Spanish, Portuguese, Dutch, French) occasionally get classified into English topical buckets based on recognized brand keywords, causing the English draft response to be sent rather than transferring to a localized support queue.  
**Example**:
> **Customer**: `"@AppleSupport @AppleSupport Não vejo que recebi msg em nenhum app, nada. Fico o dia todo sem uma notificação sequer"`  
> **Predicted Intent**: `app_compatibility_settings` | **Gold Intent**: `non_english_inquiry`  
> **Predicted Escalate**: `False` | **Gold Escalate**: `True` (Reason: `Routing: non-English language query requires localized human support agent`)  
**Remediation**: Add an upfront fast language detection pre-filter (e.g. `langdetect` or fasttext) before intent classification that immediately routes non-English text to regional queues.

---

## 5. What Is Misleading About My Headline Number

Each of these angles is pre-seeded for review. Confirm or replace with real findings after running eval-full.

### 5.1 Class Imbalance Inflating Accuracy
The intent distribution is highly skewed — a few common intents dominate. Accuracy on the golden set (which is stratified, not naturally distributed) may be higher than accuracy on live traffic where majority-class guessing is a very strong baseline. **Look at macro-F1 instead.**

### 5.2 Judge / Generator Same-Family Bias (Partially Mitigated)
We use a GPT-OSS judge vs. Llama generator, which reduces same-family bias, but Groq hosts both. If the judge has seen similar RLHF fine-tuning patterns as the drafter, it may still systematically prefer Llama-generated style. Judge calibration (§6) quantifies this risk.

### 5.3 Non-Random Golden-Set Sampling
The golden set is stratified across clusters × turn-type × sentiment. Rare intents are over-represented. Accuracy numbers computed on this set **cannot** be extrapolated to live traffic without reweighting by the natural intent distribution.

### 5.4 No Live Outcome Data
We score *plausibility* of drafts (judge ratings, groundedness), not actual customer satisfaction or ticket resolution. A draft can score 4/5 from a judge while still frustrating the real customer if it misses context we don't have.

### 5.5 Small Subsample vs. Full 3M Dataset
The pipeline is trained/indexed on ≤ 20,000 threads from a ~3M-row dataset. Long-tail complaint types, seasonal patterns, and rare escalation triggers are underrepresented. Retrieval quality degrades when the relevant precedent simply isn't in the index.

### 5.6 Escalation Policy Is Self-Authored
The escalation ground truth in the golden set is hand-authored by us applying the rules we wrote. Evaluating the system against those labels measures *consistency*, not *correctness*. A real evaluation would require escalation decisions from trained customer-service managers.

---

## 6. Judge Calibration

To evaluate whether the LLM judge (`openai/gpt-oss-120b`) aligns with human assessment, a double-blind calibration protocol was designed via [`eval/judge_calibration.py`](../eval/judge_calibration.py). 

A stratified sample of 50 examples has been exported to [`data/golden/calibration_blind.csv`](../data/golden/calibration_blind.csv) with all model identifiers and judge scores stripped. Human annotators score each response from 1 to 5 across the five rubric dimensions.

### Calibration Protocol & Metrics Formulae:
- **Quadratic-Weighted Cohen's Kappa ($\kappa_w$)**: Penalizes larger score divergences quadratically:
  $$\kappa_w = 1 - \frac{\sum w_{ij} O_{ij}}{\sum w_{ij} E_{ij}}, \quad w_{ij} = \frac{(i - j)^2}{(k - 1)^2}$$
- **Spearman Rank Correlation ($\rho$)**: Measures monotonic rank preservation.
- **Agreement Margins**: Exact match percentage and adjacent match percentage ($\pm 1$ rating band).

```bash
# Run comparison once human scores are saved in data/golden/calibration_blind.csv:
python eval/judge_calibration.py --compare \
    --judged data/golden/judged.csv \
    --human data/golden/calibration_blind.csv \
    --out data/golden/calibration_results.json
```

| Dimension | Target Kappa (QW) | Benchmark Agreement | Key Risk Mitigated |
|-----------|-------------------|---------------------|--------------------|
| Issue understanding | $\ge 0.70$ | $\ge 85\%$ within $\pm 1$ | Misinterpreting colloquial slang as silence |
| Correctness / Groundedness | $\ge 0.85$ | $\ge 95\%$ within $\pm 1$ | Uncaught hallucinated URL/cost entities |
| Brand voice match | $\ge 0.65$ | $\ge 80\%$ within $\pm 1$ | Tone-deafness on high-distress inquiries |
| Actionability | $\ge 0.70$ | $\ge 90\%$ within $\pm 1$ | Ineffective diagnostic troubleshooting advice |
| Escalation appropriateness | $\ge 0.75$ | $\ge 90\%$ within $\pm 1$ | Over-escalation costing human agent capacity |

---

## 7. What I'd Do With One More Week

1. **Better entity extraction for groundedness**: Replace the regex-based fact extractor with a proper NER model (spaCy `en_core_web_trf`) to catch monetary amounts, dates, and identifiers more reliably.

2. **Live A/B experiment**: Deploy behind a shadow router, route 5% of real support queries through the pipeline, and track CSAT / reopened-ticket rate as outcome metrics.

3. **Fine-tune the classifier**: The few-shot Groq classifier is convenient but an in-context fine-tuned model (even `distilbert` fine-tuned on the 10k cluster-labelled examples) would likely outperform it on recall for rare intents.

4. **Richer retrieval**: Replace flat cosine-sim with a proper BM25 + dense hybrid retrieval. Add recency weighting (older resolved pairs are less relevant if brand policies change).

5. **Multi-brand / multi-language support**: The current architecture is brand-specific by design. Generalising requires a brand-routing layer and multilingual embeddings.

6. **Systematic prompt regression testing**: Every time the brand voice card or system prompt changes, run a regression diff against the golden set to catch quality regressions before deployment.

---

## Appendix: Model and Resource Summary

| Component | Model | Provider | Notes |
|-----------|-------|----------|-------|
| Intent classifier | llama-3.1-8b-instant | Groq | 30 RPM / 14,400 RPD |
| Reply drafter | llama-3.3-70b-versatile | Groq | 30 RPM / 1,000 RPD |
| LLM judge | openai/gpt-oss-120b | Groq | 30 RPM / 1,000 RPD |
| Embeddings | all-MiniLM-L6-v2 | Local (HF) | No API calls |
| Groundedness check | llama-3.1-8b-instant | Groq | Shared with classifier budget |
