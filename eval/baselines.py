"""
eval/baselines.py
-----------------
Step 10: Baseline systems for comparison.

Baseline A — Trivial:
  - Intent:    always predict majority class from training distribution
  - Draft:     one fixed canned reply
  - Escalate:  never escalate

Baseline B — Simple:
  - Intent:    TF-IDF + Logistic Regression (scikit-learn)
  - Draft:     nearest-neighbor lookup — return the outbound reply of the
               single closest historical inbound (no generation)
  - Escalate:  keyword list (urgent, fraud, legal, cancel, refund + angry words)

Usage:
  python eval/baselines.py --golden data/golden/golden_set.csv \
                           --threads data/sample/brand_threads.parquet
"""

import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

SEED = 42

CANNED_REPLY = (
    "Thank you for reaching out to us! We're sorry to hear you're experiencing "
    "an issue. Please DM us your account details and we'll look into this right away."
)

ESCALATION_KEYWORDS = {
    "urgent", "lawsuit", "legal", "fraud", "scam", "police", "lawyer",
    "attorney", "cancel", "refund", "charge", "overcharged", "angry",
    "furious", "disgusted", "terrible", "worst", "horrible", "abusive",
}


# ---------------------------------------------------------------------------
# Baseline A: Trivial
# ---------------------------------------------------------------------------

class TrivialBaseline:
    """Always predict majority class; always return canned reply; never escalate."""

    def __init__(self):
        self.majority_intent = None

    def fit(self, df: pd.DataFrame) -> None:
        """Learn majority intent from labeled data."""
        if "gold_intent" in df.columns and df["gold_intent"].notna().any():
            self.majority_intent = df["gold_intent"].value_counts().idxmax()
        else:
            self.majority_intent = "other"

    def classify(self, text: str) -> dict:
        return {"intent": self.majority_intent, "confidence": 1.0}

    def draft(self, text: str, **kwargs) -> str:
        return CANNED_REPLY

    def escalate(self, **kwargs) -> dict:
        return {"escalate": False, "reason": "trivial baseline: never escalate"}

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["pred_intent"]         = self.majority_intent
        df["pred_confidence"]     = 1.0
        df["draft_reply"]         = CANNED_REPLY
        df["pred_escalate"]       = False
        df["pred_escalate_reason"] = "trivial baseline: never escalate"
        return df


# ---------------------------------------------------------------------------
# Baseline B: Simple
# ---------------------------------------------------------------------------

class SimpleBaseline:
    """TF-IDF + LR for intent; NN lookup for draft; keyword list for escalation."""

    def __init__(self):
        self.clf_pipeline: Pipeline | None = None
        self.train_texts:  list[str] = []
        self.train_replies: list[str] = []
        self._tfidf_train_matrix = None

    def fit(self, df_labeled: pd.DataFrame, df_threads: pd.DataFrame) -> None:
        """
        Fit on labeled golden set (intent) + thread pairs (draft lookup).
        """
        # Intent classifier
        mask = df_labeled["gold_intent"].notna() & (df_labeled["gold_intent"] != "")
        X    = df_labeled.loc[mask, "inbound_text"].tolist()
        y    = df_labeled.loc[mask, "gold_intent"].tolist()
        if X:
            self.clf_pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(max_features=15000, ngram_range=(1, 2),
                                          sublinear_tf=True)),
                ("lr",    LogisticRegression(max_iter=1000, random_state=SEED,
                                             C=5.0, class_weight="balanced")),
            ])
            self.clf_pipeline.fit(X, y)
        else:
            self.clf_pipeline = None

        # NN draft lookup: use all thread inbound texts as index
        self.train_texts   = df_threads["inbound_text"].fillna("").tolist()
        self.train_replies = df_threads["outbound_text"].fillna("").tolist()

        if self.train_texts:
            self._tfidf_vec = TfidfVectorizer(max_features=15000, ngram_range=(1, 2),
                                               sublinear_tf=True)
            self._tfidf_train_matrix = self._tfidf_vec.fit_transform(self.train_texts)

    def classify(self, text: str) -> dict:
        if self.clf_pipeline is None:
            return {"intent": "other", "confidence": 0.5}
        intent = self.clf_pipeline.predict([text])[0]
        proba  = self.clf_pipeline.predict_proba([text]).max()
        return {"intent": intent, "confidence": round(float(proba), 3)}

    def draft(self, text: str, **kwargs) -> str:
        if self._tfidf_train_matrix is None or not self.train_texts:
            return CANNED_REPLY
        q = self._tfidf_vec.transform([text])
        # Cosine similarity via dot product (TF-IDF vectors are L2-normalised internally)
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(q, self._tfidf_train_matrix)[0]
        best = int(np.argmax(sims))
        return self.train_replies[best]

    def escalate(self, text: str, **kwargs) -> dict:
        tokens = set(text.lower().split())
        hit    = tokens & ESCALATION_KEYWORDS
        if hit:
            return {"escalate": True, "reason": f"keyword match: {hit}"}
        return {"escalate": False, "reason": "no escalation keywords found"}

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        intents, confs, drafts, escs, esc_reasons = [], [], [], [], []
        for _, row in df.iterrows():
            text = str(row["inbound_text"])
            clf  = self.classify(text)
            dr   = self.draft(text)
            esc  = self.escalate(text=text)
            intents.append(clf["intent"])
            confs.append(clf["confidence"])
            drafts.append(dr)
            escs.append(esc["escalate"])
            esc_reasons.append(esc["reason"])

        df["pred_intent"]          = intents
        df["pred_confidence"]      = confs
        df["draft_reply"]          = drafts
        df["pred_escalate"]        = escs
        df["pred_escalate_reason"] = esc_reasons
        return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden",  default="data/golden/golden_set.csv")
    ap.add_argument("--threads", default="data/sample/brand_threads.parquet")
    args = ap.parse_args()

    golden  = pd.read_csv(args.golden)
    threads = pd.read_parquet(args.threads)

    labeled = golden[golden["gold_intent"].notna() & (golden["gold_intent"] != "")]

    # If golden labels are not yet filled, train on clustered dataset
    if len(labeled) == 0:
        clusters_path = Path("data/sample/clusters.parquet")
        if clusters_path.exists():
            from src.taxonomy import INTENT_NAMES
            cdf = pd.read_parquet(str(clusters_path))
            cdf = cdf[cdf["cluster"].isin(INTENT_NAMES.keys()) & (cdf["cluster"] != -1)].copy()
            cdf["gold_intent"] = cdf["cluster"].map(INTENT_NAMES)
            train_df = cdf[["inbound_text", "gold_intent"]]
            print(f"[baselines] Golden labels blank; training baseline classifier on {len(train_df)} taxonomy cluster examples")
        else:
            train_df = labeled
    else:
        train_df = labeled

    # Trivial baseline
    triv = TrivialBaseline()
    triv.fit(train_df)
    triv_preds = triv.predict_batch(golden)
    print(f"[baselines] Trivial majority intent: {triv.majority_intent}")

    # Simple baseline
    simp = SimpleBaseline()
    simp.fit(train_df, threads)
    simp_preds = simp.predict_batch(golden)

    if len(labeled) > 0:
        y_true = labeled["gold_intent"].tolist()
        print("\n--- Trivial baseline intent accuracy ---")
        t_acc = accuracy_score(y_true, triv_preds.loc[labeled.index, "pred_intent"])
        t_f1  = f1_score(y_true, triv_preds.loc[labeled.index, "pred_intent"],
                         average="macro", zero_division=0)
        print(f"  accuracy={t_acc:.3f}  macro-F1={t_f1:.3f}")

        print("\n--- Simple baseline intent accuracy ---")
        s_acc = accuracy_score(y_true, simp_preds.loc[labeled.index, "pred_intent"])
        s_f1  = f1_score(y_true, simp_preds.loc[labeled.index, "pred_intent"],
                         average="macro", zero_division=0)
        print(f"  accuracy={s_acc:.3f}  macro-F1={s_f1:.3f}")

    # Save predictions
    triv_out = Path("data/golden/baseline_trivial.csv")
    simp_out = Path("data/golden/baseline_simple.csv")
    triv_preds.to_csv(str(triv_out), index=False)
    simp_preds.to_csv(str(simp_out), index=False)
    print(f"\n[baselines] Saved to {triv_out} and {simp_out}")


if __name__ == "__main__":
    main()
