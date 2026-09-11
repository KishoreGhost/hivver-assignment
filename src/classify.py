"""
src/classify.py
---------------
Step 4: Intent classifier using Groq llama-3.1-8b-instant with JSON mode.

Returns: {intent: str, confidence: float}

Uses few-shot examples built from taxonomy cluster representatives.
All calls route through the shared cached rate-limited client.
"""

import json
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_client import call_llm_json

MODEL = "llama-3.1-8b-instant"

# ---------------------------------------------------------------------------
# TODO: Populate after step 3 cluster naming.
# Few-shot examples: list of {text, intent} drawn from cluster representatives.
# Include 2-3 per cluster. These will be built by build_few_shots() below.
# ---------------------------------------------------------------------------
# After running taxonomy.py and naming clusters, run:
#   python -c "from src.classify import build_few_shots; build_few_shots()"
# to auto-populate data/sample/few_shots.json, then these get loaded at runtime.
FEW_SHOTS_PATH = Path("data/sample/few_shots.json")

from src.taxonomy import INTENT_NAMES as _TAXONOMY_INTENTS
INTENT_NAMES: list[str] = [name for name in _TAXONOMY_INTENTS.values() if name]


def _load_few_shots() -> list[dict]:
    if FEW_SHOTS_PATH.exists():
        return json.loads(FEW_SHOTS_PATH.read_text())
    return []


def _build_system_prompt(intents: list[str], few_shots: list[dict]) -> str:
    intent_list = "\n".join(f"  - {i}" for i in intents)
    shots_text  = "\n".join(
        f'Customer: "{s["text"]}"\nIntent: {s["intent"]}'
        for s in few_shots[:20]   # cap to avoid token overflow
    )
    return f"""You are an intent classifier for a customer-support system.

Classify the customer message into EXACTLY ONE of these intents:
{intent_list}

Respond ONLY with a JSON object with two fields:
  "intent": one of the listed intent strings (or "other" if none fit)
  "confidence": a float between 0.0 and 1.0

Examples:
{shots_text}
"""


def classify(
    text: str,
    *,
    intents: list[str] | None = None,
    few_shots: list[dict] | None = None,
    bypass_cache: bool = False,
) -> dict:
    """
    Classify a single customer message.

    Returns {"intent": str, "confidence": float}
    """
    if intents is None:
        intents = INTENT_NAMES
    if not intents:
        raise ValueError("INTENT_NAMES is empty — run taxonomy.py first and fill in names.")
    if few_shots is None:
        few_shots = _load_few_shots()

    system = _build_system_prompt(intents, few_shots)
    messages = [
        {"role": "system",  "content": system},
        {"role": "user",    "content": f'Customer message: "{text}"'},
    ]

    result = call_llm_json(
        MODEL, messages,
        temperature=0.1,
        max_tokens=64,
        bypass_cache=bypass_cache,
    )

    # Validate fields
    intent     = str(result.get("intent", "other"))
    confidence = float(result.get("confidence", 0.5))

    if intent not in intents and intent != "other":
        intent = "other"

    return {"intent": intent, "confidence": round(confidence, 3)}


def build_few_shots(clusters_parquet: str = "data/sample/clusters.parquet",
                    n_per_cluster: int = 3) -> None:
    """
    Build few-shot example list from cluster representatives and save to JSON.
    Call this after naming intents in taxonomy.INTENT_NAMES.
    """
    import pandas as pd
    from src.taxonomy import INTENT_NAMES as cluster_names

    if not cluster_names:
        print("[classify] INTENT_NAMES in taxonomy.py is empty — fill it in first.")
        return

    df = pd.read_parquet(clusters_parquet)
    shots = []
    for cid, name in cluster_names.items():
        if cid == -1:
            continue
        subset = df[df["cluster"] == cid]["inbound_text"].dropna().tolist()
        for ex in subset[:n_per_cluster]:
            shots.append({"text": ex, "intent": name})

    FEW_SHOTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEW_SHOTS_PATH.write_text(json.dumps(shots, indent=2))
    print(f"[classify] Saved {len(shots)} few-shot examples to {FEW_SHOTS_PATH}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?",
                    default="My order hasn't arrived yet, what's happening?")
    args = ap.parse_args()
    result = classify(args.text)
    print(result)
