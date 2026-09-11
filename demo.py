"""
demo.py
-------
Interactive CLI to test the AppleSupport AI Agent pipeline.

Run:
  python demo.py
  python demo.py --case 1
  python demo.py --text "My iPhone battery dies in 2 hours"
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Ensure UTF-8 output on Windows terminal with immediate flushing
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

import argparse
from src.pipeline import handle_message

PRESET_CASES = [
    {
        "id": 1,
        "name": "Profanity / Slang (Autocorrect Bug)",
        "text": "Dude Fuck this fucking bullshit “i” shit. Get your fucking shit together @AppleSupport.",
        "turns": []
    },
    {
        "id": 2,
        "name": "Compound Inquiries (Battery vs Autocorrect)",
        "text": "Yay, I️ won the lottery with the latest iPhone glitch! And @AppleSupport still claims my bad battery doesn’t qualify for the recall so YAY for me!",
        "turns": []
    },
    {
        "id": 3,
        "name": "Hardware vs OS Glitch Ambiguity (Screen Freeze)",
        "text": "@AppleSupport my iphone's screen stopped responding in the middle of nowhere as if it got busy. its been 15 long hours yet NO hope. how come?",
        "turns": []
    },
    {
        "id": 4,
        "name": "Multi-Turn Conversation Fatigue (3 Prior Unresolved Turns)",
        "text": "@AppleSupport Oh, and that weird I️ autocorrect thing just started happening with me, too. Yesterday.",
        "turns": [
            {"role": "user", "text": "My phone is acting really weird today"},
            {"role": "agent", "text": "We'd be glad to help. Could you tell us what model phone you're using?"},
            {"role": "user", "text": "It's an iPhone 8 on iOS 11. Still having issues."}
        ]
    },
    {
        "id": 5,
        "name": "Non-English Language Routing (Portuguese)",
        "text": "@AppleSupport @AppleSupport Não vejo que recebi msg em nenhum app, nada. Fico o dia todo sem uma notificação sequer",
        "turns": []
    }
]


def print_result(user_text: str, prior_turns: list, res: dict) -> None:
    print("\n" + "=" * 70)
    print(f"👤 CUSTOMER INQUIRY:")
    print(f"   \"{user_text}\"")
    if prior_turns:
        print(f"   (Prior turns: {len(prior_turns)})")

    print("\n🎯 INTENT CLASSIFICATION:")
    print(f"   Intent     : {res['intent']}")
    print(f"   Confidence : {res['confidence']:.2f}")

    print("\n🛡️  ESCALATION GATE:")
    if res['escalate']:
        print(f"   Status     : 🔴 ESCALATED TO HUMAN AGENT")
        print(f"   Reason     : {res['escalate_reason']}")
    else:
        print(f"   Status     : 🟢 AUTOMATED RESOLUTION APPROVED")
        print(f"   Reason     : {res['escalate_reason']}")

    print("\n💬 DRAFT REPLY:")
    for line in res['draft_reply'].splitlines():
        print(f"   {line}")

    if res.get("precedents"):
        top = res["precedents"][0]
        print(f"\n🔍 TOP RETRIEVED PRECEDENT (Similarity: {top.get('score', 0):.2f}):")
        print(f"   Past Inbound  : \"{top.get('inbound', '')[:90]}...\"")
        print(f"   Past Solution : \"{top.get('outbound', '')[:90]}...\"")
    print("=" * 70 + "\n")


def run_interactive():
    print("=" * 70)
    print("   🍎 AppleSupport AI Customer Support Agent - Interactive Demo")
    print("=" * 70)
    print("Commands:")
    print("  Type any message to test the agent.")
    print("  Type 'cases' to see the 5 failure mode test cases.")
    print("  Type 'case 1' through 'case 5' to run a specific test case.")
    print("  Type 'exit' or 'quit' to stop.\n")

    thread_history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        if cmd == "cases":
            print("\nAvailable Preset Cases from Report Section 4:")
            for c in PRESET_CASES:
                print(f"  [{c['id']}] {c['name']}")
                print(f"      \"{c['text'][:80]}...\"")
            print()
            continue

        if cmd.startswith("case "):
            try:
                cid = int(cmd.split()[1])
                case = next((c for c in PRESET_CASES if c["id"] == cid), None)
                if case:
                    print(f"\nRunning Case {case['id']}: {case['name']}...")
                    res = handle_message(case["text"], thread_history=case["turns"])
                    print_result(case["text"], case["turns"], res)
                    continue
                else:
                    print("Invalid case number. Choose 1 to 5.")
                    continue
            except ValueError:
                pass

        if cmd == "clear":
            thread_history = []
            print("Conversation history cleared.")
            continue

        # Normal message
        res = handle_message(user_input, thread_history=thread_history)
        print_result(user_input, thread_history, res)

        # Track conversation history for multi-turn testing
        thread_history.append({"role": "user", "text": user_input})
        thread_history.append({"role": "agent", "text": res["draft_reply"]})


def main():
    ap = argparse.ArgumentParser(description="AppleSupport Agent Demo")
    ap.add_argument("--text", type=str, help="Single query text to test")
    ap.add_argument("--case", type=int, choices=[1, 2, 3, 4, 5], help="Run a specific test case (1-5)")
    args = ap.parse_args()

    if args.case:
        case = PRESET_CASES[args.case - 1]
        print(f"Running Case {case['id']}: {case['name']}...")
        res = handle_message(case["text"], thread_history=case["turns"])
        print_result(case["text"], case["turns"], res)
    elif args.text:
        res = handle_message(args.text)
        print_result(args.text, [], res)
    else:
        run_interactive()


if __name__ == "__main__":
    main()
