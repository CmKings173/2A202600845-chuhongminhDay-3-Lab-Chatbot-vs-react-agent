"""
Lab 3 Entry Point — run Chatbot baseline OR ReAct Agent interactively.

Usage:
    python main.py --mode chatbot
    python main.py --mode agent
    python main.py --mode compare     # run all test cases on both, print table
"""

import os
import sys
import argparse
import json
from dotenv import load_dotenv

load_dotenv()

# Make sure src/ is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.provider_factory import create_provider
from src.chatbot.chatbot import Chatbot
from src.agent.agent import ReActAgent
from src.tools.ecommerce_tools import TOOL_REGISTRY
from src.telemetry.metrics import tracker


# ---------------------------------------------------------------------------
# Test cases — designed to show where Agent wins vs Chatbot
# ---------------------------------------------------------------------------
TEST_CASES = [
    {
        "id": "TC-01",
        "type": "simple",
        "query": "What is your return policy?",
        "expected_winner": "chatbot",   # simple Q → chatbot is faster / cheaper
    },
    {
        "id": "TC-02",
        "type": "multi-step",
        "query": "How many iPhones do we have in stock and what is the unit price?",
        "expected_winner": "agent",
    },
    {
        "id": "TC-03",
        "type": "multi-step",
        "query": (
            "I want to buy 2 iPhones using coupon code 'WINNER' "
            "and ship them to Hanoi. What is the total estimated cost?"
        ),
        "expected_winner": "agent",
    },
    {
        "id": "TC-04",
        "type": "multi-step",
        "query": "What discount does the VIP50 coupon give and what is the MacBook Pro price after that discount?",
        "expected_winner": "agent",
    },
    {
        "id": "TC-05",
        "type": "tool-error",
        "query": "Check the stock for 'flying_car' and the price of 'banana'.",
        "expected_winner": "agent",   # agent handles not-found gracefully
    },
]


# ---------------------------------------------------------------------------
# Mode: Interactive chat
# ---------------------------------------------------------------------------

def run_interactive(mode: str) -> None:
    llm = create_provider()
    print(f"\n[Lab 3] Provider: {llm.model_name} | Mode: {mode.upper()}")
    print("Type 'quit' to exit.\n")

    if mode == "chatbot":
        bot = Chatbot(llm=llm)
        run_fn = bot.run
    else:
        agent = ReActAgent(llm=llm, tools=TOOL_REGISTRY, max_steps=6)
        run_fn = agent.run

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        print(f"\n{'Chatbot' if mode == 'chatbot' else 'Agent'}: ", end="", flush=True)
        response = run_fn(user_input)
        print(response)
        print()

    summary = tracker.get_summary()
    print("\n=== Session Summary ===")
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# Mode: Comparison table
# ---------------------------------------------------------------------------

def run_compare() -> None:
    llm = create_provider()
    print(f"\n[Lab 3] Comparison Mode | Provider: {llm.model_name}")
    print("=" * 70)

    chatbot = Chatbot(llm=llm)
    agent = ReActAgent(llm=llm, tools=TOOL_REGISTRY, max_steps=6)

    rows = []
    for tc in TEST_CASES:
        print(f"\n[{tc['id']}] {tc['type'].upper()} — {tc['query'][:60]}...")

        # --- Chatbot ---
        tracker.reset()
        try:
            cb_answer = chatbot.run(tc["query"])
        except Exception as exc:
            cb_answer = f"ERROR: {exc}"
        cb_metrics = tracker.get_summary()

        # --- Agent ---
        tracker.reset()
        try:
            ag_answer = agent.run(tc["query"])
        except Exception as exc:
            ag_answer = f"ERROR: {exc}"
        ag_metrics = tracker.get_summary()

        rows.append({
            "id": tc["id"],
            "type": tc["type"],
            "chatbot_tokens": cb_metrics.get("total_tokens", 0),
            "chatbot_latency_ms": cb_metrics.get("latency_avg_ms", 0),
            "chatbot_cost_usd": cb_metrics.get("total_cost_usd", 0),
            "agent_tokens": ag_metrics.get("total_tokens", 0),
            "agent_latency_ms": ag_metrics.get("latency_avg_ms", 0),
            "agent_cost_usd": ag_metrics.get("total_cost_usd", 0),
            "expected_winner": tc["expected_winner"],
        })

        print(f"  Chatbot: {cb_answer[:120]}")
        print(f"  Agent:   {ag_answer[:120]}")

    # Print comparison table
    print("\n\n=== EVALUATION TABLE ===")
    print(f"{'ID':<6} {'Type':<12} {'CB Tokens':>10} {'AG Tokens':>10} "
          f"{'CB Lat(ms)':>12} {'AG Lat(ms)':>12} "
          f"{'CB Cost':>10} {'AG Cost':>10} {'Expected Winner':<15}")
    print("-" * 105)
    for r in rows:
        print(
            f"{r['id']:<6} {r['type']:<12} "
            f"{r['chatbot_tokens']:>10} {r['agent_tokens']:>10} "
            f"{r['chatbot_latency_ms']:>12.0f} {r['agent_latency_ms']:>12.0f} "
            f"{r['chatbot_cost_usd']:>10.6f} {r['agent_cost_usd']:>10.6f} "
            f"{r['expected_winner']:<15}"
        )


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Lab 3: Chatbot vs ReAct Agent")
    parser.add_argument(
        "--mode",
        choices=["chatbot", "agent", "compare"],
        default="agent",
        help="Run mode: chatbot | agent | compare",
    )
    args = parser.parse_args()

    if args.mode == "compare":
        run_compare()
    else:
        run_interactive(args.mode)


if __name__ == "__main__":
    main()
