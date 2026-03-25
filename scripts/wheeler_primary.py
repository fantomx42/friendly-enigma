"""wheeler-primary — Wheeler-primary agent (small model as pure decoder).

In this mode Wheeler Memory is the cognitive system. The small model reads
Wheeler's attractor state and renders it as natural language.

Usage
-----
    wheeler-primary "What is quantum entanglement?"
    wheeler-primary --interactive
    wheeler-primary --model qwen2.5:1.5b --confidence-floor 0.4
    wheeler-primary --show-state "Tell me about Python decorators"
"""

from __future__ import annotations

import argparse
import sys

from wheeler_memory.decoder import (
    CONFIDENCE_FLOOR,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    WheelerPrimaryAgent,
    extract_state,
    format_state,
)
from wheeler_memory.storage import recall_memory


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="wheeler-primary",
        description="Wheeler-primary agent: Wheeler thinks, small model talks.",
    )
    p.add_argument(
        "message",
        nargs="?",
        help="Query to process. Omit for --interactive mode.",
    )
    p.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run in interactive REPL mode.",
    )
    p.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODEL,
        metavar="MODEL",
        help=f"Ollama decoder model (default: {DEFAULT_MODEL}).",
    )
    p.add_argument(
        "--ollama",
        default=DEFAULT_OLLAMA_URL,
        metavar="URL",
        help=f"Ollama base URL (default: {DEFAULT_OLLAMA_URL}).",
    )
    p.add_argument(
        "--data-dir",
        default=None,
        metavar="DIR",
        help="Wheeler Memory data directory (default: ~/.wheeler_memory).",
    )
    p.add_argument(
        "--confidence-floor",
        type=float,
        default=CONFIDENCE_FLOOR,
        metavar="F",
        help=f"Minimum mean similarity for confident responses (default: {CONFIDENCE_FLOOR}).",
    )
    p.add_argument(
        "--recall-k",
        type=int,
        default=5,
        metavar="N",
        help="Number of memories to recall per query (default: 5).",
    )
    p.add_argument(
        "--show-state",
        action="store_true",
        help="Print Wheeler's decoded state before the response.",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed pipeline info to stderr.",
    )
    return p.parse_args(argv)


def _run_with_state(agent: WheelerPrimaryAgent, message: str, show_state: bool) -> str:
    """Run the agent, optionally printing the intermediate Wheeler state."""
    if show_state:
        # Manual pipeline to show state
        hits = recall_memory(
            message,
            top_k=agent.recall_k,
            data_dir=agent.data_dir,
            use_embedding=True,
            reconstruct=agent.reconstruct,
            reconstruct_alpha=agent.reconstruct_alpha,
        )
        state = extract_state(message, hits, agent.confidence_floor)
        formatted = format_state(state)
        print("--- Wheeler State ---")
        print(formatted)
        print("--- End State ---\n")

    return agent.run(message)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    agent = WheelerPrimaryAgent(
        model=args.model,
        ollama_url=args.ollama,
        data_dir=args.data_dir,
        recall_k=args.recall_k,
        confidence_floor=args.confidence_floor,
        verbose=args.verbose,
    )

    if args.message and not args.interactive:
        try:
            reply = _run_with_state(agent, args.message, args.show_state)
            print(reply)
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        return

    if args.interactive or not args.message:
        print(f"Wheeler Primary  [model={args.model}  ollama={args.ollama}]")
        print("Wheeler thinks, model talks. Type 'exit' to leave.")
        print()
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                break

            try:
                reply = _run_with_state(agent, user_input, args.show_state)
                print(f"Wheeler: {reply}\n")
            except RuntimeError as exc:
                print(f"Error: {exc}\n", file=sys.stderr)


if __name__ == "__main__":
    main()
