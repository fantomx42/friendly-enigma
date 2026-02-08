#!/usr/bin/env python3
"""
Ralph Simple — Single-model iterative loop with Wheeler Memory.

Uses OllamaClient for persistent streaming connections so the
TCP socket stays open across iterations instead of reconnecting
every time.

Usage:
    python ralph_simple.py "Your objective here"
    RALPH_MODEL=qwen3:8b python ralph_simple.py "Your objective"
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Ensure the package root is on sys.path so imports work when
# invoked directly (python ralph_simple.py) or via ralph_loop.sh.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from ollama_client import OllamaClient, OllamaConnectionError, OllamaError

logger = logging.getLogger("ralph")

# Completion token that signals the objective is satisfied.
COMPLETION_TOKEN = "<promise>COMPLETE</promise>"

DEFAULT_MAX_ITERATIONS = 50
DEFAULT_OUTPUT_DIR = "./ralph_output"
DEFAULT_SYSTEM_PROMPT = (
    "You are Ralph, an autonomous AI agent. Work toward the objective "
    "iteratively. When the objective is fully satisfied, include the "
    "exact token <promise>COMPLETE</promise> in your response.\n\n"
    "You have a persistent memory called Wheeler Memory. Between iterations, "
    "relevant memories may be recalled and provided to you in a block tagged "
    "[Wheeler Memory Context]. Treat this context like your own past experience "
    "— use it when relevant, but recognize that older memories may be less "
    "accurate. Your outputs are also stored to memory for future recall."
)


def _wheeler_recall(prompt: str) -> str:
    """Recall relevant context from Wheeler Memory (best-effort)."""
    wheeler_recall_script = _HERE.parent / "wheeler_recall.py"
    if not wheeler_recall_script.exists():
        return ""
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, str(wheeler_recall_script), prompt],
            capture_output=True,
            text=True,
            timeout=30,
        )
        recalled = result.stdout.strip()
        if recalled:
            mem_count = recalled.count("\n---\n") + 1
            logger.info("Wheeler recalled %d memory(s)", mem_count)
            print(f"[Wheeler] Recalled {mem_count} memory(s)", flush=True)
        else:
            logger.info("Wheeler recall returned no memories")
            print("[Wheeler] No memories found", flush=True)
        return recalled
    except Exception as exc:
        logger.debug("Wheeler recall failed: %s", exc)
        return ""


def _wheeler_store(output_path: str) -> None:
    """Store iteration output in Wheeler Memory (best-effort)."""
    wheeler_store_script = _HERE.parent / "wheeler_store.py"
    if not wheeler_store_script.exists():
        return
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, str(wheeler_store_script), output_path,
             "--type", "iteration"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            logger.info("Wheeler stored iteration output: %s", output_path)
            print("[Wheeler] Stored iteration output", flush=True)
        else:
            logger.debug("Wheeler store returned non-zero: %s", result.stderr)
    except Exception as exc:
        logger.debug("Wheeler store failed: %s", exc)


def run_loop(
    objective: str,
    model: str | None = None,
    num_ctx: int | None = None,
    base_url: str | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> bool:
    """Run the Ralph iterative loop.

    Returns True if the completion promise was found, False if
    max iterations were exhausted.
    """
    os.makedirs(output_dir, exist_ok=True)

    # One client for the entire loop — the underlying Session keeps
    # the HTTP connection alive between generate() calls.
    with OllamaClient(base_url=base_url, model=model, num_ctx=num_ctx) as client:
        logger.info("Ollama client: %s", client)

        if not client.is_available():
            logger.error(
                "Ollama is not reachable at %s. Is the server running?",
                client.base_url,
            )
            return False

        logger.info("Model: %s", client.model)
        logger.info("Max iterations: %d", max_iterations)

        # Context array carries forward between generate() calls so
        # the model has conversational continuity without resending
        # all prior text.
        ollama_context = None

        for iteration in range(1, max_iterations + 1):
            print(f"\n>>> Iteration {iteration} / {max_iterations}")
            print("-" * 50)

            # Recall relevant Wheeler Memory context
            print("[Wheeler] Recalling...", flush=True)
            wheeler_context = _wheeler_recall(objective)

            # Build the prompt for this iteration
            if wheeler_context:
                prompt = (
                    f"[Wheeler Memory Context]\n{wheeler_context}\n\n"
                    f"---\n\n[Objective]\n{objective}"
                )
            else:
                prompt = f"[Objective]\n{objective}"

            if iteration > 1:
                prompt += (
                    f"\n\n[Iteration {iteration}] "
                    "Continue working. If done, include "
                    "<promise>COMPLETE</promise>."
                )

            # Save prompt for debugging
            prompt_path = Path(output_dir) / f"iteration_{iteration}_prompt.txt"
            prompt_path.write_text(prompt, encoding="utf-8")

            # Stream the response over the persistent connection
            print(f"[Ollama:{client.model}] Streaming...", flush=True)
            output_parts: list[str] = []

            try:
                for chunk in client.generate(
                    prompt,
                    system=system_prompt,
                    context=ollama_context,
                ):
                    token = chunk.get("response", "")
                    output_parts.append(token)
                    print(token, end="", flush=True)

                    # Capture the context array from the final chunk so
                    # the next iteration has conversational memory.
                    if chunk.get("done"):
                        ollama_context = chunk.get("context")

            except OllamaConnectionError as exc:
                logger.error("Connection lost: %s", exc)
                print(f"\n[ERROR] {exc}")
                return False
            except OllamaError as exc:
                logger.error("Ollama error: %s", exc)
                print(f"\n[ERROR] {exc}")
                return False

            full_output = "".join(output_parts)
            print()  # newline after streamed output

            # Save output
            output_path = Path(output_dir) / f"iteration_{iteration}_output.txt"
            output_path.write_text(full_output, encoding="utf-8")

            # Store in Wheeler Memory
            print("[Wheeler] Storing...", flush=True)
            _wheeler_store(str(output_path))

            # Check for completion
            if COMPLETION_TOKEN in full_output:
                print("\n" + "=" * 50)
                print(f"Completion promise found after {iteration} iteration(s)")
                print("=" * 50)
                return True

            # Brief pause between iterations
            time.sleep(0.5)

    print("\n" + "=" * 50)
    print(f"Max iterations reached ({max_iterations})")
    print("=" * 50)
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Ralph AI — single-model iterative loop"
    )
    parser.add_argument("objective", help="The objective to work toward")
    parser.add_argument(
        "--model", default=None,
        help="Ollama model name (default: RALPH_MODEL env or qwen3:8b)"
    )
    parser.add_argument(
        "--num-ctx", type=int, default=None,
        help="Context window size in tokens (default: RALPH_NUM_CTX env or 32768)"
    )
    parser.add_argument(
        "--base-url", default=None,
        help="Ollama server URL (default: OLLAMA_HOST env or http://localhost:11434)"
    )
    parser.add_argument(
        "--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS,
        help=f"Maximum iterations (default: {DEFAULT_MAX_ITERATIONS})"
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for iteration outputs (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    print("=" * 50)
    print("Ralph AI — Streaming Loop")
    print("=" * 50)

    success = run_loop(
        objective=args.objective,
        model=args.model,
        num_ctx=args.num_ctx,
        base_url=args.base_url,
        max_iterations=args.max_iterations,
        output_dir=args.output_dir,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
