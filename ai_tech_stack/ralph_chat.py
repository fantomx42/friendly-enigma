#!/usr/bin/env python3
"""
Ralph Chat — Interactive Chat with Wheeler Memory.

Uses OllamaClient for persistent streaming connections.
Queries Wheeler Memory for context before every turn and stores
the interaction results back to memory.

Usage:
    python ralph_chat.py
    RALPH_MODEL=qwen3:8b python ralph_chat.py
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Ensure the package root is on sys.path so imports work
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from ollama_client import OllamaClient, OllamaConnectionError, OllamaError

logger = logging.getLogger("ralph_chat")

DEFAULT_SYSTEM_PROMPT = (
    """You are Ralph, an intelligent AI assistant with a persistent memory called Wheeler Memory.

When answering:
1. Use the provided [Wheeler Memory Context] to inform your response. If you have done something before, recall how it went.
2. Be concise and helpful.
3. If the user asks you to remember something specific, acknowledge it (it will be stored automatically)."""
)

def _wheeler_recall(prompt: str) -> str:
    """Recall relevant context from Wheeler Memory (best-effort)."""
    wheeler_recall_script = _HERE.parent / "wheeler_recall.py"
    if not wheeler_recall_script.exists():
        return ""
    try:
        import subprocess
        # Search for top 3 relevant memories
        result = subprocess.run(
            [sys.executable, str(wheeler_recall_script), prompt, "--top-k", "3"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        recalled = result.stdout.strip()
        if recalled:
            mem_count = recalled.count("\n") + 1
            print(f"\n[Wheeler] Recalled {mem_count} related memories.", flush=True)
        return recalled
    except Exception as exc:
        logger.debug("Wheeler recall failed: %s", exc)
        return ""


def _wheeler_store(text_to_store: str, store_type: str = "iteration") -> None:
    """Store text in Wheeler Memory (best-effort)."""
    wheeler_store_script = _HERE.parent / "wheeler_store.py"
    if not wheeler_store_script.exists():
        return
    try:
        import subprocess
        # We pass the text directly via stdin or --text
        # Using --text for simplicity, but subprocess.input is safer for large text
        result = subprocess.run(
            [sys.executable, str(wheeler_store_script), "--text", text_to_store, "--type", store_type],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            print(f"[Wheeler] Memory stored ({store_type}).", flush=True)
        else:
            print(f"Wheeler store error: {result.stderr}", flush=True)
    except Exception as exc:
        print(f"Wheeler store failed: {exc}", flush=True)


def chat_loop(
    model: str | None = None,
    num_ctx: int | None = None,
    base_url: str | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
):
    """Run the interactive chat loop."""
    
    print("=" * 60)
    print("Ralph Chat with Wheeler Memory")
    print(f"Model: {model or 'default'}")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 60)

    with OllamaClient(base_url=base_url, model=model, num_ctx=num_ctx) as client:
        if not client.is_available():
            print(f"Error: Ollama is not reachable at {client.base_url}")
            return

        # We maintain a list of messages for the chat history
        messages = [{"role": "system", "content": system_prompt}]

        while True:
            try:
                user_input = input("\n>>> ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit"):
                    print("Goodbye!")
                    break
            except EOFError:
                break

            # 1. Recall from Wheeler based on user input
            wheeler_context = _wheeler_recall(user_input)
            
            # 2. Construct the message for this turn
            if wheeler_context:
                full_user_content = (
                    f"{user_input}\n\n"
                    f"[Wheeler Memory Context]\n{wheeler_context}"
                )
            else:
                full_user_content = user_input

            # Add user message to history
            messages.append({"role": "user", "content": full_user_content})

            # 3. Stream Response
            print("\nRalph: ", end="", flush=True)
            response_content = ""
            
            try:
                # Stream the response
                for chunk in client.chat(messages, stream=True):
                    content = chunk.get("message", {}).get("content", "")
                    print(content, end="", flush=True)
                    response_content += content
                
                print() # Newline after response

                # 4. Update history (append assistant response)
                messages.append({"role": "assistant", "content": response_content})

                # 5. Store the interaction in Wheeler
                interaction_to_store = f"User: {user_input}\nRalph: {response_content}"
                _wheeler_store(interaction_to_store, store_type="iteration")

            except KeyboardInterrupt:
                print("\nInterrupted.")
                break
            except Exception as e:
                print(f"\nError: {e}")


def main():
    parser = argparse.ArgumentParser(description="Ralph Chat Interface")
    parser.add_argument("--model", default=None, help="Ollama model name")
    parser.add_argument("--num-ctx", type=int, default=None, help="Context window size")
    parser.add_argument("--base-url", default=None, help="Ollama URL")
    
    args = parser.parse_args()

    chat_loop(
        model=args.model,
        num_ctx=args.num_ctx,
        base_url=args.base_url
    )

if __name__ == "__main__":
    main()