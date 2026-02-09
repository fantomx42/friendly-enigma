#!/usr/bin/env python3
"""
Ralph Chat API — Non-interactive interface for VS Code.

Takes user input and history via JSON, performs Wheeler recall,
queries Qwen, and returns the response + memory metadata.
"""

import json
import sys
import argparse
from pathlib import Path

# Ensure the package root is on sys.path
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from ollama_client import OllamaClient
from ralph_chat import _wheeler_recall, _wheeler_store, DEFAULT_SYSTEM_PROMPT

import contextlib

def run_chat_turn(user_input, history=None, model=None):
    history = history or []
    
    # Redirect internal prints to stderr
    with contextlib.redirect_stdout(sys.stderr):
        # 1. Wheeler Recall
        wheeler_context = _wheeler_recall(user_input)
        
        # 2. Build messages
        messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
        
        # Add previous history
        for msg in history:
            messages.append(msg)
            
        # Add current turn with Wheeler context
        if wheeler_context:
            current_content = f"{user_input}\n\n[Wheeler Memory Context]\n{wheeler_context}"
        else:
            current_content = user_input
            
        messages.append({"role": "user", "content": current_content})
        
        # 3. Query Ollama
        with OllamaClient(model=model) as client:
            response_content = client.chat_full(messages)
            
        # 4. Store in Wheeler
        interaction_to_store = f"User: {user_input}\nRalph: {response_content}"
        _wheeler_store(interaction_to_store, store_type="iteration")
    
    return {
        "response": response_content,
        "wheeler_context": wheeler_context,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", help="JSON input string {input: str, history: list, model: str}")
    args = parser.parse_args()
    
    if args.json:
        try:
            data = json.loads(args.json)
            result = run_chat_turn(
                data.get("input", ""),
                data.get("history", []),
                data.get("model")
            )
            print(json.dumps(result))
        except Exception as e:
            print(json.dumps({"error": str(e)}))
    else:
        # Read from stdin if no arg
        try:
            line = sys.stdin.readline()
            if line:
                data = json.loads(line)
                result = run_chat_turn(
                    data.get("input", ""),
                    data.get("history", []),
                    data.get("model")
                )
                print(json.dumps(result))
        except Exception as e:
            print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    main()