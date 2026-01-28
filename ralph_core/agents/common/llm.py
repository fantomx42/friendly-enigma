import requests
import sys
import os

# Ensure ralph_core is in path for absolute imports
_ralph_core_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ralph_core_path not in sys.path:
    sys.path.insert(0, _ralph_core_path)

from .config import MODELS, OLLAMA_API, LLAMA_CPP_API
from metrics import tracker

USE_V2 = os.environ.get("RALPH_V2", "0") == "1"

def call_model(role: str, prompt: str, stream: bool = False) -> str:
    """
    Generic handler to call a specific role in the swarm.
    """
    model_name = MODELS.get(role)
    if not model_name:
        raise ValueError(f"Unknown role: {role}")

    # Emit agent start marker for GUI
    print(f"[AGENT:{role.upper()}:START]", flush=True)

    try:
        if USE_V2:
            # Ralph v2 uses llama-server (Chalmers + Wiggum)
            # This uses speculative decoding automatically on the server side
            response = requests.post(
                LLAMA_CPP_API,
                json={
                    "prompt": prompt,
                    "stream": stream,
                    "temperature": 0.7 if role == "orchestrator" else 0.2,
                },
                timeout=600
            )
            response.raise_for_status()
            data = response.json()
            
            # Log Metrics (llama.cpp format)
            if not stream:
                tracker.log_llm_call(
                    model="v2-chalmers-wiggum",
                    prompt_tokens=data.get("tokens_evaluated", 0),
                    completion_tokens=data.get("tokens_predicted", 0),
                    duration_ms=int(data.get("timings", {}).get("predicted_ms", 0))
                )
            
            result = data.get("content", "").strip()

        else:
            # Legacy/V1 uses Ollama
            response = requests.post(
                OLLAMA_API,
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": stream,
                    "keep_alive": "10m",
                    "options": {
                        "temperature": 0.7 if role == "orchestrator" else 0.2
                    }
                },
                timeout=600
            )
            if response.status_code == 404:
                return f"Error: Model {model_name} not found. Run 'ollama pull {model_name}'"
            
            response.raise_for_status()
            data = response.json()
            
            # Log Metrics
            if not stream:
                tracker.log_llm_call(
                    model=model_name,
                    prompt_tokens=data.get("prompt_eval_count", 0),
                    completion_tokens=data.get("eval_count", 0),
                    duration_ms=data.get("total_duration", 0) // 1_000_000
                )
            
            result = data.get("response", "").strip()
        
        if stream:
            print(f"[AGENT:{role.upper()}:END]", flush=True)
            return "Streaming not yet implemented in swarm.py"

        # Emit agent end marker for GUI
        print(f"[AGENT:{role.upper()}:END]", flush=True)

        return result

    except Exception as e:
        print(f"[AGENT:{role.upper()}:END]", flush=True)
        return f"CRITICAL SWARM ERROR ({role}): {str(e)}"
