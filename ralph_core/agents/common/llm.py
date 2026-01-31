import requests
import sys
import os

# Ensure ralph_core is in path for absolute imports
_ralph_core_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ralph_core_path not in sys.path:
    sys.path.insert(0, _ralph_core_path)

from .config import MODELS, OLLAMA_API, LLAMA_CPP_API
from metrics import tracker

try:
    from ollama_client import OllamaClient
except ImportError:
    # Fallback for different execution environments
    from ralph_core.ollama_client import OllamaClient

# Global client instance
_ollama_client = OllamaClient()

USE_V2 = os.environ.get("RALPH_V2", "0") == "1"

def call_model(role: str, prompt: str, stream: bool = False) -> str:
    """
    Generic handler to call a specific role in the swarm.
    Now with VRAM awareness via OllamaClient.
    """
    model_name = MODELS.get(role)
    if not model_name:
        raise ValueError(f"Unknown role: {role}")

    # Emit agent start marker for GUI
    print(f"[AGENT:{role.upper()}:START]", flush=True)

    try:
        if USE_V2:
            # ... (keep existing v2 logic)
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
            result = data.get("content", "").strip()

        else:
            # Use the VRAM-aware OllamaClient
            data = _ollama_client.generate(
                model=model_name,
                prompt=prompt,
                stream=stream,
                keep_alive="10m",
                options={
                    "temperature": 0.7 if role == "orchestrator" else 0.2
                }
            )
            
            # Log Metrics
            if not stream:
                tracker.log_llm_call(
                    model=model_name,
                    prompt_tokens=data.get("prompt_eval_count", 0),
                    completion_tokens=data.get("eval_count", 0),
                    duration_ms=data.get("total_duration", 0) // 1_000_000
                )
            
            result = data.get("response", "").strip()
        
        # Emit agent end marker for GUI
        print(f"[AGENT:{role.upper()}:END]", flush=True)
        return result

    except Exception as e:
        print(f"[AGENT:{role.upper()}:END]", flush=True)
        return f"CRITICAL SWARM ERROR ({role}): {str(e)}"

    except Exception as e:
        print(f"[AGENT:{role.upper()}:END]", flush=True)
        return f"CRITICAL SWARM ERROR ({role}): {str(e)}"
