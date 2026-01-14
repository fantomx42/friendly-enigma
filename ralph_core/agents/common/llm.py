import requests
import sys
import os

# Ensure ralph_core is in path for absolute imports
_ralph_core_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ralph_core_path not in sys.path:
    sys.path.insert(0, _ralph_core_path)

from .config import MODELS, OLLAMA_API
from metrics import tracker

def call_model(role: str, prompt: str, stream: bool = False) -> str:
    """
    Generic handler to call a specific role in the swarm.
    """
    model_name = MODELS.get(role)
    if not model_name:
        raise ValueError(f"Unknown role: {role}")

    try:
        response = requests.post(
            OLLAMA_API,
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": stream,
                "options": {
                    "temperature": 0.7 if role == "orchestrator" else 0.2  # Low temp for coding
                }
            },
            timeout=600  # DeepSeek reasoning can take time
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
                duration_ms=data.get("total_duration", 0) // 1_000_000 # Convert ns to ms
            )
        
        if stream:
            # TODO: Implement generator for streaming if needed
            return "Streaming not yet implemented in swarm.py"
        
        return data.get("response", "").strip()

    except Exception as e:
        return f"CRITICAL SWARM ERROR ({role}): {str(e)}"
