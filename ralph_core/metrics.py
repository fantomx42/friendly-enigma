import json
import os
import time
from typing import Dict, Any

METRICS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "metrics.jsonl")

class Metrics:
    def __init__(self):
        self.current_session = {
            "timestamp": time.time(),
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "tasks": [],
            "gpu_usage": []
        }

    def log_llm_call(self, model: str, prompt_tokens: int, completion_tokens: int, duration_ms: int):
        entry = {
            "type": "llm_call",
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "duration_ms": duration_ms,
            "timestamp": time.time()
        }
        self.save_entry(entry)

    def log_iteration(self, iteration: int, status: str):
        entry = {
            "type": "iteration",
            "iteration": iteration,
            "status": status,
            "timestamp": time.time()
        }
        self.save_entry(entry)

    def log_gpu(self, usage: float, memory: float):
        entry = {
            "type": "gpu",
            "usage_percent": usage,
            "memory_percent": memory,
            "timestamp": time.time()
        }
        self.save_entry(entry)

    def save_entry(self, entry: Dict[str, Any]):
        try:
            # Print to stdout for GUI capture
            print(f"[METRICS] {json.dumps(entry)}", flush=True)
            
            with open(METRICS_FILE, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"[Metrics] Error saving entry: {e}")

    @staticmethod
    def get_recent_metrics(limit: int = 100):
        try:
            if not os.path.exists(METRICS_FILE):
                return []
            with open(METRICS_FILE, "r") as f:
                lines = f.readlines()
                return [json.loads(line) for line in lines[-limit:]]
        except Exception:
            return []

# Global instance
tracker = Metrics()
