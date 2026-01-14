import json
import os
from typing import List, Dict, Optional
# Import local swarm for estimation (assuming ralph_core is in path)
# We use a try-except import in case this is run from a different context where ralph_core isn't resolved yet
try:
    from ralph_core.swarm import estimate_task
except ImportError:
    # Fallback if running directly or path issues
    def estimate_task(t): return {"value": 5, "probability": 0.5}

QUEUE_FILE = "RALPH_QUEUE.json"

class QueueManager:
    def __init__(self):
        self.queue_file = QUEUE_FILE
        self._ensure_file()

    def _ensure_file(self):
        if not os.path.exists(self.queue_file):
            with open(self.queue_file, 'w') as f:
                json.dump([], f)

    def add_task(self, source: str, task: str):
        # 1. Estimate
        print(f"[Queue] Estimating priority for: {task}")
        metrics = estimate_task(task)
        value = metrics.get("value", 5)
        prob = metrics.get("probability", 0.5)
        
        # Bayesian Priority = Expected Value = Value * Probability
        priority = value * prob
        
        with open(self.queue_file, 'r') as f:
            queue = json.load(f)
        
        queue.append({
            "source": source,
            "task": task,
            "status": "pending",
            "metrics": {
                "value": value,
                "probability": prob,
                "score": priority
            },
            "timestamp": os.path.getmtime(self.queue_file)
        })
        
        with open(self.queue_file, 'w') as f:
            json.dump(queue, f, indent=2)
        print(f"[Queue] Added task. Score: {priority:.2f} (V:{value} * P:{prob})")

    def get_next_task(self) -> Optional[Dict]:
        with open(self.queue_file, 'r') as f:
            queue = json.load(f)
        
        # Filter pending
        pending = [t for t in queue if t["status"] == "pending"]
        if not pending:
            return None
            
        # Sort by Score (Desc)
        # Handle legacy tasks without metrics
        def get_score(t):
            return t.get("metrics", {}).get("score", 0)
            
        pending.sort(key=get_score, reverse=True)
        
        return pending[0]

    def mark_complete(self, task_desc: str):
        with open(self.queue_file, 'r') as f:
            queue = json.load(f)
            
        for task in queue:
            if task["task"] == task_desc:
                task["status"] = "complete"
        
        with open(self.queue_file, 'w') as f:
            json.dump(queue, f, indent=2)

# Global Instance
queue = QueueManager()
