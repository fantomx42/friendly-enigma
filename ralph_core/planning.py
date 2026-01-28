import json
import os
from typing import List, Dict, Optional

PLAN_FILE = "RALPH_PLAN.json"

class PlanManager:
    def __init__(self, master_objective: str):
        self.master_objective = master_objective
        self.plan_file = PLAN_FILE
        self.plan = self._load_plan()

    def _load_plan(self) -> Dict:
        if os.path.exists(self.plan_file):
            try:
                with open(self.plan_file, 'r') as f:
                    data = json.load(f)
                    # Only reload if it matches the current objective
                    if data.get("master_objective") == self.master_objective:
                        return data
            except json.JSONDecodeError:
                pass
        return {"master_objective": self.master_objective, "tasks": []}

    def save_plan(self):
        # Print to stdout for GUI capture
        print(f"[PLAN] {json.dumps(self.plan)}", flush=True)

        with open(self.plan_file, 'w') as f:
            json.dump(self.plan, f, indent=2)

    def set_tasks(self, tasks: List[str]):
        """Initializes the plan with a list of subtasks."""
        self.plan["tasks"] = [{"id": i+1, "description": t, "status": "pending"} for i, t in enumerate(tasks)]
        self.save_plan()

    def get_current_task(self) -> Optional[Dict]:
        """Returns the first pending task."""
        for task in self.plan["tasks"]:
            if task["status"] == "pending":
                return task
        return None

    def mark_task_complete(self, task_id: int):
        for task in self.plan["tasks"]:
            if task["id"] == task_id:
                task["status"] = "complete"
                self.save_plan()
                return

    def is_fully_complete(self) -> bool:
        return all(t["status"] == "complete" for t in self.plan["tasks"]) and len(self.plan["tasks"]) > 0

    def get_status_summary(self) -> str:
        if not self.plan["tasks"]:
            return "No plan established."
        
        summary = f"Plan for: {self.master_objective}\n"
        for t in self.plan["tasks"]:
            icon = "✅" if t["status"] == "complete" else "⏳"
            summary += f"{icon} {t['id']}. {t['description']}\n"
        return summary
