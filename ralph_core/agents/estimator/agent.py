from ..common.llm import call_model
import json

def estimate_task(task_description: str) -> dict:
    """
    The Estimator's step. Predicts value and success probability.
    """
    prompt = (
        f"TASK: {task_description}\n\n"
        f"You are the Estimator. Analyze this task for prioritization.\n"
        f"1. Value (1-10): How critical/impactful is this? (10=Critical/High ROI, 1=Trivial)\n"
        f"2. Success Probability (0.0-1.0): How likely is an autonomous agent to succeed without human help?\n"
        f"   - Simple file edits -> High (0.9)\n"
        f"   - Complex architecture/creativity -> Medium (0.5)\n"
        f"   - Unknown external APIs/hardware -> Low (0.2)\n\n"
        f'Format: JSON ONLY. {{ "value": int, "probability": float }}\n'
    )
    
    response = call_model("orchestrator", prompt)
    
    # Clean up response to ensure valid JSON
    try:
        # Strip markdown code blocks if present
        clean_json = response.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
    except Exception:
        # Fallback defaults
        return {"value": 5, "probability": 0.5}

