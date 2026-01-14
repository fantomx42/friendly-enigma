from ..common.llm import call_model

def reflect(progress_text: str) -> str:
    """
    The Reflector's step. Analyzes execution history for patterns of failure.
    """
    prompt = (
        f"HISTORY:\n{progress_text}\n\n"
        f"You are the Reflector. Analyze the recent history for repeated failures or errors.\n"
        f"Goal: Identify 1-3 specific, actionable technical lessons to prevent future errors.\n"
        f"Format: Provide ONLY a bulleted list of lessons. No intro/outro.\n"
        f"Example:\n"
        f"- Always use raw strings r'' for regex patterns.\n"
        f"- Check if a file exists before reading it.\n"
    )
    # Using 'orchestrator' (DeepSeek) for reasoning capability
    return call_model("orchestrator", prompt)
