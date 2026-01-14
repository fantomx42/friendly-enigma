from ..common.llm import call_model

def diagnose(error_log: str, recent_code: str) -> str:
    """
    The Debugger's step. Analyzes a specific runtime error and code to propose a fix.
    """
    prompt = (
        f"CODE:\n{recent_code}\n\n"
        f"ERROR LOG:\n{error_log}\n\n"
        f"You are the Debugger. The previous execution failed.\n"
        f"Task: Analyze the stack trace/error and the code.\n"
        f"Output: A concise technical explanation of the bug and the specific fix required.\n"
        f"Format: Plain text. Be direct.\n"
    )
    # DeepSeek is excellent at reasoning through bugs
    return call_model("orchestrator", prompt)

