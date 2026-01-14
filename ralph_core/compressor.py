import os
import glob
from .agents.common.llm import call_model

CHECKPOINT_DIR = "memory/checkpoints"
PROGRESS_FILE = "RALPH_PROGRESS.md"

def compress_history(iteration: int):
    """
    Compresses the RALPH_PROGRESS.md into a concise checkpoint.
    """
    if not os.path.exists(PROGRESS_FILE):
        return

    print(f"[Compressor] Compressing history at iteration {iteration}...")

    with open(PROGRESS_FILE, 'r') as f:
        history = f.read()

    prompt = (
        f"HISTORY:\n{history}\n\n"
        f"You are the Compressor. The history is getting too long.\n"
        f"Summarize the key actions, decisions, and current state into a concise 'Checkpoint'.\n"
        f"Focus on what has been DONE and what is currently BROKEN or PENDING.\n"
        f"Format: Markdown."
    )
    
    # Use Orchestrator (DeepSeek) for high-quality summary
    summary = call_model("orchestrator", prompt)
    
    checkpoint_file = os.path.join(CHECKPOINT_DIR, f"checkpoint_iter_{iteration}.md")
    with open(checkpoint_file, 'w') as f:
        f.write(f"# Checkpoint (Iteration {iteration})\n\n{summary}")
        
    print(f"[Compressor] Saved checkpoint: {checkpoint_file}")
    
    # Truncate Progress File, keeping the Checkpoint as the new base
    with open(PROGRESS_FILE, 'w') as f:
        f.write(f"# History Compressed at Iteration {iteration}\n\n")
        f.write(f"See {checkpoint_file} for details.\n\n")
        f.write(f"## Current Status Summary\n{summary}\n")

def get_latest_checkpoint() -> str:
    """Retrieves the content of the latest checkpoint."""
    files = glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_iter_*.md"))
    if not files:
        return ""
    
    # Sort by iteration number in filename
    latest = max(files, key=lambda f: int(f.split("_iter_")[1].split(".md")[0]))
    
    with open(latest, 'r') as f:
        return f.read()
