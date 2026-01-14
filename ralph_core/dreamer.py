from .agents.common.llm import call_model
import json
import random

TOPICS = [
    "Data Structures", "Algorithms", "File I/O", "Regex", "API Consumption", 
    "Data Visualization", "System Administration", "Cryptography", "Image Processing"
]

def generate_dream() -> str:
    """
    Generates a synthetic programming task to train the system.
    """
    topic = random.choice(TOPICS)
    
    prompt = (
        f"Topic: {topic}\n"
        f"You are the Dreamer. Generate a specific, self-contained coding challenge to train an AI agent.\n"
        f"Constraint: The task must be solvable with a single Python script.\n"
        f"Format: Return ONLY the task description string.\n"
        f"Example: 'Write a python script that recursively traverses a directory and finds the largest file.'\n"
    )
    
    task_description = call_model("orchestrator", prompt)
    
    # Prefix to indicate it's a dream (optional, but good for tracking)
    return f"[DREAM] {task_description}"

