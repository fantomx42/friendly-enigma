"""
The Iterative Cycle Handler.
Connects the shell loop (ralph_loop.sh) to the AI Swarm (swarm.py).
"""

import sys
import os
import re

# Ensure we can import swarm from the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from swarm import think, code, review
from memory import Memory

from swarm import think, code, review
from memory import Memory
from executor import Executor

def save_code_to_disk(text):
    # ... (rest of the function remains the same, I will include it for the replace tool)
    pattern = r"```([\w\./:-]+)?\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    saved_files = []
    for header, content in matches:
        filename = "generated_script.py"
        if header:
            header = header.strip()
            if ":" in header:
                _, possible_name = header.split(":", 1)
                filename = possible_name.strip()
            elif "." in header and not header.lower() in ["node.js", "vue.js"]:
                 filename = header.strip()
            elif header.lower() in ["python", "py"]:
                filename = "generated_script.py"
            elif header.lower() in ["bash", "sh"]:
                filename = "generated_script.sh"
            elif header.lower() in ["rust", "rs"]:
                filename = "generated_lib.rs"
        if ".." in filename or filename.startswith("/"):
            continue
        try:
            directory = os.path.dirname(filename)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(filename, "w") as f:
                f.write(content)
            saved_files.append(filename)
            print(f"[Runner] Saved file: {filename}")
        except Exception as e:
            print(f"[Runner] Error saving {filename}: {e}")
    return saved_files

def run_commands(text):
    """
    Extracts and runs commands from <execute>command</execute> tags.
    """
    executor = Executor()
    pattern = r"<execute>(.*?)</execute>"
    commands = re.findall(pattern, text, re.DOTALL)
    
    results = []
    for cmd in commands:
        res = executor.run(cmd.strip())
        results.append(res)
    return results

def main():
    if len(sys.argv) < 3:
        print("Usage: runner.py <OBJECTIVE> <ITERATION>")
        sys.exit(1)

    objective = sys.argv[1]
    iteration = sys.argv[2]
    
    # Configure logging to file (append mode)
    # This allows the UI to tail this file
    log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ralph.log")
    
    class Tee(object):
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    f = open(log_file, 'a')
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, f)
    
    print(f"[Runner] Starting Iteration {iteration} for: {objective}")

    # 1. Initialize Memory
    
    try:
        files = os.listdir(".")
        memory_state = brain.retrieve_full_state()
        context = (
            f"Current Directory Files: {files}\n"
            f"Iteration: {iteration}\n"
            f"{memory_state}"
        )
    except Exception as e:
        context = f"Error reading state: {e}"

    # 3. Orchestrator Plan
    print("[Swarm] Orchestrator is thinking...")
    plan = think(context, objective)

    # 4. Engineer Execution
    print("[Swarm] Engineer is coding...")
    code_prompt = (
        f"{plan}\n\n"
        f"IMPORTANT: Format your code with the filename: ```python:filename.py\n...```\n"
        f"If you need to run a command (e.g., test the script), use <execute>python3 script.py</execute>."
    )
    implementation = code(code_prompt, context)
    
    # 5. Designer Review & Save
    print("[Swarm] Designer is reviewing...")
    final_output = review(implementation, plan)
    
    # SAVE TO DISK
    saved = save_code_to_disk(final_output)
    
    # EXECUTE COMMANDS
    execution_results = run_commands(final_output)
    
    # 6. Verification
    exec_summary = ""
    for r in execution_results:
        status = "PASS" if r['success'] else "FAIL"
        exec_summary += f"\nCommand: {status}\nSTDOUT: {r['stdout']}\nSTDERR: {r['stderr']}\n"

    verification_prompt = (
        f"OBJECTIVE: {objective}\n"
        f"CREATED FILES: {saved}\n"
        f"EXECUTION RESULTS: {exec_summary}\n"
        f"Analyze if the objective is met.\n"
        f"If YES, output exactly: <promise>COMPLETE</promise>\n"
        f"If NO, explain what is missing or failed."
    )
    print("[Swarm] Verifying completion...")
    verdict = review(verification_prompt, "Verification Phase")

    # 7. Persistence
    with open("RALPH_PROGRESS.md", "a") as f:
        f.write(f"\n## Iteration {iteration}\n")
        f.write(f"### Result\n{final_output}\n")
        f.write(f"### Execution\n{exec_summary}\n")
        f.write(f"### Verification\n{verdict}\n")
        
    brain.context["last_iteration"] = iteration
    brain.context["last_verdict"] = verdict[:100]
    brain.save_context()

    if "<promise>COMPLETE</promise>" in verdict or "MISSION ACCOMPLISHED" in verdict:
        with open("RALPH_PROGRESS.md", "a") as f:
            f.write("\n<promise>COMPLETE</promise>\n")
        print("[Runner] Task Verified as COMPLETE.")
        sys.exit(0)
    else:
        print("[Runner] Task incomplete. Continuing...")
        sys.exit(1)

if __name__ == "__main__":
    main()
