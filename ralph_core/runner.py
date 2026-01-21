"""
The Iterative Cycle Handler.
Connects the shell loop (ralph_loop.sh) to the AI Swarm (swarm.py).

Architecture Modes:
- LEGACY (default): Sequential pipeline (think → code → review → verify)
- V2: Message-driven collaborative pipeline with Translator & ASICs

Set RALPH_V2=1 environment variable to enable V2 mode.
"""

import sys
import os
import re

# Ensure we can import swarm from the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Core swarm functions (legacy)
from swarm import think, code, review, verify, reflect, decompose, diagnose, estimate_task
# Note: generate_dream is used by ralph_daemon.py, not here

# New V2 components
try:
    from swarm import (
        translate, TaskSpec,
        code_with_revision, review_with_feedback,
        spawn_asic, get_bus, reset_bus,
        Message, MessageType,
        # Message handlers
        orchestrator_handle_message,
        engineer_handle_message,
        designer_handle_message,
        create_work_request,
        # ASIC handlers
        asic_handle_message,
        ASIC_HANDLERS,
        list_available_asics,
    )
    V2_AVAILABLE = True
except ImportError:
    V2_AVAILABLE = False
    translate = None
    orchestrator_handle_message = None
    engineer_handle_message = None
    designer_handle_message = None
    create_work_request = None
    asic_handle_message = None
    ASIC_HANDLERS = {}
    list_available_asics = lambda: []

from memory import Memory
from forklift import Forklift, forklift_lift_sync, format_forklift_context

from executor import Executor

from metrics import tracker

from planning import PlanManager

from compressor import compress_history

from git_manager import git

# Security Checkpoint - all outputs must pass through here
try:
    from security import checkpoint, is_safe_to_output, Decision
    SECURITY_ENABLED = True
except ImportError:
    SECURITY_ENABLED = False
    checkpoint = None
    is_safe_to_output = lambda c, t: (True, "Security not available")

# V2 Mode flag
USE_V2 = os.environ.get("RALPH_V2", "0") == "1" and V2_AVAILABLE



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

        # Security Checkpoint: Validate file write before proceeding
        if SECURITY_ENABLED:
            safe, reason = is_safe_to_output(content, "file")
            if not safe:
                print(f"[Security] BLOCKED file write to {filename}: {reason}")
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
    All commands pass through Security Checkpoint before execution.

    """

    executor = Executor()

    pattern = r"<execute>(.*?)</execute>"

    commands = re.findall(pattern, text, re.DOTALL)



    results = []

    for cmd in commands:
        cmd = cmd.strip()

        # Security Checkpoint: Validate shell command before execution
        if SECURITY_ENABLED:
            safe, reason = is_safe_to_output(cmd, "shell")
            if not safe:
                print(f"[Security] BLOCKED command: {reason}")
                results.append({"blocked": True, "reason": reason, "command": cmd})
                continue

        res = executor.run(cmd)

        results.append(res)

    return results



def extract_last_failure(progress_file: str):

    """Helper to extract the last error and code from history."""

    if not os.path.exists(progress_file):

        return None, None

        

    with open(progress_file, 'r') as f:

        content = f.read()

        

    # Check for recent FAIL

    if "Command: FAIL" not in content[-2000:]:

        return None, None

        

    # Naive extraction of the last execution block and result

    # We grab the last 3000 chars as context for the debugger

    return content[-3000:], content[-3000:]



def run_v2_pipeline(objective: str, context: str, plan_summary: str, brain: "Memory") -> dict:
    """
    V2 Message-Driven Pipeline with Translator and bidirectional communication.

    Flow:
        1. Translator converts human input → TaskSpec
        2. Orchestrator creates strategic plan
        3. Message Bus routes: Orchestrator → Engineer → Designer (with revision cycles)
        4. Return final output

    Args:
        objective: The current objective
        context: Project context string
        plan_summary: Status of the plan
        brain: Memory instance

    Returns:
        dict with 'output', 'execution_results', 'messages_exchanged'
    """
    print("[V2] Starting message-driven pipeline...")

    # Reset message bus for this iteration
    if reset_bus:
        reset_bus()
    bus = get_bus() if get_bus else None

    # --- STEP 1: TRANSLATOR ---
    print("[V2] Translator processing human input...")
    task_spec = None
    task_spec_dict = None
    if translate:
        task_spec = translate(objective, context)
        translated_context = task_spec.to_prompt_context()
        task_type = task_spec.task_type
        task_spec_dict = task_spec.to_dict()
        print(f"[V2] Translated: {task_spec.objective[:80]}... (type: {task_type})")
    else:
        # Fallback if translator not available
        translated_context = f"OBJECTIVE: {objective}"
        task_type = "code"
        task_spec_dict = {"objective": objective, "task_type": task_type}

    # --- STEP 2: ORCHESTRATOR ---
    print("[V2] Orchestrator planning...")
    full_context = f"{translated_context}\n\n{context}\n\n{plan_summary}"
    plan = think(full_context, objective)

    # --- STEP 3: ASIC CHECK ---
    # For certain task types, try spawning an ASIC first
    asic_result = None
    if spawn_asic and task_type in ["regex", "json", "sql", "test"]:
        print(f"[V2] Spawning ASIC for task type: {task_type}")
        try:
            options = spawn_asic(task_type, objective)
            if options and not all("Error" in o for o in options):
                asic_result = {
                    "options": options,
                    "task_type": task_type,
                }
                print(f"[V2] ASIC generated {len(options)} options")
        except Exception as e:
            print(f"[V2] ASIC error (falling back): {e}")

    # --- STEP 4: MESSAGE-DRIVEN COLLABORATION LOOP ---
    print("[V2] Starting Message Bus collaboration...")

    final_output = None

    # Agent message handlers dispatch table
    AGENT_HANDLERS = {
        "orchestrator": orchestrator_handle_message,
        "engineer": engineer_handle_message,
        "designer": designer_handle_message,
    }

    # Add ASIC handlers dynamically (asic:regex, asic:json, etc.)
    if ASIC_HANDLERS:
        AGENT_HANDLERS.update(ASIC_HANDLERS)
        print(f"[V2] Registered {len(ASIC_HANDLERS)} ASIC handlers: {list(ASIC_HANDLERS.keys())}")

    # Create initial WORK_REQUEST from Orchestrator → Engineer
    if create_work_request and bus:
        initial_msg = create_work_request(
            task_spec=task_spec_dict,
            plan=plan,
            context=full_context,
        )

        # If ASIC produced options, include them in metadata
        if asic_result:
            initial_msg.metadata["asic_options"] = asic_result["options"]
            initial_msg.metadata["asic_task_type"] = asic_result["task_type"]

        # Store plan in metadata for revision rounds
        initial_msg.metadata["plan"] = plan
        initial_msg.metadata["full_context"] = full_context

        # Send to bus
        if not bus.send(initial_msg):
            print("[V2] Circuit breaker triggered on initial message!")
            return {
                "output": "Error: Message bus circuit breaker triggered",
                "task_spec": task_spec_dict,
                "asic_used": asic_result is not None,
                "messages_exchanged": 0,
                "revision_rounds": 0,
            }

        # === MESSAGE LOOP ===
        # Process messages until COMPLETE or circuit breaker
        MAX_ITERATIONS = 20  # Safety limit
        iteration = 0

        while iteration < MAX_ITERATIONS:
            iteration += 1

            # Check each agent for pending messages
            message_processed = False

            for agent_name, handler in AGENT_HANDLERS.items():
                if handler is None:
                    continue

                while bus.has_messages(agent_name):
                    msg = bus.receive(agent_name)
                    if msg is None:
                        break

                    print(f"[V2] {agent_name} processing {msg.type.value}...")

                    # Enrich message with context for revisions
                    if msg.type == MessageType.REVISION_REQUEST:
                        msg.payload["original_plan"] = plan
                        msg.payload["original_context"] = full_context

                    # Handle message
                    response = handler(msg)
                    message_processed = True

                    if response is None:
                        # Handler returned None - check if COMPLETE
                        if msg.type == MessageType.COMPLETE:
                            final_output = msg.payload.get("final_output", "")
                            print(f"[V2] Task COMPLETE via message bus")
                            break
                        continue

                    # Store plan in response metadata for Designer
                    response.metadata["plan"] = plan
                    response.metadata["revision_round"] = msg.metadata.get("revision_round", 0)

                    # Send response to bus
                    if not bus.send(response):
                        print(f"[V2] Circuit breaker triggered!")
                        break

                if final_output is not None:
                    break

            if final_output is not None:
                break

            # If no messages processed this iteration, we're done or stuck
            if not message_processed:
                print("[V2] No messages to process, checking for completion...")
                if bus.is_complete():
                    final_output = bus.get_final_output()
                break

        # Get stats
        stats = bus.get_stats()
        print(f"[V2] Message loop complete. Stats: {stats['total_sent']} messages, {stats['revision_count']} revisions")
        print(bus.format_history())

        return {
            "output": final_output or "Error: Pipeline did not produce output",
            "task_spec": task_spec_dict,
            "asic_used": asic_result is not None,
            "messages_exchanged": stats["total_sent"],
            "revision_rounds": stats["revision_count"],
        }

    else:
        # Fallback: No bus available, use direct function calls (legacy V2)
        print("[V2] Message bus not available, falling back to direct calls...")

        MAX_REVISIONS = 3
        revision_round = 0
        messages_exchanged = 0

        # Initial engineering
        if asic_result:
            implementation = asic_result["options"][0]
            print(f"[V2] Using ASIC option as starting point")
        else:
            code_prompt = (
                f"{plan}\n\n"
                f"INSTRUCTIONS:\n"
                f"1. If the plan requires code, format it with the filename: ```python:filename.py\n...```\n"
                f"2. If the plan requires running commands, use <execute>command</execute>.\n"
                f"3. If the plan is a direct answer, just write the text directly.\n"
            )
            implementation = code(code_prompt, full_context)
            messages_exchanged += 1

        # Revision loop (legacy fallback)
        while revision_round < MAX_REVISIONS:
            print(f"[V2] Review round {revision_round + 1}...")

            if review_with_feedback:
                review_result = review_with_feedback(implementation, plan, revision_round)
                messages_exchanged += 1

                if review_result["approved"]:
                    final_output = review_result["final_output"]
                    print(f"[V2] Designer approved (round {revision_round + 1})")
                    break
                else:
                    issues = review_result["issues"]
                    severity = review_result["severity"]
                    print(f"[V2] Designer requested revision ({severity}): {issues[:2]}...")

                    if severity == "low" and revision_round > 0:
                        final_output = implementation
                        print(f"[V2] Accepting with minor issues")
                        break

                    revision_feedback = "\n".join(f"- {issue}" for issue in issues)
                    if code_with_revision:
                        implementation = code_with_revision(
                            plan=plan,
                            context=full_context,
                            revision_feedback=revision_feedback,
                            revision_round=revision_round + 1,
                        )
                    else:
                        implementation = code(
                            f"{plan}\n\nPREVIOUS ISSUES TO FIX:\n{revision_feedback}",
                            full_context
                        )
                    messages_exchanged += 1
                    revision_round += 1
            else:
                final_output = review(implementation, plan)
                messages_exchanged += 1
                break

        if final_output is None:
            final_output = review(implementation, plan) if review else implementation
            print(f"[V2] Max revisions reached, using final implementation")

        return {
            "output": final_output,
            "task_spec": task_spec_dict,
            "asic_used": asic_result is not None,
            "messages_exchanged": messages_exchanged,
            "revision_rounds": revision_round,
        }


def main():

    if len(sys.argv) < 3:

        print("Usage: runner.py <OBJECTIVE> <ITERATION>")

        sys.exit(1)



    master_objective = sys.argv[1]

    iteration = sys.argv[2]

    

    tracker.log_iteration(int(iteration), "START")

    

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

    

    print(f"[Runner] Starting Iteration {iteration} for: {master_objective}")



    # --- COMPRESSION CHECK ---

    if int(iteration) > 1 and int(iteration) % 5 == 0:

        compress_history(int(iteration))



    # 0. Initialize Plan

    plan_mgr = PlanManager(master_objective)

    

    # Check if we need to decompose (only on first run or if plan is empty)

    if not plan_mgr.plan["tasks"] and int(iteration) < 3: # Allow direct execution if user forces it

        print("[Swarm] Decomposing objective into subtasks...")

        subtasks_text = decompose(master_objective)

        

        # Parse bullets

        tasks = []

        for line in subtasks_text.split('\n'):

            line = line.strip()

            if line.startswith('-') or line.startswith('*') or line[0].isdigit():

                clean_task = line.lstrip("-*0123456789. ").strip()

                if clean_task:

                    tasks.append(clean_task)

        

        if tasks:
            # --- ESTIMATOR: Prioritize tasks by expected value ---
            if len(tasks) > 1:
                print(f"[Estimator] Analyzing {len(tasks)} tasks for prioritization...")
                prioritized = []
                for task in tasks:
                    try:
                        est = estimate_task(task)
                        expected_value = est["value"] * est["probability"]
                        prioritized.append((task, expected_value, est))
                        print(f"  - [{expected_value:.1f}] {task[:50]}...")
                    except Exception as e:
                        print(f"  - [ERR] {task[:50]}... ({e})")
                        prioritized.append((task, 5.0, {"value": 5, "probability": 1.0}))

                # Sort by expected value (highest first)
                prioritized.sort(key=lambda x: x[1], reverse=True)
                tasks = [t[0] for t in prioritized]
                print(f"[Estimator] Tasks reordered by priority")

            plan_mgr.set_tasks(tasks)
            print(f"[Planner] Created Plan with {len(tasks)} steps.")

        else:

            print("[Planner] Failed to decompose. Proceeding with monolithic objective.")



    # Get Current Subtask

    current_task_node = plan_mgr.get_current_task()

    

    if current_task_node:

        current_objective = current_task_node["description"]

        plan_summary = plan_mgr.get_status_summary()

        print(f"[Planner] Focused on Subtask {current_task_node['id']}: {current_objective}")

    else:

        current_objective = master_objective

        plan_summary = "No subtasks. Executing master objective directly."



    # 1. Initialize Memory

    brain = Memory() # Instantiation fixed

    

    # --- LEARNING PHASE ---

    if int(iteration) > 1 and os.path.exists("RALPH_PROGRESS.md"):

        print("[Swarm] Reflecting on past performance...")

        try:

            with open("RALPH_PROGRESS.md", "r") as pf:

                content = pf.read()

                # Analyze last 3000 chars to catch recent context

                recent_history = content[-3000:]

            

            lessons_learned = reflect(recent_history)

            if lessons_learned and "None" not in lessons_learned:

                # Parse bullet points

                for line in lessons_learned.split('\n'):

                    line = line.strip()

                    if line.startswith('-') or line.startswith('*'):

                        # Remove bullet and quotes

                        clean_lesson = line[1:].strip().strip('"').strip("'")

                        brain.add_lesson(clean_lesson)

                print(f"[Reflector] Analysis Complete.")

        except Exception as e:

            print(f"[Reflector] Error during reflection: {e}")



    try:
        files = os.listdir(".")

        # --- FORKLIFT PROTOCOL: Selective Memory Loading ---
        # Replaces the old retrieve_full_state() dump with intelligent selection
        forklift = Forklift(brain)
        memories = forklift.lift(
            objective=current_objective,
            task_type="code",  # Default; could be inferred from task_spec
            scope="standard",
        )
        print(f"[Forklift] Loaded {len(memories['lessons'])} lessons, {len(memories['facts'])} facts")

        # --- SELF-HEALING / DEBUGGER ---
        diagnosis_context = ""
        if int(iteration) > 1:
            last_err, last_code = extract_last_failure("RALPH_PROGRESS.md")
            if last_err:
                print("[Swarm] Detecting failure. Invoking Debugger...")
                diagnosis = diagnose(last_err, last_code)
                diagnosis_context = f"=== DEBUGGER DIAGNOSIS ===\n{diagnosis}\n"
                print(f"[Debugger] Diagnosis: {diagnosis[:100]}...")

        # Read Project Docs for Context
        project_context = ""
        for doc in ["STATUS.md", "MISSION.md"]:
            if os.path.exists(doc):
                with open(doc, 'r') as f:
                    project_context += f"\n--- {doc} ---\n{f.read()}\n"

        # --- FORMAT CONTEXT using Forklift output ---
        context = format_forklift_context(
            memories=memories,
            files=files,
            iteration=iteration,
            plan_summary=plan_summary,
            project_context=project_context,
            diagnosis_context=diagnosis_context,
        )

    except Exception as e:
        context = f"Error reading state: {e}"



    # --- FAST PATH FOR STATUS ---

    if master_objective.lower() in ["status report", "system status", "status", "system check"]:



        print("[Runner] Fast Path Activated: Reading System Status...")

        status_out = ""

        for doc in ["STATUS.md", "MISSION.md"]:

            if os.path.exists(doc):

                with open(doc, 'r') as f:

                    status_out += f"\n{f.read()}\n"

        

        print(f"\nSTART_RESPONSE_DATA\n{status_out}\nEND_RESPONSE_DATA")

        print("<promise>COMPLETE</promise>")

        with open("RALPH_PROGRESS.md", "a") as f:

            f.write(f"\n## Iteration {iteration} (Fast Path)\n{status_out}\n")

        sys.exit(0)



    # === PIPELINE EXECUTION (V2 or Legacy) ===
    if USE_V2:
        print("[Runner] Using V2 message-driven pipeline...")
        v2_result = run_v2_pipeline(current_objective, context, plan_summary, brain)
        final_output = v2_result["output"]
        plan = f"[V2 Pipeline - TaskSpec: {v2_result.get('task_spec', {}).get('task_type', 'unknown')}]"

        # SHOW OUTPUT TO USER
        print(f"\nSTART_RESPONSE_DATA\n{final_output}\nEND_RESPONSE_DATA")

        # SAVE TO DISK
        saved = save_code_to_disk(final_output)

        # EXECUTE COMMANDS
        execution_results = run_commands(final_output)

        # Build exec_summary for verification
        exec_summary = ""
        for r in execution_results:
            status = "PASS" if r['success'] else "FAIL"
            exec_summary += f"\nCommand: {status}\nSTDOUT: {r['stdout']}\nSTDERR: {r['stderr']}\n"

        if exec_summary:
            print(f"\n=== EXECUTION SUMMARY ===\n{exec_summary}\n=========================")

        # Skip to verification (already reviewed in V2 pipeline)
        print("[Swarm] Verifying completion...")
        verdict = verify(current_objective, final_output, exec_summary)
        print(f"[Verifier] Decision:\n{verdict}")

    else:
        # === LEGACY PIPELINE ===
        # 3. Orchestrator Plan
        print("[Swarm] Orchestrator is thinking...")
        plan = think(context, current_objective)

        # 4. Engineer Execution
        print("[Swarm] Engineer is coding...")
        code_prompt = (
            f"{plan}\n\n"
            f"INSTRUCTIONS:\n"
            f"1. If the plan requires code, format it with the filename: ```python:filename.py\n...```\n"
            f"2. If the plan requires running commands, use <execute>command</execute>.\n"
            f"3. If the plan is a direct answer (like a poem, joke, or explanation), just write the text directly without code blocks.\n"
        )
        implementation = code(code_prompt, context)

        # 5. Designer Review & Save
        print("[Swarm] Designer is reviewing...")
        final_output = review(implementation, plan)

        # SHOW OUTPUT TO USER
        print(f"\nSTART_RESPONSE_DATA\n{final_output}\nEND_RESPONSE_DATA")

        # SAVE TO DISK
        saved = save_code_to_disk(final_output)

        # EXECUTE COMMANDS
        execution_results = run_commands(final_output)

        # 6. Verification
        exec_summary = ""
        for r in execution_results:
            status = "PASS" if r['success'] else "FAIL"
            exec_summary += f"\nCommand: {status}\nSTDOUT: {r['stdout']}\nSTDERR: {r['stderr']}\n"

        if exec_summary:
            print(f"\n=== EXECUTION SUMMARY ===\n{exec_summary}\n=========================")

        print("[Swarm] Verifying completion...")
        verdict = verify(current_objective, final_output, exec_summary)

        # Print the verifier's decision for debugging/transparency
        print(f"[Verifier] Decision:\n{verdict}")
    # END LEGACY PIPELINE



    # 7. Persistence

    with open("RALPH_PROGRESS.md", "a") as f:

        f.write(f"\n## Iteration {iteration} - Task: {current_objective}\n")

        f.write(f"### Result\n{final_output}\n")

        f.write(f"### Execution\n{exec_summary}\n")

        f.write(f"### Verification\n{verdict}\n")

        

    brain.context["last_iteration"] = iteration

    brain.context["last_verdict"] = verdict[:100]

    brain.save_context()



    if "<promise>COMPLETE</promise>" in verdict or "MISSION ACCOMPLISHED" in verdict:

        

        # --- GIT AUTO-COMMIT ---

        print("[Git] Verifying Auto-Commit...")

        commit_res = git.commit_all()

        print(commit_res)

        

        # Task Complete logic

        if current_task_node:

            print(f"[Runner] Subtask {current_task_node['id']} verified complete.")

            plan_mgr.mark_task_complete(current_task_node['id'])

            

            # Check if WHOLE plan is done

            if plan_mgr.is_fully_complete():

                with open("RALPH_PROGRESS.md", "a") as f:

                    f.write("\n<promise>COMPLETE</promise>\n")

                print("[Runner] All subtasks complete. Mission Accomplished.")

                sys.exit(0)

            else:

                print("[Runner] Subtask complete. Continuing to next task...")

                sys.exit(0)

        else:

            # Monolithic mode

            with open("RALPH_PROGRESS.md", "a") as f:

                f.write("\n<promise>COMPLETE</promise>\n")

            print("[Runner] Task Verified as COMPLETE.")

            sys.exit(0)

    else:

        print("[Runner] Task incomplete. Continuing...")

        sys.exit(1)

if __name__ == "__main__":
    main()
