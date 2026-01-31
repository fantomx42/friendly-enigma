import time
import json
import os
import sys
from enum import Enum
from typing import Dict, List, Optional, Any
from .protocols.bus import get_bus, MessageBus
from .protocols.messages import Message, MessageType, work_request, complete_message, error_message, diagnostic_message

class LoopState(Enum):
    IDLE = "IDLE"
    TRANSLATING = "TRANSLATING"
    PLANNING = "PLANNING"
    CODING = "CODING"
    VERIFYING = "VERIFYING"
    REFLECTING = "REFLECTING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"

class RalphLoop:
    """
    The main autonomous controller for Ralph AI.
    Implements the 'Ralph Wiggum' iterative failure/refinement loop.
    """
    def __init__(self, max_iterations: int = 10, bus: Optional[MessageBus] = None, state_file: str = ".ralph_loop_state.json"):
        self.state = LoopState.IDLE
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self.task_history: List[Dict] = []
        self.strike_count = 0  # Failures for the CURRENT atomic task
        self.max_strikes = 3   # From Product Guidelines: "The 3-Strike Reflector Rule"
        self.bus = bus or get_bus()
        self.state_file = state_file
        self.load_state()
        
        # Initialize Agents (Lazy load to avoid circular imports during startup)
        self._agents_initialized = False

    def _init_agents(self):
        if self._agents_initialized: return
        from .agents.orchestrator.agent import handle_message as orch_handler
        from .agents.engineer.agent import handle_message as eng_handler
        from .agents.designer.agent import handle_message as des_handler
        
        self.agent_handlers = {
            "orchestrator": orch_handler,
            "engineer": eng_handler,
            "designer": des_handler
        }
        self._agents_initialized = True

    def save_state(self):
        """Persists the loop state to disk."""
        data = {
            "state": self.state.value,
            "current_iteration": self.current_iteration,
            "strike_count": self.strike_count,
            "task_history": self.task_history
        }
        with open(self.state_file, 'w') as f:
            json.dump(data, f, indent=2)

    def load_state(self):
        """Loads the loop state from disk if available."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self.state = LoopState(data.get("state", "IDLE"))
                    self.current_iteration = data.get("current_iteration", 0)
                    self.strike_count = data.get("strike_count", 0)
                    self.task_history = data.get("task_history", [])
            except Exception as e:
                print(f"[RalphLoop] Error loading state: {e}")

    def transition(self, next_state: LoopState):
        print(f"[RalphLoop] Transition: {self.state.value} -> {next_state.value}")
        self.state = next_state
        self.save_state()
        # Broadcast status to bus
        self.bus.send(Message(
            type=MessageType.STATUS,
            sender="controller",
            receiver="system",
            payload={"state": self.state.value, "iteration": self.current_iteration}
        ))

    def run_step(self, objective: str):
        """
        Executes a single step of the loop based on current state.
        """
        self._init_agents()

        if self.state == LoopState.IDLE:
            self.current_iteration = 1
            self.transition(LoopState.TRANSLATING)
            return self._step_translate(objective)
        
        elif self.state == LoopState.TRANSLATING:
            # For now, just skip to planning
            self.transition(LoopState.PLANNING)
            return self._step_plan(objective)

        elif self.state == LoopState.PLANNING:
            self._process_messages("orchestrator")
            # Check if planning phase produced a work request
            if any(m.type == MessageType.WORK_REQUEST for m in self.bus.get_history()):
                self.transition(LoopState.CODING)
                return "Planning complete. Moving to Coding."
            return "Orchestrator thinking..."

        elif self.state == LoopState.CODING:
            self._process_messages("engineer")
            if any(m.type == MessageType.CODE_OUTPUT for m in self.bus.get_history()):
                self.transition(LoopState.VERIFYING)
                return "Coding complete. Moving to Verifying."
            return "Engineer coding..."

        elif self.state == LoopState.VERIFYING:
            self._process_messages("designer")
            status = self._check_bus_status()
            
            if status == "COMPLETE":
                self.transition(LoopState.COMPLETE)
                return "Task Successful."
            elif status == "ERROR" or status == "REVISION":
                self.strike_count += 1
                if self.strike_count >= self.max_strikes:
                    self.transition(LoopState.REFLECTING)
                    return "3 Strikes reached. Reflecting..."
                else:
                    self.current_iteration += 1
                    if self.current_iteration > self.max_iterations:
                        self.transition(LoopState.FAILED)
                        return "Max iterations reached."
                    self.transition(LoopState.CODING) # Cycle back to coding for fix
                    return f"Iteration {self.current_iteration} (Strike {self.strike_count})..."
            else:
                return "Waiting for Designer/Verifier..."

        elif self.state == LoopState.REFLECTING:
            self._reflect()
            self.strike_count = 0
            self.transition(LoopState.PLANNING)
            return "Reflection complete. Restarting loop."

    def _process_messages(self, agent_name: str):
        """Dispatches messages to the specific agent."""
        while self.bus.has_messages(agent_name):
            msg = self.bus.receive(agent_name)
            if not msg: break
            
            handler = self.agent_handlers.get(agent_name)
            if handler:
                response = handler(msg)
                if response:
                    self.bus.send(response)

    def _step_translate(self, objective: str):
        print(f"[Ralph] Translating objective: {objective}")
        return {"status": "ok"}

    def _step_plan(self, objective: str):
        print(f"[RalphLoop] Initiating plan for: {objective}")
        from .agents.orchestrator.agent import decompose
        plan_text = decompose(objective)
        # Manually create first work request
        self.bus.send(work_request(plan=plan_text, task_spec={"objective": objective}, context=""))
        return {"status": "ok"}

    def _check_bus_status(self) -> str:
        """Checks the message bus for the outcome of the last action."""
        if self.bus.is_complete():
            return "COMPLETE"
        
        history = self.bus.get_history(limit=5)
        for msg in history:
            if msg.type == MessageType.ERROR:
                return "ERROR"
            if msg.type == MessageType.REVISION_REQUEST:
                return "REVISION"
        
        return "PENDING"

    def _reflect(self):
        print("[Ralph] Triggering REFLECTOR agent...")
        self.bus.send(diagnostic_message(
            error="Repeated failure",
            traceback="3 strikes reached in VERIFYING state",
            agent_state={"iteration": self.current_iteration},
            attempt_count=self.strike_count,
            sender="controller",
            receiver="reflector"
        ))