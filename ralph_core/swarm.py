"""
swarm.py

The Core Swarm Interface for the Ralph Protocol.
This module abstracts the multi-model architecture into a single callable interface.
Now delegates to specific agent modules in ralph_core/agents/.

Architecture (3-Tier Hierarchical Swarm):
    1. Translator - Human input → Structured TaskSpec
    2. Middle Management - Engineer ↔ Designer (bidirectional)
    3. LLM ASICs - Ultra-small specialists for micro-tasks

Message Flow:
    Orchestrator --[WORK_REQUEST]--> Engineer
    Engineer --[CODE_OUTPUT]--> Designer
    Designer --[REVISION_REQUEST]--> Engineer (if quality insufficient)
    Designer --[COMPLETE]--> Orchestrator (when satisfied)
"""

# ============================================
# Core Agent Functions (Original)
# ============================================
from agents.orchestrator.agent import think, decompose, create_work_request
from agents.orchestrator.agent import handle_message as orchestrator_handle_message
from agents.engineer.agent import code
from agents.designer.agent import review, verify
from agents.reflector.agent import reflect
from agents.debugger.agent import diagnose
from agents.estimator.agent import estimate_task
from agents.common.llm import call_model
from agents.common.config import MODELS

# ============================================
# New: Translator Agent (Tier 1)
# ============================================
try:
    from agents.translator.agent import translate, TaskSpec
except ImportError:
    translate = None
    TaskSpec = None

# ============================================
# New: Bidirectional Agent Functions
# ============================================
from agents.engineer.agent import code_with_revision, code_with_asic
from agents.designer.agent import review_with_feedback, evaluate_options

# Message handlers (for message-driven pipeline)
from agents.engineer.agent import handle_message as engineer_handle_message
from agents.designer.agent import handle_message as designer_handle_message

# ============================================
# New: ASIC System (Tier 3)
# ============================================
try:
    from asic import (
        spawn_asic, spawn_asic_parallel,
        ASIC_REGISTRY, list_available_asics,
        asic_handle_message, ASIC_HANDLERS,
    )
except ImportError:
    spawn_asic = None
    spawn_asic_parallel = None
    ASIC_REGISTRY = {}
    list_available_asics = lambda: []
    asic_handle_message = None
    ASIC_HANDLERS = {}

# ============================================
# New: Message Bus Protocol
# ============================================
try:
    from protocols import Message, MessageType, MessageBus
    from protocols.bus import get_bus, reset_bus
except ImportError:
    Message = None
    MessageType = None
    MessageBus = None
    get_bus = None
    reset_bus = None

# ============================================
# Exports
# ============================================
__all__ = [
    # Original functions
    "think", "decompose", "code", "review", "verify",
    "reflect", "diagnose", "estimate_task",
    "call_model", "MODELS",

    # Translator (Tier 1)
    "translate", "TaskSpec",

    # Bidirectional functions
    "code_with_revision", "code_with_asic",
    "review_with_feedback", "evaluate_options",

    # Message handlers (all three agents)
    "orchestrator_handle_message",
    "engineer_handle_message",
    "designer_handle_message",
    "create_work_request",

    # ASIC System (Tier 3)
    "spawn_asic", "spawn_asic_parallel",
    "ASIC_REGISTRY", "list_available_asics",
    "asic_handle_message", "ASIC_HANDLERS",

    # Message Bus
    "Message", "MessageType", "MessageBus",
    "get_bus", "reset_bus",
]
