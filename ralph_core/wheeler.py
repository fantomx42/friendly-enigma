import sys
import os
import hashlib
import threading
import numpy as np

# Add wheeler_ai_training to path so we can import the module
# File is at: ai_tech_stack/ralph_core/wheeler.py
# Root is at: ../../../
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
wheeler_path = os.path.join(project_root, "wheeler_ai_training")

if wheeler_path not in sys.path:
    sys.path.append(wheeler_path)

WHEELER_AVAILABLE = False
try:
    from wheeler_ai import WheelerAI
    WHEELER_AVAILABLE = True
except ImportError as e:
    print(f"[WheelerBridge] Could not import WheelerAI: {e}")

try:
    from .npu_engine import NPUWheelerEngine
    NPU_AVAILABLE = True
except ImportError:
    NPU_AVAILABLE = False

# Stability tracking (SCM integration)
try:
    from .wheeler_weights import stability_tracker
    STABILITY_TRACKING = True
except ImportError:
    try:
        from wheeler_weights import stability_tracker
        STABILITY_TRACKING = True
    except ImportError:
        STABILITY_TRACKING = False

class WheelerMemoryBridge:
    """
    Bridge to integrate the experimental Wheeler AI memory system
    into Ralph's core memory loop.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(WheelerMemoryBridge, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self.ai = None
        if WHEELER_AVAILABLE:
            try:
                # Initialize with standard size
                print("[WheelerBridge] Initializing Wheeler AI backend...")
                self.ai = WheelerAI(width=128, height=128)
                
                # Try to offload dynamics to NPU
                if NPU_AVAILABLE:
                    try:
                        print("[WheelerBridge] Attempting NPU offload...")
                        # Map to 128x128 as specified in WheelerAI init
                        npu_engine = NPUWheelerEngine(
                            model_path="ai_tech_stack/models/wheeler_dynamics_128.xml", 
                            device="NPU"
                        )
                        # Replace the CPU dynamics with NPU engine
                        # We need to make sure the NPU engine has a run_dynamics method 
                        # compatible with the Trajectory expectation
                        self.ai.dynamics = npu_engine
                        print(f"[WheelerBridge] NPU Acceleration Active on {npu_engine.device}.")
                    except Exception as e:
                        print(f"[WheelerBridge] NPU offload failed: {e}. Using CPU.")
                
                # Start autonomic system (dreaming/consolidation)
                self.ai.start_autonomic()
                print("[WheelerBridge] Wheeler AI Active.")
            except Exception as e:
                print(f"[WheelerBridge] Failed to initialize: {e}")
        
        self._initialized = True

    def remember(self, text: str):
        """
        Process text through Wheeler dynamics and store in 2D spatial memory.
        """
        if self.ai:
            try:
                # process() does encoding -> dynamics -> storage
                # We suppress the output here as we just want to remember
                self.ai.process(text)
            except Exception as e:
                print(f"[WheelerBridge] Error remembering: {e}")

    def recall(self, query: str) -> str:
        """
        Probe Wheeler memory for associations.

        When stability tracking is enabled, each recalled pattern
        gets a hit recorded and its stability_score is appended.
        """
        if not self.ai:
            return ""

        try:
            # 1. Encode query to frame
            frame = self.ai.codec.encode(query)

            # 2. Run dynamics to find attractor (stable thought)
            # Use fewer ticks for quick lookup
            traj = self.ai.dynamics.run_dynamics(frame, max_ticks=20)
            attractor = traj.final_frame

            # 3. Query knowledge store
            results = self.ai.knowledge.recall(attractor, top_k=3)

            if not results:
                return ""

            # Format results with stability scores
            summary = []
            for _, score, memory in results:
                if score > 0.4:  # Similarity threshold
                    stability_info = ""
                    if STABILITY_TRACKING:
                        pid = hashlib.md5(memory.text.encode()).hexdigest()[:12]
                        metrics = stability_tracker.record_hit(pid, memory.text)
                        stability_info = f" (stability={metrics.stability_score:.2f})"
                    summary.append(f"- [{score:.2f}]{stability_info} {memory.text}")

            if STABILITY_TRACKING:
                stability_tracker.flush()

            if summary:
                return "\n--- WHEELER SPATIAL ASSOCIATIONS ---\n" + "\n".join(summary) + "\n"
            return ""

        except Exception as e:
            print(f"[WheelerBridge] Error recalling: {e}")
            return ""

    def shutdown(self):
        """Stop background threads."""
        if self.ai:
            self.ai.stop_autonomic()
