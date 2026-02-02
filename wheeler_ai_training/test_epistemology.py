import sys
import os
import time

# Ensure we can import the module
sys.path.append(os.getcwd())
from wheeler_ai import WheelerAI

def test_epistemology():
    print("Initializing Wheeler AI...")
    ai = WheelerAI(width=128, height=128)
    
    fact_a = "The capital of France is Paris."
    fact_b = "The capital of France is Rome." # Contradiction/Tension
    
    print(f"\nPhase 1: Building Confidence in '{fact_a}'")
    # We force multiple stores to boost metrics
    # Note: store() updates hits if hash matches
    
    # 1. First learn
    resp = ai.process(fact_a)
    print(f"Iter 1: {resp.strip()}")
    
    # 2. Reinforce (simulate different contexts by clearing context frames slightly)
    # We manually hack the memory to boost confidence for testing speed
    print("...Reinforcing belief...")
    
    # Find the memory we just stored
    mem_idx = ai.knowledge.text_index.get(ai.knowledge.memories[0].text_hash if hasattr(ai.knowledge.memories[0], 'text_hash') else "")
    # Actually text_index stores by hash, let's find it manually or via hash
    import hashlib
    h = hashlib.sha256(fact_a.encode()).hexdigest()[:16]
    mem_idx = ai.knowledge.text_index[h]
    
    memory = ai.knowledge.memories[mem_idx]
    memory.hits = 20
    memory.reinforcement_diversity = 5
    memory.connectivity = 3
    memory.stability = 1.0
    
    print(f"Belief Status: '{memory.text}'")
    print(f"  Confidence: {memory.confidence:.2f}")
    
    if memory.confidence < 0.75:
        print("WARNING: Confidence too low for tension test. Boosting further.")
        memory.hits = 50
        print(f"  New Confidence: {memory.confidence:.2f}")

    print(f"\nPhase 2: Testing Tension with '{fact_b}'")
    resp = ai.process(fact_b)
    print(f"Response: {resp.strip()}")
    
    if "Tension detected" in resp:
        print("\nSUCCESS: Tension correctly detected!")
    else:
        print("\nFAILURE: Tension NOT detected. (Did the frames not overlap enough?)")
        
        # Debug info
        print("Debug: Checking similarity...")
        frame_a = ai.codec.encode(fact_a)
        frame_b = ai.codec.encode(fact_b)
        
        # Run dynamics
        traj_a = ai.dynamics.run_dynamics(frame_a)
        traj_b = ai.dynamics.run_dynamics(frame_b)
        
        sim = ai.reasoning.similarity(traj_a.final_frame, traj_b.final_frame)
        print(f"Similarity score: {sim:.4f}")
        print("Required range for tension: 0.85 < score < 0.99")

if __name__ == "__main__":
    test_epistemology()
