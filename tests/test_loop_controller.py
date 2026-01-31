import sys
import os

# Add ai_tech_stack to path
sys.path.append(os.path.join(os.getcwd(), "ai_tech_stack"))

def test_loop_logic():
    from ralph_core.controller import RalphLoop, LoopState
    
    print("Testing RalphLoop State Transitions...")
    loop = RalphLoop(max_iterations=5)
    
    # 1. Start
    loop.run_step("Create a sum function")
    assert loop.state == LoopState.TRANSLATING
    
    # 2. To Planning
    loop.run_step("...")
    assert loop.state == LoopState.PLANNING
    
    # 3. To Coding
    loop.run_step("...")
    assert loop.state == LoopState.CODING
    
    # 4. To Verifying
    loop.run_step("...")
    assert loop.state == LoopState.VERIFYING
    
    # 5. Fail Verification (Strike 1)
    print("Simulating Fail 1...")
    loop.run_step("...")
    assert loop.state == LoopState.CODING
    assert loop.strike_count == 1
    
    # 6. Cycle back to Verifying
    loop.run_step("...") # Transition to VERIFYING
    
    # 7. Fail Verification (Strike 2)
    print("Simulating Fail 2...")
    loop.run_step("...") # Transition back to CODING, strike = 2
    assert loop.strike_count == 2
    
    # 8. Cycle back to Verifying
    loop.run_step("...") # Transition to VERIFYING
    
    # 9. Fail Verification (Strike 3) -> REFLECT
    print("Simulating Fail 3...")
    loop.run_step("...") # Transition to REFLECTING
    assert loop.state == LoopState.REFLECTING
    print("Verified: 3-Strike Rule triggered Reflection.")
    
    # 10. Reflect -> New Plan
    loop.run_step("...") # Transition to PLANNING
    assert loop.state == LoopState.PLANNING
    assert loop.strike_count == 0
    print("Verified: Reflection reset strikes and returned to Planning.")

if __name__ == "__main__":
    try:
        test_loop_logic()
        print("\n[SUCCESS] RalphLoop Controller Logic Verified.")
    except Exception as e:
        print(f"\n[FAILURE] {e}")
        import traceback
        traceback.print_exc()
        exit(1)
