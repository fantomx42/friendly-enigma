import sys
import os

# Add ai_tech_stack to path
sys.path.append(os.path.join(os.getcwd(), "ai_tech_stack"))

def test_bus_integration():
    from ralph_core.controller import RalphLoop, LoopState
    from ralph_core.protocols.bus import MessageBus
    from ralph_core.protocols.messages import complete_message, error_message
    
    print("Testing RalphLoop with MessageBus Integration...")
    bus = MessageBus()
    loop = RalphLoop(max_iterations=5, bus=bus)
    
    # 1. Idle -> Translating
    loop.run_step("Build a robot")
    assert loop.state == LoopState.TRANSLATING
    
    # 2. Translating -> Planning
    loop.run_step("")
    assert loop.state == LoopState.PLANNING
    
    # 3. Planning -> Coding
    # This should have sent a WORK_REQUEST
    loop.run_step("")
    assert loop.state == LoopState.CODING
    assert any(m.type.value == "work_request" for m in bus.get_history())
    print("Verified: WORK_REQUEST sent to bus.")
    
    # 4. Coding -> Verifying
    loop.run_step("")
    assert loop.state == LoopState.VERIFYING
    
    # 5. Simulate ERROR from bus
    print("Simulating error message on bus...")
    bus.send(error_message("Syntax error in robot.py"))
    
    # 6. Run step - should detect error and cycle back
    loop.run_step("")
    assert loop.state == LoopState.CODING
    assert loop.strike_count == 1
    print("Verified: Loop reacted to ERROR message.")
    
    # 7. Cycle back to Verifying
    loop.run_step("")
    assert loop.state == LoopState.VERIFYING
    
    # 8. Simulate COMPLETE from bus
    print("Simulating complete message on bus...")
    bus.send(complete_message("Robot built successfully"))
    
    # 9. Run step - should detect completion
    loop.run_step("")
    assert loop.state == LoopState.COMPLETE
    print("Verified: Loop reacted to COMPLETE message.")

if __name__ == "__main__":
    try:
        test_bus_integration()
        print("\n[SUCCESS] RalphLoop Bus Integration Verified.")
    except Exception as e:
        print(f"\n[FAILURE] {e}")
        import traceback
        traceback.print_exc()
        exit(1)
