import sys
import os

# Add ai_tech_stack to path
sys.path.append(os.path.join(os.getcwd(), "ai_tech_stack"))

def test_bridge_npu_init():
    from ralph_core.wheeler import WheelerMemoryBridge
    
    print("Initializing WheelerMemoryBridge...")
    bridge = WheelerMemoryBridge()
    
    if bridge.ai and hasattr(bridge.ai, 'dynamics'):
        from ralph_core.npu_engine import NPUWheelerEngine
        is_npu = isinstance(bridge.ai.dynamics, NPUWheelerEngine)
        print(f"Dynamics engine type: {type(bridge.ai.dynamics)}")
        assert is_npu, "Bridge dynamics should be NPUWheelerEngine"
        print(f"Verified: Bridge is using NPU on {bridge.ai.dynamics.device}")
    else:
        print("Wheeler AI not available in bridge.")

if __name__ == "__main__":
    try:
        test_bridge_npu_init()
        print("Integration Test PASSED")
    except Exception as e:
        print(f"Integration Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
