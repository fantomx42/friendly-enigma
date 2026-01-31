import openvino

def test_npu_available():
    core = openvino.Core()
    devices = core.available_devices
    print(f"Available devices: {devices}")
    assert "NPU" in devices, "NPU device not found in OpenVINO available devices"

if __name__ == "__main__":
    try:
        test_npu_available()
        print("Test PASSED: NPU is visible to OpenVINO.")
    except AssertionError as e:
        print(f"Test FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)
