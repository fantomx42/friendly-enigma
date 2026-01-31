import pytest
import shutil
from unittest.mock import MagicMock, patch

def test_vram_sentinel_rocm_smi():
    """Verify that we can parse rocm-smi output or handle its absence."""
    # Check if rocm-smi is installed (we installed it earlier)
    rocm_smi_path = shutil.which("rocm-smi")
    if not rocm_smi_path:
        # If not in path, check /opt/rocm/bin
        rocm_smi_path = "/opt/rocm/bin/rocm-smi"
        import os
        if not os.path.exists(rocm_smi_path):
            pytest.skip("rocm-smi not found, skipping VRAM test")

    # Mock subprocess to simulate rocm-smi output
    with patch("subprocess.check_output") as mock_run:
        # Simulate JSON output from rocm-smi --showmeminfo vram --json
        mock_run.return_value = b'{"card0": {"VRAM Total Memory (B)": "17163091968", "VRAM Total Used Memory (B)": "456789"}}'
        
        # Import directly from file path to avoid package init issues
        import sys
        import os
        module_dir = os.path.join(os.getcwd(), "ai_tech_stack/ralph_core")
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
            
        from vram_sentinel import VRAMSentinel
        sentinel = VRAMSentinel()
        vram = sentinel.get_vram_usage()
        
        assert vram['total'] > 0
        assert vram['used'] >= 0
        assert vram['free'] == vram['total'] - vram['used']
        print(f"VRAM Sentinel Test Passed: {vram}")

if __name__ == "__main__":
    test_vram_sentinel_rocm_smi()
