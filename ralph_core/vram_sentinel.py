import shutil
import subprocess
import json
import os

class VRAMSentinel:
    """
    Monitors VRAM usage on AMD GPUs using rocm-smi.
    Ensures we have enough space to load 14B models (approx 9GB).
    """
    def __init__(self):
        self.rocm_smi_path = shutil.which("rocm-smi") or "/opt/rocm/bin/rocm-smi"
        if not os.path.exists(self.rocm_smi_path):
            print("[VRAMSentinel] Warning: rocm-smi not found. VRAM monitoring disabled.")
            self.available = False
        else:
            self.available = True

    def get_vram_usage(self) -> dict:
        """Returns {total, used, free} in bytes."""
        if not self.available:
            return {"total": 16 * 1024**3, "used": 0, "free": 16 * 1024**3} # Dummy 16GB

        try:
            # Use --json flag for reliable parsing
            output = subprocess.check_output([self.rocm_smi_path, "--showmeminfo", "vram", "--json"])
            data = json.loads(output)
            
            # Assuming single GPU (card0)
            card = list(data.keys())[0]
            total = int(data[card]["VRAM Total Memory (B)"])
            used = int(data[card]["VRAM Total Used Memory (B)"])
            
            return {
                "total": total,
                "used": used,
                "free": total - used
            }
        except Exception as e:
            print(f"[VRAMSentinel] Error reading VRAM: {e}")
            return {"total": 0, "used": 0, "free": 0}

    def check_headroom(self, required_gb: float = 9.0) -> bool:
        """Checks if there is enough free VRAM for a model."""
        usage = self.get_vram_usage()
        free_gb = usage["free"] / (1024**3)
        print(f"[VRAMSentinel] Free VRAM: {free_gb:.2f} GB (Required: {required_gb} GB)")
        return free_gb >= required_gb
