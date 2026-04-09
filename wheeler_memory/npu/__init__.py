"""Intel NPU and future accelerator scaffolding.

Provides stubs for Intel NPU (via OpenVINO) and Google Coral Edge TPU
integration. These are placeholder modules — actual implementation
depends on driver maturity and hardware availability.
"""

from __future__ import annotations


def npu_available() -> bool:
    """Check if an Intel NPU is accessible via OpenVINO."""
    try:
        from openvino.runtime import Core

        core = Core()
        return "NPU" in core.available_devices
    except Exception:
        return False


def device_info() -> dict:
    """Return NPU device information if available."""
    info: dict = {"available": False, "device": None, "backend": None}
    try:
        from openvino.runtime import Core

        core = Core()
        if "NPU" in core.available_devices:
            info["available"] = True
            info["device"] = "Intel NPU"
            info["backend"] = "OpenVINO"
            try:
                info["full_name"] = core.get_property("NPU", "FULL_DEVICE_NAME")
            except Exception:
                pass
    except ImportError:
        info["backend"] = "OpenVINO not installed"
    return info
