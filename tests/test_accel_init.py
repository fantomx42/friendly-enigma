"""Tests for wheeler_memory.accel module imports and device detection."""

import pytest


def test_accel_imports():
    """accel package imports without error."""
    from wheeler_memory import accel

    assert hasattr(accel, "gpu_available")
    assert hasattr(accel, "accel_info")


def test_accel_info_returns_dict():
    """accel_info() returns a dict with expected keys."""
    from wheeler_memory.accel import accel_info

    info = accel_info()
    assert isinstance(info, dict)
    assert "gpu" in info
    assert "npu" in info
    assert isinstance(info["gpu"], bool)
    assert isinstance(info["npu"], bool)


def test_gpu_available_returns_bool():
    """gpu_available() returns a boolean (may be True or False)."""
    from wheeler_memory.accel import gpu_available

    result = gpu_available()
    assert isinstance(result, bool)


def test_accel_ca_module_imports():
    """accel.ca bindings import without error."""
    from wheeler_memory.accel import ca

    assert hasattr(ca, "gpu_available")
    assert hasattr(ca, "gpu_evolve_single")
    assert hasattr(ca, "gpu_evolve_batch")


def test_gpu_dynamics_shim():
    """gpu_dynamics.py re-exports from accel.ca."""
    from wheeler_memory import gpu_dynamics

    assert hasattr(gpu_dynamics, "gpu_available")
    assert hasattr(gpu_dynamics, "gpu_evolve_single")
    assert hasattr(gpu_dynamics, "gpu_evolve_batch")


def test_npu_imports():
    """npu package imports without error."""
    from wheeler_memory import npu

    assert hasattr(npu, "npu_available")
    assert hasattr(npu, "device_info")


def test_npu_available_returns_bool():
    """npu_available() returns False (no NPU runtime expected in CI)."""
    from wheeler_memory.npu import npu_available

    result = npu_available()
    assert isinstance(result, bool)


def test_coral_imports():
    """Coral sub-package imports without error."""
    from wheeler_memory.npu import coral

    assert hasattr(coral, "coral_available")
    assert hasattr(coral, "dual_tpu_info")


def test_coral_available_returns_bool():
    from wheeler_memory.npu.coral import coral_available

    assert coral_available() is False  # No Coral hardware expected
