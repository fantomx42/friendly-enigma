"""Tests for wheeler_memory.accel module imports and device detection."""


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
    assert isinstance(info["gpu"], bool)


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
