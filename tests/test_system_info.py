"""Tests for `wheeler-info` (scripts/system_info.py) and the `accel` field on
`get_system_summary`. The CLI subprocess tests run on any host — `accel_info()`
returns `{"gpu": False, "gpu_version": None}` when HIP isn't built, which is
shape-compatible with what the CLI emits."""

from __future__ import annotations

import json
import subprocess
import sys

from wheeler_memory.hardware import get_system_summary


def test_get_system_summary_has_accel_key() -> None:
    summary = get_system_summary()
    assert "accel" in summary
    accel = summary["accel"]
    assert isinstance(accel, dict)
    assert "gpu" in accel
    assert "gpu_version" in accel
    assert isinstance(accel["gpu"], bool)
    assert accel["gpu_version"] is None or isinstance(accel["gpu_version"], int)


def _run_cli(*flags: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "scripts.system_info", *flags],
        capture_output=True,
        text=True,
        check=True,
    )


def test_cli_json_mode_is_pure_json() -> None:
    result = _run_cli("--json")
    parsed = json.loads(result.stdout)
    assert "accel" in parsed
    assert "optimal_device" in parsed
    assert "\x1b[" not in result.stdout


def test_cli_default_mode_has_footer() -> None:
    result = _run_cli()
    parsed, idx = json.JSONDecoder().raw_decode(result.stdout)
    assert "accel" in parsed
    remainder = result.stdout[idx:]
    assert "Wheeler Memory Auto-Config" in remainder
    assert "Embedding device (PyTorch)" in remainder
    assert "CA kernel (HIP)" in remainder
