"""Google Coral Edge TPU integration (future).

Scaffolding for dual Coral M.2 TPU inference. Not yet active —
hardware not yet installed.
"""

from __future__ import annotations


def coral_available() -> bool:
    """Check if a Coral Edge TPU is accessible."""
    try:
        from pycoral.utils.edgetpu import list_edge_tpus

        return len(list_edge_tpus()) > 0
    except Exception:
        return False


def dual_tpu_info() -> dict:
    """Return info about connected Coral TPUs."""
    info: dict = {"available": False, "devices": [], "count": 0}
    try:
        from pycoral.utils.edgetpu import list_edge_tpus

        tpus = list_edge_tpus()
        info["available"] = len(tpus) > 0
        info["devices"] = tpus
        info["count"] = len(tpus)
    except ImportError:
        info["devices"] = ["pycoral not installed"]
    return info
