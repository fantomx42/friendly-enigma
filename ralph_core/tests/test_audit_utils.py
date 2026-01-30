"""
Tests for Repository Audit Utilities
"""

import pytest
from verify_diagnostic import get_git_log, parse_status_md

def test_get_git_log_returns_list():
    """Verify that get_git_log returns a list of dictionaries."""
    logs = get_git_log(limit=5)
    assert isinstance(logs, list)
    if logs:
        assert "hash" in logs[0]
        assert "date" in logs[0]
        assert "message" in logs[0]

def test_parse_status_md():
    """Verify that parse_status_md correctly extracts updates."""
    content = """
    # RECENT UPDATES
    - [2026-01-29] Wheeler Memory Integrated
    - [2026-01-21] Tool Registry Complete
    """
    updates = parse_status_md(content)
    assert len(updates) == 2
    assert updates[0]["date"] == "2026-01-29"
    assert "Wheeler Memory Integrated" in updates[0]["message"]
