#!/usr/bin/env python3
"""
Diagnostic script for repository audit.
Synchronizes STATUS.md with git log and filesystem reality.
"""

import subprocess
import re
import os

def get_git_log(limit=20, path="."):
    """
    Retrieves the git log as a list of dictionaries.
    Format: [{'hash': '...', 'date': 'YYYY-MM-DD', 'message': '...'}]
    """
    try:
        # %h: short hash, %as: author date (YYYY-MM-DD), %s: subject
        cmd = ["git", "log", f"-n {limit}", "--format=%h|%as|%s", "--", path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logs = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("|")
            if len(parts) == 3:
                logs.append({
                    "hash": parts[0],
                    "date": parts[1],
                    "message": parts[2]
                })
        return logs
    except Exception as e:
        print(f"Error fetching git log: {e}")
        return []

def parse_status_md(content):
    """
    Parses STATUS.md content to extract entries from RECENT UPDATES.
    """
    updates = []
    # Match lines like - [2026-01-29] **Update text**
    pattern = r"^\s*-\s*\[(\d{4}-\d{2}-\d{2})\]\s*(.*)"
    matches = re.findall(pattern, content, re.MULTILINE)
    for date, message in matches:
        updates.append({
            "date": date,
            "message": message.strip()
        })
    return updates

if __name__ == "__main__":
    # Example usage
    print("Recent Git Activity:")
    for entry in get_git_log(limit=5):
        print(f"[{entry['date']}] {entry['hash']}: {entry['message']}")