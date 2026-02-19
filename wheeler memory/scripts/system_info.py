#!/usr/bin/env python3
"""CLI tool to display system hardware information and optimal device configuration."""

import json
from wheeler_memory.hardware import get_system_summary

def main():
    summary = get_system_summary()
    
    # Print formatted JSON
    print(json.dumps(summary, indent=2))
    
    # Highlight optimal device
    optimal = summary.get("optimal_device", "cpu")
    print(f"\n[ Wheeler Memory Auto-Config ]")
    print(f"Optimal Device Selected: \033[1;32m{optimal.upper()}\033[0m")
    
    # Print warnings if any
    warnings = summary.get("warnings", [])
    if warnings:
        print("\n\033[1;33m[ Warnings ]\033[0m")
        for warn in warnings:
            print(f"- {warn}")

if __name__ == "__main__":
    main()
