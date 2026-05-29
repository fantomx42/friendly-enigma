#!/usr/bin/env python3
"""CLI tool to display system hardware information and optimal device configuration."""

import argparse
import json
import os
import sys

from wheeler_memory.hardware import get_system_summary

GREEN = "\033[1;32m"
YELLOW = "\033[1;33m"
RESET = "\033[0m"


def _use_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def main() -> None:
    parser = argparse.ArgumentParser(description="Wheeler Memory system + device info")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit only the JSON object (no footer, no ANSI). For piping.",
    )
    args = parser.parse_args()

    try:
        summary = get_system_summary()
        print(json.dumps(summary, indent=2))

        if args.json:
            return

        g, y, r = (GREEN, YELLOW, RESET) if _use_color() else ("", "", "")
        optimal = summary.get("optimal_device", "cpu")
        ca_gpu = summary.get("accel", {}).get("gpu", False)
        warnings = summary.get("warnings", [])

        print("\n[ Wheeler Memory Auto-Config ]")
        print(f"Embedding device (PyTorch): {g}{optimal.upper()}{r}")
        print(f"CA kernel (HIP):            {g}{'GPU' if ca_gpu else 'CPU'}{r}")
        if warnings:
            print(f"\n{y}[ Warnings ]{r}")
            for warn in warnings:
                print(f"- {warn}")
    except ImportError as e:
        print(
            f"Error: Missing dependency — {e}\nRun: pip install -e .", file=sys.stderr
        )
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
