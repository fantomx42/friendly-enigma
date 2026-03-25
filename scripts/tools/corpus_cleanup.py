"""
Wheeler Memory Corpus Cleanup Tool

This tool removes noisy training data from the Wheeler Memory corpus
using pattern-based filtering. It's designed to be safe, transparent, and reversible
(through dry-run mode).

Usage:
    python filter_tool.py --pattern trivia --dry-run
    python filter_tool.py --pattern "dialogue|passage" --chunk code
    python filter_tool.py --pattern trivia --chunk general --backup

Recommended cleanup sequence:
    1. Run dry-runs to verify patterns match expected entries
    2. Backup corpus before deletion
    3. Remove worst offenders first (daily_tasks dialogues)
    4. Remove code chunk dialogues/passages
    5. Remove general chunk trivia
    6. Verify similarity floor improvement
"""

import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from collections import Counter

from wheeler_memory.chunking import list_existing_chunks
from wheeler_memory.eviction import _delete_memory_files


# Pattern definitions
PATTERNS = {
    "trivia": re.compile(
        r"^\s*(what|who|when|where|why|how|which|whose|whom).*\?", re.IGNORECASE
    ),
    "dialogue": re.compile(r"^(dialogue|[a-z]+:\s*)", re.IGNORECASE),
    "passage": re.compile(r"^(passage|[a-z]{3,20}:|visit|read)", re.IGNORECASE),
    "short_fragment": re.compile(r"^.{0,30}$"),
}

# Recommended cleanup by chunk
CLEANUP_RECOMMENDATIONS = {
    "daily_tasks": {
        "pattern": "dialogue",
        "reason": "77.1% of daily_tasks are dialogue training data, not personal tasks",
        "priority": "CRITICAL",
        "estimated_removal": 84,
    },
    "code": {
        "pattern": ["dialogue", "passage"],
        "reason": "12% of code are dialogue/passage format—mismatch, likely leaked from daily_tasks",
        "priority": "HIGH",
        "estimated_removal": 258,
    },
    "general": {
        "pattern": "trivia",
        "reason": "46.9% of general are trivia Q&A—causes similarity floor problem",
        "priority": "HIGH",
        "estimated_removal": 625,
    },
}


def _load_index(chunk_dir: Path) -> dict:
    """Load index.json from chunk directory."""
    index_path = chunk_dir / "index.json"
    if index_path.exists():
        return json.loads(index_path.read_text())
    return {}


def _save_index(chunk_dir: Path, index: dict) -> None:
    """Save index.json to chunk directory."""
    index_path = chunk_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2))


def find_matches(
    data_dir: Path,
    patterns: list[str] | str,
    chunk: str | None = None,
) -> dict[str, list[tuple[str, str]]]:
    """
    Find all entries matching the given patterns.

    Args:
        data_dir: Wheeler memory data directory
        patterns: Pattern name(s) from PATTERNS dict, or comma-separated list
        chunk: Optional specific chunk to search (else all chunks)

    Returns:
        Dict mapping chunk_name -> list of (hex_key, text) tuples
    """
    if isinstance(patterns, str):
        patterns = [p.strip() for p in patterns.split("|")]

    # Resolve pattern objects
    compiled_patterns = []
    for p in patterns:
        if p in PATTERNS:
            compiled_patterns.append(PATTERNS[p])
        else:
            # Treat as raw regex
            try:
                compiled_patterns.append(re.compile(p, re.IGNORECASE))
            except re.error as e:
                raise ValueError(f"Invalid regex pattern: {p}\n{e}")

    matches = {}

    chunks_to_search = [chunk] if chunk else list_existing_chunks(data_dir)

    for chunk_name in chunks_to_search:
        chunk_dir = data_dir / "chunks" / chunk_name
        if not chunk_dir.exists():
            continue

        index = _load_index(chunk_dir)
        chunk_matches = []

        for hex_key, meta in index.items():
            text = meta.get("text", "").strip()
            # Match if ANY pattern matches
            if any(pattern.match(text) for pattern in compiled_patterns):
                chunk_matches.append((hex_key, text))

        if chunk_matches:
            matches[chunk_name] = chunk_matches

    return matches


def display_matches(matches: dict[str, list], max_display: int = 5):
    """Pretty-print matching entries by chunk."""
    total = sum(len(v) for v in matches.values())
    print(f"\nFound {total} matching entries:\n")

    for chunk_name in sorted(matches.keys()):
        entries = matches[chunk_name]
        print(f"  {chunk_name.upper()}: {len(entries)} entries")
        for hex_key, text in entries[:max_display]:
            print(f"    - {text[:70]}")
        if len(entries) > max_display:
            print(f"    ... and {len(entries) - max_display} more")
        print()


def estimate_impact(
    data_dir: Path,
    matches: dict[str, list],
) -> dict[str, dict]:
    """Estimate corpus impact of removing matched entries."""
    impact = {}

    for chunk_name in list_existing_chunks(data_dir):
        chunk_dir = data_dir / "chunks" / chunk_name
        index = _load_index(chunk_dir)
        total_before = len(index)
        removed_count = len(matches.get(chunk_name, []))
        total_after = total_before - removed_count

        impact[chunk_name] = {
            "before": total_before,
            "after": total_after,
            "removed": removed_count,
            "pct_removed": 100 * removed_count / total_before
            if total_before > 0
            else 0,
        }

    return impact


def display_impact(impact: dict[str, dict]):
    """Pretty-print corpus impact estimate."""
    total_before = sum(v["before"] for v in impact.values())
    total_removed = sum(v["removed"] for v in impact.values())
    total_after = total_before - total_removed

    print(f"\nEstimated Impact:")
    print(
        f"  Total:    {total_before:,} → {total_after:,} ({100 * total_removed / total_before:.1f}% removed)"
    )
    print(f"\n  By chunk:")

    for chunk_name in sorted(impact.keys()):
        info = impact[chunk_name]
        if info["removed"] > 0:
            print(
                f"    {chunk_name:15} "
                f"{info['before']:4d} → {info['after']:4d} "
                f"(remove {info['removed']:3d}, {info['pct_removed']:5.1f}%)"
            )


def backup_corpus(data_dir: Path) -> Path:
    """Create timestamped backup of corpus."""
    backup_dir = (
        data_dir.parent
        / f"wheeler_memory_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    chunks_dir = data_dir / "chunks"

    print(f"\nBacking up corpus to: {backup_dir}")
    shutil.copytree(chunks_dir, backup_dir / "chunks", dirs_exist_ok=True)

    return backup_dir


def delete_matches(
    data_dir: Path,
    matches: dict[str, list],
    dry_run: bool = False,
) -> dict[str, int]:
    """
    Delete matched entries from corpus.

    Args:
        data_dir: Wheeler memory data directory
        matches: Output from find_matches()
        dry_run: If True, don't actually delete

    Returns:
        Dict mapping chunk_name -> count deleted
    """
    deleted_count = {}

    for chunk_name in sorted(matches.keys()):
        chunk_dir = data_dir / "chunks" / chunk_name
        entries = matches[chunk_name]

        deleted_count[chunk_name] = 0

        for hex_key, text in entries:
            if dry_run:
                deleted_count[chunk_name] += 1
            else:
                try:
                    _delete_memory_files(data_dir, chunk_name, hex_key)
                    deleted_count[chunk_name] += 1
                except Exception as e:
                    print(f"  ERROR deleting {hex_key}: {e}")

    return deleted_count


def main():
    parser = argparse.ArgumentParser(
        description="Clean Wheeler Memory corpus of training data noise"
    )
    parser.add_argument(
        "--pattern",
        required=True,
        help=(
            "Pattern(s) to match. Either preset (trivia|dialogue|passage|short_fragment) "
            "or comma-separated regex patterns"
        ),
    )
    parser.add_argument(
        "--chunk",
        help="Specific chunk to filter (default: all chunks)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without deleting",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create timestamped backup before deletion",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Data directory (default: ~/.wheeler_memory)",
    )
    parser.add_argument(
        "--show-recommendations",
        action="store_true",
        help="Show recommended cleanup strategy",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else Path.home() / ".wheeler_memory"

    if not (data_dir / "chunks").exists():
        print(f"ERROR: Corpus not found at {data_dir / 'chunks'}")
        return 1

    if args.show_recommendations:
        print("\n" + "=" * 80)
        print("RECOMMENDED CLEANUP STRATEGY")
        print("=" * 80)
        for chunk, rec in CLEANUP_RECOMMENDATIONS.items():
            print(f"\n{chunk.upper()}: {rec['priority']}")
            print(f"  Pattern: {rec['pattern']}")
            print(f"  Reason: {rec['reason']}")
            print(f"  Estimated removal: {rec['estimated_removal']} entries")
        return 0

    # Find matches
    print(f"\nSearching for pattern: {args.pattern}")
    if args.chunk:
        print(f"In chunk: {args.chunk}")
    print()

    try:
        matches = find_matches(data_dir, args.pattern, chunk=args.chunk)
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1

    if not matches:
        print("No matches found.")
        return 0

    # Display matches
    display_matches(matches, max_display=5)

    # Estimate impact
    impact = estimate_impact(data_dir, matches)
    display_impact(impact)

    # Dry run?
    if args.dry_run:
        print("\n[DRY RUN] No changes made.")
        return 0

    # Confirm
    total_removed = sum(len(v) for v in matches.values())
    response = input(f"\nRemove {total_removed} entries? [y/N] ").strip().lower()
    if response != "y":
        print("Cancelled.")
        return 0

    # Backup if requested
    if args.backup:
        backup_corpus(data_dir)

    # Delete
    print("\nRemoving entries...")
    deleted_count = delete_matches(data_dir, matches, dry_run=False)

    print("\nDeleted:")
    for chunk_name, count in sorted(deleted_count.items()):
        if count > 0:
            print(f"  {chunk_name}: {count} entries")

    print("\nCleanup complete!")
    return 0


if __name__ == "__main__":
    exit(main())
