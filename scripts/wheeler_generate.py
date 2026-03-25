"""CLI: Generate from CA trajectory resonance (IT from BIT).

Usage:
    wheeler-generate "what do you know about GPU debugging?"
    wheeler-generate "query" --chunk code --min-resonance 0.20 --dedup peak
    wheeler-generate "query" --embed --salience high
    wheeler-generate "query" --json
    wheeler-generate --interactive
"""

import argparse
import json
import sys


def _result_to_dict(result) -> dict:
    """Serialise a GenerationResult to a JSON-friendly dict."""
    return {
        "query": result.query,
        "convergence_state": result.convergence_state,
        "convergence_ticks": result.convergence_ticks,
        "sequence": result.sequence,
        "ticks": [
            {
                "tick": t.tick,
                "peak_text": t.peak_text,
                "peak_sim": round(t.peak_sim, 6),
                "emitted": t.emitted,
            }
            for t in result.ticks
        ],
    }


def _print_result(result, verbose: bool = False) -> None:
    """Human-readable output."""
    if not result.sequence:
        print(
            "(no resonances above threshold — try --min-resonance lower or store more memories)"
        )
        return

    print(f"Query : {result.query}")
    print(f"State : {result.convergence_state}  ticks={result.convergence_ticks}")
    print()

    # Find the tick where each sequence entry was first emitted
    emitted_ticks: list[tuple[int, float, str]] = []
    seen: set[str] = set()
    for t in result.ticks:
        if t.emitted and t.peak_text not in seen and t.peak_text in result.sequence:
            seen.add(t.peak_text)
            emitted_ticks.append((t.tick, t.peak_sim, t.peak_text))

    for i, (tick, sim, text) in enumerate(emitted_ticks, 1):
        print(f"  [{i}] [tick {tick:>3}]  {text}  (sim={sim:.3f})")

    if verbose:
        print()
        print("Full tick log:")
        print(f"  {'tick':>4}  {'sim':>7}  {'emitted':>7}  text")
        print("  " + "-" * 60)
        for t in result.ticks:
            flag = "*" if t.emitted else " "
            text_preview = (
                t.peak_text[:45] + "…" if len(t.peak_text) > 45 else t.peak_text
            )
            print(f"  {t.tick:>4}  {t.peak_sim:>7.4f}  {flag:>7}  {text_preview}")


def _run_generation(args, query: str):
    """Run generation for a single query and print/return result."""
    from wheeler_memory.attention import salience_from_label
    from wheeler_memory.generation import trajectory_resonance

    sal = salience_from_label(args.salience) if args.salience else None

    try:
        result = trajectory_resonance(
            query,
            data_dir=args.data_dir,
            chunk=args.chunk,
            min_resonance=args.min_resonance,
            dedup=args.dedup,
            use_embedding=args.embed,
            salience=sal,
        )
    except ImportError as e:
        if "sentence_transformers" in str(e):
            print(
                "Error: Embedding requires sentence-transformers.\n"
                "Install with: pip install -e '.[embed]'",
                file=sys.stderr,
            )
        else:
            print(f"Error: Missing dependency — {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(_result_to_dict(result), indent=2))
    else:
        _print_result(result, verbose=getattr(args, "verbose", False))

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate from CA trajectory resonance (IT from BIT)"
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Query text to generate from",
    )
    parser.add_argument(
        "--chunk",
        default=None,
        help="Restrict to this chunk (default: auto-select all chunks)",
    )
    parser.add_argument(
        "--min-resonance",
        type=float,
        default=0.15,
        help="Minimum Pearson similarity to emit at a tick (default: 0.15)",
    )
    parser.add_argument(
        "--dedup",
        choices=["first", "last", "peak", "none"],
        default="first",
        help="Deduplication strategy (default: first)",
    )
    parser.add_argument(
        "--embed",
        action="store_true",
        help="Use sentence embedding for query frame (semantic mode)",
    )
    parser.add_argument(
        "--salience",
        choices=["low", "medium", "high"],
        default=None,
        help="Attention level: low (fast), medium (default), high (deep)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output full GenerationResult as JSON",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="REPL mode: enter queries interactively",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show full per-tick log",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Data directory (default: ~/.wheeler_memory)",
    )

    args = parser.parse_args()

    if args.interactive:
        import readline  # noqa: F401 — enables line editing on supported platforms

        print("Wheeler Generate — IT from BIT  (Ctrl-D or Ctrl-C to exit)")
        print(
            f"Settings: chunk={args.chunk!r}  min_resonance={args.min_resonance}  dedup={args.dedup!r}"
        )
        print()
        while True:
            try:
                query = input("query> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not query:
                continue
            _run_generation(args, query)
            print()
        return

    if args.query is None:
        parser.error("query is required (or use --interactive)")

    _run_generation(args, args.query)


if __name__ == "__main__":
    main()
