"""
wheeler_weights.py - Stability Metrics for Wheeler Memory Patterns

Implements the Symbolic Collapse Model (SCM) by tracking three dimensions
of pattern stability:

1. Hit Count       - Activation frequency (how often a pattern is recalled)
2. Frame Persistence - Survival across context switches (how many iterations
                       a pattern remains relevant)
3. Compression Survival - Whether a pattern survived meta-rule consolidation
                          (REM Sleep compression)

These combine into a stability_score (0.0 - 1.0) per pattern, used by
context_budget.py to allocate token budget proportionally.

See /docs/SCM_AXIOMS.md for theoretical foundation.
"""

import json
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


# Persistence path for stability data
STABILITY_DB_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "memory",
    "stability_metrics.json",
)


@dataclass
class PatternMetrics:
    """Stability metrics for a single Wheeler memory pattern."""

    pattern_id: str
    text_preview: str = ""  # First 80 chars for identification

    # --- SCM Dimensions ---
    hit_count: int = 0                 # Activation frequency
    frame_persistence: int = 0         # Frames survived across context switches
    compression_survived: bool = False  # Survived REM Sleep meta-rule consolidation

    # --- Tracking State ---
    first_seen: float = 0.0       # Unix timestamp
    last_accessed: float = 0.0    # Unix timestamp
    context_switches_seen: int = 0  # Total context switches since first_seen

    @property
    def stability_score(self) -> float:
        """
        Composite stability score (0.0 - 1.0).

        Weights:
          - Hit count:             40% (capped at 20 hits -> 1.0)
          - Frame persistence:     35% (ratio of frames survived / total switches)
          - Compression survival:  25% (binary: survived or not)

        These weights reflect SCM axiom priorities:
          - Demonstrated usage (hits) is the strongest signal
          - Persistence under pressure (frame survival) is next
          - Compression survival is a strong but binary signal
        """
        # Hit count component: logarithmic scaling, saturates at ~20 hits
        if self.hit_count <= 0:
            hit_score = 0.0
        else:
            import math
            hit_score = min(1.0, math.log(self.hit_count + 1) / math.log(21))

        # Frame persistence: ratio of frames survived vs context switches
        if self.context_switches_seen <= 0:
            persist_score = 0.0 if self.frame_persistence == 0 else 0.5
        else:
            persist_score = min(1.0, self.frame_persistence / self.context_switches_seen)

        # Compression survival: binary
        compress_score = 1.0 if self.compression_survived else 0.0

        return round(
            0.40 * hit_score + 0.35 * persist_score + 0.25 * compress_score,
            4,
        )


class StabilityTracker:
    """
    Tracks and persists stability metrics for Wheeler memory patterns.

    Thread-safe singleton - shared across the Ralph pipeline.
    """

    _instance: Optional["StabilityTracker"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._patterns: Dict[str, PatternMetrics] = {}
        self._dirty = False
        self._load()
        self._initialized = True

    # --- Public API ---

    def record_hit(self, pattern_id: str, text: str = "") -> PatternMetrics:
        """Record an activation (recall) of a pattern. Returns updated metrics."""
        metrics = self._get_or_create(pattern_id, text)
        metrics.hit_count += 1
        metrics.last_accessed = time.time()
        self._dirty = True
        return metrics

    def record_context_switch(self) -> None:
        """
        Called when the Ralph loop starts a new iteration.

        Increments context_switches_seen for all tracked patterns.
        Patterns that were accessed in the *previous* iteration get
        their frame_persistence incremented (they survived the switch).
        """
        now = time.time()
        # A pattern "survived" if it was accessed within the last iteration window
        # Use a 10-minute window as a heuristic for recency
        recency_threshold = now - 600

        for metrics in self._patterns.values():
            metrics.context_switches_seen += 1
            if metrics.last_accessed >= recency_threshold:
                metrics.frame_persistence += 1

        self._dirty = True

    def record_compression_survival(self, pattern_id: str) -> None:
        """
        Mark a pattern as having survived REM Sleep compression.

        Called by the Sleeper agent after consolidation identifies
        patterns that persisted through meta-rule generation.
        """
        if pattern_id in self._patterns:
            self._patterns[pattern_id].compression_survived = True
            self._dirty = True

    def get_stability(self, pattern_id: str) -> float:
        """Get stability score for a pattern. Returns 0.0 if unknown."""
        if pattern_id in self._patterns:
            return self._patterns[pattern_id].stability_score
        return 0.0

    def get_metrics(self, pattern_id: str) -> Optional[PatternMetrics]:
        """Get full metrics for a pattern."""
        return self._patterns.get(pattern_id)

    def get_all_scored(self) -> List[Dict]:
        """
        Get all patterns with their stability scores, sorted descending.

        Returns list of dicts with keys:
          pattern_id, text_preview, stability_score, hit_count,
          frame_persistence, compression_survived
        """
        results = []
        for pid, metrics in self._patterns.items():
            results.append({
                "pattern_id": pid,
                "text_preview": metrics.text_preview,
                "stability_score": metrics.stability_score,
                "hit_count": metrics.hit_count,
                "frame_persistence": metrics.frame_persistence,
                "compression_survived": metrics.compression_survived,
            })
        results.sort(key=lambda x: x["stability_score"], reverse=True)
        return results

    def flush(self) -> None:
        """Persist metrics to disk if dirty."""
        if self._dirty:
            self._save()
            self._dirty = False

    # --- Internal ---

    def _get_or_create(self, pattern_id: str, text: str = "") -> PatternMetrics:
        if pattern_id not in self._patterns:
            self._patterns[pattern_id] = PatternMetrics(
                pattern_id=pattern_id,
                text_preview=text[:80] if text else "",
                first_seen=time.time(),
                last_accessed=time.time(),
            )
        return self._patterns[pattern_id]

    def _load(self) -> None:
        if not os.path.exists(STABILITY_DB_FILE):
            return
        try:
            with open(STABILITY_DB_FILE, "r") as f:
                data = json.load(f)
            for entry in data.get("patterns", []):
                pid = entry.get("pattern_id", "")
                if pid:
                    self._patterns[pid] = PatternMetrics(**entry)
        except Exception as e:
            print(f"[StabilityTracker] Load error: {e}")

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(STABILITY_DB_FILE), exist_ok=True)
            data = {
                "version": 1,
                "updated": time.time(),
                "patterns": [asdict(m) for m in self._patterns.values()],
            }
            with open(STABILITY_DB_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[StabilityTracker] Save error: {e}")


# Module-level singleton
stability_tracker = StabilityTracker()
