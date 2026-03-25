"""Experiential grid — episodic memory with temporal context.

The experiential grid stores WHEN and HOW you encountered something.
Each episode bundles the semantic content with temporal context:
preceding query state, SCM state at encoding time, and timestamp.

Properties:
  - Evolved under loose rules (low push, high flow) for fast mixing.
  - Aggressive decay (EXPERIENTIAL_HALF_LIFE_DAYS = 2).
  - Consolidation (sleep): temporal context stripped, attractor re-projected
    into corpus space under tight rules.  Result: "I know this but don't
    remember learning it."
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


@dataclass
class ExperientialMeta:
    """Temporal context bundled with an experiential memory."""

    encoding_timestamp: str = ""
    preceding_query_hex: str = ""
    scm_snapshot_hash: str = ""
    consolidated: bool = False

    def __post_init__(self) -> None:
        if not self.encoding_timestamp:
            self.encoding_timestamp = datetime.now(timezone.utc).isoformat()

    def save(self, path: Path) -> None:
        """Write metadata to JSON."""
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> ExperientialMeta:
        """Load metadata from JSON."""
        data = json.loads(path.read_text())
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def scm_snapshot_hash(scm_grid: np.ndarray) -> str:
    """Compute a short hash of the current SCM state for provenance tracking."""
    return hashlib.sha256(scm_grid.tobytes()).hexdigest()[:16]


def experiential_dir(chunk_dir: Path) -> Path:
    """Return (and ensure) the experiential subdirectory for a chunk."""
    d = chunk_dir / "experiential"
    d.mkdir(parents=True, exist_ok=True)
    return d


def experiential_bricks_dir(chunk_dir: Path) -> Path:
    """Return (and ensure) the experiential bricks subdirectory for a chunk."""
    d = chunk_dir / "experiential_bricks"
    d.mkdir(parents=True, exist_ok=True)
    return d
