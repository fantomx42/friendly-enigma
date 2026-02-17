"""Wheeler Memory: cellular automata-based associative memory system."""

from .brick import MemoryBrick
from .chunking import (
    CHUNK_KEYWORDS,
    DEFAULT_CHUNK,
    find_brick_across_chunks,
    list_existing_chunks,
    select_chunk,
    select_recall_chunks,
)
from .dynamics import apply_ca_dynamics, evolve_and_interpret
from .hashing import hash_to_frame, text_to_hex
from .oscillation import detect_oscillation, get_cell_roles
from .rotation import store_with_rotation_retry
from .storage import list_memories, recall_memory, store_memory

__all__ = [
    "hash_to_frame",
    "text_to_hex",
    "apply_ca_dynamics",
    "evolve_and_interpret",
    "get_cell_roles",
    "detect_oscillation",
    "MemoryBrick",
    "store_memory",
    "recall_memory",
    "list_memories",
    "store_with_rotation_retry",
    "CHUNK_KEYWORDS",
    "DEFAULT_CHUNK",
    "select_chunk",
    "select_recall_chunks",
    "find_brick_across_chunks",
    "list_existing_chunks",
]
