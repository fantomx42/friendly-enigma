"""Wheeler Memory: cellular automata-based associative memory system."""

from .attention import (
    AttentionBudget,
    compute_attention_budget,
    salience_from_label,
    salience_from_temperature,
)
from .brick import MemoryBrick
from .chunking import (
    CHUNK_KEYWORDS,
    DEFAULT_CHUNK,
    find_brick_across_chunks,
    list_existing_chunks,
    select_chunk,
    select_recall_chunks,
)
from .dynamics import (
    apply_ca_dynamics,
    apply_ca_dynamics_parameterized,
    evolve_and_interpret,
    evolve_with_params,
)
from .hashing import hash_to_frame, text_to_hex
from .oscillation import detect_oscillation, get_cell_roles
from .rotation import store_with_rotation_retry
from .storage import list_memories, recall_memory, store_memory
from .temperature import (
    HALF_LIFE_DAYS,
    HIT_SATURATION,
    MAX_ATTRACTORS,
    MAX_WARMTH,
    TIER_DEAD,
    TIER_FADING,
    TIER_HOT,
    TIER_WARM,
    WARMTH_HALF_LIFE_DAYS,
    WARMTH_HOP1,
    WARMTH_HOP2,
    compute_temperature,
    compute_warmth,
    effective_temperature,
    temperature_tier,
)
from .eviction import (
    EvictionResult,
    forget_by_text,
    forget_memory,
    score_memories,
    sweep_and_evict,
)
from .consolidation import (
    ConsolidationResult,
    consolidate_brick,
    consolidate_experiential_to_corpus,
    consolidation_stats,
    select_keyframes,
    sleep_consolidate,
)
from .warming import get_neighbors, load_associations

# GPU backend (optional — available only when a HIP .so is built)
try:
    from .accel.ca import (
        gpu_available,
        gpu_evolve_batch,
        gpu_evolve_single,
        gpu_version,
        gpu_query_vram,
    )
except ImportError:
    gpu_available = lambda: False
    gpu_evolve_single = None
    gpu_evolve_batch = None
    gpu_version = lambda: None
    gpu_query_vram = lambda *_: None

# Embedding backend (optional — requires sentence-transformers)
try:
    from .embedding import embed_available, embed_to_frame, embed_to_frame_batch
except ImportError:
    embed_available = lambda: False
    embed_to_frame = None
    embed_to_frame_batch = None

# SCM grid (spatial trust topology)
from .scm_grid import SCMGrid

# Experiential grid (episodic memory with temporal context)
from .experiential import ExperientialMeta

# Three-grid interference engine
from .interference import (
    ABSORBED,
    CONTESTED,
    GROUNDED,
    SILENT,
    UNCONSOLIDATED,
    ConsistencyResult,
    InterferenceResult,
    compute_interference,
    interference_score,
    recall_with_interference,
    self_consistency_check,
)

# Similarity functions (basin separation)
from .similarity import (
    hybrid_similarity,
    pearson_similarity,
    spatial_cosine_similarity,
)

# Reconstructive recall
from .reconstruction import reconstruct, reconstruct_batch

# Two-tier recall API (recognize / reconstruct from seed) and per-basin T
from .recall_api import (
    BasinSeed,
    Pattern,
    recognize,
    recognize_top_k,
    set_learning_enabled,
)
from .recall_api import reconstruct as reconstruct_from_seed

# FCAS — Fractal Cube Address Space (core address layer)
from .fcas import (
    Address,
    address_of,
    cross_cube_interference,
    evolve_cube,
    expand_cube,
    portal_hash,
    recognize_address,
    resolve,
    traverse,
)

# LLM agent (optional — requires Ollama running locally)
from .agent import WheelerAgent

# Wheeler-primary decoder (small model as pure codec)
from .decoder import DecoderState, WheelerPrimaryAgent, extract_state, format_state

# Corpus crystallization pipeline
from .crystallization import CrystallizationResult, crystallize, load_corpus

# Generative engine (IT from BIT)
from .generation import GenerationResult, TickResult, trajectory_resonance

__all__ = [
    "hash_to_frame",
    "text_to_hex",
    "apply_ca_dynamics",
    "apply_ca_dynamics_parameterized",
    "evolve_and_interpret",
    "evolve_with_params",
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
    "compute_temperature",
    "temperature_tier",
    "HALF_LIFE_DAYS",
    "HIT_SATURATION",
    "MAX_ATTRACTORS",
    "MAX_WARMTH",
    "TIER_DEAD",
    "TIER_FADING",
    "TIER_HOT",
    "TIER_WARM",
    "WARMTH_HALF_LIFE_DAYS",
    "WARMTH_HOP1",
    "WARMTH_HOP2",
    "compute_warmth",
    "effective_temperature",
    "get_neighbors",
    "load_associations",
    # Eviction
    "sweep_and_evict",
    "forget_memory",
    "forget_by_text",
    "score_memories",
    "EvictionResult",
    # Consolidation
    "sleep_consolidate",
    "consolidate_brick",
    "select_keyframes",
    "consolidation_stats",
    "ConsolidationResult",
    # GPU (optional)
    "gpu_available",
    "gpu_evolve_single",
    "gpu_evolve_batch",
    "gpu_version",
    "gpu_query_vram",
    # Embedding (optional)
    "embed_available",
    "embed_to_frame",
    "embed_to_frame_batch",
    # SCM grid
    "SCMGrid",
    # Three-grid interference recall
    "recall_with_interference",
    # Reconstructive recall
    "reconstruct",
    "reconstruct_batch",
    # Two-tier recall API
    "BasinSeed",
    "Pattern",
    "recognize",
    "recognize_top_k",
    "reconstruct_from_seed",
    "set_learning_enabled",
    # FCAS address layer
    "Address",
    "address_of",
    "cross_cube_interference",
    "evolve_cube",
    "expand_cube",
    "portal_hash",
    "recognize_address",
    "resolve",
    "traverse",
    # Attention model
    "AttentionBudget",
    "compute_attention_budget",
    "salience_from_label",
    "salience_from_temperature",
    # LLM agent
    "WheelerAgent",
    # Wheeler-primary decoder
    "WheelerPrimaryAgent",
    "DecoderState",
    "extract_state",
    "format_state",
    # Crystallization
    "CrystallizationResult",
    "crystallize",
    "load_corpus",
    # Generative engine
    "GenerationResult",
    "TickResult",
    "trajectory_resonance",
    # Similarity functions
    "pearson_similarity",
    "spatial_cosine_similarity",
    "hybrid_similarity",
]
