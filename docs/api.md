# Python API Reference

## Quick imports

```python
from wheeler_memory import (
    store_with_rotation_retry,
    recall_memory,
    list_memories,
    AttentionBudget,
    compute_attention_budget,
    salience_from_label,
    salience_from_temperature,
)
from wheeler_memory.reconstruction import reconstruct
from wheeler_memory.embedding import embed_to_frame
from wheeler_memory.temperature import compute_temperature
from wheeler_memory.hardware import get_system_summary
```

---

## `store_with_rotation_retry`

```python
def store_with_rotation_retry(
    text: str,
    max_rotations: int = 4,
    save: bool = True,
    data_dir: str | Path | None = None,
    *,
    chunk: str | None = None,
    use_embedding: bool = False,
    encoder: str | None = None,
    salience: float | None = None,
) -> dict:
```

Encode `text` as a 64×64 CA seed frame, evolve it to an attractor, and store
the result. If the initial evolution does not converge, the seed frame is
rotated by 90°, 180°, and 270° successively until convergence or all rotations
are exhausted.

**Parameters**

| Parameter | Default | Description |
|---|---|---|
| `text` | — | Text to store |
| `max_rotations` | `4` | How many rotation angles to try (1–4) |
| `save` | `True` | Write attractor + brick to disk |
| `data_dir` | `~/.wheeler_memory` | Override the storage root |
| `chunk` | `None` | Force a specific chunk; auto-routes if `None` |
| `use_embedding` | `False` | Use sentence embedding instead of SHA-256 (legacy; prefer `encoder`) |
| `encoder` | `None` | Encoder name: `hash`, `hippocampus`, `embedding`, `blended`, `word`, `word-blended`, `context`. Overrides `use_embedding`. Default: `blended`. |
| `salience` | `None` | Salience score `[0, 1]` or `None` for default (0.5). Controls CA budget. |

**Returns** a `dict` with:

| Key | Type | Description |
|---|---|---|
| `state` | `str` | `CONVERGED`, `OSCILLATING`, `DEGENERATE`, `CHAOTIC`, or `FAILED_ALL_ROTATIONS` |
| `attractor` | `np.ndarray` | Final 64×64 attractor frame |
| `convergence_ticks` | `int` | Ticks until convergence |
| `history` | `list[np.ndarray]` | All frames from seed to attractor |
| `metadata` | `dict` | Includes `rotation_used`, `attempts`, `wall_time_seconds`, `salience`, `attention_label`, `stability_threshold` |

**Example**

```python
result = store_with_rotation_retry("fix the auth bug in login.py")
print(result["state"])          # CONVERGED
print(result["metadata"])       # {'rotation_used': 0, 'attempts': 1, ...}

# Semantic store
result = store_with_rotation_retry(
    "the sky is blue",
    use_embedding=True,
    chunk="science",
)

# High-salience store (deeper attractor)
result = store_with_rotation_retry(
    "critical architectural decision",
    salience=0.9,
)
print(result["metadata"]["attention_label"])  # "high"
```

---

## `recall_memory`

```python
def recall_memory(
    text: str,
    top_k: int = 5,
    data_dir: str | Path | None = None,
    *,
    chunk: str | None = None,
    temperature_boost: float = 0.0,
    use_embedding: bool = False,
    encoder: str | None = None,
    reconstruct: bool = False,
    reconstruct_alpha: float = 0.3,
    salience: float | None = None,
) -> list[dict]:
```

Evolve the query `text` to an attractor, then search stored attractors by
Pearson correlation. Returns the `top_k` best matches, sorted by
`effective_similarity`.

**Parameters**

| Parameter | Default | Description |
|---|---|---|
| `text` | — | Query text |
| `top_k` | `5` | Maximum results to return |
| `data_dir` | `~/.wheeler_memory` | Override the storage root |
| `chunk` | `None` | Search only this chunk; searches all matching chunks if `None` |
| `temperature_boost` | `0.0` | Adds `boost × temperature` to ranking score |
| `use_embedding` | `False` | Use sentence embedding for the query frame (legacy; prefer `encoder`) |
| `encoder` | `None` | Encoder name: `hash`, `hippocampus`, `embedding`, `blended`, `word`, `word-blended`, `context`. Overrides `use_embedding`. |
| `reconstruct` | `False` | Apply Darman reconstruction to each result |
| `reconstruct_alpha` | `0.3` | Blend factor for reconstruction (0 = pure stored, 1 = pure query) |
| `salience` | `None` | Salience score `[0, 1]` or `None` for default. Controls CA budget for query evolution and reconstruction. |

**Returns** a `list[dict]`, each entry containing:

| Key | Type | Description |
|---|---|---|
| `text` | `str` | Original stored text |
| `similarity` | `float` | Pearson correlation with query attractor |
| `temperature` | `float` | Current temperature `[0, 1]` |
| `temperature_tier` | `str` | `hot`, `warm`, or `cold` |
| `effective_similarity` | `float` | `similarity + temperature_boost × temperature` |
| `state` | `str` | Convergence state when the memory was stored |
| `chunk` | `str` | Which chunk this memory lives in |
| `timestamp` | `str` | ISO-8601 timestamp of when it was stored |
| `hex_key` | `str` | SHA-256 hex key (file identifier) |

When `reconstruct=True`, each result also includes:

| Key | Type | Description |
|---|---|---|
| `reconstructed_attractor` | `np.ndarray` | The context-blended attractor |
| `reconstruction_state` | `str` | Convergence state of the reconstruction |
| `reconstruction_ticks` | `int` | Ticks for reconstruction to converge |
| `reconstruction_alpha` | `float` | The alpha used |
| `correlation_with_stored` | `float` | Pearson between reconstruction and stored attractor |
| `correlation_with_query` | `float` | Pearson between reconstruction and query attractor |

Every recalled memory has its `hit_count` incremented and `last_accessed`
updated automatically.

**Example**

```python
# Basic recall
matches = recall_memory("authentication error", top_k=3)
for m in matches:
    print(f"[{m['temperature_tier']}] {m['text']}  sim={m['similarity']:.3f}")

# Fuzzy semantic recall with temperature ranking bonus
matches = recall_memory(
    "what was I debugging yesterday",
    use_embedding=True,
    temperature_boost=0.2,
)

# Reconstructive recall (Darman architecture)
matches = recall_memory(
    "machine learning tools",
    reconstruct=True,
    reconstruct_alpha=0.3,
)
for m in matches:
    print(m["text"])
    print(f"  correlation with stored: {m['correlation_with_stored']:.3f}")
    print(f"  correlation with query:  {m['correlation_with_query']:.3f}")
```

---

## `list_memories`

```python
def list_memories(
    data_dir: str | Path | None = None,
    *,
    chunk: str | None = None,
) -> list[dict]:
```

Return all stored memories from the index without running any CA evolution.
Temperatures are computed lazily at list time.

**Parameters**

| Parameter | Default | Description |
|---|---|---|
| `data_dir` | `~/.wheeler_memory` | Override the storage root |
| `chunk` | `None` | List only this chunk; lists all chunks if `None` |

**Returns** a `list[dict]` with the same fields as `recall_memory` results
(minus similarity / reconstruction fields).

**Example**

```python
memories = list_memories()
for m in sorted(memories, key=lambda x: x["temperature"], reverse=True):
    print(f"[{m['temperature_tier']:4}] {m['temperature']:.2f}  {m['text']}")

# List just the code chunk
code_mems = list_memories(chunk="code")
```

---

## `reconstruct`

```python
# wheeler_memory.reconstruction
def reconstruct(
    stored_attractor: np.ndarray,
    query_attractor: np.ndarray,
    alpha: float = 0.3,
) -> dict:
```

The **Darman architecture**: blend a stored memory with the current query
context and re-evolve through the CA. The same stored memory reconstructs
differently depending on what you're thinking about.

```
blended = (1 − α) × stored + α × query
result  = CA_evolve(blended)
```

`alpha=0` returns the stored memory unchanged (after re-evolution).
`alpha=1` ignores the stored memory entirely.
`alpha=0.3` (default) is memory-dominant but context-aware.

**Returns** a `dict`:

| Key | Type | Description |
|---|---|---|
| `attractor` | `np.ndarray` | Reconstructed 64×64 attractor |
| `state` | `str` | Convergence state of reconstruction |
| `convergence_ticks` | `int` | Ticks to re-converge |
| `alpha` | `float` | The alpha used |
| `correlation_with_stored` | `float` | Pearson(reconstructed, stored) |
| `correlation_with_query` | `float` | Pearson(reconstructed, query) |

**Example**

```python
import numpy as np
from wheeler_memory import recall_memory
from wheeler_memory.reconstruction import reconstruct
from wheeler_memory.storage import DEFAULT_DATA_DIR

# Retrieve stored and query attractors manually
results = recall_memory("Python libraries", top_k=1)
hex_key = results[0]["hex_key"]
chunk   = results[0]["chunk"]

stored_att = np.load(DEFAULT_DATA_DIR / "chunks" / chunk / "attractors" / f"{hex_key}.npy")

from wheeler_memory.hashing import hash_to_frame
from wheeler_memory.dynamics import evolve_and_interpret

query_att = evolve_and_interpret(hash_to_frame("machine learning"))["attractor"]

recon = reconstruct(stored_att, query_att, alpha=0.3)
print(f"state: {recon['state']}")
print(f"correlation with stored: {recon['correlation_with_stored']:.3f}")
print(f"correlation with query:  {recon['correlation_with_query']:.3f}")
```

---

## Encoders

Wheeler Memory provides multiple text-to-frame encoders. All produce a 64×64 float32 frame in [-1, +1].

### `embed_to_frame` (sentence-transformer)

```python
# wheeler_memory.embedding  (requires pip install -e ".[embed]")
def embed_to_frame(text: str, size: int = 64) -> np.ndarray:
```

Convert `text` to a 64×64 CA frame via sentence embedding and random projection.

1. Encode text → 384-dim vector (`all-MiniLM-L6-v2`)
2. Project 384 → 4096 via a fixed Gaussian random matrix (seed `0xDEADBEEF`)
3. Apply `tanh(x × 3)` to map to `(−1, 1)`
4. Reshape to `(64, 64)`

```python
from wheeler_memory.embedding import embed_to_frame, embed_available

if embed_available():
    frame = embed_to_frame("The Eiffel Tower is in Paris")
```

### `hippocampus_to_frame` (native n-gram)

```python
# wheeler_memory.hippocampus
def hippocampus_to_frame(text: str, size: int = 64) -> np.ndarray:
```

Character 3-gram and 4-gram random indexing. No pretrained models required. Lexically similar text produces similar frames. Default native encoder.

### `context_to_frame` (distributional semantics)

```python
# wheeler_memory.word_encoder
def context_to_frame(text: str, size: int = 64) -> np.ndarray:
```

Context-window random indexing encoder. Requires pre-trained vectors (via `train_context_ri()`). Trained on WikiText-103 (1.16M lines, 500K vocab, 384-dim). Blends with hippocampus via `CONTEXT_RI_BLEND` (default 0.9).

### `hash_to_frame` (deterministic)

```python
# wheeler_memory.hashing
def hash_to_frame(text: str, size: int = 64) -> np.ndarray:
```

SHA-256 hash → PCG64 RNG → uniform(-1, 1) grid. Exact match only — changing one character completely changes the frame (avalanche effect).

---

## `compute_temperature`

```python
# wheeler_memory.temperature
def compute_temperature(
    hit_count: int,
    last_accessed: str | datetime,
    now: datetime | None = None,
) -> float:
```

Compute the temperature scalar `[0, 1]` for a memory from its access history.

```
base_from_hits  = min(1.0,  0.3 + 0.7 × (hit_count / 10))
decay_from_time = 2 ^ (−days_since_last_access / 7)
temp            = base_from_hits × decay_from_time
```

`last_accessed` accepts an ISO-8601 string or a `datetime` object (timezone-aware).
`now` defaults to `datetime.now(timezone.utc)`.

**Example**

```python
from wheeler_memory.temperature import compute_temperature, temperature_tier
from datetime import datetime, timezone, timedelta

yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()

temp = compute_temperature(hit_count=5, last_accessed=yesterday)
print(f"temp={temp:.4f}  tier={temperature_tier(temp)}")
# temp=0.5743  tier=warm
```

---

## `get_system_summary`

```python
# wheeler_memory.hardware
def get_system_summary() -> dict:
```

Aggregate hardware information useful for debugging environment issues and
understanding which accelerator will be used.

**Returns** a `dict` with keys:

| Key | Type | Description |
|---|---|---|
| `os` | `str` | OS name (`Linux`, `Windows`, …) |
| `release` | `str` | Kernel/OS release string |
| `cpu` | `dict` | `architecture`, `processor`, `physical_cores`, `total_cores`, `frequency_mhz` |
| `memory` | `dict` | `total_gb`, `available_gb`, `used_gb`, `percent_used` |
| `storage` | `dict` | `total_gb`, `used_gb`, `free_gb`, `percent_used` for `/` |
| `gpu_npu` | `dict` | `nvidia_gpu` list or string, `pci_devices` list from `lspci` |
| `optimal_device` | `str` | `"cuda"`, `"mps"`, or `"cpu"` |
| `warnings` | `list[str]` | Mismatch warnings (e.g. GPU found but PyTorch using CPU) |

**Example**

```python
from wheeler_memory.hardware import get_system_summary

info = get_system_summary()
print(info["optimal_device"])       # cpu / cuda / mps
for w in info["warnings"]:
    print("WARNING:", w)
```

---

## Store → Recall → Reconstruct workflow

```python
from wheeler_memory import store_with_rotation_retry, recall_memory

# 1. Store memories
store_with_rotation_retry("Python has great libraries for data science")
store_with_rotation_retry("scikit-learn is perfect for classical ML")
store_with_rotation_retry("PyTorch is used for deep learning")

# 2. Recall with semantic embedding
results = recall_memory(
    "machine learning tools",
    top_k=3,
    use_embedding=True,        # fuzzy semantic match
    reconstruct=True,          # Darman reconstruction
    reconstruct_alpha=0.3,     # 70% stored, 30% query context
    temperature_boost=0.1,     # favour recently accessed memories
)

# 3. Inspect results
for r in results:
    print(f"[{r['temperature_tier']:4}] sim={r['similarity']:.3f}  {r['text']}")
    if "correlation_with_stored" in r:
        print(f"         stored_corr={r['correlation_with_stored']:.3f}"
              f"  query_corr={r['correlation_with_query']:.3f}")
```

---

## Eviction / Forgetting

```python
from wheeler_memory import (
    sweep_and_evict,
    forget_memory,
    forget_by_text,
    score_memories,
    EvictionResult,
    TIER_FADING,
    TIER_DEAD,
    MAX_ATTRACTORS,
)
```

### `sweep_and_evict`

```python
def sweep_and_evict(
    data_dir: str | Path,
    dry_run: bool = False,
) -> EvictionResult:
```

Run all three eviction phases and return a report.

1. **Fade** — delete `.npz` bricks for memories below `TIER_FADING` (0.05)
2. **Evict** — fully remove memories below `TIER_DEAD` (0.01)
3. **Capacity** — if over `MAX_ATTRACTORS` (10,000), evict bottom 10% cold memories

Memories younger than 1 day are never affected.

**Returns** an `EvictionResult`:

| Field | Type | Description |
|---|---|---|
| `bricks_deleted` | `list[dict]` | Memories whose bricks were faded |
| `memories_evicted` | `list[dict]` | Memories fully removed |
| `total_before` | `int` | Memory count before sweep |
| `total_after` | `int` | Memory count after sweep |

**Example**

```python
result = sweep_and_evict("~/.wheeler_memory")
print(f"Faded {len(result.bricks_deleted)} bricks")
print(f"Evicted {len(result.memories_evicted)} memories")
print(f"Total: {result.total_before} → {result.total_after}")

# Dry run — inspect without deleting
result = sweep_and_evict("~/.wheeler_memory", dry_run=True)
```

---

### `forget_memory` / `forget_by_text`

```python
def forget_memory(hex_key: str, data_dir: str | Path) -> bool:
def forget_by_text(text: str, data_dir: str | Path) -> bool:
```

Immediately delete a specific memory. Returns `True` if found.

```python
forget_by_text("fix the python debug error", "~/.wheeler_memory")
forget_memory("a1b2c3d4...", "~/.wheeler_memory")
```

---

### `score_memories`

```python
def score_memories(data_dir: str | Path) -> list[dict]:
```

Score all memories by effective temperature, sorted coldest-first.
Each entry contains `hex_key`, `chunk`, `text`, `temperature`, `age_days`, `hit_count`.

---

## Sleep Consolidation

```python
from wheeler_memory import (
    sleep_consolidate,
    consolidate_brick,
    select_keyframes,
    consolidation_stats,
    ConsolidationResult,
)
```

### `sleep_consolidate`

```python
def sleep_consolidate(
    data_dir: str | Path,
    dry_run: bool = False,
    chunk: str | None = None,
) -> ConsolidationResult:
```

Sweep all chunks (or a specific one) and prune redundant frames within each
brick. Hot bricks are skipped, warm bricks get light pruning, cold bricks
get aggressive pruning. Already-consolidated bricks are skipped (idempotent).

**Returns** a `ConsolidationResult`:

| Field | Type | Description |
|---|---|---|
| `memories_consolidated` | `list[dict]` | Memories that were pruned |
| `memories_skipped` | `list[dict]` | Memories skipped (with reason) |
| `total_frames_before` | `int` | Total frames before consolidation |
| `total_frames_after` | `int` | Total frames after consolidation |

Each consolidated entry contains `hex_key`, `chunk`, `text`, `frames_before`,
`frames_after`, `tier`.

**Example**

```python
# Dry run to preview
result = sleep_consolidate("~/.wheeler_memory", dry_run=True)
for m in result.memories_consolidated:
    print(f"{m['tier']} {m['frames_before']} -> {m['frames_after']}  {m['text']}")

# Actual consolidation
result = sleep_consolidate("~/.wheeler_memory")
print(f"Frames: {result.total_frames_before} -> {result.total_frames_after}")
```

---

### `consolidate_brick`

```python
def consolidate_brick(
    brick: MemoryBrick,
    delta_threshold: float,
    role_threshold: float,
) -> MemoryBrick:
```

Return a new `MemoryBrick` with pruned history. Already-consolidated bricks
and bricks with fewer than 5 frames are returned unchanged.

Adds metadata: `consolidated`, `consolidated_at`, `original_frame_count`,
`retained_frame_count`, `frames_pruned`, and the thresholds used.

---

### `select_keyframes`

```python
def select_keyframes(
    history: list[np.ndarray],
    delta_threshold: float,
    role_threshold: float,
) -> list[int]:
```

Pure computation: return sorted list of frame indices to keep. A frame is
kept if the mean absolute delta or the role-change fraction (vs. last kept
frame) exceeds the threshold. Frame 0 and the final frame are always kept.

---

### `consolidation_stats`

```python
def consolidation_stats(
    data_dir: str | Path,
    chunk: str | None = None,
) -> list[dict]:
```

Read-only: per-memory frame counts and potential savings. Each entry contains
`hex_key`, `chunk`, `text`, `frame_count`, `temperature`, `tier`,
`consolidated`, `potential_frames`.

---

## Attention Model (Variable Tick Rates)

```python
from wheeler_memory import (
    AttentionBudget,
    compute_attention_budget,
    salience_from_label,
    salience_from_temperature,
)
```

### `compute_attention_budget`

```python
def compute_attention_budget(salience: float) -> AttentionBudget:
```

Map a salience score `[0, 1]` to an `AttentionBudget` controlling CA evolution
depth. Salience is clamped to `[0, 1]`.

Interpolation is piecewise linear for `max_iters` and log-linear for
`stability_threshold` (which spans orders of magnitude).

**Anchor points**

| Salience | max_iters | threshold | Label |
|----------|-----------|-----------|-------|
| 0.0 | 200 | 5e-4 | low |
| 0.5 | 1000 | 1e-4 | medium |
| 1.0 | 3000 | 1e-6 | high |

`salience=0.5` produces **exactly** the pre-attention-model defaults, so
omitting salience changes nothing.

**Returns** an `AttentionBudget`:

| Field | Type | Description |
|---|---|---|
| `max_iters` | `int` | Maximum CA iterations |
| `stability_threshold` | `float` | Convergence threshold |
| `salience` | `float` | The (clamped) salience used |
| `label` | `str` | `"low"`, `"medium"`, or `"high"` (property) |

**Example**

```python
budget = compute_attention_budget(0.9)
print(budget.max_iters)            # 2600
print(budget.stability_threshold)  # ~3.16e-6
print(budget.label)                # "high"
```

---

### `salience_from_label`

```python
def salience_from_label(label: str) -> float:
```

Convert a human label to a numeric salience: `"low"` → 0.2, `"medium"` → 0.5,
`"high"` → 0.9. Unknown labels return 0.5 (default).

---

### `salience_from_temperature`

```python
def salience_from_temperature(temperature: float) -> float:
```

Derive salience from a memory's temperature. Used during reconstruction so
that hot memories automatically get more computational attention.

```
salience = 0.1 + 0.9 × temperature
```

`temp=0` → salience 0.1, `temp=1` → salience 1.0.

---

## Reconstructive Recall (Batch)

```python
from wheeler_memory import reconstruct_batch

results = reconstruct_batch(
    stored_attractors=[att1, att2, att3],
    query_attractor=query_att,
    alpha=0.3,
)
```

`reconstruct_batch` applies `reconstruct()` to multiple stored attractors in one call. Returns a list of result dicts (same format as `reconstruct()`).

---

## Embedding (Batch)

```python
from wheeler_memory import embed_to_frame_batch, embed_available

if embed_available():
    frames = embed_to_frame_batch(["text one", "text two", "text three"])
    # Returns list of (64, 64) numpy arrays
```

`embed_to_frame_batch` encodes multiple texts in a single pass through the sentence transformer for efficiency. `embed_available()` returns `True` if sentence-transformers is installed.

---

## Crystallization

```python
from wheeler_memory import crystallize, load_corpus, CrystallizationResult
```

### `crystallize`

```python
def crystallize(
    corpus_path: Path,
    data_dir: Path | None = None,
    batch_size: int = 32,
    chunk: str | None = None,
    use_embedding: bool = True,
    max_items: int | None = None,
    resume: bool = True,
    fmt: str = "auto",
    verbose: bool = False,
) -> CrystallizationResult:
```

Feed a text corpus through Wheeler's encode-evolve-store pipeline at scale. Forms an attractor landscape before deployment so the system boots with crystallized knowledge.

**Parameters**

| Parameter | Default | Description |
|---|---|---|
| `corpus_path` | — | Path to corpus file (JSONL, CSV, TXT, or Parquet) |
| `data_dir` | `~/.wheeler_memory` | Override the storage root |
| `batch_size` | `32` | Texts per batch for embedding efficiency |
| `chunk` | `None` | Force all texts into this chunk; auto-routes if `None` |
| `use_embedding` | `True` | Use sentence embedding; if `False`, uses SHA-256 hashing |
| `max_items` | `None` | Stop after N texts (for validation runs) |
| `resume` | `True` | Skip texts whose hex key is already stored |
| `fmt` | `"auto"` | Format: `"jsonl"`, `"csv"`, `"txt"`, `"parquet"`, or `"auto"` |
| `verbose` | `False` | Print progress to stderr |

**Returns** a `CrystallizationResult`:

| Field | Type | Description |
|---|---|---|
| `stored` | `int` | Memories successfully stored |
| `skipped` | `int` | Already-stored entries skipped (resume) |
| `errors` | `int` | Entries that failed to store |
| `saturation_pct` | `float` | Percentage of MAX_ATTRACTORS capacity used |
| `elapsed_seconds` | `float` | Wall-clock time |
| `chunks_used` | `dict[str, int]` | Per-chunk store counts |

**Example**

```python
from pathlib import Path
from wheeler_memory import crystallize

result = crystallize(Path("corpus.jsonl"), max_items=10_000, verbose=True)
print(result)
# Crystallization complete:
#   stored:     2711
#   skipped:    0
#   errors:     0
#   saturation: 27.1%
#   elapsed:    45.3s
```

---

### `load_corpus`

```python
def load_corpus(path: Path, fmt: str = "auto") -> Iterator[str]:
```

Stream texts from a corpus file. Supports JSONL, CSV, TXT, and Parquet formats. Auto-detects format from file extension.

**Example**

```python
from pathlib import Path
from wheeler_memory import load_corpus

for text in load_corpus(Path("corpus.jsonl")):
    print(text[:80])
```

---

## Wheeler-Primary Decoder

```python
from wheeler_memory import WheelerPrimaryAgent, DecoderState, extract_state, format_state
```

### `WheelerPrimaryAgent`

```python
class WheelerPrimaryAgent:
    def __init__(
        self,
        model: str = "qwen2.5:1.5b",
        ollama_url: str = "http://localhost:11434",
        data_dir: str | Path | None = None,
        recall_k: int = 5,
        confidence_floor: float = 0.18,
        reconstruct: bool = True,
        reconstruct_alpha: float = 0.3,
        verbose: bool = False,
    ) -> None: ...

    def run(self, user_message: str) -> str: ...
    def run_stream(self, user_message: str) -> Iterator[dict]: ...
```

Wheeler-primary agent where Wheeler Memory is the cognitive system and the small model is a pure language renderer. Unlike `WheelerAgent`, this agent does not reason, plan, or use tools — it reads Wheeler's attractor state and renders it as natural language.

**`run()`** executes the full pipeline: recall from Wheeler, extract structured state, format prompt, decode via small model.

**`run_stream()`** yields typed events: `{"type": "recall"}`, `{"type": "state"}`, `{"type": "token"}`, `{"type": "done"}`.

**Example**

```python
from wheeler_memory import WheelerPrimaryAgent

agent = WheelerPrimaryAgent(verbose=True)
reply = agent.run("What is quantum entanglement?")
print(reply)
```

---

### `extract_state`

```python
def extract_state(
    query: str,
    recall_results: list[dict],
    confidence_floor: float = 0.18,
) -> DecoderState:
```

Extract structured state from `recall_memory()` results. Computes confidence from max similarity, detects co-activation from shared chunks, and marks the state as uncertain if confidence falls below the floor.

---

### `format_state`

```python
def format_state(state: DecoderState) -> str:
```

Serialize a `DecoderState` into structured text for the small model. The output includes query, confidence label, ranked active memories with CA metadata, co-activation pairs, and uncertainty instructions.

---

### `DecoderState`

```python
@dataclass
class DecoderState:
    query: str
    attractors: list[dict]
    confidence: float = 0.0
    co_activated: list[tuple[str, str]]
    uncertain: bool = True
```

Structured representation of Wheeler's recall state for the decoder.

---

## Three-Grid Interference

```python
from wheeler_memory.interference import recall_with_interference
```

### `recall_with_interference`

```python
def recall_with_interference(
    text: str,
    top_k: int = 5,
    data_dir: str | Path | None = None,
    *,
    encoder: str | None = None,
) -> list[dict]:
```

Default recall path since v0.3.1. Scores candidates using three-grid interference: corpus Pearson similarity × experiential similarity × SCM openness gating. Degrades gracefully to pure Pearson when no experiential data exists.

Each result includes `interference_state` (`GROUNDED`, `ABSORBED`, `UNCONSOLIDATED`, or `CONTESTED`).

---

## Cortex

```python
from wheeler_memory.cortex import cortex_reason, build_graph
from wheeler_memory.cortex_scm import compute_scm
from wheeler_memory.cortex_classifier import classify, init_weights, load_weights
```

### `cortex_reason`

Builds L1 graph (Pearson adjacency matrix, clusters, bridges, contradictions) and runs L2 settlement CA (opinion diffusion). Returns graph reasoning + settled opinions.

### `compute_scm`

Computes 7-layer SCM score: Temperature, Salience, Energy, Integration, Polarity, Net Warrant, Explanation Readiness. Returns `SCMResult` with per-layer scores and unified classification.

### `classify`

L3 classifier: takes settlement opinions (K=10), choice similarities (4), and SCM layers (7) → 4-class softmax probabilities. Returns `(predicted_index, confidence)`.

```python
weights = load_weights("~/.wheeler_memory/cortex_classifier.npz")
predicted, confidence = classify(settlement, choice_sims, scm_layers, weights)
```
