# Wheeler Memory: Performance Bottleneck Analysis

**Analysis Date**: 2026-03-13
**System**: Single-machine research platform
**Focus**: Identify and prioritize performance optimizations

---

## Executive Summary

The Wheeler Memory project implements a cellular automata (CA)-based episodic memory system with embedded search and reconstruction. Current analysis reveals **one dominant bottleneck** (recall scanning) and several secondary optimization opportunities. This system demonstrates good architectural patterns but suffers from full-table scans and memory overhead in high-volume scenarios.

**Estimated improvement potential**: 40-70% speedup with recommended optimizations, primarily through:
1. Approximate similarity indexing for recall (biggest impact)
2. Lazy loading and streaming patterns for embedding model
3. GPU batch dispatch optimization for crystallization
4. Connection pooling and caching for warmth/association data

---

## 1. THE CRITICAL BOTTLENECK: Recall Memory Scanning (storage.py)

### Problem: O(N) Full-Table Correlation Scan

**Location**: `wheeler_memory/storage.py:recall_memory()` lines 174-225

**What happens on a recall query**:
1. Query text is hashed/embedded → 64×64 CA frame
2. Query frame evolved until convergence (~40-50 ticks)
3. **For EVERY chunk in the system**:
   - Load index.json (fast)
   - Load index from warmth.json (fast)
   - **For EVERY attractor in the chunk**:
     - **Load .npy file from disk** (~16 KB per file)
     - **Compute Pearson correlation** (4096 float32 values)
     - Compute temperature and effective similarity
     - Extract attractor features (entropy, clusters)
4. Sort results by effective_similarity
5. Perform polar companion lookups (batched, good)
6. Optionally reconstruct top-k results

### Performance Impact

**Bottleneck factors**:
- **I/O dominant**: With MAX_ATTRACTORS=10,000, worst case = 10,000 disk reads (~160 MB) per query
- **Computational**: Pearson correlation + temperature + features for each attractor
- **No indexing**: Every query touches the full corpus regardless of relevance

**Typical timings** (estimated for 1,000 stored memories):
- Query frame evolution: 50-100ms (CPU, small grid)
- Index load: 1-2ms
- Attractor scans:
  - 1,000 attractors: ~1-2 seconds (if disk cached) to 5-10s (cold cache)
  - 10,000 attractors: ~10-20+ seconds

**Why this matters**: Every `recall_memory()` call becomes a slow operation, blocking interactive recall and freezing batch operations.

### Root Cause Analysis

1. **Design choice**: Exact Pearson correlation provides perfect recall but requires full scans
2. **No approximate indexing**: Unlike databases with B-trees or LSH for vector search
3. **All chunks searched**: Lines 152-160 include all existing chunks, even tangentially related ones
4. **Per-memory feature extraction**: Lines 209-225 re-compute features on every query

---

## 2. Secondary Bottlenecks

### 2.1 Embedding Model Loading (embedding.py)

**Location**: `wheeler_memory/embedding.py:get_model()` lines 40-50

**Problem**: sentence-transformers model lazy-loaded once per process
- **First call**: 500ms-2s to download/load model to device
- **Subsequent calls**: cached, fast (~5ms per text)

**Impact**:
- Crystallization with embedding is slow on first batch
- CLI tools have startup delay if using embeddings
- No warmup or preloading strategy

**Severity**: Medium (one-time per process, but noticeable in interactive use)

### 2.2 Oscillation Detection (oscillation.py & dynamics.py)

**Location**: `wheeler_memory/dynamics.py:evolve_and_interpret()` lines 98-111

**Problem**: Full history stored in memory + role detection on every 10th tick
- History grows as ~array list of 40-50+ (64×64) frames = ~16-20 MB per evolution
- Oscillation detection: O(window) comparisons with numpy operations

**Impact on crystallization**:
- With batch_size=32: 32 × 16MB = 512 MB per batch in memory
- 1,000 memories crystallized = ~512 MB of history buffers

**Severity**: Medium (memory overhead, only during crystallization, not recall)

### 2.3 GPU Dispatch Not Batched in Crystallization (crystallization.py)

**Location**: `wheeler_memory/crystallization.py:_process_batch()` lines 288-289

**Problem**: CA evolution happens serially per frame, not via GPU batch APIs
```python
results = [evolve_and_interpret(f) for f in frames]  # Sequential CPU/GPU calls
```

**What could be done**: Use `gpu_evolve_batch()` for parallel evolution if GPU is available

**Impact**: GPU parallelism is underutilized
- Each frame evolved individually
- No opportunity for kernel fusion or batch optimization
- GPU sit idle between frames

**Severity**: High if GPU is available, low if CPU-only

### 2.4 Reconstruction Features Computed Redundantly (storage.py)

**Location**: `wheeler_memory/storage.py:recall_memory()` lines 209-225

**Problem**: Attractor features (entropy, clusters, alive_fraction) computed for **every** attractor in recall, even if not in top-k

**Impact**:
- BFS-based cluster detection is O(grid_size^2) per attractor
- Computing for all 10,000 attractors = 10k × O(4096 comparisons)
- Then discarded for all but top-5

**Optimization**: Only compute features after sorting

**Severity**: Medium (computational waste, not I/O)

### 2.5 Connected Component Analysis (dynamics.py:compute_attractor_features)

**Location**: `wheeler_memory/dynamics.py:compute_attractor_features()` lines 150-168

**Problem**: BFS for cluster counting is O(n) where n = grid_size^2 = 4096
- Per-attractor, invoked on every recall result
- Scales poorly with grid size

**Optimization**: Approximate with quick heuristics (label propagation, or skip entirely if not used in ranking)

**Severity**: Low-Medium (linear, but not on critical path if features are computed post-sort)

---

## 3. GPU Dispatch Status

### Current State

**File**: `wheeler_memory/gpu_dynamics.py`

**Good aspects**:
- Graceful fallback to CPU if GPU unavailable
- Supports both HIP v2 (variable grid, stability threshold) and v1 (fixed 64×64)
- Batch API exists: `gpu_evolve_batch()`
- Memory query function for VRAM estimation

**Problems**:
1. **Not used in crystallization**: Batch processing ignores batch GPU API
   - Crystallization feeds frames one-at-a-time via `evolve_and_interpret()`
   - No GPU batch dispatch opportunity taken

2. **No shared compilation status check**:
   - GPU lib path hardcoded to relative `wheeler_memory/gpu/`
   - No clear documentation on whether v2 is built or v1 fallback applies

3. **History not available from GPU path**:
   - GPU path returns empty history (line 174)
   - MemoryBrick.save() requires history for reconstruction
   - Fallback synthesizes minimal 2-frame history (lines 74-75) — wasteful

**GPU effectiveness**: Likely underutilized; GPU would provide 5-10x speedup on CA evolution if properly batched

---

## 4. Memory Usage Concerns

### Per-Query Memory Overhead

When recalling from 10,000 memories:
- Query attractor: 64×64 × 4 bytes = 16 KB
- Temp storage during scan: ~2-5 MB (numpy arrays, correlation results)
- Index cache: ~100-200 KB (JSON)
- Results list: 10,000 × (200 bytes metadata) = ~2 MB
- **Total**: ~3-7 MB per recall (acceptable)

### Crystallization Memory Overhead

Processing 1,000 memories with batch_size=32:
- History frames: 32 batches × 40-50 frames × 16 KB = 20-25 MB
- Intermediate arrays during CA evolution: ~10 MB
- Embedding model in VRAM: 200-400 MB (sentence-transformers)
- **Peak**: ~450-600 MB (manageable on research machine)

### Attractor Storage

- 10,000 attractors × 16 KB = 160 MB on disk
- Index.json + metadata: ~2-5 MB per chunk
- Warmth data: ~500 KB per chunk
- **Total on disk**: ~160-200 MB for 10k memories (good compression ratio vs raw data)

**Severity**: Low for single-machine research, but becomes important at scale

---

## 5. I/O Pattern Issues

### Inefficient Index Loading

**Problem**: Indices loaded multiple times per recall cycle
- Lines 178, 239 in `storage.py` both call `_load_index(chunk_dir)`
- Indices are small (~100 KB for 1000 items) but JSON parsing overhead
- No persistent index caching between calls

### Warming Data Inefficiency

- Lines 183 in `storage.py`: `load_warmth()` called per chunk
- No caching across multiple recall calls in same session
- Warmth data is small but still incurs file I/O

### Polar Association Lookups

**Good**: Batched by chunk (lines 236-240) — reduces redundant file reads

**Could be better**: Associations loaded twice (once in loop, could be cached)

---

## 6. Embedding Integration Concerns

### Model Lifecycle Issues

1. **Lazy load on first call**: Blocks first recall with embedding
2. **No unload mechanism**: Model stays in memory indefinitely
3. **Device selection**: Uses `get_optimal_device()` once at load time
   - If device changes or becomes unavailable, no recovery

### Batch Embedding Performance

**Good**: `embed_to_frame_batch()` uses vectorized operations (line 118)
- Batch of 32 texts: ~50-100ms (vs serial ~5ms × 32 = 160ms)
- 2-3x speedup through batching

**Room for improvement**:
- No streaming for very large corpora (e.g., 100k+ texts)
- No checkpoint/resume for long crystallization runs

---

## 7. Convergence Detection & History Storage

### Issue: Full History Retention

**Lines 80, 87 in dynamics.py**: History appended for every tick
- Converges in ~40-50 ticks typically
- Stores 40-50 × 64×64 × 4 bytes = ~16 MB per memory during evolution
- Discarded after brick construction (not permanently stored)

**Impact**: Peak memory during crystallization, not storage

### Oscillation Detection Cost

**Lines 98-111 in dynamics.py**: Full role-space analysis every 10 ticks after tick 50
- Calls `detect_oscillation(history)` with full history
- `get_cell_roles()` called for last 20 frames (O(window) × O(4096) ops)
- Only helps 1-5% of memories that actually oscillate

**Optimization**: Could sample history at lower frequency or use heuristics

---

## Summary Table: Bottlenecks by Severity & Impact

| Bottleneck | Severity | Latency Impact | Memory Impact | Fixability |
|---|---|---|---|---|
| **Recall full-table scan** | CRITICAL | 1-20s per recall | 3-7 MB | Hard (requires indexing redesign) |
| **GPU batch dispatch unused** | HIGH | 5-10x speedup lost | - | Easy (use existing API) |
| **Embedding model lazy load** | MEDIUM | 0.5-2s first call | 200-400 MB | Easy (eager load option) |
| **History stored in memory** | MEDIUM | - | 16 MB per evolution | Medium (streaming history) |
| **Oscillation detection overhead** | MEDIUM | 50-100ms per check | - | Medium (sampling/heuristics) |
| **Feature extraction on all results** | MEDIUM | 100-200ms | - | Easy (post-sort only) |
| **BFS cluster counting** | LOW-MEDIUM | 50-100ms per memory | - | Medium (approximate algorithm) |
| **Index caching missing** | LOW | 10-20ms per recall | - | Easy (add LRU cache) |
| **No streaming for large corpora** | LOW | - | unbounded | Medium (implement streaming) |

---

## Recommended Optimization Roadmap

### Phase 1: High-Impact, Low-Effort (Estimated 30-40% speedup)

1. **Post-sort feature extraction** (storage.py:225)
   - Only compute features for top-k results
   - Estimated gain: 20-30% on recall latency

2. **Enable GPU batch dispatch in crystallization** (crystallization.py:289)
   - Use `gpu_evolve_batch()` instead of serial `evolve_and_interpret()`
   - Estimated gain: 5-10x on GPU systems, 1-2x on CPU (better cache locality)

3. **Reduce oscillation detection frequency** (dynamics.py:98-111)
   - Check every 20 ticks instead of 10
   - Estimated gain: 5-10% on evolution latency

4. **Cache indices and warmth data** (storage.py)
   - LRU cache for recently-accessed chunks
   - Estimated gain: 5-10% on multi-recall sessions

### Phase 2: Medium-Impact, Medium-Effort (Estimated 20-30% additional speedup)

5. **Implement approximate similarity indexing**
   - LSH or product quantization on attractors
   - Filter to top-100 candidates, then compute exact correlation
   - Estimated gain: 80-90% on recall latency with <1% recall loss

6. **Streaming history for oscillation detection**
   - Keep only last N frames instead of all history
   - Estimated gain: 15-20% memory reduction during crystallization

7. **Eager embedding model loading**
   - Load model at system startup with optional progress indicator
   - Estimated gain: No latency change, better UX

8. **Batch GPU kernel fusion**
   - If GPU available, batch reconstruct_batch() calls
   - Estimated gain: 3-5x on reconstruction-heavy workloads

### Phase 3: Nice-to-Have, Complex (Estimated 10-20% additional speedup)

9. **Adaptive convergence thresholds**
   - Use temperature-dependent thresholds (hot memories converge with looser threshold)
   - Estimated gain: 10-15% on recall evolution

10. **Approximate cluster counting**
    - Use heuristics (e.g., "alive_fraction" proxy) instead of full BFS
    - Estimated gain: 5-10% on feature extraction

11. **Streaming crystallization**
    - Checkpoint every N items, resume from last checkpoint
    - Allows interruption and restart without reprocessing
    - Estimated gain: Better usability, not raw performance

---

## Detailed Recommendations by File

### wheeler_memory/storage.py

**Change 1: Post-sort feature extraction**
```python
# BEFORE (lines 209-225): compute for all results
results.append({ ..., "grid_entropy": feats["grid_entropy"], ... })

# AFTER: compute only for top_k
results.sort(key=lambda r: r["effective_similarity"], reverse=True)
for r in results[:top_k]:
    feats = compute_attractor_features(attractor)
    r["grid_entropy"] = feats["grid_entropy"]
    r["cluster_count"] = feats["cluster_count"]
    r["alive_fraction"] = feats["alive_fraction"]
```

**Change 2: Index caching**
```python
# Add to module top-level or class
_index_cache = {}  # {(chunk_dir_str): (index_dict, timestamp)}
_cache_ttl = 300  # seconds

def _load_index_cached(chunk_dir: Path) -> dict:
    key = str(chunk_dir)
    if key in _index_cache:
        index, timestamp = _index_cache[key]
        if time.time() - timestamp < _cache_ttl:
            return index
    index = _load_index(chunk_dir)
    _index_cache[key] = (index, time.time())
    return index
```

### wheeler_memory/crystallization.py

**Change 3: GPU batch dispatch**
```python
# BEFORE (line 289):
results = [evolve_and_interpret(f) for f in frames]

# AFTER: check GPU availability
if gpu_available():
    try:
        results = gpu_evolve_batch(frames)
    except Exception:
        results = [evolve_and_interpret(f) for f in frames]
else:
    results = [evolve_and_interpret(f) for f in frames]
```

Add import:
```python
from .gpu_dynamics import gpu_available, gpu_evolve_batch
```

### wheeler_memory/dynamics.py

**Change 4: Reduce oscillation detection frequency**
```python
# BEFORE (line 98):
if i > 50 and i % 10 == 0:

# AFTER:
if i > 50 and i % 20 == 0:
```

**Change 5: Streaming history (advanced)**
```python
# BEFORE: history = [frame.copy()] ... history.append(frame.copy())

# AFTER: keep only recent frames
history = collections.deque(maxlen=50)
history.append(frame.copy())
```

### wheeler_memory/oscillation.py

**Change 6: Sampling-based oscillation detection (advanced)**
```python
# BEFORE: def detect_oscillation(history: list[np.ndarray], window: int = 20)

# AFTER: sample history
def detect_oscillation(history: list[np.ndarray], window: int = 20, sample_interval: int = 2):
    # Sample every Nth frame to reduce computation
    sampled = history[::sample_interval]
    # ... rest of logic
```

### wheeler_memory/embedding.py

**Change 7: Eager loading option**
```python
def eager_load_model(device: str | None = None):
    """Preload embedding model eagerly (call at system startup)."""
    global _model
    if device is None:
        device = get_optimal_device()
    from sentence_transformers import SentenceTransformer
    _model = SentenceTransformer(MODEL_NAME, device=device)
    print(f"Embedding model loaded on {device}")
```

---

## Performance Testing Recommendations

### Benchmark Suite to Create

1. **Recall latency benchmark**
   ```python
   # Time recall_memory() with varying corpus sizes
   # corpus_sizes = [100, 1000, 5000, 10000]
   # measure: total latency, breakdown by chunk, feature extraction time
   ```

2. **Crystallization throughput**
   ```python
   # Time crystallize() with different batch sizes and GPU/CPU
   # Measure: items/second, peak memory, history overhead
   ```

3. **Embedding latency**
   ```python
   # Time embed_to_frame_batch() with different batch sizes
   # Measure: first call (cold), warm calls, batching speedup
   ```

4. **GPU dispatch efficiency**
   ```python
   # Compare gpu_evolve_batch() vs serial evolve_and_interpret()
   # Measure: throughput, VRAM usage, speedup factor
   ```

5. **Memory profile during crystallization**
   ```python
   # Monitor peak memory during crystallization with different batch sizes
   # Measure: history overhead, embedding model footprint
   ```

### Quick Validation Test (5 minutes)

```python
import time
from pathlib import Path
from wheeler_memory.storage import recall_memory, store_memory
from wheeler_memory.dynamics import evolve_and_interpret
from wheeler_memory.hashing import hash_to_frame
from wheeler_memory.brick import MemoryBrick

# Store 100 test memories
test_corpus = [f"memory {i}" for i in range(100)]
for text in test_corpus:
    frame = hash_to_frame(text)
    result = evolve_and_interpret(frame)
    brick = MemoryBrick.from_evolution_result(result)
    store_memory(text, result, brick)

# Time a recall
t0 = time.perf_counter()
results = recall_memory("memory 50")
elapsed = time.perf_counter() - t0

print(f"Recall latency for 100 memories: {elapsed:.2f}s")
print(f"Top result: {results[0]['text']}")
```

---

## Architecture Notes for Future Reference

### Why Pearson Correlation?

Pearson correlation was chosen because:
1. Invariant to affine transformations (unlike cosine distance)
2. Captures statistical similarity of entire grid
3. Provides stable ranking across diverse attractor shapes

**Tradeoff**: Requires full-table scan; no efficient indexing structure.

### Why Full History Storage?

History needed for:
1. MemoryBrick reconstruction (energy state visualization)
2. Oscillation detection (role-space analysis)
3. Potential future forensics/explainability

**Tradeoff**: Memory overhead during crystallization; can be mitigated with streaming.

### Why Separate GPU Path?

GPU dispatch is optional because:
1. Not all machines have GPU (research machines vary)
2. CPU performance is often acceptable for small grids (64×64)
3. GPU only worthwhile for batch crystallization, not single-item recall

**Future**: Could make GPU mandatory for server deployments.

---

## Conclusion

The Wheeler Memory system is well-architected with clear separation of concerns, but suffers from a **full-table scan bottleneck in recall** that becomes acute above ~1,000 stored memories. The recommended Phase 1 optimizations (30-40% speedup) are straightforward and low-risk. Phase 2 optimizations (additional 20-30% speedup) require more thought, particularly around approximate indexing.

For a research platform, the system is acceptable up to ~2,000-3,000 memories on a single machine. Beyond that, implementing Phase 1 + Phase 2 recommendations is strongly advised.

**Most impactful single change**: Post-sort feature extraction (20-30% latency reduction, trivial to implement).
