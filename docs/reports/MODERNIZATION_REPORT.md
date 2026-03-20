# Wheeler Memory: Python 3.11+ Modernization Assessment

**Date:** 2025-02-17
**Scope:** Code review for Python 3.11+ modernization opportunities
**Target:** Production-grade research codebase

---

## Executive Summary

Wheeler Memory is well-architected for a research project with solid separation of concerns. The codebase is **Python 3.11+ compatible** but has **significant modernization opportunities** that would improve type safety, readability, and maintainability without increasing complexity.

**Key findings:**
- ✅ Modern patterns already in use (dataclasses, unions, generators, context managers)
- ⚠️ **Type hints are inconsistent**: public APIs lack comprehensive coverage
- ⚠️ **urllib → httpx/aiohttp**: Current HTTP clients could benefit from async
- ⚠️ **Dataclass optimization**: Several classes could use `slots=True` and `frozen=True`
- ✅ **Dependency management**: Well-scoped, no bloat
- ⚠️ **Anti-patterns**: Mutable default factories, inconsistent error handling

---

## 1. Type Hint Coverage

### Current State
Type hints are **partially applied**:
- ✅ Good: Storage functions have parameter type hints
- ✅ Good: Dataclasses fully typed
- ⚠️ Inconsistent: Return types missing from many functions
- ⚠️ Missing: Generic collections not fully typed (e.g., `dict` vs `dict[str, Any]`)

### Specific Issues

#### 1.1 Storage Module (`wheeler_memory/storage.py`)
```python
# Current — missing return type and internal hints
def _load_index(chunk_dir: Path) -> dict:  # Should be dict[str, dict]
    index_path = chunk_dir / "index.json"
    ...

def _bump_recalled_memories(data_dir: Path, results: list[dict]) -> None:
    # results needs structure: list[dict[str, ...]]
    by_chunk: dict[str, list[str]] = {}
    ...
```

**Recommendation:**
```python
def _load_index(chunk_dir: Path) -> dict[str, dict[str, Any]]:
    ...

def _bump_recalled_memories(data_dir: Path, results: list[dict[str, Any]]) -> None:
    by_chunk: dict[str, list[str]] = {}
    ...
```

#### 1.2 Agent Module (`wheeler_memory/agent.py`)
Missing return types and generic hints:

```python
# Current
def _ollama_chat(
    messages: list[dict],
    model: str,
    tools: list[dict],
    base_url: str,
    stream: bool = False,
) -> dict:  # Should specify structure

def _ollama_chat_stream(
    messages: list[dict],
    model: str,
    tools: list[dict],
    base_url: str,
):  # Missing return type entirely
```

**Recommendation:**
```python
from typing import Any, Generator

def _ollama_chat(
    messages: list[dict[str, Any]],
    model: str,
    tools: list[dict[str, Any]],
    base_url: str,
    stream: bool = False,
) -> dict[str, Any]:
    ...

def _ollama_chat_stream(
    messages: list[dict[str, Any]],
    model: str,
    tools: list[dict[str, Any]],
    base_url: str,
) -> Generator[dict[str, Any], None, None]:
    ...
```

#### 1.3 Decoder Module (`wheeler_memory/decoder.py`)
```python
# Current — run_stream missing full type hint
def run_stream(self, user_message: str) -> Iterator[dict]:  # Too generic
```

**Recommendation:**
```python
from typing import TypedDict, Literal

class RecallEvent(TypedDict):
    type: Literal["recall"]
    hits: list[dict[str, Any]]

class StateEvent(TypedDict):
    type: Literal["state"]
    confidence: float
    uncertain: bool

class TokenEvent(TypedDict):
    type: Literal["token"]
    content: str

DecoderEvent = RecallEvent | StateEvent | TokenEvent

def run_stream(self, user_message: str) -> Generator[DecoderEvent, None, None]:
    ...
```

#### 1.4 Temperature Module (`wheeler_memory/temperature.py`)
Generally well-typed, but a few gaps:

```python
# Current
def ensure_access_fields(entry: dict, creation_timestamp: str) -> dict:
    # entry should have structure hint
```

**Recommendation:**
```python
from typing import TypedDict, Required

class AccessMetadata(TypedDict):
    hit_count: int
    last_accessed: str

class IndexEntry(TypedDict, total=False):
    text: str
    metadata: Required[AccessMetadata]
    timestamp: str
    state: str

def ensure_access_fields(entry: IndexEntry, creation_timestamp: str) -> IndexEntry:
    ...
```

### Impact Assessment
- **Effort:** Medium (1-2 hours per file)
- **Benefit:** Enables mypy strict mode, IDE support, and documentation
- **Risk:** None — backward compatible

---

## 2. Modern Python 3.11+ Features

### 2.1 Exception Groups (Available in Python 3.11+)
Currently **not used**. The rotation retry logic in `wheeler_memory/rotation.py` could benefit:

```python
# Current pattern (if used)
# Try multiple strategies, collect errors, raise them all

# Modern 3.11+
from itertools import count

def store_with_rotation_retry(...):
    errors = []
    for attempt in count():
        try:
            # attempt store
            return result
        except SpecificError as e:
            errors.append(e)
            if attempt >= max_retries:
                # Python 3.11+ feature
                raise ExceptionGroup("All rotation strategies failed", errors)
```

**Note:** Only beneficial if current code has unhandled exception groups. Check `rotation.py`.

### 2.2 TaskGroups for Concurrent Operations
Current: No async code path.
**Potential use case** if GPU/embedding batch operations become async:

```python
# Future possibility (not needed now)
import asyncio
from asyncio import TaskGroup

async def gpu_evolve_batch_async(frames, ...):
    async with TaskGroup() as tg:
        tasks = [tg.create_task(gpu_evolve_single_async(f)) for f in frames]
    return [t.result() for t in tasks]
```

**Status:** Low priority for this project.

### 2.3 tomllib for Parsing pyproject.toml
Current: Using setuptools build system (static pyproject.toml).
**Potential use:** If CLI tools need dynamic config from pyproject.toml.

```python
# Python 3.11+ — no external dependency
import tomllib

with open("pyproject.toml", "rb") as f:
    config = tomllib.load(f)
```

**Status:** Not urgent, but clean if implemented.

### 2.4 Match Expressions
Some branching logic could be cleaner with pattern matching (PEP 634, Python 3.10+):

#### Current
```python
# wheeler_memory/dynamics.py
if _GPU_READY and _gpu_evolve is not None:
    try:
        result = _gpu_evolve(...)
    except Exception as e:
        logging.warning("GPU evolution failed, falling back to CPU: %s", e)
# CPU fallback...
```

#### With Match (Python 3.10+)
```python
match (_GPU_READY, _gpu_evolve):
    case (True, gpu_fn) if gpu_fn is not None:
        try:
            result = gpu_fn(...)
        except Exception as e:
            logging.warning("GPU evolution failed, falling back to CPU: %s", e)
            result = _cpu_evolve(...)
    case _:
        result = _cpu_evolve(...)
```

**Status:** Nice-to-have, improves readability on complex branching.

---

## 3. HTTP Clients: urllib → httpx / aiohttp

### Current State
**Both `agent.py` and `decoder.py` use `urllib`:**

```python
# agent.py & decoder.py
import urllib.request
import urllib.error

def _ollama_chat(...):
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())

def _ollama_chat_stream(...):
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw_line in resp:
            ...
```

### Issues
1. **Verbose API**: urllib is low-level and verbose
2. **Stream handling**: Manual line parsing, no built-in chunking
3. **Retry logic**: Not implemented; manual retry needed
4. **Async not possible**: urllib is sync-only, blocks on I/O
5. **Connection pooling**: No built-in support (creates new socket per request)

### Recommendation: httpx

**Why httpx over aiohttp:**
- ✅ Sync + async in one package
- ✅ HTTP/2 support (aiohttp is HTTP/1.1 only)
- ✅ Type-safe, Pydantic integration
- ✅ Built-in retry and timeout management
- ✅ Drop-in for urllib (simpler migration)

**Migration path:**

```python
# pyproject.toml
dependencies = [
    ...
    "httpx>=0.25.0",
    ...
]
```

**Current code:**
```python
def _ollama_chat(messages, model, tools, base_url, stream=False):
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "stream": stream,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as exc:
        raise RuntimeError(...) from exc
```

**With httpx:**
```python
import httpx

def _ollama_chat(
    messages: list[dict[str, Any]],
    model: str,
    tools: list[dict[str, Any]],
    base_url: str,
    stream: bool = False,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "stream": stream,
    }
    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(
                f"{base_url.rstrip('/')}/api/chat",
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {base_url}. "
            "Is it running? Try: ollama serve"
        ) from exc
```

**Benefits:**
- Cleaner API (no manual Request object)
- Automatic JSON serialization/deserialization
- `raise_for_status()` for error handling
- Connection pooling (reuse connections across calls)

**For streaming:**
```python
def _ollama_chat_stream(
    messages: list[dict[str, Any]],
    model: str,
    tools: list[dict[str, Any]],
    base_url: str,
) -> Generator[dict[str, Any], None, None]:
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "stream": True,
    }
    try:
        with httpx.stream("POST",
            f"{base_url.rstrip('/')}/api/chat",
            json=payload,
            timeout=120.0,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line.strip():
                    yield json.loads(line)
    except httpx.HTTPError as exc:
        raise RuntimeError(...) from exc
```

### Impact Assessment
- **Effort:** Low (30 min for both modules)
- **Benefit:** Cleaner code, connection pooling, future async support
- **Risk:** Low (httpx is battle-tested, stable API)
- **Breaking changes:** None (same return signatures)

---

## 4. Dataclass Optimizations

### Current State
Dataclasses are used well but don't leverage optimization flags.

```python
# Current
@dataclass
class MemoryBrick:
    evolution_history: list[np.ndarray]
    final_attractor: np.ndarray
    convergence_ticks: int
    state: str
    metadata: dict = field(default_factory=dict)

@dataclass
class AttentionBudget:
    max_iters: int
    stability_threshold: float
    salience: float

@dataclass
class DecoderState:
    query: str
    attractors: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    co_activated: list[tuple[str, str]] = field(default_factory=list)
    uncertain: bool = True
```

### Optimizations

#### 4.1 `slots=True` for Memory Efficiency
**MemoryBrick** is created once per memory, so slots don't matter much.
**AttentionBudget** is created frequently (per query), **good candidate for slots**:

```python
@dataclass(slots=True, frozen=True)
class AttentionBudget:
    """CA evolution budget derived from salience."""
    max_iters: int
    stability_threshold: float
    salience: float

    @property
    def label(self) -> str:
        """Human-readable label: low / medium / high."""
        ...
```

**Benefits:**
- ~40% less memory per instance
- Prevents dynamic attribute assignment (guards against bugs)
- Immutability (frozen=True) prevents accidental mutations

#### 4.2 `frozen=True` for Immutable Records
**DecoderState** is read by the model, never modified:

```python
@dataclass(frozen=True)
class DecoderState:
    """Structured representation of Wheeler's recall state for the decoder."""
    query: str
    attractors: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    co_activated: list[tuple[str, str]] = field(default_factory=list)
    uncertain: bool = True
```

**Benefits:**
- Hashable (can be used as dict keys)
- Signals immutability to developers
- Runtime protection against accidental mutation

#### 4.3 CrystallizationResult (from crystallization.py)
Check if it's mutable; if not, apply `frozen=True`:

```python
@dataclass(frozen=True)
class CrystallizationResult:
    ...
```

### Impact Assessment
- **Effort:** Minimal (1 line per dataclass)
- **Benefit:** Memory efficiency, immutability guards, hashability
- **Risk:** None (frozen=True is backward compatible for read-only code)
- **Priority:** Medium

---

## 5. Error Handling & Custom Exceptions

### Current State
Error handling is **informal**:

```python
# agent.py
except urllib.error.URLError as exc:
    raise RuntimeError(
        f"Could not reach Ollama at {base_url}. "
        "Is it running? Try: ollama serve"
    ) from exc
```

```python
# storage.py (in recall_memory)
try:
    embed_fn = _get_embed_to_frame()
    query_frame = embed_fn(text)
except Exception:
    return None, []  # Silent failure in _build_recall_context
```

### Issues
1. **Generic RuntimeError**: Callers can't distinguish between different failures
2. **Silent failures**: Embedding import errors swallowed, hard to debug
3. **No retry policy**: HTTP failures have no backoff logic
4. **Inconsistent logging**: Some errors logged, others silent

### Recommendation

#### 5.1 Define Custom Exception Hierarchy
```python
# wheeler_memory/exceptions.py (new file)
"""Wheeler Memory exception hierarchy."""

class WheelerError(Exception):
    """Base exception for all Wheeler Memory errors."""
    pass

class StorageError(WheelerError):
    """Raised when storage operations fail (disk full, permissions, etc.)."""
    pass

class RecallError(WheelerError):
    """Raised when recall fails (no memories found, chunk missing, etc.)."""
    pass

class CAError(WheelerError):
    """Raised when cellular automata evolution fails."""
    pass

class OllamaError(WheelerError):
    """Raised when Ollama communication fails."""

    def __init__(self, base_url: str, original_error: Exception):
        self.base_url = base_url
        self.original_error = original_error
        super().__init__(
            f"Could not reach Ollama at {base_url}. "
            f"Is it running? Try: ollama serve. "
            f"Details: {original_error}"
        )

class EmbeddingError(WheelerError):
    """Raised when embedding backend fails (usually ImportError or inference error)."""
    pass
```

#### 5.2 Use Custom Exceptions in agent.py
```python
from .exceptions import OllamaError

def _ollama_chat(...) -> dict[str, Any]:
    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(...)
            resp.raise_for_status()
            return resp.json()
    except (httpx.HTTPError, httpx.TimeoutException) as exc:
        raise OllamaError(base_url, exc) from exc
```

#### 5.3 Use Custom Exceptions in storage.py
```python
from .exceptions import EmbeddingError, RecallError

def _get_embed_to_frame():
    try:
        from .embedding import embed_to_frame
        return embed_to_frame
    except ImportError as exc:
        raise EmbeddingError(
            "sentence-transformers not installed. "
            "Install with: pip install -e '.[embed]'"
        ) from exc

def recall_memory(...):
    if use_embedding:
        try:
            embed_fn = _get_embed_to_frame()
            query_frame = embed_fn(text)
        except EmbeddingError as exc:
            logging.warning("Embedding failed, falling back to hashing: %s", exc)
            use_embedding = False
            query_frame = hash_to_frame(text)
```

### Impact Assessment
- **Effort:** Low (2-3 hours for full integration)
- **Benefit:** Better error handling, easier debugging, clearer API
- **Risk:** None (backward compatible for catch-all `except WheelerError`)
- **Priority:** Medium

---

## 6. Anti-Patterns & Code Smells

### 6.1 Mutable Default Arguments (Avoided ✅)
Project **correctly uses** `field(default_factory=...)`:
```python
@dataclass
class MemoryBrick:
    metadata: dict = field(default_factory=dict)  # ✅ Correct
```

### 6.2 Dict.get() with Side Effects
No issues found; the project uses safe patterns like:
```python
meta = entry.setdefault("metadata", {})
```

### 6.3 Implicit Type Coercion
Minor issue in `dynamics.py`:
```python
# Current
state=str(data["state"])  # Unnecessary str() if already string
metadata=json.loads(str(data["metadata_json"]))  # Overly defensive
```

**Better:**
```python
state: str = data["state"]
metadata = json.loads(data["metadata_json"])
```

### 6.4 Index Loading/Saving Duplication
Both `storage.py` and `eviction.py` duplicate:
```python
def _load_index(chunk_dir: Path) -> dict:
    index_path = chunk_dir / "index.json"
    if index_path.exists():
        return json.loads(index_path.read_text())
    return {}

def _save_index(chunk_dir: Path, index: dict) -> None:
    index_path = chunk_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2))
```

**Solution:** Move to `chunking.py` and import everywhere.

### Impact Assessment
- **Effort:** Very low (consolidation exercise)
- **Benefit:** DRY principle, easier maintenance
- **Risk:** None
- **Priority:** Low

---

## 7. Dependency Management

### Current Dependencies
```toml
dependencies = [
    "numpy>=2.0",
    "scipy>=1.14",
    "matplotlib>=3.9",
    "psutil>=5.9.0",
]

[project.optional-dependencies]
embed = ["sentence-transformers>=3.0"]
train = ["scikit-learn>=1.0"]
```

### Assessment
✅ **Well-scoped:**
- Core: numpy, scipy, matplotlib (essential for CA & visualization)
- Optional: sentence-transformers (embedding), scikit-learn (training)
- No bloat, no circular dependencies

### Recommendation: Add httpx
```toml
dependencies = [
    "numpy>=2.0",
    "scipy>=1.14",
    "matplotlib>=3.9",
    "psutil>=5.9.0",
    "httpx>=0.25.0",  # For cleaner HTTP client
]
```

**No version conflicts expected** — httpx is a pure HTTP library with minimal deps.

---

## 8. Summary Table

| Category | Current | Recommendation | Effort | Impact | Priority |
|----------|---------|-----------------|--------|--------|----------|
| **Type Hints** | Partial | Full coverage with TypedDict/Literal | 2-3h | High | HIGH |
| **urllib → httpx** | urllib | Switch to httpx | 30min | Medium | HIGH |
| **Dataclass slots** | No slots | Add slots=True to AttentionBudget | 5min | Medium | MEDIUM |
| **Exception hierarchy** | Generic | Custom WheelerError + subclasses | 2-3h | Medium | MEDIUM |
| **Match expressions** | If-else | Pattern matching (3.10+) | 1-2h | Low | LOW |
| **ExceptionGroups** | Not used | Use if rotation logic has groups | 30min | Low | LOW |
| **DRY (indices)** | Duplication | Centralize in chunking.py | 30min | Low | LOW |

---

## 9. Implementation Roadmap

### Phase 1: High-Impact (Week 1)
1. **Add comprehensive type hints** to `storage.py`, `agent.py`, `decoder.py`
   - Use `TypedDict` for complex dicts
   - Mark `run_stream()` with proper `Generator[...]` types
2. **Migrate urllib → httpx** in agent.py and decoder.py
   - Update pyproject.toml
   - Test streaming with httpx.stream()

### Phase 2: Medium-Impact (Week 2)
3. **Create exceptions.py** with custom hierarchy
4. **Apply slots=True** to AttentionBudget, frozen=True to DecoderState
5. **Consolidate index I/O** into chunking.py

### Phase 3: Nice-to-Have (Backlog)
6. Adopt match expressions for complex branching
7. Add ExceptionGroup support if rotation logic needs it
8. Use tomllib for dynamic pyproject.toml parsing

---

## 10. Code Examples

### Example 1: Storage Module Type Hints

**Before:**
```python
def _load_index(chunk_dir: Path) -> dict:
    index_path = chunk_dir / "index.json"
    if index_path.exists():
        return json.loads(index_path.read_text())
    return {}
```

**After:**
```python
from typing import Any

IndexType = dict[str, dict[str, Any]]  # Type alias

def _load_index(chunk_dir: Path) -> IndexType:
    index_path = chunk_dir / "index.json"
    if index_path.exists():
        return json.loads(index_path.read_text())
    return {}
```

### Example 2: HTTP Client Migration

**Before (urllib):**
```python
def _ollama_chat_stream(messages, model, tools, base_url):
    payload = {...}
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            for raw_line in resp:
                if raw_line.strip():
                    yield json.loads(raw_line)
    except urllib.error.URLError as exc:
        raise RuntimeError(...) from exc
```

**After (httpx):**
```python
import httpx

def _ollama_chat_stream(
    messages: list[dict[str, Any]],
    model: str,
    tools: list[dict[str, Any]],
    base_url: str,
) -> Generator[dict[str, Any], None, None]:
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "stream": True,
    }
    try:
        with httpx.stream(
            "POST",
            f"{base_url.rstrip('/')}/api/chat",
            json=payload,
            timeout=120.0,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line.strip():
                    yield json.loads(line)
    except httpx.HTTPError as exc:
        raise OllamaError(base_url, exc) from exc
```

### Example 3: Custom Exceptions

**New file: `wheeler_memory/exceptions.py`**
```python
"""Wheeler Memory exception hierarchy."""

class WheelerError(Exception):
    """Base exception for all Wheeler Memory errors."""
    pass

class OllamaError(WheelerError):
    """Raised when Ollama communication fails."""

    def __init__(self, base_url: str, original_error: Exception) -> None:
        self.base_url = base_url
        self.original_error = original_error
        super().__init__(
            f"Could not reach Ollama at {base_url}. "
            f"Is it running? Try: ollama serve"
        )

class EmbeddingError(WheelerError):
    """Raised when embedding backend is unavailable or fails."""
    pass
```

### Example 4: Dataclass Optimization

**Before:**
```python
@dataclass
class AttentionBudget:
    max_iters: int
    stability_threshold: float
    salience: float

    @property
    def label(self) -> str:
        ...
```

**After:**
```python
@dataclass(slots=True, frozen=True)
class AttentionBudget:
    """CA evolution budget derived from salience."""
    max_iters: int
    stability_threshold: float
    salience: float

    @property
    def label(self) -> str:
        ...
```

---

## 11. Testing Recommendations

After implementing changes:

```bash
# Type checking
mypy wheeler_memory/ --strict

# Test import and basic functionality
python -c "from wheeler_memory import *; print('OK')"

# Test HTTP client changes
pytest tests/test_agent.py -v

# Test dataclass changes (ensure serialization still works)
pytest tests/test_brick.py -v
```

---

## Conclusion

Wheeler Memory is **already modern and well-designed**. The recommended changes are **incremental improvements** that would:
1. Increase type safety and IDE support
2. Improve error handling and debugging
3. Simplify HTTP client code
4. Optimize memory usage for frequently-created objects

All changes are **backward compatible** and can be implemented incrementally without disrupting the research workflow.
