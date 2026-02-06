# Ralph Progress Log

## February 5, 2026 - SCM Architecture Overhaul

### Completed
- **SCM Axioms Foundation**: Created `/docs/SCM_AXIOMS.md` with 8 core axioms of the Symbolic Collapse Model
- **Wheeler Stability Scoring**: Added `recall_with_stability()` to `ralph_core/wheeler.py`
  - hit_count (40%): Sigmoid-normalized activation frequency
  - frame_persistence (30%): Re-encode comparison
  - compression_survival (30%): 10-tick dynamics survival
- **Single-Model Architecture**: Replaced multi-agent hierarchy with `qwen3-coder-next`
  - Created `ralph_simple.py` with Wheeler Memory integration
  - Replaced `ralph_loop.sh` to delegate to new pipeline
  - Updated `wheeler_recall.py` to output stability scores
- **Documentation**: Updated `ARCHITECTURE.md` with:
  - SCM Axioms Foundation section
  - Single-Model Architecture diagram
  - Wheeler Stability Metrics table
- **Model Configuration**:
  - Installed `qwen3-coder-next` (80B MoE, 51GB)
  - Removed all other models except `nomic-embed-text`
  - Updated Ollama to v0.15.5-rc4 (pre-release)

### Current Configuration
| Component | Value |
|-----------|-------|
| Model | qwen3-coder-next (80B params, 3B active) |
| Model Size | 51GB |
| VRAM | 16GB (offloads to RAM) |
| Inference | Slow (~30-60s per iteration) |
| Wheeler | Stability-weighted context budgeting |

### Known Limitations
- Model requires 44.5GB RAM; runs in mixed VRAM+RAM mode
- Inference is slow due to RAM offload on 16GB VRAM system
- Wheeler memory may contain legacy escape codes (filtered in `get_weighted_context`)

<promise>COMPLETE</promise>

---

## February 4, 2026

### Iteration 0 - Task: Update GEMINI.md files with latest status and features.
- Updated root `GEMINI.md` with latest project features and technical stack.
- Updated `ai_tech_stack/GEMINI.md` with Librarian agent and OpenVINO optimization details.
- Updated global memories in `~/.gemini/GEMINI.md` with Epistemological Independence capability.
- Verified consistency across documentation and global memories.

<promise>COMPLETE</promise>

---

## February 2, 2026

### Iteration 0 - Initialization
- Analyzed codebase for critical fixes.
- Identified duplicate exception handler in `agents/common/llm.py`.
- Identified import path error in `runner.py`.
- Identified hardcoded path in `vector_db.py`.
- Identified missing OpenVINO models for NPU/iGPU engines.
- Implemented Epistemological Independence track (Confidence Tracking & Tension Detection).
- Fixed duplicate exception handler.
- Fixed import paths.
- Fixed hardcoded paths (switched to `RALPH_BRAIN_DIR`).
- Created model export toolchain (`export_dynamics.py`, `setup_models.sh`).
- Verified setup with new unit tests.
- Updated `STATUS.md` and `CHANGELOG.md`.

<promise>COMPLETE</promise>
