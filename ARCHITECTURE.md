# Ralph AI - Compound Architecture

This document defines the target architecture for Ralph AI - a secure, hierarchical autonomous agent system.

## The Ralph Compound

```
╔═════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                    THE RALPH COMPOUND                                            ║
║                                                                                                  ║
║    ┌───────────────────────────┐                                                                 ║
║    │        GATE IN            │                                                                 ║
║    │    (External Inputs)      │                                                                 ║
║    │                           │                                                                 ║
║    │  ┌────────┐ ┌──────────┐  │                                                                 ║
║    │  │  CLI   │ │  Chrome  │  │                                                                 ║
║    │  └────────┘ └──────────┘  │                                                                 ║
║    │  ┌────────┐ ┌──────────┐  │                                                                 ║
║    │  │ Voice  │ │   API    │  │                                                                 ║
║    │  └────────┘ └──────────┘  │                                                                 ║
║    │  ┌────────┐ ┌──────────┐  │                                                                 ║
║    │  │Keyboard│ │  Mouse   │  │                                                                 ║
║    │  └────────┘ └──────────┘  │                                                                 ║
║    │  ┌────────┐ ┌──────────┐  │                                                                 ║
║    │  │ Camera │ │   Mic    │  │                                                                 ║
║    │  └────────┘ └──────────┘  │                                                                 ║
═════╪═══════════════════════════╪═════════════════════════════════════════════════════════════════║
║    │                           │                         PERIMETER WALL                          ║
║    │                           ▼                                                                 ║
║    │                                                                                             ║
║    │    ┌──────────────────────────────┐          ┌─────────────────────────┐                    ║
║    │    │         WAREHOUSE            │          │      R&D BUILDING       │                    ║
║    │    │       (Memory/VectorDB)      │◄════════►│    (Learning Lab)       │                    ║
║    │    │                              │          │                         │                    ║
║    └───►│  ┌───────────┐ ┌───────────┐ │          │  ┌───────────────────┐  │                    ║
║         │  │ Long-term │ │  Context  │ │          │  │    Reflector      │  │                    ║
║         │  │  Storage  │ │   Cache   │ │          │  │  (Learn from      │  │                    ║
║         │  └───────────┘ └───────────┘ │          │  │   past runs)      │  │                    ║
║         │                              │          │  └───────────────────┘  │                    ║
║         │  ┌────────────────────────┐  │          │  ┌───────────────────┐  │                    ║
║         │  │    Warehouse Workers   │  │          │  │     Dreamer       │  │                    ║
║         │  │   (Memory Handlers)    │  │          │  │  (Background      │  │                    ║
║         │  └───────────┬────────────┘  │          │  │   experiments)    │  │                    ║
║         └──────────────┼───────────────┘          │  └───────────────────┘  │                    ║
║                        │                          │  ┌───────────────────┐  │                    ║
║                   ═════╪═════                     │  │    Estimator      │  │                    ║
║                   FORKLIFT                        │  │  (Value/Priority) │  │                    ║
║                   TRANSFER                        │  └───────────────────┘  │                    ║
║                        │                          └────────────┬────────────┘                    ║
║                        ▼                                       │                                 ║
║    ┌──────────────────────────────┐    ┌───────────────────────┴───────────────────────────┐     ║
║    │     PROCESSING PLANT         │    │              MAIN HQ SKYSCRAPER                   │     ║
║    │         (ASICs)              │    │             (Thinking Agents)                     │     ║
║    │                              │    │                                                   │     ║
║    │  ┌──────┐ ┌──────┐ ┌──────┐  │    │  ══════════════════════════════════════════════   │     ║
║    │  │regex │ │ json │ │ sql  │  │    │  ║║║║║║  PNEUMATIC AIR TUBE SYSTEM  ║║║║║║║║║║   │     ║
║    │  └──────┘ └──────┘ └──────┘  │    │  ══════════════════════════════════════════════   │     ║
║    │  ┌──────┐ ┌──────┐ ┌──────┐  │    │           │         │         │         │        │     ║
║    │  │ test │ │ fix  │ │ doc  │  │    │  ┌────────┴─────────┴─────────┴─────────┴─────┐  │     ║
║    │  └──────┘ └──────┘ └──────┘  │    │  │ FLOOR 5: Translator (Human → TaskSpec)    │  │     ║
║    │  ┌──────────┐ ┌──────────┐   │    │  └────────────────────────┬──────────────────┘  │     ║
║    │  │tiny_code │ │ sm_code  │   │    │                      ◄────┼────►                │     ║
║    │  └──────────┘ └──────────┘   │    │  ┌────────────────────────┴──────────────────┐  │     ║
║    │                              │    │  │ FLOOR 4: Orchestrator (Strategy/Planning) │  │     ║
║    │  ══════════════════════════  │    │  └────────────────────────┬──────────────────┘  │     ║
║    │        ASIC BUS              │    │                      ◄────┼────►                │     ║
║    │  ══════════════════════════  │    │  ┌────────────────────────┴──────────────────┐  │     ║
║    │              │               │    │  │ FLOOR 3: Middle Management                │  │     ║
║    │              │               │    │  │       Engineer ◄─────────► Designer       │  │     ║
║    │              │               │    │  └────────────────────────┬──────────────────┘  │     ║
║    │              │               │    │                      ◄────┼────►                │     ║
║    │              │               │◄═══╪═══════════════════════════╪════════════════╗    │     ║
║    │              │               │    │        GROUND LEVEL CONNECTION             ║    │     ║
║    │              │               │═══►╪═══════════════════════════╪════════════════╝    │     ║
║    │              │               │    │                      ◄────┼────►                │     ║
║    │              │               │    │  ┌────────────────────────┴──────────────────┐  │     ║
║    │              │               │    │  │ FLOOR 1: Executor Interface              │  │     ║
║    │              │               │    │  │      (Sandbox/Shell dispatch)            │  │     ║
║    │              │               │    │  └────────────────────────┬──────────────────┘  │     ║
║    │              │               │    │                      ◄────┼────►                │     ║
║    │              │               │    │  ┌────────────────────────┴──────────────────┐  │     ║
║    │              │               │    │  │ BASEMENT: Verifier (Results → back UP)   │  │     ║
║    │              │               │    │  └───────────────────────────────────────────┘  │     ║
║    │              │               │    └─────────────────────────────────────────────────┘     ║
║    │              │               │                                                            ║
║    └──────────────┼───────────────┘                                                            ║
║                   │                                                                            ║
║                   ▼                                                                            ║
║    ┌──────────────────────────────────────────────────────────────────────────────────────┐    ║
║    │                           SECURITY CHECKPOINT                                        │    ║
║    │  ┌─────────┐  ┌─────────┐  ┌─────────────────────────┐  ┌─────────┐  ┌─────────┐     │    ║
║    │  │ TOWER   │  │  DOGS   │  │   SECURITY GUARDS       │  │  DOGS   │  │ TOWER   │     │    ║
║    │  │ (Audit) │  │(Sniffers│  │  (Validation Layer)     │  │(Sniffers│  │ (Audit) │     │    ║
║    │  │         │  │ Malware)│  │  - Type checking        │  │ Secrets)│  │         │     │    ║
║    │  └─────────┘  └─────────┘  │  - Permission verify    │  └─────────┘  └─────────┘     │    ║
║    │                            │  - Content sanitize     │                               │    ║
║    │                            │  - Rate limiting        │                               │    ║
║    │                            └─────────────────────────┘                               │    ║
║    └──────────────────────────────────────────────────────────────────────────────────────┘    ║
║                                            │                                                   ║
═════════════════════════════════════════════╪═══════════════════════════════════════════════════║
║                                            │                      PERIMETER WALL              ║
║    ┌───────────────────────────────────────┴─────────────────────────────────────────────┐     ║
║    │                              GATE OUT                                               │     ║
║    │                         (External Outputs)                                          │     ║
║    │                                                                                     │     ║
║    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐               │     ║
║    │  │ Terminal │  │  Files   │  │ Browser  │  │   API    │  │  Voice   │               │     ║
║    │  │  Output  │  │  System  │  │  Actions │  │ Response │  │  Output  │               │     ║
║    │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘               │     ║
║    └─────────────────────────────────────────────────────────────────────────────────────┘     ║
║                                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════════════════════╝
```

---

## Component Mapping

### GATE IN (External Inputs)
The single entry point into the compound. All external inputs must pass through here.

| Input Type | Description | Implementation |
|------------|-------------|----------------|
| CLI | Command line interface | `ralph_loop.sh`, `runner.py` |
| Chrome | Browser automation | MCP tools, web.py |
| Voice | Speech input | `ralph_voice.py`, voice.py |
| API | HTTP/REST endpoints | `ralph_ui/backend/main.py` |
| Keyboard | Direct text input | stdin |
| Mouse | GUI interactions | Future |
| Camera | Visual input | vision.py |
| Mic | Audio input | voice.py |

---

### WAREHOUSE (Memory/VectorDB)
Persistent storage for all data. Nothing gets processed without first being logged here.

| Component | Description | Implementation |
|-----------|-------------|----------------|
| Long-term Storage | Persistent facts, lessons | `memory/`, `~/.ralph/global_memory/` |
| Context Cache | Short-term working memory | `context.json` |
| Warehouse Workers | Memory handlers | `memory.py`, `vector_db.py` |

**Connections:**
- Receives from: Gate IN
- Sends to: Processing Plant (via Forklift), R&D Building

---

### R&D BUILDING (Learning Lab)
Research and development - learns from past runs, experiments with improvements.

| Component | Description | Implementation |
|-----------|-------------|----------------|
| Reflector | Analyzes past runs, extracts lessons | `agents/reflector/` |
| Dreamer | Background experiments, tries new approaches | `dreamer.py` |
| Estimator | Prioritizes tasks by value/complexity | `agents/estimator/` |

**Connections:**
- Bidirectional with: Warehouse (reads history, stores insights)
- Sends to: Main HQ (improvements, lessons)

---

### PROCESSING PLANT (ASICs)
The factory floor - fast, specialized micro-workers that do specific tasks.

| ASIC | Task Type | Model |
|------|-----------|-------|
| regex | Regular expressions | tinyllama:1.1b |
| json | JSON parsing/generation | deepseek-coder:1.3b |
| sql | SQL queries | deepseek-coder:1.3b |
| test | Unit test generation | qwen2.5-coder:1.5b |
| fix | Bug fixes | qwen2.5-coder:1.5b |
| doc | Docstrings | tinyllama:1.1b |
| tiny_code | Small code snippets | deepseek-coder:1.3b |
| sm_code | Medium code tasks | qwen2.5-coder:1.5b |

**ASIC BUS**: Internal communication between ASICs and the Skyscraper.

**Connections:**
- Receives from: Warehouse (via Forklift)
- Bidirectional with: Main HQ (Ground Level Connection)
- Sends to: Security Checkpoint

---

### MAIN HQ SKYSCRAPER (Thinking Agents)
The brain of the operation - hierarchical thinking from human interface down to execution.

| Floor | Agent | Purpose | Model |
|-------|-------|---------|-------|
| 5 | Translator | Human input → TaskSpec | phi3:mini |
| 4 | Orchestrator | Strategy, planning, decomposition | deepseek-r1:14b |
| 3 | Engineer | Code generation | qwen2.5-coder:14b |
| 3 | Designer | Code review, verification | mistral-nemo:12b |
| 1 | Executor | Shell/sandbox dispatch | N/A (system) |
| B | Verifier | Results validation → back UP | Uses Orchestrator |

**PNEUMATIC AIR TUBE SYSTEM**: The Message Bus that enables fast communication between floors.
- Implementation: `protocols/bus.py`, `protocols/messages.py`
- Message Types: WORK_REQUEST, CODE_OUTPUT, REVISION_REQUEST, ASIC_REQUEST, ASIC_RESPONSE, COMPLETE, ERROR

**GROUND LEVEL CONNECTION**: Bidirectional link between Processing Plant and HQ.
- ASICs receive micro-tasks from Engineer (Floor 3)
- ASICs return results back to the Skyscraper

---

### SECURITY CHECKPOINT
Heavy security before anything leaves the compound.

| Component | Metaphor | Implementation |
|-----------|----------|----------------|
| Guard Towers | Audit/logging | `metrics.py`, audit logs |
| Dogs (Malware) | Malware detection | Code scanning, sandboxing |
| Dogs (Secrets) | Secret detection | Pattern matching for keys/passwords |
| Security Guards | Validation layer | Type checking, permissions, sanitization, rate limiting |

**Security Functions:**
- Type checking: Validate output format
- Permission verify: Ensure action is authorized
- Content sanitize: Remove dangerous content
- Rate limiting: Prevent abuse

---

### GATE OUT (External Outputs)
The single exit point from the compound. All outputs must pass security first.

| Output Type | Description | Implementation |
|-------------|-------------|----------------|
| Terminal Output | Console/stdout | print(), logging |
| File System | Write files | tools.py write_file |
| Browser Actions | Web automation | MCP tools |
| API Response | HTTP responses | FastAPI endpoints |
| Voice Output | Speech synthesis | Future TTS |

---

## Data Flow

```
GATE IN ──► WAREHOUSE ──► FORKLIFT ──► PROCESSING PLANT ◄══► HQ SKYSCRAPER
                │                              │                    │
                │                              │                    │
                └──────► R&D BUILDING ◄────────┴────────────────────┘
                              │
                              ▼
                    (Improvements fed back)
                              │
                              ▼
                     SECURITY CHECKPOINT
                              │
                              ▼
                          GATE OUT
```

**Key Principles:**
1. Everything enters through Gate IN
2. Everything is stored in Warehouse first
3. Processing Plant (ASICs) does fast micro-work
4. HQ Skyscraper does strategic thinking
5. R&D learns and improves the system
6. Security validates before exit
7. Everything exits through Gate OUT

---

## Implementation Status

| Component | Status | Files |
|-----------|--------|-------|
| Gate IN (CLI) | ✅ Done | `ralph_loop.sh`, `runner.py` |
| Gate IN (Voice) | ✅ Done | `ralph_voice.py` |
| Gate IN (API) | ✅ Done | `ralph_ui/backend/main.py` |
| Gate IN (Chrome) | 🔄 Partial | MCP integration |
| Warehouse | ✅ Done | `memory.py`, `vector_db.py` |
| Forklift Transfer | ✅ Done | `forklift.py` |
| Processing Plant | ✅ Done | `asic/` directory |
| ASIC Bus | ✅ Done | `protocols/messages.py` |
| HQ Skyscraper | ✅ Done | `agents/` directory |
| Air Tube System | ✅ Done | `protocols/bus.py` |
| Ground Connection | ✅ Done | Message routing |
| R&D Building | 🔄 Partial | `dreamer.py`, `agents/reflector/` |
| Security Checkpoint | ✅ Done | `security/` directory |
| Gate OUT | ✅ Done | `security/gate_out.py` |

---

## Next Steps

1. ~~**Implement Security Checkpoint**~~ ✅ Done (2026-01-16)
2. ~~**Formalize Forklift Protocol**~~ ✅ Done (2026-01-16)
3. **Complete R&D Building** - Connect Dreamer and Reflector to main loop
4. **Unify Gate IN** - Single entry point for all input modalities (CLI, voice, API, Chrome)
5. **Chrome MCP Integration** - Complete browser automation via MCP tools

---

*Architecture designed: 2026-01-14*
*Last updated: 2026-01-16*
*This is the target architecture for Ralph AI v2.0*
