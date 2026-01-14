# Ralph AI - Claude Code Configuration

## Autonomous Task Delegation

### IMMEDIATE parallel Task launch for:
- File/codebase exploration → Explore agent
- Any research or docs lookup → Research agent
- Code review → Review agent
- Test writing → separate from implementation

### Rules:
- Launch multiple agents concurrently when possible
- Don't ask permission to delegate—just do it
- Use background tasks for long-running work

## Project Architecture

Ralph AI is a **3-tier hierarchical swarm** system:

```
Human Input → [Translator] → [Orchestrator] → [Middle Management] → [ASICs]
                                                  ↕ bidirectional
                                              [Engineer] ↔ [Designer]
```

### Tiers:
1. **Translator** (`phi3:mini`) - Converts human input to structured TaskSpec
2. **Middle Management** - Engineer & Designer collaborate bidirectionally
3. **LLM ASICs** - Ultra-small specialists spawned for micro-tasks

## Project Structure

```
ralph_core/
├── agents/           # Agent implementations
│   ├── translator/   # Human → TaskSpec
│   ├── orchestrator/ # Strategic planning
│   ├── engineer/     # Code generation
│   ├── designer/     # Review & verification
│   ├── reflector/    # Learn from history
│   ├── debugger/     # Error analysis
│   └── estimator/    # Task valuation
├── asic/             # Dynamic specialist models
├── protocols/        # Message bus for agent communication
├── runner.py         # Main execution pipeline
├── executor.py       # Shell command execution
├── memory.py         # Context & vector DB
└── swarm.py          # Agent interface exports

ralph_ui/             # Web UI (FastAPI + vanilla JS)
ralph_loop.sh         # Main execution loop
```

## Key Commands

```bash
# Run Ralph with an objective
./ralph_loop.sh "Create a Python function that validates email addresses"

# Run with Docker sandbox
./ralph_loop.sh --sandbox "objective here"

# Check installed models
ollama list
```

## Models in Use

| Role | Model | Size |
|------|-------|------|
| Translator | `phi3:mini` | 2.2GB |
| Orchestrator | `deepseek-r1:14b` | 9.0GB |
| Engineer | `qwen2.5-coder:14b` | 9.0GB |
| Designer | `mistral-nemo:12b` | 7.1GB |
| ASIC (tiny code) | `deepseek-coder:1.3b` | 776MB |
| ASIC (small code) | `qwen2.5-coder:1.5b` | 986MB |
| ASIC (ultra-tiny) | `tinyllama:1.1b` | 637MB |
| Embeddings | `nomic-embed-text` | 274MB |

## Development Guidelines

- Follow the Ralph Wiggum Method: iterate until `<promise>COMPLETE</promise>`
- Git is memory - commit frequently
- Lessons are stored in `~/.ralph/global_memory/lessons.json`
- Use vector DB for semantic recall across projects
