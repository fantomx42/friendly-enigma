# Ralph AI

**A 3-tier hierarchical swarm system for autonomous AI task execution using local LLMs**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Ollama](https://img.shields.io/badge/Ollama-Required-green.svg)](https://ollama.ai/)

Ralph AI implements the "Ralph Wiggum Method" - a fault-tolerant autonomous system that iterates until success through predictable failure and cyclical refinement. Rather than expecting perfect execution on the first attempt, Ralph embraces failure as diagnostic information and continuously refines solutions until the `<promise>COMPLETE</promise>` signal is achieved.

## Architecture Overview

Ralph AI is structured as a **3-tier hierarchical swarm** with bidirectional communication:

```
┌─────────────────────────────────────────────────────────────┐
│                      HUMAN INPUT                            │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  TIER 1: TRANSLATOR (phi3:mini - 2.2GB)                     │
│  Converts natural language → Structured TaskSpec JSON       │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  ORCHESTRATOR (deepseek-r1:14b - 9.0GB)                     │
│  Strategic planning & task decomposition                    │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  TIER 2: MIDDLE MANAGEMENT (Bidirectional Revision)         │
│  ┌────────────────────┐  ◄──────►  ┌──────────────────┐    │
│  │  ENGINEER          │             │  DESIGNER        │    │
│  │  qwen2.5-coder:14b │             │  mistral-nemo:12b│    │
│  │  Code Generation   │             │  Review & Verify │    │
│  └────────────────────┘             └──────────────────┘    │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  TIER 3: LLM ASICs (Ultra-Small Specialists)                │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐  │
│  │  Regex   │  JSON    │  SQL     │  Test    │  Fix     │  │
│  │  Parser  │  Validate│  Query   │  Gen     │  Typo    │  │
│  └──────────┴──────────┴──────────┴──────────┴──────────┘  │
│  Models: tinyllama:1.1b, deepseek-coder:1.3b, qwen:1.5b    │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

#### Message Bus Architecture (V2)
- **Protocol-based Communication**: All agents communicate through a structured message bus
- **Circuit Breakers**: Automatic failure recovery and retry logic
- **Async I/O**: Non-blocking agent interactions for parallel execution
- **Type Safety**: Structured message schemas with validation

#### ASIC System (Application-Specific Integrated Circuits)
Ralph dynamically spawns ultra-small specialist models for micro-tasks:

| ASIC Specialist | Model | Use Case |
|----------------|-------|----------|
| `regex_parser` | qwen2.5-coder:1.5b | Pattern matching & extraction |
| `json_validator` | deepseek-coder:1.3b | JSON schema validation |
| `sql_query` | qwen2.5-coder:1.5b | Database query generation |
| `test_generator` | qwen2.5-coder:1.5b | Unit test creation |
| `fix_typo` | tinyllama:1.1b | Quick typo corrections |
| `docstring` | tinyllama:1.1b | Documentation generation |
| `log_parser` | deepseek-coder:1.3b | Log analysis |
| `error_classifier` | tinyllama:1.1b | Error categorization |
| `import_fixer` | qwen2.5-coder:1.5b | Import statement repair |
| `linter_suggestion` | deepseek-coder:1.3b | Code style fixes |

ASICs are spawned on-demand with:
- **Context compression**: Only essential information passed
- **Fast execution**: <1s response times for micro-tasks
- **Resource efficiency**: ~600MB - 1GB per ASIC instance

#### Core Agents

- **Translator** (`ralph_core/agents/translator/`) - Converts human objectives to machine-readable TaskSpec
- **Orchestrator** (`ralph_core/agents/orchestrator/`) - Strategic planning and task decomposition
- **Engineer** (`ralph_core/agents/engineer/`) - Code generation and implementation
- **Designer** (`ralph_core/agents/designer/`) - Code review, verification, and quality assurance
- **Reflector** (`ralph_core/agents/reflector/`) - Learns from execution history
- **Debugger** (`ralph_core/agents/debugger/`) - Error analysis and fix suggestions
- **Estimator** (`ralph_core/agents/estimator/`) - Task complexity evaluation

## Features

- **Autonomous Iteration**: Continues until `<promise>COMPLETE</promise>` is achieved (max 10 iterations)
- **Vector Memory**: Semantic search across project history using `nomic-embed-text`
- **Git Integration**: Automatic commit tracking for memory persistence
- **Docker Sandbox**: Optional isolated execution environment
- **Voice Interface**: Speech-to-text input via `ralph_voice.py`
- **Vision Support**: Image analysis using LLaVA models
- **Web Search**: Integrated web research capabilities
- **Real-time UI**: Web interface at http://ralph.ai (FastAPI backend + vanilla JS frontend)
- **Metrics & Logging**: JSONL-based execution tracking

## Installation

### Prerequisites

- **Hardware**:
  - Minimum 32GB RAM (64GB recommended)
  - GPU with 16GB+ VRAM (tested on RX 9070 XT)
  - Multi-core CPU (tested on Ryzen 5800X3D)
- **OS**: Linux (tested on CachyOS/Arch Linux)
- **Python**: 3.8 or higher
- **Ollama**: [Install from ollama.ai](https://ollama.ai/)

### 1. Clone the Repository

```bash
git clone git@github.com:fantomx42/friendly-enigma.git ralph-ai
cd ralph-ai
```

### 2. Install Required Models

Ralph requires specific models with precise parameter sizes. Pull them with Ollama:

```bash
# Tier 1: Translator
ollama pull phi3:mini

# Orchestrator
ollama pull deepseek-r1:14b

# Tier 2: Middle Management
ollama pull qwen2.5-coder:14b
ollama pull mistral-nemo:12b

# Tier 3: ASICs
ollama pull tinyllama:1.1b
ollama pull deepseek-coder:1.3b
ollama pull qwen2.5-coder:1.5b

# Embeddings
ollama pull nomic-embed-text

# Optional: Vision
ollama pull llava:13b
```

**Total storage required**: ~35GB

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

Expected dependencies:
- `ollama`
- `chromadb` (vector database)
- `fastapi` & `uvicorn` (web UI)
- `pydantic` (data validation)

### 4. Verify Installation

```bash
# Check installed models
ollama list

# Test basic execution
./ralph_loop.sh "Say hello and exit successfully"
```

## Usage

### Basic Execution

```bash
./ralph_loop.sh "Your objective here"
```

Example:
```bash
./ralph_loop.sh "Create a Python function that validates email addresses with regex"
```

### Sandbox Mode (Docker)

For safe execution of untrusted or destructive operations:

```bash
./ralph_loop.sh --sandbox "Your objective here"
```

This runs the task in an isolated Docker container with limited filesystem access.

### Voice Interface

```bash
python ralph_voice.py
# Speak your objective, Ralph will transcribe and execute
```

### Web UI

Start the web interface:

```bash
./start_ui.sh
# Visit http://ralph.ai (or http://localhost:8000)
```

The UI provides:
- Real-time task progress visualization
- Iteration history
- Agent communication logs
- Git commit tracking

## How It Works

### The Ralph Loop

1. **Parse Objective**: Translator converts human input to TaskSpec
2. **Plan**: Orchestrator decomposes task into subtasks
3. **Execute**: Engineer generates code/solution
4. **Review**: Designer validates output
5. **Verify**: Run tests, linters, and checks
6. **Iterate**: If not complete, feed errors back to Engineer
7. **Complete**: Exit when `<promise>COMPLETE</promise>` is detected

### Iteration Example

```
Iteration 1: Engineer creates function (syntax error)
            ↓
Iteration 2: Fix syntax, Designer finds logic bug
            ↓
Iteration 3: Fix logic, tests fail
            ↓
Iteration 4: Fix tests, linter warnings
            ↓
Iteration 5: Fix warnings, all checks pass
            ↓
        <promise>COMPLETE</promise>
```

### TaskSpec Format

Translator produces structured JSON:

```json
{
  "objective": "Create email validator",
  "constraints": ["Use regex", "Handle edge cases"],
  "success_criteria": [
    "Passes test_email_validator.py",
    "No linter warnings"
  ],
  "estimated_complexity": 3
}
```

## Configuration

### Environment Variables

Create a `.env` file (optional):

```bash
RALPH_MAX_ITERATIONS=10        # Max loop iterations
RALPH_SANDBOX=0                # Enable sandbox by default
RALPH_LOG_LEVEL=INFO           # Logging verbosity
RALPH_MEMORY_PATH=~/.ralph     # Memory storage location
```

### Model Customization

Edit model assignments in `ralph_core/swarm.py`:

```python
MODELS = {
    "translator": "phi3:mini",
    "orchestrator": "deepseek-r1:14b",
    "engineer": "qwen2.5-coder:14b",
    "designer": "mistral-nemo:12b",
}
```

## Project Structure

```
ralph-ai/
├── ralph_core/               # Core swarm implementation
│   ├── agents/               # Agent implementations
│   │   ├── translator/       # Human → TaskSpec
│   │   ├── orchestrator/     # Strategic planning
│   │   ├── engineer/         # Code generation
│   │   ├── designer/         # Review & verification
│   │   ├── reflector/        # Historical learning
│   │   ├── debugger/         # Error analysis
│   │   └── estimator/        # Task complexity
│   ├── asic/                 # ASIC specialist system
│   │   ├── handler.py        # ASIC execution handler
│   │   ├── registry.py       # ASIC definitions
│   │   └── spawner.py        # Dynamic ASIC spawning
│   ├── protocols/            # Message bus
│   │   ├── bus.py            # Message routing & circuit breakers
│   │   └── messages.py       # Message schemas
│   ├── runner.py             # Main execution pipeline
│   ├── executor.py           # Shell command execution
│   ├── memory.py             # Context & vector DB
│   ├── swarm.py              # Agent interface exports
│   ├── tools.py              # Utility functions
│   ├── vector_db.py          # ChromaDB integration
│   ├── vision.py             # LLaVA vision support
│   ├── voice.py              # Speech-to-text
│   └── web.py                # Web search integration
├── ralph_ui/                 # Web interface
│   ├── backend/
│   │   └── main.py           # FastAPI server
│   └── frontend/
│       └── index.html        # Vanilla JS UI
├── sandbox/                  # Docker sandbox config
├── ralph_loop.sh             # Main execution loop
├── ralph_voice.py            # Voice interface entry point
├── ralph_daemon.py           # Background service mode
└── README.md                 # This file
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Areas for improvement:
- Additional ASIC specialists
- Support for more model backends (OpenAI, Anthropic, etc.)
- Enhanced memory compression strategies
- Multi-project context switching
- Performance optimizations

## Roadmap

- [ ] Windows/macOS support
- [ ] Multi-language support (currently Python-focused)
- [ ] Cloud deployment options (AWS, GCP, Azure)
- [ ] Plugin system for custom agents
- [ ] Benchmark suite for iteration efficiency
- [ ] Integration with popular IDEs (VSCode, JetBrains)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

Built with:
- [Ollama](https://ollama.ai/) - Local LLM inference
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [DeepSeek](https://www.deepseek.com/), [Qwen](https://qwenlm.github.io/), [Mistral AI](https://mistral.ai/) - Model providers

Inspired by:
- The Ralph Wiggum Method (predictable failure → iterative success)
- Swarm intelligence and hierarchical control systems
- ASIC design principles applied to LLM architecture

## Support

- **Issues**: [GitHub Issues](https://github.com/fantomx42/friendly-enigma/issues)
- **Discussions**: [GitHub Discussions](https://github.com/fantomx42/friendly-enigma/discussions)

---

**"Me fail English? That's unpossible!"** - Ralph Wiggum (and Ralph AI's philosophy)
