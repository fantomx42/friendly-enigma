# Project Ralph
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

> **Distributed Heterogeneous Agent Swarm**

Project Ralph implements the "Ralph Wiggum" methodology: a paradigm predicated upon the utility of predictable failure and cyclical refinement. It operates as a persistent, self-correcting loop where autonomous agents iterate to solve problems.

## 📚 Documentation

Detailed documentation is available in the `docs/` directory:

- [**Mission & Config**](docs/MISSION.md): Hardware specs and operating protocols.
- [**Architecture**](docs/projectralph.md): Detailed architectural specification of "The Ralph Protocol".
- [**Status**](docs/STATUS.md): Current project state and roadmap.
- [**Tools**](docs/TOOLS.md): Available tools and commands.

## 📂 Project Structure

| Directory | Description |
| :--- | :--- |
| [`ralph_core/`](ralph_core/) | Core logic and runner for the agent swarm. |
| [`ralph_ui/`](ralph_ui/) | Backend and frontend for the Ralph UI. |
| [`docs/`](docs/) | Project documentation and specifications. |
| [`data/`](data/) | Runtime data, tasks, and progress logs. |
| [`legacy/`](legacy/) | Deprecated or old code. |
| [`memory/`](memory/) | Long-term memory storage. |

## 🚀 Getting Started

### Prerequisites
- Python 3.x
- High-Performance Workstation (Recommended for full swarm capabilities)

### Running the Loop

The core of Project Ralph is the autonomous loop.

```bash
./ralph_loop.sh "Your Objective Here"
```

This will:
1. Initialize the task in `data/RALPH_TASK.md`.
2. Start the iterative cycle of generation, verification, and critique.
3. Continue until the objective is met or max iterations are reached.

### Running the UI

To start the web interface:

```bash
./start_ui.sh
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

See [docs/STATUS.md](docs/STATUS.md) for current blockers and active goals.
