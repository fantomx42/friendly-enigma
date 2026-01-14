# Contributing to Ralph AI

Thank you for your interest in contributing to Ralph AI! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Commit Message Convention](#commit-message-convention)
- [Pull Request Process](#pull-request-process)
- [Adding a New ASIC Specialist](#adding-a-new-asic-specialist)
- [Documentation](#documentation)
- [Community](#community)

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone git@github.com:YOUR_USERNAME/friendly-enigma.git
   cd friendly-enigma
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream git@github.com:fantomx42/friendly-enigma.git
   ```
4. **Create a branch** for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Prerequisites

- Python 3.8+
- Ollama installed with required models
- 32GB+ RAM (64GB recommended)
- GPU with 16GB+ VRAM (for optimal performance)

### Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install development dependencies
pip install flake8 pylint mypy black isort pytest

# Pull required Ollama models
ollama pull phi3:mini
ollama pull deepseek-r1:14b
ollama pull qwen2.5-coder:14b
ollama pull mistral-nemo:12b
ollama pull tinyllama:1.1b
ollama pull deepseek-coder:1.3b
ollama pull qwen2.5-coder:1.5b
ollama pull nomic-embed-text
```

### Verify Installation

```bash
# Test basic functionality
./ralph_loop.sh "Say hello and complete successfully"

# Run Python import checks
python -c "from ralph_core import runner, executor, memory, swarm"
```

## How to Contribute

### Types of Contributions

We welcome various types of contributions:

- **Bug fixes**: Fix issues reported in GitHub Issues
- **New features**: Implement features from the roadmap or feature requests
- **ASIC specialists**: Add new specialized micro-task models
- **Documentation**: Improve README, guides, or code comments
- **Testing**: Add test coverage or improve existing tests
- **Performance**: Optimize iteration speed, memory usage, or model switching
- **Refactoring**: Improve code quality and maintainability

### Finding Work

- Browse [Issues](https://github.com/fantomx42/friendly-enigma/issues) labeled `good first issue` or `help wanted`
- Check the [Roadmap](README.md#roadmap) for planned features
- Read [GitHub Discussions](https://github.com/fantomx42/friendly-enigma/discussions) for ideas

## Coding Standards

### Python Style

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- Maximum line length: 127 characters
- Use 4 spaces for indentation
- Use `snake_case` for functions and variables
- Use `PascalCase` for classes
- Use `UPPER_CASE` for constants

### Code Formatting

Use `black` and `isort` for automatic formatting:

```bash
# Format code
black ralph_core/
isort ralph_core/

# Check without modifying
black --check ralph_core/
isort --check-only ralph_core/
```

### Linting

Run linters before committing:

```bash
# Quick syntax check
flake8 ralph_core/ --select=E9,F63,F7,F82

# Full linting (warnings okay, but minimize them)
flake8 ralph_core/ --max-line-length=127
```

### Type Hints

Use type hints for function signatures:

```python
from typing import Dict, List, Optional

def process_task(task: Dict[str, str], iterations: int = 5) -> Optional[str]:
    """Process a task with specified iterations.

    Args:
        task: Task specification dictionary
        iterations: Maximum number of iterations

    Returns:
        Result string or None if failed
    """
    pass
```

## Testing Guidelines

### Running Tests

```bash
# Run all tests (when available)
pytest tests/

# Run specific test file
pytest tests/test_executor.py

# Run with coverage
pytest --cov=ralph_core tests/
```

### Manual Testing Checklist

For any change, test at minimum:

1. **Basic execution**:
   ```bash
   ./ralph_loop.sh "Create a simple Python function"
   ```

2. **Sandbox mode**:
   ```bash
   ./ralph_loop.sh --sandbox "Create a test file"
   ```

3. **Multiple iterations**: Verify the change works across 3+ iterations

4. **Error handling**: Test with invalid inputs or edge cases

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_<module>.py`
- Use descriptive test function names: `test_executor_runs_simple_command()`
- Include both happy path and error cases

## Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `test`: Adding or updating tests
- `chore`: Maintenance tasks, dependencies, etc.
- `ci`: CI/CD changes

### Examples

```bash
feat(asic): add yaml_validator ASIC specialist

Added new ASIC for YAML validation using qwen2.5-coder:1.5b.
Response time: ~800ms. Integrated with Engineer and Designer agents.

Closes #123
```

```bash
fix(executor): handle timeout errors gracefully

Added 30s timeout to shell command execution with proper error
messaging. Prevents infinite hangs on stuck processes.

Fixes #456
```

## Pull Request Process

### Before Submitting

1. **Sync with upstream**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run quality checks**:
   ```bash
   # Format code
   black ralph_core/
   isort ralph_core/

   # Lint
   flake8 ralph_core/ --select=E9,F63,F7,F82

   # Test
   ./ralph_loop.sh "Test objective"
   ```

3. **Update documentation**:
   - Update README.md if adding features
   - Update CLAUDE.md if changing architecture
   - Add docstrings to new functions
   - Update CHANGELOG.md for significant changes

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat(component): your change description"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

### Submitting the PR

1. Go to [Pull Requests](https://github.com/fantomx42/friendly-enigma/pulls)
2. Click "New Pull Request"
3. Select your fork and branch
4. Fill out the PR template completely
5. Link related issues
6. Request review

### PR Review Process

- Maintainers will review within 3-5 business days
- Address review comments by pushing additional commits
- Once approved, a maintainer will merge your PR
- You can delete your branch after merging

### After Merge

Celebrate! You've contributed to Ralph AI. Update your fork:

```bash
git checkout main
git pull upstream main
git push origin main
```

## Adding a New ASIC Specialist

ASIC specialists are a key feature of Ralph AI. Follow these steps to add one:

### 1. Define the ASIC

Create an issue using the [ASIC Proposal template](.github/ISSUE_TEMPLATE/asic_proposal.yml) to discuss your idea first.

### 2. Choose a Model

Select an appropriate ultra-small model:
- `tinyllama:1.1b` - Basic text tasks (<637MB)
- `deepseek-coder:1.3b` - Code-focused tasks (~776MB)
- `qwen2.5-coder:1.5b` - Advanced code tasks (~986MB)

### 3. Register the ASIC

Add to `ralph_core/asic/registry.py`:

```python
ASIC_REGISTRY = {
    # ... existing ASICs
    "your_asic_name": {
        "model": "qwen2.5-coder:1.5b",
        "description": "Brief description of what this ASIC does",
        "prompt_template": """You are a specialized ASIC for [task].
Given {input}, output {format}.
Be concise and fast.""",
        "max_tokens": 200,
        "temperature": 0.1,
        "timeout": 5.0
    }
}
```

### 4. Implement Handler Logic

Add handler in `ralph_core/asic/handler.py`:

```python
def handle_your_asic_name(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle your_asic_name ASIC requests.

    Args:
        input_data: Input dictionary with required fields

    Returns:
        Output dictionary with results
    """
    # Validate input
    # Process with model
    # Return structured output
    pass
```

### 5. Integration Points

Update agents to call your ASIC:
- Modify `ralph_core/agents/engineer/` to invoke ASIC
- Update `ralph_core/agents/designer/` to validate ASIC output

### 6. Document

- Add to README.md ASIC table
- Add usage example in docstring
- Update model requirements

### 7. Test

```bash
# Test ASIC directly
python -c "from ralph_core.asic import spawner; print(spawner.spawn_asic('your_asic_name', {...}))"

# Test in full loop
./ralph_loop.sh "Objective that triggers your ASIC"
```

## Documentation

### Code Documentation

- Add docstrings to all public functions and classes
- Use Google-style docstring format:

```python
def example_function(param1: str, param2: int) -> bool:
    """Brief description of function.

    Longer description if needed. Explain the purpose and behavior.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When param2 is negative

    Example:
        >>> example_function("test", 42)
        True
    """
    pass
```

### Updating README

When adding features, update:
- Feature list
- Architecture diagram (if structural change)
- Usage examples
- Model requirements
- Configuration options

## Community

### Getting Help

- **GitHub Discussions**: Ask questions and share ideas
- **Issues**: Report bugs or request features
- **Discord/Slack**: (Add links when available)

### Reporting Bugs

Use the [Bug Report template](.github/ISSUE_TEMPLATE/bug_report.yml) and include:
- Clear reproduction steps
- Expected vs actual behavior
- Environment details (OS, Python version, Ollama version)
- Relevant logs from `ralph.log`

### Feature Requests

Use the [Feature Request template](.github/ISSUE_TEMPLATE/feature_request.yml) and explain:
- Problem being solved
- Proposed solution
- Use cases
- Priority

## Recognition

Contributors will be recognized in:
- GitHub Contributors list
- CHANGELOG.md for significant contributions
- README.md acknowledgments (for major features)

## License

By contributing to Ralph AI, you agree that your contributions will be licensed under the MIT License.

---

## Questions?

If you have questions not covered here, feel free to:
- Open a [Discussion](https://github.com/fantomx42/friendly-enigma/discussions)
- Comment on a related Issue
- Reach out to maintainers

Thank you for contributing to Ralph AI!
