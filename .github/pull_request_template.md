## Description

<!-- Provide a clear and concise description of what this PR does -->

## Type of Change

<!-- Mark the relevant option with an "x" -->

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] ASIC specialist addition
- [ ] Performance improvement
- [ ] Code refactoring
- [ ] CI/CD improvement

## Related Issue

<!-- Link to the issue this PR addresses -->
Closes #(issue number)

## Component(s) Affected

<!-- Check all that apply -->

- [ ] Translator Agent
- [ ] Orchestrator Agent
- [ ] Engineer Agent
- [ ] Designer Agent
- [ ] Reflector Agent
- [ ] Debugger Agent
- [ ] Estimator Agent
- [ ] ASIC System
- [ ] Message Bus (protocols)
- [ ] Memory/Vector DB
- [ ] Web UI
- [ ] Voice Interface
- [ ] Vision Module
- [ ] Docker Sandbox
- [ ] Core Infrastructure (runner, executor)
- [ ] Documentation

## Changes Made

<!-- Detailed list of changes -->

-
-
-

## Testing Performed

<!-- Describe the testing you've done -->

### Manual Testing

- [ ] Tested with `./ralph_loop.sh` with objective: `"<your test objective>"`
- [ ] Verified in sandbox mode: `./ralph_loop.sh --sandbox "<objective>"`
- [ ] Tested web UI functionality
- [ ] Verified with multiple iterations
- [ ] Tested with different models (if applicable)

### Automated Testing

- [ ] All existing tests pass
- [ ] Added new tests for new functionality
- [ ] No new linter warnings introduced

### Test Results

```
Paste relevant test output here
```

## Performance Impact

<!-- If applicable, describe performance implications -->

- Expected iteration time change: [none/faster/slower]
- Memory usage impact: [none/increase/decrease]
- Model switching frequency: [none/more/less]

## Screenshots/Logs

<!-- If applicable, add screenshots or log excerpts -->

<details>
<summary>Example output</summary>

```
Paste example output here
```

</details>

## Checklist

<!-- Ensure all items are checked before requesting review -->

### Code Quality

- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] My changes generate no new warnings or errors
- [ ] I have verified all imports are correct

### Documentation

- [ ] I have updated relevant documentation (README.md, CLAUDE.md, etc.)
- [ ] I have added docstrings to new functions/classes
- [ ] I have updated the CHANGELOG.md (if applicable)

### Testing

- [ ] I have tested this with required Ollama models installed
- [ ] I have verified the change works across multiple iterations
- [ ] I have tested edge cases and error conditions

### Integration

- [ ] My changes don't break existing functionality
- [ ] I have tested integration with other agents/components
- [ ] Message bus communication works correctly (if applicable)
- [ ] ASIC spawning works correctly (if applicable)

### Deployment

- [ ] My changes work in both local and sandbox modes
- [ ] I have considered backward compatibility
- [ ] No hardcoded paths or user-specific configurations
- [ ] Environment variables are documented (if new ones added)

## Additional Notes

<!-- Any other information reviewers should know -->

## For ASIC Specialist PRs

<!-- Only fill out if adding a new ASIC specialist -->

- [ ] ASIC is registered in `ralph_core/asic/registry.py`
- [ ] Handler logic added in `ralph_core/asic/handler.py`
- [ ] Model requirement documented in README.md
- [ ] Response time tested and is <2 seconds
- [ ] Integration points with agents tested
- [ ] Input/output format is well-defined

## Breaking Changes

<!-- If this is a breaking change, describe migration steps -->

**Migration Required:**

```
List steps users need to take to adapt to this change
```

---

**Review Focus Areas:**

<!-- Help reviewers by highlighting areas needing special attention -->

-
-

**Reviewer Checklist:**

- [ ] Code is clear and maintainable
- [ ] Tests are adequate and passing
- [ ] Documentation is complete and accurate
- [ ] No security concerns
- [ ] Performance is acceptable
- [ ] Integration with Ralph AI philosophy maintained
