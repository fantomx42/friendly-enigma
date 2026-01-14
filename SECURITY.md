# Security Policy

## Supported Versions

Ralph AI is currently in active development. Security updates are provided for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| main    | :white_check_mark: |
| < 1.0   | :x:                |

Once v1.0 is released, this table will be updated with specific version support timelines.

## Security Considerations

### Execution Environment

Ralph AI executes code autonomously and can run shell commands. This presents inherent security risks:

- **Code Execution**: The Engineer agent generates and executes code that may interact with your filesystem
- **Shell Commands**: The Executor module runs shell commands with your user's permissions
- **Model Trust**: Output from LLMs is non-deterministic and should not be blindly trusted
- **Sandbox Isolation**: Use `--sandbox` mode for untrusted or destructive operations

### Recommended Security Practices

1. **Use Sandbox Mode** for untrusted operations:
   ```bash
   ./ralph_loop.sh --sandbox "potentially dangerous objective"
   ```

2. **Review Generated Code** before production use, especially for:
   - File system operations
   - Network requests
   - Database modifications
   - System configuration changes

3. **Limit Filesystem Access** by running Ralph AI in a restricted directory:
   ```bash
   cd /path/to/safe/workspace
   ./ralph_loop.sh "objective"
   ```

4. **Monitor Logs** regularly:
   - Check `ralph.log` for unexpected behavior
   - Review `metrics.jsonl` for anomalous patterns
   - Inspect Git commits made by Ralph AI

5. **Environment Variables**: Keep sensitive data in `.env` files that are `.gitignore`d
   - Never commit API keys, passwords, or secrets
   - Use environment variable substitution for sensitive config

6. **Docker Sandbox**: For production or high-risk tasks:
   - Configure Docker with minimal permissions
   - Use read-only mounts where possible
   - Limit network access in sandbox containers

### Known Security Limitations

- **LLM Hallucinations**: Models may generate incorrect or unsafe code
- **Prompt Injection**: User objectives may be manipulated to bypass safety checks
- **Memory Leakage**: Vector DB and git history may contain sensitive information
- **Resource Exhaustion**: Runaway iterations could consume excessive CPU/GPU/memory
- **No Input Sanitization**: User objectives are passed directly to models

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please follow these steps:

### 1. Do Not Create a Public Issue

**DO NOT** open a public GitHub Issue for security vulnerabilities. This could put users at risk.

### 2. Report Privately

Send a detailed report to:

**[To be added: security contact email, e.g., security@ralph-ai.org]**

Alternatively, use GitHub's private vulnerability reporting:
1. Go to the [Security tab](https://github.com/fantomx42/friendly-enigma/security)
2. Click "Report a vulnerability"
3. Fill out the form with details

### 3. Include in Your Report

Please provide as much information as possible:

- **Description**: Clear description of the vulnerability
- **Impact**: What could an attacker do? Who is affected?
- **Reproduction Steps**: Detailed steps to reproduce the issue
- **Affected Components**: Which parts of Ralph AI are affected?
  - Specific agent(s)
  - Executor module
  - Message bus
  - Web UI
  - Docker sandbox
- **Proof of Concept**: Code or commands demonstrating the vulnerability
- **Suggested Fix**: If you have ideas on how to fix it
- **Disclosure Timeline**: When you plan to publicly disclose (if at all)

### Example Report Template

```
Subject: [SECURITY] Brief description of vulnerability

## Summary
One-paragraph summary of the vulnerability.

## Impact
- Who is affected?
- What can an attacker do?
- What is the severity? (Critical/High/Medium/Low)

## Reproduction Steps
1. Step one
2. Step two
3. Observe the vulnerability

## Affected Versions
- main branch (commit SHA: abc123)
- Version 0.x.x

## Environment
- OS: Ubuntu 22.04
- Python: 3.10.12
- Ollama: 0.x.x

## Suggested Fix
(If applicable)

## Disclosure Timeline
I plan to disclose this on [date] unless you request more time.
```

### 4. Response Timeline

We will acknowledge receipt of your report within **48 hours** and provide:
- Confirmation of the vulnerability
- Estimated timeline for a fix
- Credit preferences (if you want to be acknowledged)

### 5. Responsible Disclosure

We follow coordinated vulnerability disclosure:

1. **Initial Response**: Within 48 hours
2. **Investigation**: Within 7 days
3. **Fix Development**: Within 30 days (for critical issues)
4. **Public Disclosure**: After fix is released and users have had time to update

We appreciate researchers who:
- Give us reasonable time to fix vulnerabilities
- Do not exploit vulnerabilities beyond proof-of-concept
- Do not publicly disclose until we've released a fix
- Act in good faith to avoid privacy violations and data destruction

## Security Updates

Security patches will be released as follows:

1. **Critical Vulnerabilities** (RCE, arbitrary code execution):
   - Fixed within 24-48 hours
   - Immediate patch release
   - Security advisory published

2. **High Severity** (data leakage, privilege escalation):
   - Fixed within 7 days
   - Patch release scheduled
   - Security advisory published

3. **Medium/Low Severity**:
   - Fixed in next regular release
   - Documented in CHANGELOG.md
   - May be publicly disclosed before fix if risk is minimal

## Security Advisories

Published security advisories are available at:
- [GitHub Security Advisories](https://github.com/fantomx42/friendly-enigma/security/advisories)
- CHANGELOG.md (for less critical issues)

Subscribe to GitHub notifications to receive security updates.

## Bug Bounty Program

We currently do not offer a bug bounty program. However, we deeply appreciate security researchers and will publicly acknowledge contributors (with permission) in:
- Security advisories
- CHANGELOG.md
- README.md acknowledgments section

## Security-Related Configuration

### Recommended Docker Sandbox Config

```yaml
# sandbox/docker-compose.yml
services:
  ralph-sandbox:
    image: ralph-ai-sandbox
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    read_only: true
    tmpfs:
      - /tmp
    networks:
      - isolated
    mem_limit: 8g
    cpus: 2
```

### Filesystem Permissions

```bash
# Restrict ralph.log permissions
chmod 600 ralph.log

# Protect memory directory
chmod 700 ~/.ralph/

# Ensure scripts are not world-writable
chmod 755 ralph_loop.sh
```

### Environment Variables

```bash
# .env (never commit this file)
RALPH_MAX_ITERATIONS=10        # Prevent runaway loops
RALPH_SANDBOX=1                # Force sandbox mode
RALPH_LOG_LEVEL=INFO           # Avoid verbose logging of sensitive data
RALPH_EXEC_TIMEOUT=30          # Kill long-running commands
```

## Scope

This security policy covers:
- Ralph AI core (`ralph_core/`)
- Execution loop (`ralph_loop.sh`)
- Web UI (`ralph_ui/`)
- Docker sandbox configuration
- Documentation and examples

Out of scope:
- Vulnerabilities in upstream dependencies (report to respective projects)
- Vulnerabilities in Ollama (report to Ollama team)
- LLM model security (report to model providers)
- Issues requiring physical access to the machine
- Social engineering attacks

## Security Best Practices for Contributors

When contributing code:

1. **Validate Inputs**: Sanitize user objectives and command outputs
2. **Least Privilege**: Request minimum necessary permissions
3. **Avoid Hardcoded Secrets**: Use environment variables or config files
4. **Timeout Long Operations**: Prevent resource exhaustion
5. **Log Sanitization**: Don't log sensitive data (API keys, passwords, PII)
6. **Dependency Review**: Vet new dependencies for known vulnerabilities
7. **Code Review**: Security-sensitive changes require maintainer review

## Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [LLM Security Best Practices](https://llmsecurity.net/)
- [Prompt Injection Mitigation](https://simonwillison.net/2023/Apr/14/worst-that-can-happen/)
- [Secure Python Coding](https://python.readthedocs.io/en/stable/library/security_warnings.html)

## Questions?

For security-related questions that are not vulnerability reports:
- Open a [GitHub Discussion](https://github.com/fantomx42/friendly-enigma/discussions)
- Tag with "security" label
- Avoid sharing specific vulnerability details publicly

---

Thank you for helping keep Ralph AI secure!

**Last Updated**: 2026-01-14
