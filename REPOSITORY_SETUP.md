# Ralph AI - Repository Setup Guide

This guide provides instructions for configuring the GitHub repository for optimal discoverability, security, and community engagement.

## Repository Settings

### 1. General Settings

Navigate to **Settings > General**:

#### Repository Name & Description

- **Repository Name**: `ralph-ai` (consider renaming from `friendly-enigma`)
- **Description**:
  ```
  3-tier hierarchical swarm system for autonomous AI task execution using local LLMs. Implements iterative refinement with Ollama models.
  ```
- **Website**: `http://ralph.ai` (or documentation URL when available)

#### Topics/Tags

Add the following topics for discoverability:

```
ai
llm
ollama
autonomous-agents
swarm-intelligence
local-llm
deepseek
qwen
mistral
phi3
agent-framework
code-generation
hierarchical-agents
asic
vector-database
chromadb
python
fastapi
docker
self-healing
```

**How to add**: Settings > General > Topics (click the gear icon)

#### Features

Enable these features:
- ✅ **Wikis** - For extended documentation
- ✅ **Issues** - Bug reports and feature requests
- ✅ **Discussions** - Community Q&A and ideas
- ✅ **Projects** - Roadmap and task tracking
- ❌ **Sponsorships** - (Optional, enable if seeking sponsorship)
- ✅ **Preserve this repository** - Archive for long-term availability

#### Pull Requests

Configure merge options:
- ✅ **Allow merge commits** - Standard merging
- ✅ **Allow squash merging** - Clean history for small changes
- ✅ **Allow rebase merging** - Linear history
- ✅ **Automatically delete head branches** - Keep branches clean
- ✅ **Allow auto-merge** - After approvals
- ❌ **Allow update branch** - Prevent accidental updates

### 2. Branch Protection Rules

Navigate to **Settings > Branches > Add branch protection rule**:

#### For `main` branch:

```
Branch name pattern: main

Protect matching branches:
  ✅ Require a pull request before merging
      ✅ Require approvals: 1
      ✅ Dismiss stale pull request approvals when new commits are pushed
      ✅ Require review from Code Owners (create CODEOWNERS file)

  ✅ Require status checks to pass before merging
      ✅ Require branches to be up to date before merging
      Required status checks:
        - Python Syntax & Import Validation
        - Project Structure Validation
        - Security Scan

  ✅ Require conversation resolution before merging

  ✅ Require linear history (optional - enforces rebase/squash)

  ✅ Include administrators (apply rules to admins too)

  ❌ Allow force pushes (prevent history rewriting)
  ❌ Allow deletions (prevent branch deletion)
```

#### For `develop` branch (if using):

Same as above but:
- Require approvals: 1
- Can be more lenient for experimental features

### 3. Security Settings

Navigate to **Settings > Security**:

#### Code Security and Analysis

```
✅ Private vulnerability reporting
    Enable this to receive security reports via GitHub

✅ Dependency graph
    Automatically track dependencies

✅ Dependabot alerts
    Get notified of vulnerable dependencies

✅ Dependabot security updates
    Automatically create PRs for security updates

✅ Dependabot version updates (optional)
    Create dependabot.yml:
```

Create `.github/dependabot.yml`:

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 5
    labels:
      - "dependencies"
      - "python"

  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "monthly"
    labels:
      - "dependencies"
      - "ci"

  - package-ecosystem: "docker"
    directory: "/sandbox"
    schedule:
      interval: "weekly"
    labels:
      - "dependencies"
      - "docker"
```

#### Code Scanning

```
✅ CodeQL analysis
    - Enable in Security > Code scanning > Set up > CodeQL
    - Uses .github/workflows/codeql.yml (auto-generated)
```

#### Secret Scanning

```
✅ Secret scanning
    Automatically detect committed secrets

✅ Push protection
    Prevent secrets from being committed
```

## Labels

### Create Standard Labels

Navigate to **Issues > Labels** and create/modify:

#### Component Labels (Blue/Purple: `#0E8A16` style)

```
agent:translator       - Translator agent (phi3:mini)
agent:orchestrator     - Orchestrator agent (deepseek-r1:14b)
agent:engineer         - Engineer agent (qwen2.5-coder:14b)
agent:designer         - Designer agent (mistral-nemo:12b)
agent:reflector        - Reflector agent
agent:debugger         - Debugger agent
agent:estimator        - Estimator agent
asic                   - ASIC specialist system
message-bus            - Message bus & protocols
memory                 - Memory & vector DB
ui                     - Web UI (frontend/backend)
voice                  - Voice interface
vision                 - Vision module (LLaVA)
sandbox                - Docker sandbox
core                   - Core infrastructure (runner, executor)
```

#### Type Labels (Green: `#28A745`)

```
bug                    - Something isn't working
enhancement            - New feature or request
documentation          - Improvements or additions to docs
refactor               - Code refactoring (no behavior change)
performance            - Performance improvement
test                   - Adding or updating tests
ci                     - CI/CD related changes
dependencies           - Dependency updates
```

#### Priority Labels (Red/Orange/Yellow)

```
priority:critical      - Critical issue (security, data loss) - Red #D73A4A
priority:high          - High priority - Orange #FF9900
priority:medium        - Medium priority - Yellow #FFCC00
priority:low           - Low priority - Light Gray #CCCCCC
```

#### Status Labels (Gray: `#6C757D`)

```
needs-triage           - Needs initial review
needs-reproduction     - Cannot reproduce bug
needs-information      - More information needed
blocked                - Blocked by external issue
wontfix                - Will not be worked on
duplicate              - Duplicate of another issue
```

#### Difficulty Labels (Light Blue: `#0075CA`)

```
good first issue       - Good for newcomers
help wanted            - Extra attention needed
advanced               - Requires deep system knowledge
asic-proposal          - Proposal for new ASIC specialist
```

#### Special Labels

```
breaking-change        - Introduces breaking changes - Red #D73A4A
security               - Security-related issue - Red #D73A4A
model-requirement      - Requires new Ollama model - Purple #6F42C1
hardware-intensive     - Requires high-end hardware - Purple #6F42C1
```

### Bulk Label Creation Script

Create `scripts/create_labels.sh`:

```bash
#!/bin/bash
# Usage: ./scripts/create_labels.sh OWNER REPO TOKEN

OWNER=$1
REPO=$2
TOKEN=$3

create_label() {
    local name=$1
    local color=$2
    local description=$3

    curl -X POST \
        -H "Authorization: token $TOKEN" \
        -H "Accept: application/vnd.github.v3+json" \
        "https://api.github.com/repos/$OWNER/$REPO/labels" \
        -d "{\"name\":\"$name\",\"color\":\"$color\",\"description\":\"$description\"}"
}

# Component Labels
create_label "agent:translator" "0E8A16" "Translator agent (phi3:mini)"
create_label "agent:orchestrator" "0E8A16" "Orchestrator agent (deepseek-r1:14b)"
create_label "agent:engineer" "0E8A16" "Engineer agent (qwen2.5-coder:14b)"
create_label "agent:designer" "0E8A16" "Designer agent (mistral-nemo:12b)"
create_label "asic" "6F42C1" "ASIC specialist system"
create_label "message-bus" "0E8A16" "Message bus & protocols"
create_label "memory" "0E8A16" "Memory & vector DB"
create_label "ui" "0E8A16" "Web UI"
create_label "sandbox" "0E8A16" "Docker sandbox"

# Type Labels
create_label "enhancement" "28A745" "New feature or request"
create_label "documentation" "0075CA" "Documentation improvements"
create_label "refactor" "FEF2C0" "Code refactoring"
create_label "performance" "28A745" "Performance improvement"

# Priority Labels
create_label "priority:critical" "D73A4A" "Critical issue"
create_label "priority:high" "FF9900" "High priority"
create_label "priority:medium" "FFCC00" "Medium priority"
create_label "priority:low" "CCCCCC" "Low priority"

# Status Labels
create_label "needs-triage" "6C757D" "Needs initial review"
create_label "needs-information" "6C757D" "More information needed"
create_label "blocked" "6C757D" "Blocked by external issue"

# Difficulty Labels
create_label "good first issue" "7057FF" "Good for newcomers"
create_label "help wanted" "008672" "Extra attention needed"
create_label "advanced" "0075CA" "Requires deep system knowledge"

# Special Labels
create_label "breaking-change" "D73A4A" "Introduces breaking changes"
create_label "security" "D73A4A" "Security-related issue"
create_label "model-requirement" "6F42C1" "Requires new Ollama model"
```

## CODEOWNERS File

Create `.github/CODEOWNERS`:

```
# Ralph AI Code Owners
# These owners will be automatically requested for review when PRs modify these paths

# Default owner for everything
*                           @fantomx42

# Core System
/ralph_core/runner.py       @fantomx42
/ralph_core/executor.py     @fantomx42
/ralph_core/swarm.py        @fantomx42

# Agents
/ralph_core/agents/         @fantomx42

# ASIC System
/ralph_core/asic/           @fantomx42

# Message Bus (critical)
/ralph_core/protocols/      @fantomx42

# Security-sensitive
/ralph_core/executor.py     @fantomx42
/sandbox/                   @fantomx42
/.github/workflows/         @fantomx42

# Documentation
/README.md                  @fantomx42
/CONTRIBUTING.md            @fantomx42
/SECURITY.md                @fantomx42

# Configuration
/ralph_loop.sh              @fantomx42
/.github/                   @fantomx42
```

## GitHub Projects

### Create Project Board

Navigate to **Projects > New project**:

**Project Name**: Ralph AI Roadmap

**Template**: Board

**Columns**:
1. **Backlog** - Ideas and future work
2. **To Do** - Prioritized next tasks
3. **In Progress** - Currently being worked on
4. **Review** - In PR review
5. **Done** - Completed

**Automation**:
- Move to "In Progress" when PR is opened
- Move to "Review" when PR is ready for review
- Move to "Done" when PR is merged or issue is closed

### Initial Project Items

Add cards for:
- [ ] Windows/macOS support
- [ ] Multi-language support
- [ ] Cloud deployment options
- [ ] Plugin system
- [ ] Benchmark suite
- [ ] IDE integrations

## GitHub Discussions

Enable and create categories:

Navigate to **Settings > Features > Enable Discussions**

**Categories**:
1. **Announcements** - Project updates (maintainers only)
2. **General** - General discussion about Ralph AI
3. **Ideas** - Feature ideas and brainstorming
4. **Q&A** - Questions and answers
5. **Show and Tell** - Share your Ralph AI projects
6. **ASIC Proposals** - Discuss new ASIC specialists
7. **Performance** - Performance optimization discussions
8. **Hardware** - Hardware setup and requirements

## Social Preview

Create a social preview image (1280x640 pixels):

Upload at: **Settings > General > Social preview**

Suggested content:
- Ralph AI logo/name
- "3-Tier Hierarchical Swarm System"
- "Autonomous AI Task Execution"
- Key tech: Ollama, Python, FastAPI

Tools for creation:
- Canva
- Figma
- [GitHub Social Preview Generator](https://github.com/apertureless/github-social-image-generator)

## Repository Insights

### Traffic Analysis

Monitor at: **Insights > Traffic**
- Track views and clones
- Identify popular content
- Measure growth

### Community Metrics

Monitor at: **Insights > Community**
- Check community health percentage
- Ensure all recommended files are present

Target: 100% community health score

## Visibility & Discoverability

### README Optimization

Ensure README includes:
- Clear description in first 3 lines
- Architecture diagram (visual)
- Installation instructions
- Usage examples
- Links to documentation
- Badges for build status, version, license

### GitHub Search Optimization

Optimize for GitHub search:
- Use keywords in repository description
- Add comprehensive topics
- Maintain active issues and PRs
- Regular commits and releases
- Complete documentation

### External Promotion

Consider:
- Blog post announcing Ralph AI
- Reddit posts (r/LocalLLaMA, r/opensource, r/selfhosted)
- Hacker News submission
- Twitter/X announcement
- YouTube demo video
- LinkedIn article

## Analytics & Monitoring

### GitHub Insights

Monitor:
- **Pulse**: Weekly activity summary
- **Contributors**: Contributor graph
- **Commits**: Commit frequency
- **Code frequency**: Additions/deletions over time
- **Network**: Fork network graph
- **Forks**: Fork count and usage

### External Tools

Consider integrating:
- **Shields.io**: More badges (downloads, stars, forks)
- **CodeCov**: Code coverage reporting
- **LGTM**: Code quality analysis
- **Libraries.io**: Dependency monitoring

## Checklist

Use this checklist when setting up the repository:

### Essential
- [ ] Update repository name to `ralph-ai`
- [ ] Set clear description
- [ ] Add topics/tags (minimum 10)
- [ ] Enable Discussions
- [ ] Enable Issues with templates
- [ ] Configure branch protection for `main`
- [ ] Create standard labels
- [ ] Add CODEOWNERS file
- [ ] Enable Dependabot
- [ ] Enable secret scanning
- [ ] Upload social preview image

### Recommended
- [ ] Create project board
- [ ] Set up CodeQL scanning
- [ ] Configure Dependabot version updates
- [ ] Create discussion categories
- [ ] Set up wiki homepage
- [ ] Add comprehensive README
- [ ] Create CONTRIBUTING.md
- [ ] Create CODE_OF_CONDUCT.md
- [ ] Create SECURITY.md
- [ ] Set up CI/CD workflow

### Optional
- [ ] Custom domain for GitHub Pages
- [ ] Sponsor button
- [ ] Release automation
- [ ] Code coverage integration
- [ ] Performance benchmarking CI
- [ ] Docker Hub integration
- [ ] Package registry publishing

## Maintenance

Regular maintenance tasks:

**Weekly**:
- Review and triage new issues
- Respond to discussions
- Merge approved PRs
- Update project board

**Monthly**:
- Review Dependabot PRs
- Update documentation
- Close stale issues
- Publish release notes

**Quarterly**:
- Review security advisories
- Update roadmap
- Analyze traffic and engagement
- Community health check

---

**Last Updated**: 2026-01-14

For questions about repository setup, open a [GitHub Discussion](https://github.com/fantomx42/friendly-enigma/discussions).
