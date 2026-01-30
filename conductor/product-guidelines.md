# Product Guidelines

## Prose Style & Tone
- **Technical & Precise (Primary)**: Documentation, architectural plans, and core agent logs must prioritize accuracy, clarity, and performance metrics. The complexity of the hierarchical swarm requires unambiguous communication.
- **Narrative Layer (Character-Driven)**: In user-facing interfaces (GUI/CLI dialogue), adopt a **Character-Driven** persona. Agents (Orchestrator, Engineer, Librarian) should interact as distinct entities within a team, providing a narrative context to their technical actions. This adds a layer of "team discussion" to the operation without sacrificing technical precision in the underlying logs.

## User Experience (UX) Principles
- **Adaptive Transparency (A/C Hybrid)**:
    - **Default View (Minimal Cognitive Load)**: Present high-level status updates, current objectives, and "traffic light" health indicators to avoid overwhelming the user.
    - **Deep Dive Mode (Transparency First)**: Provide a seamless toggle to expose the raw "thought stream," message bus traffic, and tool execution logs for auditing and debugging. The system must capture everything, even if it doesn't always show it.
- **Fail Fast & Explicitly (Core Philosophy)**: Errors are not failures; they are data. The UX must frame errors as "Diagnostic Events" that trigger the next iteration cycle. Never swallow an error; present it clearly as the input for the next attempt.

## Security & Safety Strategy
- **User-Defined Sovereignty**: Recognizing that security needs vary by user context, the system implements a **Tiered Security Model**:
    - **Strict Mode (Default)**: Sandboxed execution (Docker), strict local boundaries (no external calls), and mandatory human approval for filesystem modifications.
    - **Trust Mode**: Configurable by the user to relax specific constraints (e.g., allowing specific API domains, bypassing approval for non-destructive writes) based on their specific workflow needs.
- **The "Safety Checkpoint"**: A dedicated architectural component that intercepts all tool calls against the user's configured policy before execution.

## Visual Identity & Design
- **Phase 1: Character-Driven Interface**: The initial GUI will focus on visualizing the "Swarm" as a team of avatars with distinct roles. This builds the mental model of a hierarchical system.
- **Future Roadmap**:
    - **Industrial Mode**: A high-density, data-centric dashboard for power users.
    - **Adaptive Modes**: Context-aware UI shifts based on the active task complexity (e.g., "War Room" view for critical debugging vs. "Status Board" for background tasks).
