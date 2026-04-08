━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROJECT RALPH · VISION DOCUMENT
Wheeler Memory + Cortex Architecture
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CORE GOAL
─────────
Build a genuinely novel AI system that:
  • Has no LLM anywhere in the stack
  • Does not rely on anyone else's model
  • Learns by itself through experience
  • Knows what it doesn't know
  • Cannot hallucinate by architecture
  • Runs entirely on local silicon

CURRENT STATE (updated 2026-04-08)
──────────────────────────────────
  • Wheeler Memory = working CA-based memory system (v0.3.1)
  • Three-grid interference = default recall path (v0.3.1)
  • Cortex L1/L2/L3 = implemented and benchmarked
  • Context-RI encoder = distributional semantics (SimLex rho +0.101)
  • wheeler-bench quality score = 0.009 (target achieved)
  • MMLU = 25.9% with L3 classifier (near chance — limited by encoding)
  • SimLex-999 = +0.101 rho (first native positive semantic signal)
  • Next focus = CA dynamics that preserve/amplify distributional signal

MMLU ROADMAP
────────────
  24.3%  zero-shot cortex (no stored knowledge)        ✓ measured
  25.3%  cortex + learned facts (1,812 attractors)      ✓ measured
  25.9%  cortex + L3 classifier (11K params)            ✓ measured
  ~35%   reconstruction scoring (active research)
  higher richer encoder + deeper cortex features
  goal   meaningful above chance without any LLM

ARCHITECTURE PRINCIPLES
───────────────────────
  • Knowledge lives in Wheeler only
  • Cortex does geometry not language
  • Certainty is architectural not guessed
  • Forgetting is a feature not a bug
  • Every query is a write
  • Sleep consolidation keeps it lean
  • SCM measures coherence not truth
  • Truth comes from quality of training data

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PIPELINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                    [ QUERY ]
                        │
                        ▼
                   [ ENCODER ]
              blended (default):
            hippocampus n-gram + context-RI
            random projection → 64×64
                   │        │
                   ▼        ▼
        ┌──────────────┐  ┌──────────────┐
        │ HIPPOCAMPUS  │  │   LANGUAGE   │
        │   WHEELER    │  │   WHEELER    │
        │              │  │              │
        │ facts/memory │  │ English/     │
        │ dynamic      │  │ grammar      │
        │ always live  │  │ FROZEN       │
        └──────┬───────┘  └──────┬───────┘
               │                 │
               └────────┬────────┘
                        ▼
            ┌───────────────────────┐
            │  FRACTAL CUBE SPACE   │  MEMORY
            │  hierarchy of 64×64   │
            │  K grids × 4096 vals  │
            │  depth = specificity  │
            └───────────┬───────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │     SCM CHECK         │
            │                       │
            │  high SCM → KNOWN     │
            │  cortex reasons over  │
            │  existing attractors  │
            │                       │
            │  low SCM → UNKNOWN    │
            │  cortex switches to   │
            │  generative mode      │
            │  synthesizes new frame│
            │  from neighbors       │
            │  stores provisionally │
            └───────────┬───────────┘
                        │
            ┌───────────▼───────────┐
            │  L1 · GRAPH REASONING │
            │  top-K frames         │
            │  Pearson edges        │
            │  clusters · bridges   │
            ├───────────────────────┤
            │  L2 · CORTEX CA       │  CORTEX
            │  CA over graph output │
            │  dynamic consensus    │
            │  neighborhood settling│
            ├───────────────────────┤
            │  SCM SCORING          │
            │  T·S·E·I·P·NW·ERF    │
            │  confidence gate      │
            │  certainty determines │
            │  output confidence    │
            ├───────────────────────┤
            │  L3 · CLASSIFIER      │
            │  ~11K params          │
            │  swappable per task   │
            │  trained on Wheeler   │
            │  output only          │
            └───────────┬───────────┘
                        │
            ┌───────────▼───────────┐
            │   PARALLEL DECODER    │  DECODER
            │  K×4096 → tokens      │
            │  whole thought at once│
            │  no autoregressive    │
            │  no LLM               │
            └───────────┬───────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │       OUTPUT          │
            │                       │
            │  + confidence score   │
            │  + uncertainty flag   │
            │    if new frame was   │
            │    synthesized        │
            └───────────┬───────────┘
                        │
              (periodically during
               low activity / sleep)
                        │
                        ▼
        ┌───────────────────────────┐
        │       SLEEP PASS          │  CONSOLIDATION
        │                           │
        │  scan all attractors      │
        │  find overlapping basins  │
        │  merge similar memories   │
        │  prune low hit-count      │
        │  prune unconfirmed frames │
        │  strip episode → schema   │
        │  migrate hip → long-term  │
        └───────────────────────────┘
                        │
                        ▼
             Wheeler wakes smaller
             and smarter than before

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LEARNING MODES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  PASSIVE     using the system teaches it
              every query is a write
              hit counts reinforce useful attractors

  ACTIVE      system notices low SCM gaps
              seeks information to fill them
              autoresearch loop

  REFLECTIVE  system evaluates its own outputs
              internal consistency as feedback
              reinforces or weakens attractors

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FAILURE MODES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  PREVENTED   random hallucination
              (low SCM gates uncertain output)

  REMAINING   confidently wrong beliefs
              if training data was wrong
              (different problem, auditable)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NO LLM · NO AUTOREGRESSIVE · NO CLOUD
SPATIAL REASONING · PARALLEL DECODE
SLEEP CONSOLIDATION · SELF LEARNING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
