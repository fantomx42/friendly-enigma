# Wheeler Memory Architecture

## Chunked Memory Architecture

Memories are routed to domain-specific chunks via keyword matching — inspired by how different cortical regions handle different domains:

```
~/.wheeler_memory/
  chunks/
    code/           attractors/ bricks/ index.json metadata.json
    hardware/       ...
    daily_tasks/    ...
    science/        ...
    meta/           ...
    general/        ...  (default catch-all)
  rotation_stats.json
```

Routing is automatic: `"fix the python debug error"` → **code**, `"buy groceries and schedule dentist"` → **daily_tasks**, `"something random"` → **general**.

On recall, the system searches across all matching chunks plus general, merging results by similarity.

## The Brick

Every memory stores its full temporal evolution as a "brick" — a 3D structure (width × height × ticks) that you can scrub through like a video timeline:

```
Tick 0:  Noisy seed     → ████░░██░░██
Tick 20: Forming        → █░░█░░░█░░░█
Tick 43: Converged      → █░░█░░░█░░░█  (frozen attractor)
```

This gives you complete debuggability — you can see exactly how and when each memory formed.

## Rotation Retry

When the initial CA evolution lands in a bad attractor basin, the system rotates the seed frame (90°/180°/270°) and retries. Rotation changes the neighbor topology, creating a different dynamical trajectory from the same information. Per-angle success statistics are tracked for future optimization.

## Temperature Dynamics

Every memory has a **temperature** reflecting how recently and frequently it's been accessed. Temperature is computed lazily on recall and listing — no background processes.

```
temp = base_from_hits × decay_from_time

base_from_hits = min(1.0, 0.3 + 0.7 × (hit_count / 10))
decay_from_time = 2 ^ (-days_since_last_access / 7.0)
```

**Tiers:**

| Tier | Range | Meaning |
|------|-------|---------|
| hot  | >= 0.6 | Frequently accessed, recent |
| warm | >= 0.3 | Default for new or moderately accessed |
| cold | < 0.3  | Stale, candidate for archival |

New memories start at temperature 0.3 (warm). Each recall bumps hit_count and resets last_accessed, keeping active memories hot. Unused memories decay with a 7-day half-life.
