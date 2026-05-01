# Plans

Active plans live at the top level. Historical plans (shipped, superseded, or
unstarted) are kept under `archive/` for reference.

## Active

- `recall_migration_audit.csv` — Per-call-site classification of `recall_memory` vs
  `recognize`/`reconstruct_from_seed`, produced for the v0.3.6 two-tier recall
  migration. Each row tracks classification (`RECOGNITION_ONLY` / `RECONSTRUCTION` /
  `BOTH`), whether it was migrated this round, and the rationale for any deferred
  sites. Use as the checklist when the next round of migrations is ready.

## Archive

- `archive/reconstruction_scoring.md` — Plan for "Reconstruction Scoring (It From Bit)".
  Partly shipped: `wheeler-mmlu --mode recall-text` is the answer-letter extraction
  variant. The broader scoring vision is superseded by the v0.3.6 two-tier API.
- `archive/ternary_dynamics_and_daydream.md` — Two ideas in one doc. "Ternary
  dynamics" shipped (`wheeler-mmlu --mode ternary` and friends). "Daydreaming" — an
  offline replay/consolidation pass — has not shipped; the closest current work is
  the per-basin Temporal-Stability EMA in v0.3.6 (recall-driven, not offline).
- `archive/fractal_cube_address_space.md` — Theoretical exploration of replacing the
  flat 64×64 grid with a fractal cube address space. No implementation traces in the
  repo; treated as parking-lot theory.
- `archive/session_2026_03_17.md` — Working notes from the 2026-03-17 session that
  generated the four other plan files.
- `archive/priority.md` — Meta-index that prioritised the 2026-03-17 plans. Superseded
  by what actually shipped in v0.3.1–v0.3.6.
- `archive/planned-theories-test/` — Two markdown files cataloguing 12 theoretical
  models considered on 2026-03-17. Some shipped (trajectory similarity, ternary),
  others remain open ideas.
