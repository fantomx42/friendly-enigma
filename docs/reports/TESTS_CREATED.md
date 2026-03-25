# Wheeler Memory Unit Tests

This document summarizes the unit test files created for the Wheeler Memory project.

## Created Test Files

All test files are located in `/home/tristan/projects/wheeler-memory/tests/` and follow pytest conventions for pure computation testing (no disk I/O, no fixtures).

### 1. test_hashing.py (58 lines, 8 tests)
Tests for SHA-256 hashing and deterministic frame generation.

**Classes:**
- `TestTextToHex` (3 tests)
  - `test_text_to_hex_deterministic` — Same input produces identical hex output
  - `test_text_to_hex_different_inputs` — Different inputs produce different hex
  - `test_text_to_hex_is_hex_string` — Result is 64-char lowercase hex string

- `TestHashToFrame` (5 tests)
  - `test_hash_to_frame_shape` — Returns 64×64 array
  - `test_hash_to_frame_dtype` — Array dtype is float32
  - `test_hash_to_frame_range` — Values in [-1, 1]
  - `test_hash_to_frame_deterministic` — Same input → identical frames
  - `test_hash_to_frame_different_inputs` — Different texts → different frames

**Imports:**
- `hash_to_frame`, `text_to_hex` from `wheeler_memory.hashing`

---

### 2. test_temperature.py (144 lines, 18 tests)
Tests for temperature dynamics computation.

**Classes:**
- `TestComputeTemperature` (6 tests)
  - `test_compute_temperature_zero_hits` — 0 hits, just accessed ≈ 0.3
  - `test_compute_temperature_saturated_hits` — 10 hits, just accessed ≈ 1.0
  - `test_compute_temperature_half_life` — 10 hits, 7 days ago ≈ 0.5
  - `test_compute_temperature_two_weeks` — 0 hits, 14 days ago ≈ 0.075
  - `test_compute_temperature_accepts_datetime` — Accepts datetime objects
  - `test_compute_temperature_accepts_iso_string` — Accepts ISO-8601 strings

- `TestTemperatureTier` (5 tests)
  - `test_temperature_tier_hot` — 0.8 → "hot"
  - `test_temperature_tier_warm` — 0.4 → "warm"
  - `test_temperature_tier_cold` — 0.1 → "cold"
  - `test_temperature_tier_boundary_hot` — Exactly 0.6 → "hot"
  - `test_temperature_tier_boundary_warm` — Exactly 0.3 → "warm"

- `TestComputeWarmth` (4 tests)
  - `test_compute_warmth_at_t0` — Just applied → ≈0.05
  - `test_compute_warmth_at_t1d` — 1 day ago → ≈0.025 (half-life)
  - `test_compute_warmth_at_t10d` — 10 days ago → 0.0 (below floor)
  - `test_compute_warmth_accepts_iso_string` — Accepts ISO strings

- `TestEffectiveTemperature` (3 tests)
  - `test_effective_temperature_adds_warmth` — base + warmth = effective
  - `test_effective_temperature_capped_at_1` — High values capped at 1.0
  - `test_effective_temperature_no_warmth` — No warmth → equals base

**Imports:**
- Constants: `HALF_LIFE_DAYS`, `HIT_SATURATION`, `TIER_HOT`, `TIER_WARM`, `WARMTH_FLOOR`, `WARMTH_HALF_LIFE_DAYS`
- Functions: `compute_temperature`, `compute_warmth`, `effective_temperature`, `temperature_tier`

---

### 3. test_attention.py (159 lines, 18 tests)
Tests for attention model with variable tick rates.

**Classes:**
- `TestBudgetAnchors` (3 tests)
  - `test_budget_anchor_low` — Salience 0.0 → LOW tier
  - `test_budget_anchor_med` — Salience 0.5 → MED tier
  - `test_budget_anchor_high` — Salience 1.0 → HIGH tier

- `TestBudgetMonotonicity` (2 tests)
  - `test_budget_max_iters_nondecreasing` — Monotonic increase
  - `test_budget_threshold_nonincreasing` — Monotonic decrease

- `TestBudgetClamping` (2 tests)
  - `test_clamp_negative` — Negative → 0.0
  - `test_clamp_over_one` — Over 1.0 → 1.0

- `TestLabelConversion` (6 tests)
  - `test_label_low`, `test_label_medium`, `test_label_high`
  - `test_label_case_insensitive` — "HIGH" == "high"
  - `test_label_unknown_returns_default` — Unknown → SALIENCE_DEFAULT
  - `test_budget_label_property` — Budget.label property

- `TestTemperatureToSalience` (2 tests)
  - `test_salience_from_temperature_endpoints` — 0.0 → 0.1, 1.0 → 1.0
  - `test_salience_from_temperature_monotonic` — Monotonic increase

- `TestBackwardsCompat` (2 tests)
  - `test_backwards_compat_default_matches_explicit` — Default == 1000 iters
  - `test_default_budget_values` — Default threshold == 1e-4

- `TestHighSalienceDepth` (1 test)
  - `test_high_salience_uses_more_ticks` — High budget ≥ ticks than low

**Imports:**
- Functions: `compute_attention_budget`, `salience_from_label`, `salience_from_temperature`
- Constants: `SALIENCE_DEFAULT`, `SALIENCE_MAX_ITERS_{LOW,MED,HIGH}`, `SALIENCE_THRESHOLD_{LOW,MED,HIGH}`

---

### 4. test_oscillation.py (67 lines, 8 tests)
Tests for oscillation detection in cellular automata evolution.

**Classes:**
- `TestGetCellRoles` (4 tests)
  - `test_get_cell_roles_uniform_frame` — Uniform values → all roles 0
  - `test_get_cell_roles_local_max` — Center max → role 1 at center
  - `test_get_cell_roles_local_min` — Center min → role -1 at center
  - `test_get_cell_roles_shape` — Returns same shape as input

- `TestDetectOscillation` (4 tests)
  - `test_detect_oscillation_no_history` — Empty history → oscillating=False
  - `test_detect_oscillation_single_frame` — Single frame → oscillating=False
  - `test_detect_oscillation_static` — Identical frames → oscillating=False
  - `test_detect_oscillation_returns_dict_keys` — Has required dict keys

**Imports:**
- Functions: `detect_oscillation`, `get_cell_roles` from `wheeler_memory.oscillation`

---

### 5. test_polarity.py (57 lines, 8 tests)
Tests for dual-polarity encoding helpers.

**Classes:**
- `TestPolarWeight` (5 tests)
  - `test_polar_weight_zero_decays` — decay_count=0 → 1.0
  - `test_polar_weight_one_decay` → 0.7
  - `test_polar_weight_three_decays` → 0.7^3 ≈ 0.343
  - `test_polar_weight_legacy_field` — Backward compat with safe_recall_count
  - `test_polar_weight_no_count` — Missing field → defaults to 0

- `TestIsNeutralized` (3 tests)
  - `test_is_neutralized_fresh` — decay_count=0 → False
  - `test_is_neutralized_after_many_decays` — decay_count=10 → True
  - `test_polar_weight_decays_below_threshold` — Finds N where 0.7^N < 0.1

**Imports:**
- Constants: `POLAR_WEIGHT_DECAY` (0.7), `POLAR_DECAY_THRESHOLD` (0.1)
- Functions: `is_neutralized`, `polar_weight`

---

### 6. test_chunking.py (59 lines, 9 tests)
Tests for domain-based memory chunking.

**Classes:**
- `TestSelectChunk` (6 tests)
  - `test_select_chunk_code_keywords` → "code"
  - `test_select_chunk_science_keywords` → "science"
  - `test_select_chunk_hardware_keywords` → "hardware"
  - `test_select_chunk_daily_tasks_keywords` → "daily_tasks"
  - `test_select_chunk_meta_keywords` → "meta"
  - `test_select_chunk_general_default` → DEFAULT_CHUNK ("general")

- `TestSelectRecallChunks` (3 tests)
  - `test_select_recall_chunks_always_includes_general` — Always includes "general"
  - `test_select_recall_chunks_code_text` — Includes "code" and "general"
  - `test_select_recall_chunks_multiple_domains` — Multi-domain support

**Imports:**
- Constants: `DEFAULT_CHUNK` ("general")
- Functions: `select_chunk`, `select_recall_chunks`

---

### 7. test_dynamics.py (112 lines, 14 tests)
Tests for cellular automata dynamics engine.

**Classes:**
- `TestApplyCADynamics` (4 tests)
  - `test_apply_ca_dynamics_shape_preserved` → 64×64
  - `test_apply_ca_dynamics_range_preserved` → [-1, 1]
  - `test_apply_ca_dynamics_local_max_increases` → Center increases
  - `test_apply_ca_dynamics_local_min_decreases` → Center decreases

- `TestEvolveAndInterpret` (10 tests)
  - `test_evolve_result_keys` — Has required keys
  - `test_evolve_attractor_shape` → 64×64
  - `test_evolve_attractor_range` → [-1, 1]
  - `test_evolve_state_valid` → One of three states
  - `test_evolve_convergence_ticks_positive` ≥ 1
  - `test_evolve_convergence_ticks_respects_max_iters`
  - `test_evolve_deterministic` — Same input → same output
  - `test_evolve_typical_input_converges` → "CONVERGED"
  - `test_evolve_history_length` — Correct frame count
  - `test_evolve_first_frame_is_input` — History[0] == input

**Imports:**
- Functions: `apply_ca_dynamics`, `evolve_and_interpret` from `wheeler_memory.dynamics`
- Helper: `hash_to_frame` from `wheeler_memory.hashing`

---

## Summary Statistics

- **Total Test Files:** 7
- **Total Test Classes:** 20
- **Total Test Methods:** 83
- **Total Lines of Code:** 656

## Test Coverage

| Module | Function(s) Tested | Tests |
|--------|-------------------|-------|
| hashing | text_to_hex, hash_to_frame | 8 |
| temperature | compute_temperature, temperature_tier, compute_warmth, effective_temperature | 18 |
| attention | compute_attention_budget, salience_from_label, salience_from_temperature | 18 |
| oscillation | get_cell_roles, detect_oscillation | 8 |
| polarity | polar_weight, is_neutralized | 8 |
| chunking | select_chunk, select_recall_chunks | 9 |
| dynamics | apply_ca_dynamics, evolve_and_interpret | 14 |

## Running the Tests

```bash
cd "/home/tristan/projects/wheeler-memory"
pytest tests/test_hashing.py -v
pytest tests/test_temperature.py -v
pytest tests/test_attention.py -v
pytest tests/test_oscillation.py -v
pytest tests/test_polarity.py -v
pytest tests/test_chunking.py -v
pytest tests/test_dynamics.py -v

# Run all tests
pytest tests/ -v
```

## Design Principles

1. **Pure Computation:** All tests use only pure functions with no disk I/O
2. **No Fixtures Needed:** Tests are self-contained and don't depend on conftest.py
3. **Exact Signatures:** All function calls match source module signatures exactly
4. **Deterministic:** All tests are deterministic and produce consistent results
5. **Fast Execution:** No network calls, no external services, all in-memory
6. **Well Documented:** Each test has a docstring explaining what it validates

