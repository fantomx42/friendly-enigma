"""Centralized tunable parameters for Wheeler Memory.

This is the single file to edit during autoresearch-style parameter tuning.
All tunable constants are defined here; modules import from this file so that
changing a value here propagates everywhere automatically.

Parameter guide
---------------
CA_DYNAMICS
  MAX_PUSH_STRENGTH    [0.20 .. 0.50]  How forcefully local max/min cells pull
                                        toward ±1.  Higher → sharper attractors.
  SLOPE_FLOW_STRENGTH  [0.10 .. 0.30]  How strongly slope cells flow toward
                                        their max neighbor.  Higher → faster mixing.

ATTENTION (controls CA tick budget per memory)
  SALIENCE_*_LOW/MED/HIGH  Piecewise boundaries for the variable-tick model.
  Salience 0.5 → MED (the default path for most store/recall operations).

TEMPERATURE (memory lifecycle)
  HALF_LIFE_DAYS   [3 .. 14]  Recency decay; 7 = halves every week.
  HIT_SATURATION   [5 .. 20]  Accesses until temperature plateaus at 1.0.

WARMING (spreading activation)
  ASSOCIATION_THRESHOLD  [0.40 .. 0.70]  Min Pearson r to form a store-time edge.
  WARMTH_HOP1 / HOP2     Boost applied at hop distance 1 and 2 on recall.
"""

# ---------------------------------------------------------------------------
# CA Dynamics
# ---------------------------------------------------------------------------
MAX_PUSH_STRENGTH = 0.30       # Local max/min pull toward ±1
SLOPE_FLOW_STRENGTH = 0.20     # Gradient descent rate for slope cells

# ---------------------------------------------------------------------------
# Attention / salience (variable tick rates)
# ---------------------------------------------------------------------------
SALIENCE_DEFAULT = 0.5

SALIENCE_MAX_ITERS_LOW = 200
SALIENCE_MAX_ITERS_MED = 1000
SALIENCE_MAX_ITERS_HIGH = 3000

SALIENCE_THRESHOLD_LOW = 5e-4
SALIENCE_THRESHOLD_MED = 1e-4
SALIENCE_THRESHOLD_HIGH = 1e-6

# ---------------------------------------------------------------------------
# Temperature / memory lifecycle
# ---------------------------------------------------------------------------
HALF_LIFE_DAYS = 7.0
HIT_SATURATION = 10

TIER_HOT = 0.6
TIER_WARM = 0.3
TIER_FADING = 0.05          # Below this → brick eligible for deletion
TIER_DEAD = 0.01            # Below this → full eviction

MAX_ATTRACTORS = 10_000     # Global capacity across all chunks
EVICTION_RATIO = 0.10       # Remove bottom 10 % when over capacity
MIN_AGE_DAYS = 1.0          # Never evict memories younger than this

# ---------------------------------------------------------------------------
# Sleep consolidation
# ---------------------------------------------------------------------------
CONSOLIDATION_DELTA_WARM = 0.02   # Mean abs delta threshold for warm bricks
CONSOLIDATION_DELTA_COLD = 0.05   # Mean abs delta threshold for cold bricks
CONSOLIDATION_ROLE_WARM = 0.05    # Role-change fraction threshold for warm bricks
CONSOLIDATION_ROLE_COLD = 0.10    # Role-change fraction threshold for cold bricks
CONSOLIDATION_MIN_FRAMES = 3      # Minimum frames to keep: seed + 1 keyframe + attractor
CONSOLIDATION_MIN_HISTORY = 5     # Don't consolidate bricks with fewer frames

# ---------------------------------------------------------------------------
# Warming / association
# ---------------------------------------------------------------------------
ASSOCIATION_THRESHOLD = 0.5       # Minimum Pearson r for a store-time association edge
WARMTH_HALF_LIFE_DAYS = 1.0       # Fast decay — warmth is short-term priming
WARMTH_HOP1 = 0.05                # Boost for direct neighbors
WARMTH_HOP2 = 0.025               # Boost for neighbors-of-neighbors
MAX_WARMTH = 0.15                 # Cap to prevent runaway accumulation
WARMTH_FLOOR = 0.001              # Below this, warmth is garbage-collected
CROSS_CHUNK_DECAY = 0.5           # Warmth decays by half when crossing a chunk boundary
