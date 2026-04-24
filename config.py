import torch

HEADER_BYTES = 8
PAGE_SIZE    = 128
EPS          = 1e-8
GROUP_SIZE   = 16
HEAD_DIM     = 128
NUM_GROUPS   = HEAD_DIM // GROUP_SIZE

BR = 8   # bits per group radius (fixed for all tiers)

GLOBAL_BUDGET_BITS = 5_000_000

SINK_TOKENS   = 4
RECENT_WINDOW = 32
 
LAMBDA_THETA = {1: 0.05, 2: 0.10, 3: 0.30}
LAMBDA_R     = {1: 0.01, 2: 0.01, 3: 0.01}
ETA          = 10.0   # drop penalty (large so drops are last resort)
 
EMA_BETA = 0.9   # smoothing factor β
EMA_R    = 8     # max-over-window width R (recent attention steps)
 
REFRESH_CADENCE = 16
 
UPGRADE_KU = 4096
 
# Asymmetric efficiency thresholds for hysteresis
RHO_DOWN = float("inf")
RHO_UP   = 0.002
 
# after a tier change we are not touching these tokens for the next "COOLDOWN_STEPS" steps
COOLDOWN_STEPS = 2

ALPHA_THETA = 1.0
ALPHA_R     = 1.0
BITS_PER_TOKEN = 34.9

# ── Per-token meta-bits (Algorithm 1: R(t) includes b_meta) ───────────
# Covers flags, offsets, tier-id per token in the page.
B_META = 2   # bits per token (conservative default)

# ── Anti-thrashing stickiness regularizer (C.4 optional) ──────────────
# D_stick_i(b) = D_i(b) + GAMMA * 1[b != b_i(t-)]
# Set to 0.0 to disable.
GAMMA = 0.001

# ── Lagrangian prices for per-layer/per-head caps (C.3) ───────────────
# T_PI: number of price-update iterations per controller update
LAGRANGE_T_PI      = 1
LAGRANGE_STEP_SIZE = 0.01
LAGRANGE_CLIP      = 1.0
# Per-layer budget cap as fraction of global budget / num_layers.
# Set to None to disable per-layer caps entirely.
PER_LAYER_CAP_FRACTION  = None   # e.g. 1.2 = 120% of fair share
PER_HEAD_CAP_FRACTION   = None   # e.g. 1.5 = 150% of fair share

# ── Page alignment overhead (App A.2) ─────────────────────────────────
# Extra bits per token to account for page headers, pointer table entries,
# and alignment padding.  Added to tier cost in budget accounting.
PAGE_ALIGNMENT_BITS_PER_TOKEN = 2