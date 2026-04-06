"""
At prefill there is no attention history, so distortion is estimated from
head-level scalars (reuse proxy û, stability proxy ŝ) and the segment the
token belongs to (prefix / retrieved / recent):

    w_theta_i = ALPHA_THETA * û_{l,h} * ω_seg(i)
    w_r_i     = ALPHA_R    * (1 - ŝ_{l,h}) * ω_seg(i)
    D_i(t)    = w_theta_i * ε_theta(t)  +  w_r_i * ε_r(t)

where:
    ε_theta(t) = π / 2^b_theta(t)   (angle quantisation error at tier t)
    ε_r(t)     = r_max / 2^b_r(t)   (radius quantisation error at tier t)
    ω_seg      = {prefix: 1.0, retrieved: 1.5, recent: 1.2}  (fixed)
    û, ŝ       = per-head scalars from prefill attention statistics

DECODE
────────────────────────
During decode, per-token attention history is available via the EMA weight
ω_i.  The proxy switches to a radius-magnitude model:

    δ_i(b) = (1/√d) · Σ_j |r̂_j^(i)| · (λ_r(b) + λ_θ(b))
    D_i(b) = ω_i · δ_i(b)

where r̂_j^(i) are the stored per-group radii for token i and the λ
coefficients are tier-specific (calibrated in inference_config.py).

Which proxy is used when
─────────────────────────
At PREFILL:  token.omega == 1.0 (no history).  We detect this and use the
             Algorithm 1 proxy if reuse/stability tensors are provided.

At DECODE:   token.omega has been updated by record_attn().  We use the
             Appendix C.2 proxy.

compute_distortion() accepts both calling conventions:
  compute_distortion(token, tier, r_max, reuse, stability)   ← prefill
  compute_distortion(token, tier)                            ← decode
"""

import math
import torch

from config import (
    HEAD_DIM,
    ALPHA_THETA, ALPHA_R,
    LAMBDA_THETA, LAMBDA_R,
    ETA,
)

SEGMENT_WEIGHTS = {
    0: 1.0,   # prefix
    1: 1.5,   # retrieved evidence — up-weighted
    2: 1.2,   # recent suffix
}

# _INV_SQRT_D = 1.0 / math.sqrt(HEAD_DIM)


def _eps_theta(b_theta: int) -> float:
    """Angle quantisation error: π / 2^b_theta (= π when b_theta=0)."""
    return math.pi if b_theta == 0 else math.pi / (2 ** b_theta)


def _eps_r(r_max: float, b_radius: int) -> float:
    """Radius quantisation error: r_max / 2^b_radius (= r_max when b_radius=0)."""
    return r_max if b_radius == 0 else r_max / (2 ** b_radius)


def compute_distortion(
    token,
    tier,
    r_max=None,
    reuse=None,
    stability=None,
) -> float:
    """
    Unified distortion proxy — selects Algorithm 1 (prefill) or C.2 (decode)
    based on whether reuse/stability tensors are provided.

    Parameters
    ----------
    token     : TokenState
    tier      : Tier  (tier_id == 0 → drop penalty ETA)
    r_max     : float  max radius across tokens (prefill only)
    reuse     : Tensor [L, H]  û head-level reuse proxy (prefill only)
    stability : Tensor [L, H]  ŝ head-level stability proxy (prefill only)
    """

    if tier.tier_id == 0:
        return float(ETA)

    use_prefill_proxy = (reuse is not None and stability is not None
                         and r_max is not None)

    if use_prefill_proxy:
        return _prefill_proxy(token, tier, r_max, reuse, stability)
    else:
        return _decode_proxy(token, tier)


def _prefill_proxy(token, tier, r_max, reuse, stability) -> float:
    """
    D_i(t) = w_theta * eps_theta(t) + w_r * eps_r(t)

    where:
        w_theta = ALPHA_THETA * û_{l,h} * ω_seg(i)
        w_r     = ALPHA_R    * (1 - ŝ_{l,h}) * ω_seg(i)
    """
    u_hat  = float(reuse[token.layer, token.head])
    s_hat  = float(stability[token.layer, token.head])
    seg_w  = SEGMENT_WEIGHTS.get(token.segment_id, 1.0)

    w_theta = ALPHA_THETA * u_hat * seg_w
    w_r     = ALPHA_R * (1.0 - s_hat) * seg_w

    e_theta = _eps_theta(tier.b_theta)
    # e_r     = _eps_r(r_max, tier.br)        # tier.br = 8 for all non-drop tiers
    r_hat_sum = float(token.r_groups.abs().sum())  # token-specific magnitude
    e_r = r_hat_sum / (2 ** tier.br) 

    return w_theta * e_theta + w_r * e_r


def _decode_proxy(token, tier) -> float:
    """
    Appendix C.2.

    D_i(b) = ω_i · δ_i(b)
    δ_i(b) = (1/√d) · Σ_j |r̂_j| · (λ_r(b) + λ_θ(b))
    """
    G_b    = tier.G
    G_fine = len(token.r_groups)
    r      = token.r_groups

    if G_b == G_fine:
        r_sum = float(r.abs().sum())
    elif G_b < G_fine:
        # Coarsen: sum adjacent pairs (e.g. b3: G_b=4, G_fine=8)
        r_t   = r.float()
        r_sum = float((r_t[0::2] + r_t[1::2]).abs().sum())
    else:
        r_sum = float(r.abs().sum())

    lam_r  = LAMBDA_R.get(tier.tier_id,  0.01)
    lam_th = LAMBDA_THETA.get(tier.tier_id, 0.10)

    _INV_SQRT_D = 1.0 / math.sqrt(token.r_groups.shape[0] * tier.g)
    delta_i = _INV_SQRT_D * r_sum * (lam_r + lam_th)
    return token.omega * delta_i