import math
import torch

from config import (
    HEAD_DIM,
    ALPHA_THETA, ALPHA_R,
    LAMBDA_THETA, LAMBDA_R,
    ETA, GAMMA,
)

SEGMENT_WEIGHTS = {
    0: 1.0,   # prefix
    1: 3,   # retrieved evidence -- up-weighted
    2: 1.2,   # recent suffix
}

_INV_SQRT_D = 1.0 / math.sqrt(HEAD_DIM)


def compute_distortion(
    token,
    tier,
    r_max=None,
    reuse=None,
    stability=None,
) -> float:
    if tier.tier_id == 0:
        return float(ETA)

    use_prefill_proxy = (reuse is not None and stability is not None
                         and r_max is not None)

    if use_prefill_proxy:
        return _prefill_proxy(token, tier, r_max, reuse, stability)
    else:
        dist = _decode_proxy(token, tier)
        if GAMMA > 0.0 and tier.tier_id != token.new_tier_id:
            dist += GAMMA
        return dist


def _prefill_proxy(token, tier, r_max, reuse, stability) -> float:
    u_hat  = float(reuse[token.layer, token.head])
    s_hat  = float(stability[token.layer, token.head])
    seg_w  = SEGMENT_WEIGHTS.get(token.segment_id, 1.0)

    w_theta = ALPHA_THETA * u_hat * seg_w
    w_r     = ALPHA_R * (1.0 - s_hat) * seg_w

    lam_th = LAMBDA_THETA.get(tier.tier_id, 0.10)
    lam_r  = LAMBDA_R.get(tier.tier_id, 0.01)

    r_hat_sum = float(token.r_groups.abs().sum())

    return w_theta * lam_th + w_r * (r_hat_sum * lam_r)


def _decode_proxy(token, tier) -> float:
    G_b    = tier.G
    G_fine = len(token.r_groups)
    r      = token.r_groups

    if G_b == G_fine:
        r_sum = float(r.abs().sum())
    elif G_b < G_fine:
        r_t   = r.float()
        r_sum = float((r_t[0::2] + r_t[1::2]).abs().sum())
    else:
        r_sum = float(r.abs().sum())

    lam_r  = LAMBDA_R.get(tier.tier_id,  0.01)
    lam_th = LAMBDA_THETA.get(tier.tier_id, 0.10)

    delta_i = _INV_SQRT_D * r_sum * (lam_r + lam_th)

    seg_w = SEGMENT_WEIGHTS.get(token.segment_id, 1.0)
    importance = max(float(token.omega), 0.1)
    return seg_w * importance * delta_i
