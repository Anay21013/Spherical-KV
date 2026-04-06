import torch
from config import EPS

def compute_reuse_proxy(attn_weights):
    """
    attn_weigths: [B, L, H, T, S] -> Prefill attention weights
    Returns: Resue proxy per head
    """
    if attn_weights.dim() != 5:
        raise ValueError("attn_weights must be [B, L, H, T, S]")

    #FOR CALCULATING MEAN ATTENTION MASS PER (LAYER, HEAD)
    head_mass = attn_weights.mean(dim=(0, 3, 4))
    total = head_mass.sum(dim=1, keepdim=True) + EPS
    reuse = head_mass / total
    return reuse