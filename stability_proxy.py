import torch
from config import EPS

def compute_margin(logits):
    """
    logits: [B, V]
    Returns scalar mean margin
    """
    top2 = torch.topk(logits, 2, dim=-1).values
    margin = top2[:, 0] - top2[:, 1]
    return margin.mean()

def compute_stability_proxy(head_outputs, logits):
    """
    head_outputs: [B, L, H, d_h]
    logits: [B, V]

    Returns:
        stability[L, H]
    """
    margin = compute_margin(logits)
    norms = head_outputs.pow(2).sum(dim=-1).mean(dim=0)  # [L,H]
    instability = 1.0 / (margin + EPS)
    stability = instability * norms
    stability = stability / (stability.sum(dim=1, keepdim=True) + EPS)

    return stability