import math
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import config as _cfg
from spherical_kv_pipeline import (
    SphericalKVPipeline, _encode_keys_with_codebooks, nvtx_range,
)
from codebook_loader import get_codebook
from tiers import build_tiers


def reconstruct_dense_K_from_codes(
    r_codes:     torch.Tensor,   # [N, G] per-group radii
    theta_codes: torch.Tensor,   # [N, G] codebook indices
    codebooks:   torch.Tensor,   # [G, cb_size, g]
    group_size:  int,
    num_groups:  int,
    device:      torch.device,
) -> torch.Tensor:
    """
    Reconstruct dense K vectors from spherical codes.
    K_recon[i] = concat_g( r_codes[i,g] * codeword[g, theta_codes[i,g]] )
    Returns [N, dh] dense float32.
    """
    N = r_codes.shape[0]
    r_c  = r_codes.to(device)         # [N, G]
    th_c = theta_codes.to(device).long()  # [N, G]
    cb   = codebooks.to(device)       # [G, cb_size, g]

    # Gather codewords: [G, N, g]
    cw = cb[
        torch.arange(num_groups, device=device).unsqueeze(1),  # [G, 1]
        th_c.T,                                                 # [G, N]
    ]

    # Scale by radii: [G, N, g] * [G, N, 1]
    K_groups = cw * r_c.T.unsqueeze(-1)  # [G, N, g]

    # Rearrange to [N, G, g] -> [N, dh]
    K_dense = K_groups.permute(1, 0, 2).reshape(N, num_groups * group_size)

    return K_dense


def recon_attention_batched(
    pipeline:  SphericalKVPipeline,
    layer_idx: int,
    kv_head:   int,
    q_batch:   torch.Tensor,   # [num_q, dh]
    k_new:     torch.Tensor,   # [dh]
    v_new:     torch.Tensor,   # [dh]
) -> torch.Tensor:
    """
    SphKV-Recon: reconstruct dense K from codes, then standard dot-product.
    Same codes and tier assignments as full SphKV, but pays the densification tax.
    """
    key    = (layer_idx, kv_head)
    device = q_batch.device
    num_q  = q_batch.shape[0]
    q      = q_batch.float()
    kn     = k_new.float().to(device)
    vn     = v_new.float().to(device)
    dh     = q.shape[1]

    _bt_to_tid = {t.b_theta: t.tier_id
                  for t in pipeline.tiers if t.tier_id != 0}

    K_ctx_parts = []
    V_ctx_parts = []

    ph = pipeline.per_head_pages.get(key, [])
    for (pt, ptt, b_theta, n_tokens, V_tier, K_tier) in ph:
        tier_id  = _bt_to_tid.get(b_theta, 1)
        cb_lh    = get_codebook(pipeline.codebooks, layer_idx, kv_head, tier_id)
        tier_obj = pipeline.tiers[tier_id]

        if K_tier is not None:
            r_codes, th_codes = K_tier
            # Reconstruct dense K (the densification tax)
            K_dense = reconstruct_dense_K_from_codes(
                r_codes, th_codes, cb_lh,
                tier_obj.g, tier_obj.G, device
            )
            K_ctx_parts.append(K_dense)
        V_ctx_parts.append(V_tier)

    # Staging buffer
    stg_r     = pipeline.stg_r.get(key)
    stg_theta = pipeline.stg_theta.get(key)
    stg_V     = pipeline.stg_V.get(key)
    if stg_r is not None and stg_r.shape[0] > 0:
        b3 = pipeline.tiers[3]
        cb_b3 = get_codebook(pipeline.codebooks, layer_idx, kv_head, b3.tier_id)
        K_stg = reconstruct_dense_K_from_codes(
            stg_r, stg_theta, cb_b3, b3.g, b3.G, device
        )
        K_ctx_parts.append(K_stg)
        V_ctx_parts.append(stg_V)

    if K_ctx_parts:
        K_ctx = torch.cat(K_ctx_parts, dim=0)  # [total_ctx, dh]
        V_ctx = torch.cat(V_ctx_parts, dim=0)  # [total_ctx, dh]
    else:
        K_ctx = torch.zeros(0, dh, device=device)
        V_ctx = torch.zeros(0, dh, device=device)

    # Standard dot-product attention (reconstruct-then-dot)
    # ctx logits: [num_q, total_ctx]
    ctx_logits = (q @ K_ctx.T) / math.sqrt(dh) if K_ctx.shape[0] > 0 \
        else torch.zeros(num_q, 0, device=device)
    new_logits = (q @ kn) / math.sqrt(dh)  # [num_q]

    all_logits = torch.cat([ctx_logits, new_logits.unsqueeze(1)], dim=1)
    attn = torch.softmax(all_logits, dim=1)

    V_full = torch.cat([V_ctx, vn.unsqueeze(0)], dim=0)
    attn_out = attn @ V_full

    return attn_out.to(q_batch.dtype)



MODES = {
    "dense":          "Dense KV baseline (no SphericalKV)",
    "sphkv":          "Full SphericalKV (A0: ADA + RDR joint)",
    "sphkv_recon":    "SphKV-Recon (A1: same codes, reconstruct-then-dot)",
    "sphkv_angle":    "SphKV-AngleOnly (A2-B: ADA kernel, uniform tier, no RDR)",
    "sphkv_rd":       "SphKV-RDOnly (A2-C: RDR + reconstruct-then-dot)",
}


def run_mode_decode(
    mode:       str,
    model,
    pipeline:   Optional[SphericalKVPipeline],
    prefill_ids: torch.Tensor,
    n_warm:     int,
    n_meas:     int,
    device:     torch.device,
) -> Tuple[List[int], float, float]:
    """
    Run n_warm + n_meas decode steps in the specified mode.
    Returns (generated_ids, elapsed_seconds, hbm_bytes_proxy).
    """
    is_cuda = device.type == "cuda"
    current_ids = prefill_ids.to(device).clone()
    generated = []

    # Warmup
    for _ in range(n_warm):
        with torch.no_grad():
            out = model(input_ids=current_ids[:, -1:],
                        use_cache=False, return_dict=True)
        next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
        current_ids = torch.cat([current_ids, next_id], dim=-1)

    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        mem_before = torch.cuda.max_memory_allocated(device)
        t_start = torch.cuda.Event(enable_timing=True)
        t_end   = torch.cuda.Event(enable_timing=True)
        t_start.record()
    else:
        import time
        t_wall = time.perf_counter()

    # Measurement
    for _ in range(n_meas):
        with torch.no_grad():
            out = model(input_ids=current_ids[:, -1:],
                        use_cache=False, return_dict=True)
        next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
        generated.append(next_id.item())
        current_ids = torch.cat([current_ids, next_id], dim=-1)

    if is_cuda:
        t_end.record()
        torch.cuda.synchronize()
        elapsed = t_start.elapsed_time(t_end) / 1e3
        mem_after = torch.cuda.max_memory_allocated(device)
        hbm_proxy = (mem_after - mem_before) / max(n_meas, 1)
    else:
        import time
        elapsed = time.perf_counter() - t_wall
        hbm_proxy = 0.0

    return generated, elapsed, hbm_proxy