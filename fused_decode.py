import torch
from torch.utils.cpp_extension import load as _load_ext
import os as _os
_here = _os.path.dirname(_os.path.abspath(__file__))
fused_decode_cuda = _load_ext(
    "fused_decode_cuda",
    sources=[_os.path.join(_here, "fused_decode.cpp"),
             _os.path.join(_here, "decode_kernel.cu")],
    verbose=True,
)

def sphkv_logits(q_all, cb_flat, tier_G, tier_g,
                 theta_codes, radius_codes, r_scales,
                 block_table, bits_table, ctx_lens, logits_out,
                 num_q_heads, num_kv_heads, kv_groups, num_tiers,
                 max_blocks, page_size, dh, G_max, cb_max, g_max,
                 sm_scale, max_ctx):
    fused_decode_cuda.forward(
        q_all.contiguous(), cb_flat.contiguous(),
        tier_G.contiguous(), tier_g.contiguous(),
        theta_codes.contiguous(), radius_codes.contiguous(),
        r_scales.contiguous(), block_table.contiguous(),
        bits_table.contiguous(), ctx_lens.contiguous(),
        logits_out.contiguous(),
        num_q_heads, num_kv_heads, kv_groups, num_tiers,
        max_blocks, page_size, dh, G_max, cb_max, g_max,
        sm_scale, max_ctx)
    return logits_out

def fused_decode(*a, **k):
    return sphkv_logits(*a, **k)
