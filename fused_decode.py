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

def sphkv_decode(q_all, cb_flat, tier_G, tier_g,
                 theta_codes, radius_codes, r_scales, v_pool,
                 block_table, bits_table, ctx_lens, out,
                 partial_scratch,
                 num_q_heads, num_kv_heads, kv_groups, num_tiers,
                 max_blocks, page_size, dh, G_max, cb_max, g_max,
                 sm_scale, max_ctx):
    fused_decode_cuda.forward(
        q_all, cb_flat, tier_G, tier_g,
        theta_codes, radius_codes, r_scales, v_pool,
        block_table, bits_table, ctx_lens, out, partial_scratch,
        num_q_heads, num_kv_heads, kv_groups, num_tiers,
        max_blocks, page_size, dh, G_max, cb_max, g_max,
        sm_scale, max_ctx)
    return out

def sphkv_encode_append(k_post, v_new, decode_cb,
                        theta_codes, radius_codes, r_scales, v_pool,
                        pids, slot, G, g, C, G_max, dh, page_size,
                        is_new_page):
    fused_decode_cuda.encode_append(
        k_post, v_new, decode_cb,
        theta_codes, radius_codes, r_scales, v_pool,
        pids, slot, G, g, C, G_max, dh, page_size, is_new_page)

# Backwards compat
def sphkv_logits(*a, **k):
    return sphkv_decode(*a, **k)
def fused_decode(*a, **k):
    return sphkv_decode(*a, **k)
