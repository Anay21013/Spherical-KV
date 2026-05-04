"""
SphericalKV logit kernel + PyTorch V accumulation.

Split approach:
  1. CUDA kernel: massively parallel logit computation (1 thread per token)
  2. PyTorch softmax (optimized CUDA kernel)
  3. PyTorch einsum V matmul (cuBLAS, all SMs)
"""
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


def sphkv_logits(
    q_lut, theta_codes, radius_codes, r_scales,
    block_table, bits_table, ctx_lens, logits_out,
    num_q_heads, num_kv_heads, kv_groups, num_tiers,
    max_blocks, page_size, G_max, cb_size_max,
    sm_scale, max_ctx,
):
    """Launch massively parallel logit kernel."""
    fused_decode_cuda.forward(
        q_lut.contiguous(),
        theta_codes.contiguous(),
        radius_codes.contiguous(),
        r_scales.contiguous(),
        block_table.contiguous(),
        bits_table.contiguous(),
        ctx_lens.contiguous(),
        logits_out.contiguous(),
        num_q_heads, num_kv_heads, kv_groups, num_tiers,
        max_blocks, page_size, G_max, cb_size_max,
        sm_scale, max_ctx,
    )
    return logits_out


# Keep old name for pipeline's _fused_available probe
def fused_decode(*args, **kwargs):
    return sphkv_logits(*args, **kwargs)
