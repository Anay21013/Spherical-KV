import os
import torch
from torch.utils.cpp_extension import load

this_dir = os.path.dirname(os.path.abspath(__file__))

fused_decode_cuda = load(
    name="fused_decode_cuda",
    sources=[
        os.path.join(this_dir, "fused_decode.cpp"),
        os.path.join(this_dir, "decode_kernel.cu"),
    ],
    verbose=True,
)


def fused_decode(
    pages,          # uint8  packed byte stream
    pointer_table,  # [P, 3] int32
    positions,      # [P * page_size] int32, -1 for padding
    cos_table,      # [max_pos, dh] float32
    sin_table,      # [max_pos, dh] float32
    q,              # [num_q, dh]   float32, post-RoPE at current_pos
    codebooks,      # [G, 1<<b_theta, g] float32 unit-norm
    dh,
    groups,
    group_size,
    b_theta,
    page_size,
):
    """
    Returns: [num_q, num_pages * page_size] float32 logits.
    Padding slots (beyond per-page valid count) are zeroed.
    """
    assert q.dim() == 2, f"q must be [num_q, dh]; got {tuple(q.shape)}"
    num_q     = q.shape[0]
    num_pages = pointer_table.shape[0]

    logits = torch.zeros(
        (num_q, num_pages, page_size),
        device=q.device, dtype=torch.float32,
    )

    fused_decode_cuda.forward(
        pages.contiguous(),
        pointer_table.contiguous(),
        positions.contiguous(),
        cos_table.contiguous(),
        sin_table.contiguous(),
        q.contiguous(),
        codebooks.contiguous(),
        logits,
        dh, groups, group_size, b_theta, page_size,
    )
    return logits.view(num_q, num_pages * page_size)