import torch
from torch.utils.cpp_extension import load
import os

this_dir = os.path.dirname(os.path.abspath(__file__))

fused_decode_cuda = load(
    name="fused_decode_cuda",
    sources=[
        os.path.join(this_dir, "fused_decode.cpp"),
        os.path.join(this_dir, "decode_kernel.cu"),
    ],
    verbose=True
)


def fused_decode(
    pages,
    pointer_table,
    q,              # [num_q, dh]  — ALWAYS 2D
    codebooks,
    dh,
    groups,
    group_size,
    b_theta,
    page_size
):
    """
    q must be 2D: [num_q, dh].
    Returns [num_q, num_pages * page_size].
    """
    num_q     = q.shape[0]
    num_pages = pointer_table.shape[0]

    logits = torch.zeros(
        (num_q, num_pages, page_size),
        device=q.device,
        dtype=torch.float32
    )

    fused_decode_cuda.forward(
        pages, pointer_table, q, codebooks, logits,
        dh, groups, group_size, b_theta, page_size
    )

    return logits.view(num_q, num_pages * page_size)