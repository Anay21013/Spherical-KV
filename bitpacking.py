import torch

import torch

def pack_bits(values, bits):
    assert values.dtype == torch.int32

    n = values.numel()
    total_bits  = n * bits
    total_bytes = (total_bits + 7) // 8

    out = torch.zeros(total_bytes, dtype=torch.uint8, device=values.device)

    bit_idx    = torch.arange(bits, device=values.device)         
    bit_matrix = ((values.unsqueeze(1) >> bit_idx) & 1).bool()  

    global_bits = (
        torch.arange(n, device=values.device).unsqueeze(1) * bits
        + bit_idx
    )                                                               

    byte_idx  = (global_bits >> 3).long().reshape(-1)              
    bit_shift = (global_bits &  7).long().reshape(-1)              
    mask      = bit_matrix.reshape(-1)                             

    contributions = (mask.to(torch.uint8) << bit_shift.to(torch.uint8))
    out.scatter_add_(0, byte_idx, contributions)

    return out
