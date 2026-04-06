import torch
import math

def quantize_theta(phi, bits):
    max_val = 2**bits - 1
    return torch.clamp(
        (phi / math.pi * max_val).round(),
        0,
        max_val
    ).to(torch.int32)

def quantize_radius(r, scale):
    """
    symmetric int8 quantization
    """
    codes = torch.clamp(
        torch.round(r / scale * 127),
        -127,
        127
    ).to(torch.int8)

    return codes