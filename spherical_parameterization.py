import torch
from config import EPS

def spherical_parameterize(K):
    """
    K: [N, dh]
    Returns:
        r: [N]
        phi: [N, dh-1]
    """
    N, dh = K.shape
    r = torch.norm(K, dim=-1) + EPS
    x = K / r.unsqueeze(-1)

    phi = []
    sq_sum = torch.zeros(N, device=K.device)

    for i in range(dh - 1):
        numerator = x[:, i]
        denom = torch.sqrt(1.0 - sq_sum + EPS)
        ratio = numerator / denom
        ratio = torch.clamp(ratio, -1 + 1e-7, 1 - 1e-7)
        angle = torch.acos(ratio)
        phi.append(angle)
        sq_sum += numerator ** 2

    phi = torch.stack(phi, dim=-1)

    return r, phi