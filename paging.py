import torch
from config import HEADER_BYTES   # = 8 (fixed fields only)


class PageHeader:

    def __init__(self, tier_id: int, count: int, r_scales: torch.Tensor):
        self.tier_id  = tier_id
        self.count    = count
        self.r_scales = r_scales   # [G]

    @property
    def G(self) -> int:
        return len(self.r_scales)

    @property
    def header_bytes(self) -> int:
        return HEADER_BYTES + self.G * 4

    def to_tensor(self) -> torch.Tensor:
        device = self.r_scales.device
        buf = torch.zeros(self.header_bytes, dtype=torch.uint8, device=device)

        buf[0] = self.tier_id & 0xFF
        buf[1] = self.count   & 0xFF
        # bytes 2-7: padding / future flags – left as zero

        scale_bytes = self.r_scales.to(torch.float32).view(torch.uint8)
        buf[HEADER_BYTES : HEADER_BYTES + len(scale_bytes)] = scale_bytes

        return buf