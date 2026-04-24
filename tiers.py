from config import BR, HEAD_DIM, B_META

class Tier:
    def __init__(self, tier_id: int, g: int, b_theta: int, dh: int,
                 br: int = BR, m: int = 1, b_meta: int = B_META):
        self.tier_id  = tier_id
        self.g        = g                        # group size
        self.b_theta  = b_theta                  # bits per codebook index
        self.dh       = dh
        self.br       = br                       # bits per group radius (always 8)
        self.m        = m                        # indices per group (always 1)
        self.G        = (dh // g) if g > 0 else 0  # number of groups
        self.b_meta   = b_meta                   # meta-bits per token (flags/offsets)

        self.b_radius = br
 
    def token_bits(self) -> int:
        if self.tier_id == 0:
            return 0
        return self.G * (self.br + self.m * self.b_theta) + self.b_meta
 
    @property
    def codebook_size(self) -> int:
        """Number of codewords: 2^b_theta."""
        return 2 ** self.b_theta if self.b_theta > 0 else 0
 
    def __repr__(self) -> str:
        return (f"Tier(id={self.tier_id}, g={self.g}, G={self.G}, "
                f"b_theta={self.b_theta}, bits={self.token_bits()})")
 
 
DROP_TIER = Tier(0, g=0, b_theta=0, dh=1, br=0)
 
 
def build_tiers(dh: int = HEAD_DIM):
    return [
        Tier(0, g=0,  b_theta=0, dh=dh, br=0),   # Drop  – no storage
        Tier(1, g=16, b_theta=6, dh=dh),           # b1 High – 112 bits, 14 B
        Tier(2, g=16, b_theta=4, dh=dh),           # b2 Mid  –  96 bits, 12 B
        Tier(3, g=32, b_theta=3, dh=dh),           # b3 Low  –  44 bits,  6 B
    ]