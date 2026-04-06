import torch

from config  import HEADER_BYTES
from paging  import PageHeader
from bitpacking import pack_bits


def build_page(
    r_groups:    torch.Tensor,   
    theta_codes: torch.Tensor,   
    tier,
    segment_id:  int,
) -> torch.Tensor:
    N, G  = r_groups.shape
    device = r_groups.device

    r_scale = r_groups.abs().max(dim=0).values       

    safe_scale = r_scale.clamp(min=1e-8).unsqueeze(0)           # [1, G]
    r_codes    = (r_groups / safe_scale * 127).round()
    r_codes    = r_codes.clamp(-127, 127).to(torch.int8)        # [N, G]

    packed_theta = pack_bits(
        theta_codes.flatten().to(torch.int32),
        tier.b_theta,
    )                                                            # [ceil(N*G*b_theta/8)]

    packed_r = r_codes.flatten().view(torch.uint8)              # [N*G]

    header = PageHeader(tier.tier_id, N, r_scale).to_tensor()  # [H+G*4]

    page = torch.cat([header, packed_theta, packed_r])
    return page


def build_page_from_codes(
    r_groups,
    theta_codes,
    tier,
    segment_id,
):
    return build_page(r_groups, theta_codes, tier, segment_id)