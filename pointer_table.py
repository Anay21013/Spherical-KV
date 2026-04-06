import torch


class PointerTable:
    def __init__(self):
        self.page_offsets = []
        self.theta_ptrs = []
        self.radius_ptrs = []

    def add_page(self, page_offset, theta_offset, radius_offset):
        self.page_offsets.append(page_offset)
        self.theta_ptrs.append(theta_offset)
        self.radius_ptrs.append(radius_offset)

    def to_tensor(self):
        return torch.tensor(
            list(zip(self.page_offsets,
                     self.theta_ptrs,
                     self.radius_ptrs)),
            dtype=torch.int32
        )