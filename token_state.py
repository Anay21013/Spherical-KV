from collections import deque
import torch
from config import NUM_GROUPS, EMA_BETA, EMA_R, COOLDOWN_STEPS

class TokenState:

    __slots__ = (
        "layer", "head", "index",
        "r", "phi",
        "r_groups", "omega", "_attn_window",
        "segment_id", "age",
        "prev_tier_id", "new_tier_id",
        "protected", "cooldown",
    )

    def __init__(
        self,
        layer,
        head,
        index,
        r,
        phi,
        segment_id,
        age,
        prev_tier_id=0,
        protected=False,
        r_groups=None,
        omega=1.0,
    ):
        self.layer = layer
        self.head  = head
        self.index = index
        self.r     = r        
        self.phi   = phi     

        #per-group radii & importance weight
        if r_groups is not None:
            self.r_groups = r_groups
        else:
            self.r_groups = torch.full(
                (NUM_GROUPS,), float(r) / max(NUM_GROUPS, 1),
                dtype=torch.float32,
            )
        self.omega        = float(omega)
        self._attn_window = deque(maxlen=EMA_R)

        self.segment_id   = int(segment_id)
        self.age          = int(age)
        self.prev_tier_id = int(prev_tier_id)
        self.new_tier_id  = int(prev_tier_id)
        self.protected    = protected
        self.cooldown = 0

    def assign_tier(self, tier_id):
        """Set the new tier and start the cooldown window."""
        if int(tier_id) != self.new_tier_id:
            self.cooldown = COOLDOWN_STEPS
        self.new_tier_id = int(tier_id)

    def assign_tier_protected(self, tier_id):
        """Assign tier for protected tokens - no cooldown needed."""
        self.new_tier_id = int(tier_id)

    def commit_tier(self):
        self.prev_tier_id = self.new_tier_id

    @property
    def is_frozen(self):
        """True if this token is in its cooldown window (C.4)."""
        return self.cooldown > 0

    def tick_cooldown(self):
        """Decrement cooldown counter once per controller update."""
        if self.cooldown > 0:
            self.cooldown -= 1

    #EMA
    def record_attn(self, attn_weight):
        self._attn_window.append(attn_weight)
        w_max = max(self._attn_window)
        self.omega = EMA_BETA * self.omega + (1.0 - EMA_BETA) * w_max

    def update_omega(self, attn_weight, beta = EMA_BETA):
        self.record_attn(attn_weight)

    def __repr__(self):
        return (f"TokenState(l={self.layer}, h={self.head}, i={self.index}, "
                f"tier={self.new_tier_id}, ω={self.omega:.3f}, "
                f"cd={self.cooldown}, prot={self.protected})")