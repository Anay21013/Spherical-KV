import heapq
from collections import defaultdict
from typing import List, Dict, Tuple
 
import torch
import config as _cfg
from config import ( HEADER_BYTES, PAGE_SIZE, NUM_GROUPS,
    SINK_TOKENS, RECENT_WINDOW,
    COOLDOWN_STEPS, UPGRADE_KU, RHO_UP,
)
from distortion_proxy import compute_distortion
from pointer_table import PointerTable
 
 
 
def _build_next_tier_map(tiers):
    """Map each tier_id to the next cheaper tier (for sequential downgrade)."""
    # tiers = [drop, b1, b2, b3]  indexed by tier_id
    nondrop = [t for t in tiers if t.tier_id != 0]
    nondrop.sort(key=lambda t: -t.token_bits())  # most expensive first: b1,b2,b3
    nxt = {}
    for i, t in enumerate(nondrop):
        nxt[t.tier_id] = nondrop[i + 1] if i + 1 < len(nondrop) else tiers[0]
    return nxt
 
 
def _classify(tokens, T: int, sink_n: int, recent_w: int, prefill_len: int = 0):
    """
    We have 3 types of tokens -> sink (always pinned at highest), recent (stored at lowest), controllable (these are the ones that get downgraded/upgraded)
    """
    sinks, recents, ctrl = [], [], []
    for tok in tokens:
        if tok.protected or tok.index < sink_n:
            sinks.append(tok)
        elif tok.index >= T - recent_w or (prefill_len > 0 and tok.index >= prefill_len):
            recents.append(tok)
        else:
            ctrl.append(tok)
    return sinks, recents, ctrl
 
 
def _seq_len(tokens) -> int:
    return max(t.index for t in tokens) + 1 if tokens else 0
 
 
class _DCache:
    def __init__(self, r_max=None, reuse=None, stability=None):
        self._c: Dict[Tuple, float] = {}
        self.r_max     = r_max
        self.reuse     = reuse
        self.stability = stability
 
    def get(self, tok, tier) -> float:
        key = (id(tok), tier.tier_id)
        if key not in self._c:
            self._c[key] = compute_distortion(
                tok, tier,
                r_max=self.r_max,
                reuse=self.reuse,
                stability=self.stability,
            )
        return self._c[key]
 
    def rho(self, tok, from_t, to_t) -> float:
        dD = self.get(tok, to_t) - self.get(tok, from_t)
        dC = from_t.token_bits() - to_t.token_bits()
        return dD / max(dC, 1)
 

 
def _build_downgrade_heap(ctrl_tokens, init_tier, next_tier_map, dc: _DCache):
    """
    Push the first downgrade move for every controllable token onto a min-heap.
    Entry: (rho, layer, head, index, tok_id, tok, from_tier, to_tier)
    Tie-break fields (layer, head, index) ensure deterministic ordering (C.3).
    """
    heap = []
    for tok in ctrl_tokens:
        to_t = next_tier_map.get(init_tier.tier_id)
        if to_t is None:
            continue
        rho = dc.rho(tok, init_tier, to_t)
        heapq.heappush(heap,
            (rho, tok.layer, tok.head, tok.index, id(tok), tok, init_tier, to_t))
    return heap
 
 
def _greedy_downgrade(
    heap, cur_tier: dict, next_tier_map, dc: _DCache,
    current_bits: int, budget: int,
    no_freeze: bool = False,
) -> int:
    while current_bits > budget and heap:
        rho, _l, _h, _i, tok_id, tok, from_t, to_t = heapq.heappop(heap)

        if cur_tier.get(tok_id) is not from_t:
            continue
        if not no_freeze and tok.is_frozen:
            continue
 
        # Apply downgrade
        delta = from_t.token_bits() - to_t.token_bits()
        if no_freeze:
            tok.assign_tier_protected(to_t.tier_id)
        else:
            tok.assign_tier(to_t.tier_id)            
        cur_tier[tok_id] = to_t
        current_bits -= delta

        next_t = next_tier_map.get(to_t.tier_id)
        if next_t is not None:
            rho2 = dc.rho(tok, to_t, next_t)
            heapq.heappush(heap,
                (rho2, tok.layer, tok.head, tok.index, tok_id, tok, to_t, next_t))
 
    return current_bits
 

 
def _greedy_upgrade(ctrl_tokens, cur_tier: dict, prev_tier_map, dc: _DCache,
                    current_bits: int, budget: int):
    """
    If budget has slack, upgrade top-UPGRADE_KU tokens by omega
    Skips frozen tokens.
    Returns updated current_bits.
    """
    slack = budget - current_bits
    if slack <= 0:
        return current_bits
 
    # Rank candidates by omega (descending importance)
    candidates = [tok for tok in ctrl_tokens
                  if not tok.is_frozen
                  and prev_tier_map.get(cur_tier[id(tok)].tier_id) is not None]
    candidates.sort(key=lambda t: -t.omega)
    candidates = candidates[:UPGRADE_KU]
 
    for tok in candidates:
        tok_id      = id(tok)
        curr_t      = cur_tier[tok_id]
        upgrade_t   = prev_tier_map.get(curr_t.tier_id)
        if upgrade_t is None:
            continue
 
        added_bits = upgrade_t.token_bits() - curr_t.token_bits()
        if added_bits > slack:
            continue
 
        # Asymmetric threshold: only upgrade if benefit/cost >= RHO_UP
        rho_up = dc.rho(tok, upgrade_t, curr_t)  # distortion saved per added bit
        if rho_up < RHO_UP:
            continue
 
        tok.assign_tier(upgrade_t.tier_id)
        cur_tier[tok_id] = upgrade_t
        current_bits += added_bits
        slack        -= added_bits
 
    return current_bits
 
 
def allocate(tokens: list, tiers: list,
             reuse=None, stability=None) -> list:
    """
    Prefill-time RDR allocation
    """
    if not tokens:
        return []
 
    drop_tier = tiers[0]
    b1 = tiers[1]
    b3 = tiers[3]
 
    next_tier_map = _build_next_tier_map(tiers)
    # prev_tier_map: cheaper → more expensive (for upgrade pass)
    prev_tier_map = {v.tier_id: k_tier
                     for k_tier, v in
                     [(tiers[tid], nxt) for tid, nxt in next_tier_map.items()
                      if nxt.tier_id != 0]}
 
    T = _seq_len(tokens)
    sinks, recents, ctrl = _classify(tokens, T, SINK_TOKENS, RECENT_WINDOW)
 
    for tok in sinks:
        tok.assign_tier_protected(b1.tier_id)
    for tok in recents:
        tok.assign_tier_protected(b3.tier_id)
 
    prot_bits = (sum(b1.token_bits() for _ in sinks) +
                 sum(b3.token_bits() for _ in recents))
    B_eff = _cfg.GLOBAL_BUDGET_BITS - prot_bits
 
    if B_eff <= 0:
        return [t for t in sinks + recents if t.new_tier_id != 0]
    for tok in ctrl:
        tok.assign_tier_protected(b1.tier_id)
 
    current_bits = len(ctrl) * b1.token_bits()
 
    if current_bits <= B_eff:
        return sinks + recents + ctrl
 
    r_max = max((float(tok.r) for tok in tokens), default=1.0)
    dc       = _DCache(r_max=r_max, reuse=reuse, stability=stability)
    cur_tier = {id(tok): b1 for tok in ctrl}
    heap     = _build_downgrade_heap(ctrl, b1, next_tier_map, dc)
    current_bits = _greedy_downgrade(heap, cur_tier, next_tier_map, dc,
                                     current_bits, B_eff,
                                     no_freeze=True)
 
    retained = []
    for tok in sinks + recents + ctrl:
        if tok.new_tier_id != 0:
            retained.append(tok)
    return retained
 
 
 
def online_refresh(retained_tokens: list, tiers: list,
                   prefill_len: int = 0) -> list:
    """
    Decode-time controller refresh 
    """
    if not retained_tokens:
        return retained_tokens
 
    tokens = retained_tokens
 
    drop_tier = tiers[0]
    b1 = tiers[1]
    b3 = tiers[3]
 
    next_tier_map = _build_next_tier_map(tiers)
    prev_tier_map = {}
    for from_id, to_tier in next_tier_map.items():
        if to_tier.tier_id != 0:
            prev_tier_map[to_tier.tier_id] = tiers[from_id]
 
    T = _seq_len(tokens)
    sinks, recents, ctrl = _classify(tokens, T, SINK_TOKENS, RECENT_WINDOW,
                                     prefill_len=prefill_len)
 
    for tok in ctrl:
        tok.tick_cooldown()
 
    for tok in sinks:
        tok.assign_tier_protected(b1.tier_id)
    for tok in recents:
        tok.assign_tier_protected(b3.tier_id)
 
    prot_bits = (sum(b1.token_bits() for _ in sinks) +
                 sum(b3.token_bits() for _ in recents))
    B_eff = _cfg.GLOBAL_BUDGET_BITS - prot_bits
 
    if B_eff <= 0:
        return [t for t in sinks + recents if t.new_tier_id != 0]
 
    tier_by_id = {t.tier_id: t for t in tiers}
    cur_tier   = {id(tok): tier_by_id[tok.new_tier_id] for tok in ctrl}
 
    current_bits = sum(tier_by_id[tok.new_tier_id].token_bits() for tok in ctrl)
 
    dc = _DCache()
    if current_bits > B_eff:
        heap = []
        for tok in ctrl:
            curr_t = cur_tier[id(tok)]
            next_t = next_tier_map.get(curr_t.tier_id)
            if next_t is not None and not tok.is_frozen:
                rho = dc.rho(tok, curr_t, next_t)
                heapq.heappush(heap,
                    (rho, tok.layer, tok.head, tok.index, id(tok),
                     tok, curr_t, next_t))
        current_bits = _greedy_downgrade(heap, cur_tier, next_tier_map, dc,
                                         current_bits, B_eff)
    #Needs to be verified whether it is in accordance with the implementation mentioned in the appendix
    elif current_bits < B_eff:
        current_bits = _greedy_upgrade(ctrl, cur_tier, prev_tier_map, dc,
                                       current_bits, B_eff)
 
    retained = []
    for tok in sinks + recents + ctrl:
        if tok.new_tier_id != 0:
            retained.append(tok)
    return retained
 
def pack_retained_pages(retained_tokens: list, tiers: list):
    from pagebuilder import build_page
    groups = defaultdict(list)
    for token in retained_tokens:
        groups[(token.layer, token.head, token.new_tier_id)].append(token)
 
    pages         = []
    pointer_table = PointerTable()
    offset        = 0
 
    for (layer, head, tier_id), tok_list in groups.items():
        tok_list.sort(key=lambda t: t.index)
        tier = tiers[tier_id]
        G    = tier.G
 
        for i in range(0, len(tok_list), PAGE_SIZE):
            chunk        = tok_list[i : i + PAGE_SIZE]
            r            = torch.stack([t.r   for t in chunk], dim=0)
            phi          = torch.stack([t.phi for t in chunk], dim=0)
            seg          = chunk[0].segment_id
            page         = build_page(r, phi, tier, seg)
            header_size  = HEADER_BYTES + G * 4
            theta_bytes  = ((len(chunk) * G * tier.b_theta) + 7) // 8
            theta_off    = offset + header_size
            radius_off   = theta_off + theta_bytes
            pages.append(page)
            pointer_table.add_page(offset, theta_off, radius_off)
            offset += page.numel()
 
    if not pages:
        return torch.zeros(0, dtype=torch.uint8), pointer_table
    return torch.cat(pages), pointer_table