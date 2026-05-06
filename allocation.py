import heapq
from collections import defaultdict
from typing import List, Dict, Tuple
 
import torch
import config as _cfg
from config import ( HEADER_BYTES, PAGE_SIZE, NUM_GROUPS,
    SINK_TOKENS, RECENT_WINDOW,
    COOLDOWN_STEPS, UPGRADE_KU, RHO_UP,
    PAGE_ALIGNMENT_BITS_PER_TOKEN,
    PER_LAYER_CAP_FRACTION, PER_HEAD_CAP_FRACTION,
    LAGRANGE_T_PI, LAGRANGE_STEP_SIZE, LAGRANGE_CLIP,
)
from distortion_proxy import compute_distortion
from pointer_table import PointerTable
 
 
 
def _build_next_tier_map(tiers):
    nondrop = [t for t in tiers if t.tier_id != 0]
    nondrop.sort(key=lambda t: -t.token_bits())  # most expensive first: b1,b2,b3
    nxt = {}
    for i, t in enumerate(nondrop):
        nxt[t.tier_id] = nondrop[i + 1] if i + 1 < len(nondrop) else tiers[0]
    return nxt
 
 
def _classify(tokens, T: int, sink_n: int, recent_w: int, prefill_len: int = 0):
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
 
 
def _effective_bits(tier) -> int:
    tb = tier.token_bits()
    if tb == 0:
        return 0
    return tb + PAGE_ALIGNMENT_BITS_PER_TOKEN


class _DCache:
    def __init__(self, r_max=None, reuse=None, stability=None,
                 lagrange_prices=None):
        self._c: Dict[Tuple, float] = {}
        self.r_max     = r_max
        self.reuse     = reuse
        self.stability = stability
        # Lagrangian prices: {layer: pi_l} and {(layer,head): pi_lh}
        self.pi_layer = {}
        self.pi_head  = {}
        if lagrange_prices is not None:
            self.pi_layer = lagrange_prices.get('layer', {})
            self.pi_head  = lagrange_prices.get('head', {})
 
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
        dC = _effective_bits(from_t) - _effective_bits(to_t)
        base_rho = dD / max(dC, 1)
        # Lagrangian price modifier (C.3):
        #   rho_pi = rho * (1 + pi_l + pi_{l,h})
        pi_l  = self.pi_layer.get(tok.layer, 0.0)
        pi_lh = self.pi_head.get((tok.layer, tok.head), 0.0)
        return base_rho * (1.0 + pi_l + pi_lh)
 

 
def _build_downgrade_heap(ctrl_tokens, init_tier, next_tier_map, dc: _DCache):
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
        delta = _effective_bits(from_t) - _effective_bits(to_t)
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
 
        added_bits = _effective_bits(upgrade_t) - _effective_bits(curr_t)
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
        tok.assign_tier_protected(b1.tier_id)
 
    prot_bits = (sum(_effective_bits(b1) for _ in sinks) +
                 sum(_effective_bits(b1) for _ in recents))
    B_eff = _cfg.GLOBAL_BUDGET_BITS - prot_bits
 
    if B_eff <= 0:
        return [t for t in sinks + recents if t.new_tier_id != 0]
    for tok in ctrl:
        tok.assign_tier_protected(b1.tier_id)
 
    current_bits = len(ctrl) * _effective_bits(b1)
 
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
        tok.assign_tier_protected(b1.tier_id)
 
    prot_bits = (sum(_effective_bits(b1) for _ in sinks) +
                 sum(_effective_bits(b1) for _ in recents))
    B_eff = _cfg.GLOBAL_BUDGET_BITS - prot_bits
 
    if B_eff <= 0:
        return [t for t in sinks + recents if t.new_tier_id != 0]
 
    tier_by_id = {t.tier_id: t for t in tiers}
    cur_tier   = {id(tok): tier_by_id[tok.new_tier_id] for tok in ctrl}
 
    current_bits = sum(_effective_bits(tier_by_id[tok.new_tier_id]) for tok in ctrl)

    # ── Lagrangian prices for per-layer/per-head caps (C.3) ──────────
    lagrange_prices = _compute_lagrange_prices(ctrl, cur_tier, tier_by_id, B_eff)

    dc = _DCache(lagrange_prices=lagrange_prices)
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
    elif current_bits < B_eff:
        current_bits = _greedy_upgrade(ctrl, cur_tier, prev_tier_map, dc,
                                       current_bits, B_eff)
 
    retained = []
    for tok in sinks + recents + ctrl:
        if tok.new_tier_id != 0:
            retained.append(tok)
    return retained


def _compute_lagrange_prices(ctrl_tokens, cur_tier, tier_by_id, B_eff):
    if PER_LAYER_CAP_FRACTION is None and PER_HEAD_CAP_FRACTION is None:
        return None

    # Count layers and heads
    layers = set()
    heads_per_layer = defaultdict(set)
    for tok in ctrl_tokens:
        layers.add(tok.layer)
        heads_per_layer[tok.layer].add(tok.head)

    num_layers = max(len(layers), 1)
    pi_layer = {}
    pi_head  = {}

    for _ in range(LAGRANGE_T_PI):
        # Compute per-layer and per-head costs
        layer_cost = defaultdict(float)
        head_cost  = defaultdict(float)
        for tok in ctrl_tokens:
            tid = cur_tier[id(tok)]
            cost = _effective_bits(tid)
            layer_cost[tok.layer] += cost
            head_cost[(tok.layer, tok.head)] += cost

        # Per-layer price update
        if PER_LAYER_CAP_FRACTION is not None:
            B_layer = B_eff * PER_LAYER_CAP_FRACTION / num_layers
            for l in layers:
                violation = layer_cost[l] - B_layer
                pi_l = pi_layer.get(l, 0.0)
                pi_l = max(0.0, min(pi_l + LAGRANGE_STEP_SIZE * violation,
                                    LAGRANGE_CLIP))
                pi_layer[l] = pi_l

        # Per-head price update
        if PER_HEAD_CAP_FRACTION is not None:
            for l in layers:
                n_heads = max(len(heads_per_layer[l]), 1)
                B_head = B_eff * PER_HEAD_CAP_FRACTION / (num_layers * n_heads)
                for h in heads_per_layer[l]:
                    violation = head_cost[(l, h)] - B_head
                    pi_lh = pi_head.get((l, h), 0.0)
                    pi_lh = max(0.0, min(pi_lh + LAGRANGE_STEP_SIZE * violation,
                                         LAGRANGE_CLIP))
                    pi_head[(l, h)] = pi_lh

    return {'layer': pi_layer, 'head': pi_head}
 
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
