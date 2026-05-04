from __future__ import annotations
import torch
from typing import Dict
from fused_decode import sphkv_decode, sphkv_encode_append


def make_tier_idx_map(tiers) -> Dict[int, int]:
    out, rank = {}, 0
    for t in tiers:
        if getattr(t, "tier_id", 0) == 0:
            continue
        out[t.tier_id] = rank
        rank += 1
    return out


class LayerPool:
    def __init__(self, pipeline, layer_idx, page_size, G_max, cb_size_max,
                 decode_capacity, device, tiers, tier_idx_map,
                 num_q, num_kv, kv_groups, dh):
        self.page_size = page_size
        self.G_max = G_max
        self.cb_size_max = cb_size_max
        self.tier_idx_map = tier_idx_map
        self.num_kv = num_kv
        self.num_q = num_q
        self.kv_groups = kv_groups
        self.dh = dh
        self.device = device

        self.g_max = max(t.g for t in tiers if getattr(t, "tier_id", 0) != 0)

        page_data = []
        head_blocks = {h: [] for h in range(num_kv)}
        ctx_lens = [0] * num_kv

        for h in range(num_kv):
            ph = pipeline.per_head_pages.get((layer_idx, h), [])
            n_tok = 0
            for ph_tuple in ph:
                _pt, _ptt, b_theta, n_tokens, V_tier, K_tier = ph_tuple[:6]
                r_full, theta_full = K_tier
                tier_obj = next(t for t in tiers
                                if getattr(t, "tier_id", 0) != 0
                                and t.b_theta == b_theta)
                t_idx = tier_idx_map[tier_obj.tier_id]
                G_t = tier_obj.G
                n_pages = (n_tokens + page_size - 1) // page_size
                for pg in range(n_pages):
                    s = pg * page_size
                    e = min(s + page_size, n_tokens)
                    npg = e - s
                    theta_pg = torch.zeros(page_size, G_max, dtype=torch.uint8, device=device)
                    rad_pg = torch.zeros(page_size, G_max, dtype=torch.uint8, device=device)
                    rscl_pg = torch.zeros(G_max, dtype=torch.float32, device=device)
                    r_chunk = r_full[s:e]
                    rscl = r_chunk.amax(dim=0).clamp(min=1e-8)
                    r_q = (r_chunk / rscl * 255.0).round().clamp(0, 255).to(torch.uint8)
                    theta_pg[:npg, :G_t] = theta_full[s:e].to(torch.uint8).to(device)
                    rad_pg[:npg, :G_t] = r_q.to(device)
                    rscl_pg[:G_t] = rscl.to(device)
                    v_pg = torch.zeros(page_size, dh, dtype=torch.float16, device=device)
                    v_pg[:npg] = V_tier[s:e].to(device).half()
                    pid = len(page_data)
                    page_data.append((theta_pg, rad_pg, rscl_pg, v_pg))
                    head_blocks[h].append((pid, t_idx))
                    n_tok += npg
            ctx_lens[h] = len(head_blocks[h]) * page_size

        n_used = len(page_data)
        n_total = n_used + decode_capacity
        max_blk = max((len(v) for v in head_blocks.values()), default=0)
        max_blk += (decode_capacity // max(num_kv, 1)) + 2
        self.max_blocks = max_blk
        self.num_pages_used = n_used

        self.theta_codes = torch.zeros((n_total, page_size, G_max), dtype=torch.uint8, device=device)
        self.radius_codes = torch.zeros((n_total, page_size, G_max), dtype=torch.uint8, device=device)
        self.r_scales = torch.zeros((n_total, G_max), dtype=torch.float32, device=device)
        self.v_pool = torch.zeros((n_total, page_size, dh), dtype=torch.float16, device=device)

        for pid, (t_pg, r_pg, rs_pg, v_pg) in enumerate(page_data):
            self.theta_codes[pid] = t_pg
            self.radius_codes[pid] = r_pg
            self.r_scales[pid] = rs_pg
            self.v_pool[pid] = v_pg

        self.ctx_lens_cpu = list(ctx_lens)
        self.block_table_cpu = [[-1] * max_blk for _ in range(num_kv)]
        self.bits_table_cpu = [[0] * max_blk for _ in range(num_kv)]
        for h, blist in head_blocks.items():
            for lb, (pid, ti) in enumerate(blist):
                self.block_table_cpu[h][lb] = pid
                self.bits_table_cpu[h][lb] = ti

        self.block_table_gpu = torch.tensor(self.block_table_cpu, dtype=torch.int32, device=device)
        self.bits_table_gpu = torch.tensor(self.bits_table_cpu, dtype=torch.int32, device=device)
        self.ctx_lens_gpu = torch.tensor(self.ctx_lens_cpu, dtype=torch.int32, device=device)

        num_tiers = len(tier_idx_map)
        self.cb_flat = torch.zeros(
            (num_kv, num_tiers, G_max, cb_size_max, self.g_max),
            dtype=torch.float32, device=device)

        self.decode_cb = None
        self._decode_cb_flat = None  # flattened for encode kernel
        from codebook_loader import get_codebook
        for tier_obj in tiers:
            tid = getattr(tier_obj, "tier_id", 0)
            if tid == 0:
                continue
            t_idx = tier_idx_map[tid]
            G = tier_obj.G
            g = tier_obj.g
            C = 2 ** tier_obj.b_theta
            heads = []
            for h_kv in range(num_kv):
                cb = get_codebook(pipeline.codebooks, layer_idx, h_kv, tid)
                if cb is None:
                    break
                heads.append(cb.to(device).float())
            if len(heads) == num_kv:
                stacked = torch.stack(heads)
                self.cb_flat[:, t_idx, :G, :C, :g] = stacked
                if tid == 1:                              # FIXED: was 3 (recent->b1)
                    self.decode_cb = stacked  # [num_kv, G, C, g]
                    self._decode_cb_flat = stacked.contiguous()

        active_tiers = [t for t in tiers if getattr(t, "tier_id", 0) != 0]
        self.tier_G_gpu = torch.tensor(
            [t.G for t in active_tiers], dtype=torch.int32, device=device)
        self.tier_g_gpu = torch.tensor(
            [t.g for t in active_tiers], dtype=torch.int32, device=device)

        self._pids = torch.zeros(num_kv, dtype=torch.long, device=device)
        self._dec_tier_idx = tier_idx_map[1]   # recent tokens go to b1 (paper §C.1)

        # Decode tier params (constant)
        b_dec = next(t for t in tiers if getattr(t, "tier_id", 0) == 1)   # FIXED: recent->b1
        self._G_dec = b_dec.G
        self._g_dec = b_dec.g
        self._C_dec = 2 ** b_dec.b_theta

        # Pre-allocated scratch
        self._out_buf = torch.zeros(num_q, dh, dtype=torch.float32, device=device)
        self._partial_scratch = torch.zeros(num_q, max_blk, dh + 2,
                                            dtype=torch.float32, device=device)
        self._uniform_ctx = ctx_lens[0] if ctx_lens else 0

    def append_and_compute(self, q_post, k_post, v_new, tiers, sm_scale):
        device = self.device

        # ── Page management (pure CPU, ~2μs) ──
        cur_len = self._uniform_ctx
        slot = cur_len % self.page_size
        is_new_page = (slot == 0)

        if is_new_page:
            pg = cur_len // self.page_size
            start_pid = self.num_pages_used
            self.num_pages_used += self.num_kv
            for h in range(self.num_kv):
                self.block_table_cpu[h][pg] = start_pid + h
                self.bits_table_cpu[h][pg] = self._dec_tier_idx
            self.block_table_gpu = torch.tensor(
                self.block_table_cpu, dtype=torch.int32, device=device)
            self.bits_table_gpu = torch.tensor(
                self.bits_table_cpu, dtype=torch.int32, device=device)
            self._pids[:] = torch.arange(start_pid, start_pid + self.num_kv,
                                         dtype=torch.long, device=device)

        self._uniform_ctx = cur_len + 1

        # ── Dispatch 1: fused encode + append (1 kernel, replaces 12 ops) ──
        sphkv_encode_append(
            k_post, v_new, self._decode_cb_flat,
            self.theta_codes, self.radius_codes,
            self.r_scales, self.v_pool,
            self._pids,
            slot, self._G_dec, self._g_dec, self._C_dec,
            self.G_max, self.dh, self.page_size,
            1 if is_new_page else 0,
        )

        # ── Dispatch 2: update ctx_lens (in-place, tiny) ──
        self.ctx_lens_gpu.add_(1)

        # ── Dispatch 3: fused decode (2 kernels internally) ──
        max_ctx = self._uniform_ctx
        num_tiers = len(self.tier_idx_map)

        sphkv_decode(
            q_post, self.cb_flat,
            self.tier_G_gpu, self.tier_g_gpu,
            self.theta_codes, self.radius_codes,
            self.r_scales, self.v_pool,
            self.block_table_gpu, self.bits_table_gpu,
            self.ctx_lens_gpu, self._out_buf,
            self._partial_scratch,
            self.num_q, self.num_kv, self.kv_groups, num_tiers,
            self.max_blocks, self.page_size, self.dh,
            self.G_max, self.cb_size_max, self.g_max,
            sm_scale, max_ctx,
        )

        return self._out_buf
