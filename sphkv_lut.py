"""
Split-kernel decode with INLINE dot product (no Q-LUT precomputation).

Per-layer hot path:
  1. Encode new token (1 einsum + 1 argmax)
  2. Append to pools (batched writes)
  3. ONE kernel launch: inline dot product logits
  4. Softmax (PyTorch)
  5. V gather + matmul (cuBLAS)
"""
from __future__ import annotations
import torch
from typing import Dict
from fused_decode import sphkv_logits


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

        # g_max = max group size across tiers (for cb_flat padding)
        self.g_max = max(t.g for t in tiers if getattr(t, "tier_id", 0) != 0)

        # ── Build pages from per_head_pages ──
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
                    v_pg = torch.zeros(page_size, dh, dtype=torch.float32, device=device)
                    v_pg[:npg] = V_tier[s:e].to(device).float()
                    pid = len(page_data)
                    page_data.append((theta_pg, rad_pg, rscl_pg, v_pg))
                    head_blocks[h].append((pid, t_idx))
                    n_tok += npg
            # ctx_len = total SLOTS so kernel processes ALL pages
            ctx_lens[h] = len(head_blocks[h]) * page_size

        n_used = len(page_data)
        n_total = n_used + decode_capacity
        max_blk = max((len(v) for v in head_blocks.values()), default=0)
        max_blk += (decode_capacity // max(num_kv, 1)) + 2
        self.max_blocks = max_blk
        self.num_pages_used = n_used

        # Pool tensors
        self.theta_codes = torch.zeros((n_total, page_size, G_max), dtype=torch.uint8, device=device)
        self.radius_codes = torch.zeros((n_total, page_size, G_max), dtype=torch.uint8, device=device)
        self.r_scales = torch.zeros((n_total, G_max), dtype=torch.float32, device=device)
        self.v_pool = torch.zeros((n_total, page_size, dh), dtype=torch.float32, device=device)

        for pid, (t_pg, r_pg, rs_pg, v_pg) in enumerate(page_data):
            self.theta_codes[pid] = t_pg
            self.radius_codes[pid] = r_pg
            self.r_scales[pid] = rs_pg
            self.v_pool[pid] = v_pg

        # Paging tables
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

        # ── Pre-stack codebooks: [H_kv, num_tiers, G_max, cb_max, g_max] ──
        # Kernel reads cb_flat[hkv, tier, g, code, :] for inline dot product
        num_tiers = len(tier_idx_map)
        self.cb_flat = torch.zeros(
            (num_kv, num_tiers, G_max, cb_size_max, self.g_max),
            dtype=torch.float32, device=device)

        self.decode_cb = None  # tier 3 codebook for decode-token encoding
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
                stacked = torch.stack(heads)  # [H_kv, G, C, g]
                # Write into cb_flat with proper padding
                self.cb_flat[:, t_idx, :G, :C, :g] = stacked
                if tid == 3:
                    self.decode_cb = stacked

        # ── Tier parameter tensors (constant, on GPU) ──
        active_tiers = [t for t in tiers if getattr(t, "tier_id", 0) != 0]
        self.tier_G_gpu = torch.tensor(
            [t.G for t in active_tiers], dtype=torch.int32, device=device)
        self.tier_g_gpu = torch.tensor(
            [t.g for t in active_tiers], dtype=torch.int32, device=device)

        self._pids = torch.zeros(num_kv, dtype=torch.long, device=device)
        self._slots = torch.zeros(num_kv, dtype=torch.long, device=device)

    def append_and_compute(self, q_post, k_post, v_new, tiers, sm_scale):
        device = self.device
        num_kv = self.num_kv
        b_dec = tiers[3]
        G_dec, g_dec = b_dec.G, b_dec.g

        # ── Encode new token ──
        K_grp = k_post.float().view(num_kv, G_dec, g_dec)
        r_real = K_grp.norm(dim=-1).clamp(min=1e-8)
        K_dir = K_grp / r_real.unsqueeze(-1)
        sims = torch.einsum('hgi,hgci->hgc', K_dir, self.decode_cb)
        theta_all = sims.argmax(dim=-1).to(torch.uint8)

        # ── Append to pools ──
        bt_dirty = False
        for h_kv in range(num_kv):
            cur_len = self.ctx_lens_cpu[h_kv]
            pg = cur_len // self.page_size
            slot = cur_len % self.page_size
            if slot == 0:
                pid = self.num_pages_used
                self.num_pages_used += 1
                self.block_table_cpu[h_kv][pg] = pid
                self.bits_table_cpu[h_kv][pg] = self.tier_idx_map[b_dec.tier_id]
                bt_dirty = True
                self.r_scales[pid, :G_dec] = r_real[h_kv]
            else:
                pid = self.block_table_cpu[h_kv][pg]
            self._pids[h_kv] = pid
            self._slots[h_kv] = slot
            self.ctx_lens_cpu[h_kv] = cur_len + 1

        cur_scales = self.r_scales[self._pids, :G_dec]
        r_q_all = (r_real / cur_scales * 255.0).round().clamp(0, 255).to(torch.uint8)
        self.theta_codes[self._pids, self._slots, :G_dec] = theta_all
        self.radius_codes[self._pids, self._slots, :G_dec] = r_q_all
        self.v_pool[self._pids, self._slots, :] = v_new.float()

        if bt_dirty:
            self.block_table_gpu = torch.tensor(
                self.block_table_cpu, dtype=torch.int32, device=device)
            self.bits_table_gpu = torch.tensor(
                self.bits_table_cpu, dtype=torch.int32, device=device)
        self.ctx_lens_gpu = torch.tensor(
            self.ctx_lens_cpu, dtype=torch.int32, device=device)

        # ── Logit kernel (inline dot product, no Q-LUT) ──
        max_ctx = max(self.ctx_lens_cpu)
        num_tiers = len(self.tier_idx_map)
        logits = torch.full((self.num_q, max_ctx), -1e9,
                            dtype=torch.float32, device=device)

        sphkv_logits(
            q_post,                 # [H_q, dh] — raw Q vector
            self.cb_flat,           # [H_kv, num_tiers, G_max, cb_max, g_max]
            self.tier_G_gpu,        # [num_tiers]
            self.tier_g_gpu,        # [num_tiers]
            self.theta_codes,       # [P, page_size, G_max]
            self.radius_codes,      # [P, page_size, G_max]
            self.r_scales,          # [P, G_max]
            self.block_table_gpu,   # [H_kv, max_blocks]
            self.bits_table_gpu,    # [H_kv, max_blocks]
            self.ctx_lens_gpu,      # [H_kv]
            logits,                 # [H_q, max_ctx] output
            self.num_q, self.num_kv, self.kv_groups, num_tiers,
            self.max_blocks, self.page_size, self.dh,
            self.G_max, self.cb_size_max, self.g_max,
            sm_scale, max_ctx,
        )

        # ── Softmax + V matmul ──
        attn = torch.softmax(logits, dim=-1)
        n_blocks_ctx = (max_ctx + self.page_size - 1) // self.page_size
        page_ids = self.block_table_gpu[:, :n_blocks_ctx].long()
        V_paged = self.v_pool[page_ids]
        V_flat = V_paged.reshape(self.num_kv, -1, self.dh)[:, :max_ctx, :]
        attn_grouped = attn.view(self.num_kv, self.kv_groups, max_ctx)
        out = torch.einsum('hgn,hnd->hgd', attn_grouped, V_flat)
        return out.reshape(self.num_q, self.dh)
