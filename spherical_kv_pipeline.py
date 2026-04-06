# from __future__ import annotations
 
# import contextlib
# import math
# import os
# import sys
# from collections import defaultdict
# from typing import Dict, List, Optional, Tuple
 
# import torch
 
 
# @contextlib.contextmanager
# def nvtx_range(name: str):
#     """
#     NVTX range for Nsight Compute / Nsight Systems.
#     Algorithm instrumentation (line 73): page_lookup, kv_read,
#     angle_logits, softmax, proj.  No-ops when CUDA unavailable.
#     """
#     if torch.cuda.is_available():
#         torch.cuda.nvtx.range_push(name)
#     try:
#         yield
#     finally:
#         if torch.cuda.is_available():
#             torch.cuda.nvtx.range_pop()
 
# _HERE = os.path.dirname(os.path.abspath(__file__))
# if _HERE not in sys.path:
#     sys.path.insert(0, _HERE)
 
# from allocation import allocate, online_refresh
# from codebook_loader import get_codebook
# from config import (EPS, GROUP_SIZE, NUM_GROUPS, PAGE_SIZE, HEADER_BYTES,
#                                REFRESH_CADENCE, SINK_TOKENS, RECENT_WINDOW,
#                                COOLDOWN_STEPS)
# from llama_hooks import (
#     aggregate_proxy_to_kv_heads,
#     build_attn_weights_tensor,
#     build_head_outputs_tensor,
#     capture_prefill_pass,
#     patch_for_decode,
#     unpatch_decode,
# )
# from pagebuilder import build_page_from_codes
# from pointer_table import PointerTable
# from resuse_proxy import compute_reuse_proxy
# from spherical_parameterization import spherical_parameterize
# from stability_proxy import compute_stability_proxy
# from tiers import build_tiers
# from token_state import TokenState
 
 
# # ---------------------------------------------------------------------------
# # Reference ADA decode (pure torch -- fallback when CUDA kernel unavailable)
# # Vectorized: no Python loop over groups.
# # ---------------------------------------------------------------------------
 
# def reference_codebook_decode(
#     q:          torch.Tensor,   # [dh]
#     codebooks:  torch.Tensor,   # [G, codebook_size, group_size]
#     K_ctx:      torch.Tensor,   # [T_ctx, dh]
#     group_size: int,
#     num_groups: int,
# ) -> torch.Tensor:              # [T_ctx]
#     """
#     Vectorized reference decode.
#     Matches decode_kernel.cu:
#         dot   = q^(g) . cw^(g)          (raw q group, not normalised)
#         acc   = sum_g r_hat^(g) * dot^(g)
#         logit = acc / sqrt(dh)
#     """
#     T_ctx, dh = K_ctx.shape
#     device    = q.device
 
#     q_groups  = q.view(num_groups, group_size)                          # [G, g]
#     K_grouped = K_ctx.view(T_ctx, num_groups, group_size)               # [T, G, g]
#     r_groups  = K_grouped.norm(dim=-1)                                  # [T, G]
#     K_dir     = K_grouped / (r_groups.unsqueeze(-1) + EPS)              # [T, G, g]
 
#     # vectorized nearest-neighbour: [T, G, cb_size]
#     sims     = torch.einsum('tgi,gci->tgc', K_dir, codebooks)
#     best_idx = sims.argmax(dim=-1)                                      # [T, G]
 
#     # gather codewords: [T, G, g]
#     cw = codebooks[
#         torch.arange(num_groups, device=device).unsqueeze(0),           # [1, G]
#         best_idx                                                         # [T, G]
#     ]
 
#     # dot product and weighted sum
#     dots   = (cw * q_groups.unsqueeze(0)).sum(-1)                       # [T, G]
#     logits = (r_groups * dots).sum(-1) / math.sqrt(dh)                  # [T]
 
#     return logits
 
 
# # ---------------------------------------------------------------------------
# # Fused decode availability + dispatch
# # ---------------------------------------------------------------------------
 
# _fused_ok: Optional[bool] = None
 
# def _fused_available() -> bool:
#     global _fused_ok
#     if _fused_ok is None:
#         try:
#             from fused_decode import fused_decode   # noqa: F401
#             _fused_ok = True
#         except Exception:
#             _fused_ok = False
#     return _fused_ok
 
 
# def _call_fused(
#     pages_tensor:  torch.Tensor,    # packed uint8
#     ptable_tensor: torch.Tensor,    # [P, 3] int32
#     q:             torch.Tensor,    # [dh] OR [num_q, dh]
#     codebooks_lh:  torch.Tensor,    # [G, cb_size, g]
#     b_theta:       int,
#     dh:            int,
#     num_groups:    int,
#     group_size:    int,
# ) -> torch.Tensor:
#     """
#     Dispatches to the batched CUDA kernel.
#     If q is 1D [dh], unsqueezes to [1, dh] and squeezes result back.
#     If q is 2D [num_q, dh], returns [num_q, num_pages * page_size].
#     """
#     from fused_decode import fused_decode
#     squeeze = (q.dim() == 1)
#     if squeeze:
#         q = q.unsqueeze(0)   # [1, dh]
#     result = fused_decode(
#         pages_tensor,
#         ptable_tensor,
#         q.float(),
#         codebooks_lh.float(),
#         dh=dh,
#         groups=num_groups,
#         group_size=group_size,
#         b_theta=b_theta,
#         page_size=PAGE_SIZE,
#     )
#     # result: [num_q, num_pages * page_size]
#     if squeeze:
#         return result.squeeze(0)   # [num_pages * page_size]
#     return result
 
 
# # ---------------------------------------------------------------------------
# # Codebook encoding  (vectorized, no Python loops)
# # ---------------------------------------------------------------------------
 
# def _encode_keys_with_codebooks(
#     K_chunk:    torch.Tensor,   # [N, dh]
#     codebooks:  torch.Tensor,   # [G, codebook_size, group_size]  tier-specific
#     group_size: int,             # g for this tier
#     num_groups: int,             # G for this tier
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Encode N key vectors using tier-specific grouped VQ (Appendix B.1).
 
#     Returns
#     -------
#     r_groups    : [N, G]  float32 -- per-group L2 norms
#     theta_codes : [N, G]  int32   -- nearest codebook index per group
#     """
#     N         = K_chunk.shape[0]
#     K_grouped = K_chunk.view(N, num_groups, group_size)      # [N, G, g]
#     r_groups  = K_grouped.norm(dim=-1)                       # [N, G]
#     K_dir     = K_grouped / (r_groups.unsqueeze(-1) + EPS)   # [N, G, g]
 
#     cb   = codebooks.to(K_dir.device)                        # [G, cb_size, g]
#     sims = torch.einsum('ngi,gci->ngc', K_dir, cb)           # [N, G, cb_size]
#     theta_codes = sims.argmax(dim=-1).to(torch.int32)        # [N, G]
#     return r_groups, theta_codes
 
 
# # ---------------------------------------------------------------------------
# # Per-(layer, kv_head) page builder  (prefill only)
# # ---------------------------------------------------------------------------
 
# def _build_per_head_pages(
#     retained_tokens: List[TokenState],
#     tiers,
#     K_all:      Dict[Tuple[int, int], torch.Tensor],
#     V_all:      Dict[Tuple[int, int], torch.Tensor],
#     codebooks:  Dict[Tuple[int, int], torch.Tensor],
#     group_size: int,
#     num_groups: int,
#     device,
# ) -> Dict[Tuple[int, int], List]:
#     """
#     Build tier-homogeneous, codebook-encoded pages per (layer, kv_head).
 
#     Each entry in the returned list is a 6-tuple:
#         (pages_tensor, ptable_tensor, b_theta, n_tokens, V_tier, K_tier)
 
#     pages_tensor  -- packed uint8 bitstream for all pages in this tier.
#     ptable_tensor -- [P, 3] int32 pointer table.
#     b_theta       -- bits per codebook index for this tier.
#     n_tokens      -- exact count of valid tokens across all pages.
#     V_tier        -- [n_tokens, dh] float on device.
#     K_tier        -- [n_tokens, dh] float on CPU (reference path only).
#     """
#     groups: Dict[Tuple[int, int, int], List[TokenState]] = defaultdict(list)
#     for ts in retained_tokens:
#         groups[(ts.layer, ts.head, ts.new_tier_id)].append(ts)
 
#     per_head: Dict[Tuple[int, int], List] = defaultdict(list)
 
#     for (layer, head, tier_id), tokens in groups.items():
#         tokens.sort(key=lambda t: t.index)
#         tier  = tiers[tier_id]
#         K_lh  = K_all[(layer, head)]
#         V_lh  = V_all[(layer, head)]
#         # ── Tier-specific codebook: keyed by (layer, head, tier_id) ─────
#         # Appendix B.1: b1/b2 use g=16, G=8 codebooks;
#         #               b3 uses g=32, G=4 codebooks.
#         cb_key = (layer, head, tier_id)
#         if cb_key in codebooks:
#             cb_lh = codebooks[cb_key]           # [G, cb_size, g]  tier-aware
#         elif (layer, head) in codebooks:
#             cb_lh = codebooks[(layer, head)]    # legacy format fallback
#         else:
#             continue                            # no codebook, skip tier
 
#         tier_g  = tier.g    # group size for this tier
#         tier_G  = tier.G    # number of groups for this tier
 
#         page_list:    List[torch.Tensor] = []
#         K_tier_parts: List[torch.Tensor] = []
#         V_tier_parts: List[torch.Tensor] = []
#         ptable = PointerTable()
#         offset = 0
 
#         for i in range(0, len(tokens), PAGE_SIZE):
#             chunk   = tokens[i : i + PAGE_SIZE]
#             N       = len(chunk)
#             indices = [t.index for t in chunk]
 
#             K_chunk = K_lh[indices]
#             V_chunk = V_lh[indices]
#             r_groups, theta_codes = _encode_keys_with_codebooks(
#                 K_chunk, cb_lh, tier_g, tier_G   # tier-specific g and G
#             )
 
#             page = build_page_from_codes(
#                 r_groups, theta_codes, tier, chunk[0].segment_id
#             )
 
#             header_size   = HEADER_BYTES + tier_G * 4
#             theta_bytes   = ((N * tier_G * tier.b_theta) + 7) // 8
#             theta_offset  = offset + header_size
#             radius_offset = theta_offset + theta_bytes
 
#             page_list.append(page)
#             K_tier_parts.append(K_chunk)
#             V_tier_parts.append(V_chunk)
#             ptable.add_page(offset, theta_offset, radius_offset)
#             offset += page.numel()
 
#         per_head[(layer, head)].append((
#             torch.cat(page_list).to(device),
#             ptable.to_tensor().to(device),
#             tier.b_theta,
#             len(tokens),
#             torch.cat(V_tier_parts, dim=0).to(device),
#             torch.cat(K_tier_parts, dim=0),         # CPU, reference path only
#         ))
 
#     return per_head
 
 
# # ---------------------------------------------------------------------------
# # Pipeline
# # ---------------------------------------------------------------------------
 
# class SphericalKVPipeline:
#     """
#     Spherical KV Cache inference pipeline.
 
#     Usage
#     -----
#     pipeline = SphericalKVPipeline(model, tokenizer, codebooks, device)
#     pipeline.prefill(input_ids)
#     out = model.generate(input_ids, ...)
#     pipeline.uninstall()
#     """
 
#     def __init__(
#         self,
#         model,
#         tokenizer,
#         codebooks: Dict[Tuple[int, int], torch.Tensor],
#         device:     torch.device = torch.device("cpu"),
#         head_dim:   Optional[int] = None,
#         group_size: Optional[int] = None,
#         sink_tokens: int = 4,
#         use_fused:   bool = True,
#     ):
#         self.model      = model
#         self.tokenizer  = tokenizer
#         self.codebooks  = {k: v.to(device) for k, v in codebooks.items()}
#         self.device     = device
#         self.sink_tokens = sink_tokens
#         self.use_fused  = use_fused and _fused_available()
 
#         cfg = model.config
#         self.num_layers   = cfg.num_hidden_layers
#         self.num_q_heads  = cfg.num_attention_heads
#         self.num_kv_heads = getattr(cfg, "num_key_value_heads", self.num_q_heads)
#         self.head_dim     = head_dim or getattr(
#             cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
#         )
#         self.group_size = group_size or GROUP_SIZE
#         self.num_groups = self.head_dim // self.group_size
#         self.tiers      = build_tiers(self.head_dim)
 
#         # Prefill compressed pages: per (layer, kv_head) list of 6-tuples
#         self.per_head_pages: Dict[Tuple[int, int], List] = {}
 
#         # Decode staging buffers: compressed codes for recent decode tokens
#         # not yet flushed into a full page.  At most PAGE_SIZE-1 tokens each.
#         # stg_r     : [N_stg, G] float32  -- per-group radii
#         # stg_theta : [N_stg, G] int32    -- codebook indices
#         # stg_V     : [N_stg, dh] float32 -- value vectors
#         self.stg_r:     Dict[Tuple[int, int], torch.Tensor] = {}
#         self.stg_theta: Dict[Tuple[int, int], torch.Tensor] = {}
#         self.stg_V:     Dict[Tuple[int, int], torch.Tensor] = {}
 
#         self.reuse:     Optional[torch.Tensor] = None
#         self.stability: Optional[torch.Tensor] = None
#         self.seq_len:   int = 0
 
#         # ── Online refresh state (Appendix C.3) ──────────────────────
#         # All retained TokenState objects from prefill, kept alive so
#         # record_attn() and the controller refresh can use them.
#         self._retained_tokens: list = []
#         self._decode_step: int = 0   # counts decode steps since prefill
 
#         # ctx_token_order[(layer, head)]: retained TokenState objects in the
#         # same positional order they appear in ctx_logits.  Built at prefill
#         # so record_attn() can map attention position → TokenState in O(1).
#         self._ctx_token_order: Dict[Tuple[int, int], List] = {}
 
#         # Dense prefill keys kept for page rebuild after online refresh.
#         # _K_all[(layer, head)] = Tensor[T, dh] float32 CPU.
#         self._K_all = {}
#         self._V_all = {}
 
#         self._original_forwards: Dict[int, object] = {}
#         self._patched = False
 
#         mode = "fused CUDA" if self.use_fused else "reference (pure-torch)"
#         print(
#             f"[SphericalKVPipeline] layers={self.num_layers}  "
#             f"kv_heads={self.num_kv_heads}  head_dim={self.head_dim}  "
#             f"groups={self.num_groups}  decode_mode={mode}"
#         )
 
#     # ------------------------------------------------------------------
#     # Prefill
#     # ------------------------------------------------------------------
 
#     def prefill(
#         self,
#         input_ids:      torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#     ) -> None:
#         if self._patched:
#             self.uninstall()
#         self._reset_state()
 
#         input_ids = input_ids.to(self.device)
#         B, T = input_ids.shape
#         self.seq_len = T
 
#         print(f"[prefill] Forward pass on {T} tokens ...")
#         kv_pairs, attn_list, ho_list, logits = capture_prefill_pass(
#             self.model, input_ids, attention_mask
#         )
 
#         attn_stacked = build_attn_weights_tensor(attn_list)
#         ho_stacked   = build_head_outputs_tensor(ho_list)
 
#         if attn_stacked is not None:
#             reuse_kv = aggregate_proxy_to_kv_heads(
#                 compute_reuse_proxy(attn_stacked),
#                 self.num_q_heads, self.num_kv_heads,
#             )
#         else:
#             reuse_kv = torch.ones(self.num_layers, self.num_kv_heads) / self.num_kv_heads
 
#         if ho_stacked is not None:
#             stab_kv = aggregate_proxy_to_kv_heads(
#                 compute_stability_proxy(
#                     ho_stacked[:, :, :, -1, :], logits[:, -1, :]
#                 ),
#                 self.num_q_heads, self.num_kv_heads,
#             )
#         else:
#             stab_kv = torch.ones(self.num_layers, self.num_kv_heads) / self.num_kv_heads
 
#         self.reuse     = reuse_kv.cpu()
#         self.stability = stab_kv.cpu()
 
#         del attn_stacked, ho_stacked, attn_list, ho_list
#         torch.cuda.empty_cache()
 
#         print("[prefill] Building TokenStates ...")
#         all_tokens: List[TokenState] = []
#         K_all: Dict[Tuple[int, int], torch.Tensor] = {}
#         V_all: Dict[Tuple[int, int], torch.Tensor] = {}
 
#         for li, (K, V) in enumerate(kv_pairs):
#             for h in range(self.num_kv_heads):
#                 K_lh = K[0, h].float().cpu()    # float32 for encoding
#                 V_lh = V[0, h].half().cpu()     # fp16 to match dense baseline
#                 K_all[(li, h)] = K_lh
#                 V_all[(li, h)] = V_lh
 
#                 r, phi = spherical_parameterize(K_lh)
 
#                 # Appendix B.1 / C.2: compute per-group radii at finest
#                 # granularity (g=16, G=8) for the distortion proxy.
#                 G_fine   = self.num_groups        # 8
#                 g_fine   = self.group_size        # 16
#                 K_grouped = K_lh.view(T, G_fine, g_fine)
#                 r_groups_all = K_grouped.norm(dim=-1)   # [T, G_fine]
 
#                 for t in range(T):
#                     # Segment assignment (Algorithm 1, line 19):
#                     #   seg=0 (prefix)    : default
#                     #   seg=1 (retrieved) : middle block in RAG prompts
#                     #                       (not detectable here without prompt metadata;
#                     #                        set via token.segment_id after construction
#                     #                        if retrieval boundaries are known)
#                     #   seg=2 (recent)    : last RECENT_WINDOW positions
#                     if t >= T - RECENT_WINDOW:
#                         seg = 2   # recent suffix
#                     else:
#                         seg = 0   # prefix (default; caller can override for RAG)
 
#                     all_tokens.append(TokenState(
#                         layer=li, head=h, index=t,
#                         r=r[t], phi=phi[t],
#                         segment_id=seg,
#                         age=T - t,   # Algorithm 1 line 19: a_i = Tp - i
#                         prev_tier_id=0,
#                         protected=(t < self.sink_tokens),
#                         r_groups=r_groups_all[t],
#                     ))
 
#         del kv_pairs
 
#         print(f"[prefill] Allocating tiers for {len(all_tokens)} token-slots ...")
#         # Budget scales with context length (bits per token per head).
#         # A fixed global budget ignores context length — at short T it
#         # may never trigger compression (all tokens fit), at long T it
#         # may be too tight and drop most tokens.
#         import config as _cfg
#         _cfg.GLOBAL_BUDGET_BITS = 5_000_000
#         print(f"[prefill] Budget: {_cfg.GLOBAL_BUDGET_BITS:,} bits (fixed global budget)")
#         # _bpt = getattr(_cfg, 'BITS_PER_TOKEN', 30)
#         # _cfg.GLOBAL_BUDGET_BITS = _bpt * T * self.num_layers * self.num_kv_heads
#         # print(f"[prefill] Budget: {_cfg.GLOBAL_BUDGET_BITS:,} bits "
#         #       f"({_bpt} bpt × T={T} × L={self.num_layers} × H={self.num_kv_heads})")
#         retained = allocate(all_tokens, self.tiers, self.reuse, self.stability)
#         for ts in retained:
#             ts.commit_tier()
#         # Keep reference for online refresh and EMA updates
#         self._retained_tokens = retained
#         self._decode_step     = 0
#         pct = 100 * len(retained) / max(len(all_tokens), 1)
#         print(f"[prefill] Retained {len(retained)}/{len(all_tokens)} ({pct:.1f}%)")
 
#         print("[prefill] Encoding keys with codebooks and building pages ...")
#         self.per_head_pages = _build_per_head_pages(
#             retained_tokens=retained,
#             tiers=self.tiers,
#             K_all=K_all,
#             V_all=V_all,
#             codebooks=self.codebooks,
#             group_size=self.group_size,
#             num_groups=self.num_groups,
#             device=self.device,
#         )
 
#         # ── Store dense keys for online refresh page rebuild ─────────
#         self._K_all = {k: v.cpu() for k, v in K_all.items()}
#         self._V_all = {k: v.cpu() for k, v in V_all.items()} 
#         del V_all
#         torch.cuda.empty_cache()
 
#         # ── Build ctx_token_order (positional map for record_attn) ───
#         # Mirrors _build_per_head_pages: tokens grouped by (layer,head,tier_id),
#         # sorted by index within each tier group, then tiers ordered ascending.
#         self._ctx_token_order.clear()
#         lh_tier_map: Dict[Tuple[int,int,int], List] = defaultdict(list)
#         for ts in retained:
#             lh_tier_map[(ts.layer, ts.head, ts.new_tier_id)].append(ts)
#         lh_map: Dict[Tuple[int,int], List] = defaultdict(list)
#         for (layer, head, tid), toks in lh_tier_map.items():
#             toks.sort(key=lambda t: t.index)
#             lh_map[(layer, head)].extend(toks)
#         self._ctx_token_order = dict(lh_map)
 
#         del K_all
 
#         print("[prefill] Patching attention layers for decode ...")
#         self._original_forwards = patch_for_decode(self.model, self)
#         self._patched = True
#         print("[prefill] Done -- model ready for compressed decode.")
 
#     # ------------------------------------------------------------------
#     # Attention: batched over q heads (primary path)
#     # ------------------------------------------------------------------
 
#     def _compressed_head_attention_batched(
#         self,
#         layer_idx: int,
#         kv_head:   int,
#         q_batch:   torch.Tensor,   # [num_q, dh]
#         k_new:     torch.Tensor,   # [dh]
#         v_new:     torch.Tensor,   # [dh]
#     ) -> torch.Tensor:             # [num_q, dh]
#         """
#         Compute compressed attention for ALL q heads sharing this kv_head
#         in a single batched kernel launch.
 
#         Compressed prefill pages -> one _call_fused per tier (one launch).
#         Staging buffer           -> vectorized einsum, zero kernel launches.
#         New token                -> batched dot product.
#         """
#         key    = (layer_idx, kv_head)
#         device = q_batch.device
#         num_q  = q_batch.shape[0]
#         q      = q_batch.float()    # [num_q, dh]
#         kn     = k_new.float().to(device)
#         vn     = v_new.float().to(device)
#         dh     = q.shape[1]
 
#         with nvtx_range("page_lookup"):
#             ph = self.per_head_pages.get(key, [])
#             # Build b_theta → tier_id map once per call
#             _bt_to_tid = {t.b_theta: t.tier_id
#                           for t in self.tiers if t.tier_id != 0}

#         # ctx_logits_parts : each [num_q, n_tokens_for_this_tier]
#         ctx_logits_parts: List[torch.Tensor] = []
#         V_ctx_parts:      List[torch.Tensor] = []

#         with nvtx_range("angle_logits"):
#             for (pt, ptt, b_theta, n_tokens, V_tier, K_tier) in ph:
#                 # Resolve per-tier codebook and geometry
#                 tier_id  = _bt_to_tid.get(b_theta, 1)
#                 cb_lh    = get_codebook(self.codebooks, layer_idx, kv_head, tier_id)
#                 tier_obj = self.tiers[tier_id]
#                 tier_G   = tier_obj.G
#                 tier_g   = tier_obj.g

#                 if self.use_fused:
#                     # one kernel launch for all num_q heads at once
#                     raw = _call_fused(
#                         pt, ptt, q, cb_lh, b_theta,
#                         dh, tier_G, tier_g,
#                     )
#                     # raw: [num_q, num_pages * page_size]
#                     ctx_logits_parts.append(raw[:, :n_tokens])  # [num_q, n_tokens]
#                 else:
#                     # reference path: vectorized over q heads
#                     logits_q = [
#                         reference_codebook_decode(
#                             q[qi], cb_lh, K_tier,
#                             tier_g, tier_G,
#                         )
#                         for qi in range(num_q)
#                     ]
#                     ctx_logits_parts.append(
#                         torch.stack(logits_q, dim=0)  # [num_q, n_tokens]
#                     )
#                 V_ctx_parts.append(V_tier)  # [n_tokens, dh]

#             # ── staging buffer: decode tokens are always b3 (recent window)
#             stg_r     = self.stg_r.get(key)
#             stg_theta = self.stg_theta.get(key)
#             stg_V     = self.stg_V.get(key)
#             if stg_r is not None and stg_r.shape[0] > 0:
#                 N_stg     = stg_r.shape[0]
#                 # Staging tokens encoded at b3 — use b3 geometry
#                 b3        = self.tiers[3]
#                 stg_G     = b3.G
#                 stg_g     = b3.g
#                 cb_b3     = get_codebook(self.codebooks, layer_idx, kv_head, b3.tier_id)
#                 q_groups  = q.view(num_q, stg_G, stg_g)  # [num_q, G_b3, g_b3]

#                 # gather codewords for all staging tokens: [G_b3, N_stg, g_b3]
#                 cw = cb_b3[
#                     torch.arange(stg_G, device=device).unsqueeze(1),  # [G_b3, 1]
#                     stg_theta.long().T                                  # [G_b3, N_stg]
#                 ]

#                 # dot products: [num_q, G_b3, N_stg]
#                 dots = torch.einsum('qgi,gni->qgn', q_groups, cw)

#                 # weighted sum over groups: [num_q, N_stg]
#                 stg_logits = (stg_r.T.unsqueeze(0) * dots).sum(1) / math.sqrt(dh)

#                 ctx_logits_parts.append(stg_logits)   # [num_q, N_stg]
#                 V_ctx_parts.append(stg_V)
 
#         if ctx_logits_parts:
#             ctx_logits = torch.cat(ctx_logits_parts, dim=1)  # [num_q, total_ctx]
#             V_ctx      = torch.cat(V_ctx_parts, dim=0)       # [total_ctx, dh]
#         else:
#             ctx_logits = torch.zeros(num_q, 0, device=device)
#             V_ctx      = torch.zeros(0, dh, device=device)
 
#         # new token logit for each q head: [num_q]
#         new_logits = (q @ kn) / math.sqrt(dh)
 
#         # full logits: [num_q, total_ctx + 1]
#         all_logits = torch.cat([ctx_logits, new_logits.unsqueeze(1)], dim=1)
 
#         with nvtx_range("softmax"):
#             attn = torch.softmax(all_logits, dim=1)   # [num_q, total_ctx + 1]
 
#         # ── C.2: update EMA importance weights ───────────────────────
#         # attn[:, :-1] are weights over the ctx tokens (all q heads).
#         # We average over q heads (GQA) and record per retained token.
#         # ctx_token_count tells us how many ctx tokens exist.
#         n_ctx = ctx_logits.shape[1] if ctx_logits_parts else 0
#         if n_ctx > 0:
#             # C.2: update EMA importance weights using per-position attention.
#             # Average over q heads (GQA) → [n_ctx] scalar per ctx position.
#             attn_ctx_mean = attn[:, :n_ctx].mean(dim=0).detach().cpu()
#             # _ctx_token_order gives the exact positional ordering used when
#             # pages were built — O(1) lookup, no linear scan over all tokens.
#             order = self._ctx_token_order.get((layer_idx, kv_head), [])
#             for pos, tok in enumerate(order):
#                 if pos < n_ctx:
#                     tok.record_attn(float(attn_ctx_mean[pos]))
 
#         V_full = torch.cat([V_ctx, vn.unsqueeze(0)], dim=0)  # [total_ctx + 1, dh]
 
#         with nvtx_range("kv_read"):
#             attn_out = attn @ V_full   # [num_q, dh]
 
#         return attn_out.to(q_batch.dtype)
 
#     def _compressed_head_attention(
#         self,
#         layer_idx: int,
#         kv_head:   int,
#         q_vec:     torch.Tensor,   # [dh]
#         k_new:     torch.Tensor,   # [dh]
#         v_new:     torch.Tensor,   # [dh]
#     ) -> torch.Tensor:             # [dh]
#         """Single-q wrapper used by evaluate.py."""
#         out = self._compressed_head_attention_batched(
#             layer_idx, kv_head,
#             q_vec.unsqueeze(0),   # [1, dh]
#             k_new, v_new,
#         )
#         return out.squeeze(0)     # [dh]
 
#     # ------------------------------------------------------------------
#     # Decode staging buffer management
#     # ------------------------------------------------------------------
 
#     def _append_decode_kv(
#         self,
#         layer_idx: int,
#         kv_head:   int,
#         k_new:     torch.Tensor,   # [dh]
#         v_new:     torch.Tensor,   # [dh]
#     ) -> None:
#         """
#         Compress the new decode token and append to the staging buffer.
#         Flush to a compressed page when PAGE_SIZE tokens accumulate.
#         Dense K is NEVER stored -- codes are computed immediately.
#         """
#         key    = (layer_idx, kv_head)
#         device = self.device
#         kn     = k_new.float().to(device).unsqueeze(0)   # [1, dh]
#         vn     = v_new.float().to(device).unsqueeze(0)   # [1, dh]
 
#         # Decode tokens are in the recent window → always tier b3 (Low)
#         decode_tier   = self.tiers[3]   # b3 (Low = bmin for recent tokens)
#         cb_key        = (layer_idx, kv_head, decode_tier.tier_id)
#         cb_lh         = self.codebooks.get(cb_key)
#         if cb_lh is None:
#             cb_lh = get_codebook(self.codebooks, layer_idx, kv_head, decode_tier.tier_id)
#         r, theta = _encode_keys_with_codebooks(
#             kn, cb_lh, decode_tier.g, decode_tier.G
#         )   # r: [1, G_b3], theta: [1, G_b3] int32
 
#         if key not in self.stg_r:
#             self.stg_r[key]     = r
#             self.stg_theta[key] = theta
#             self.stg_V[key]     = vn
#         else:
#             self.stg_r[key]     = torch.cat([self.stg_r[key],     r],     dim=0)
#             self.stg_theta[key] = torch.cat([self.stg_theta[key], theta],  dim=0)
#             self.stg_V[key]     = torch.cat([self.stg_V[key],     vn],    dim=0)
 
#         if self.stg_r[key].shape[0] >= PAGE_SIZE:
#             self._flush_decode_page(key)
 
#         # Trigger periodic controller refresh (Appendix C.3)
#         # Only fires on the first (layer=0, head=0) call per decode step
#         # to avoid refreshing N_layers × N_heads times per step.
#         if layer_idx == 0 and kv_head == 0:
#             self._maybe_refresh()
 
#     def _flush_decode_page(self, key: Tuple[int, int]) -> None:
#         """
#         Pack the staging buffer into a compressed page and add it to
#         per_head_pages.  Called automatically when staging buffer fills.
#         """
#         device    = self.device
#         r_buf     = self.stg_r[key]      # [PAGE_SIZE, G]
#         theta_buf = self.stg_theta[key]  # [PAGE_SIZE, G]
#         V_buf     = self.stg_V[key]      # [PAGE_SIZE, dh]
#         n         = r_buf.shape[0]
#         tier      = self.tiers[3]   # b3 (Low) for all decode tokens
 
#         page = build_page_from_codes(
#             r_buf.cpu(), theta_buf.cpu(), tier, segment_id=2
#         )
 
#         header_size   = HEADER_BYTES + tier.G * 4
#         theta_bytes   = ((n * tier.G * tier.b_theta) + 7) // 8
#         theta_offset  = header_size
#         radius_offset = theta_offset + theta_bytes
 
#         ptable = PointerTable()
#         ptable.add_page(0, theta_offset, radius_offset)
 
#         if key not in self.per_head_pages:
#             self.per_head_pages[key] = []
 
#         self.per_head_pages[key].append((
#             page.to(device),
#             ptable.to_tensor().to(device),
#             tier.b_theta,
#             n,
#             V_buf,
#             None,   # no dense K stored — paper compliant
#         ))
 
#         del self.stg_r[key]
#         del self.stg_theta[key]
#         del self.stg_V[key]
 
#     # ------------------------------------------------------------------
#     # Online controller refresh  (Appendix C.3)
#     # ------------------------------------------------------------------
 
#     def _maybe_refresh(self) -> None:
#         """
#         Re-run the RDR controller on the current retained token set.
#         Called every REFRESH_CADENCE decode steps.
#         Skipped if REFRESH_CADENCE == 0 (prefill-only mode).
 
#         What happens:
#           1. allocate() re-scores all retained tokens using updated omega.
#           2. Tokens that now exceed the budget are downgraded or dropped.
#           3. Tokens below the budget get upgrade consideration.
#           4. Frozen tokens (cooldown > 0) are skipped (C.4 hysteresis).
#         """
#         if REFRESH_CADENCE == 0 or not self._retained_tokens:
#             return
#         # _decode_step is incremented in _append_decode_kv (layer=0,head=0).
#         # Here we only check whether we should act.
#         if self._decode_step % REFRESH_CADENCE != 0:
#             return
 
#         # online_refresh from allocation.py: ticks cooldowns, runs downgrade
#         # + upgrade passes, respects frozen tokens (C.3 + C.4).
#         old_tiers = {id(ts): ts.new_tier_id for ts in self._retained_tokens}
#         updated   = online_refresh(self._retained_tokens, self.tiers)
 
#         # Commit and detect which (layer, head) pairs changed tier
#         changed_lh = set()
#         for tok in updated:
#             if tok.new_tier_id != old_tiers.get(id(tok)):
#                 changed_lh.add((tok.layer, tok.head))
#             tok.commit_tier()
 
#         self._retained_tokens = updated
 
#         # Rebuild compressed pages for (layer, head) pairs that changed
#         if changed_lh:
#             self._rebuild_pages_for(changed_lh)
 
#     # ------------------------------------------------------------------
#     # Lifecycle
#     # ------------------------------------------------------------------
 
#     def _rebuild_pages_for(self, changed_lh: set) -> None:
#         """
#         After an online refresh, re-encode and re-page the (layer, head)
#         pairs in changed_lh using the current tier assignments in
#         self._retained_tokens and the stored dense keys in self._K_all.
#         Updates self.per_head_pages and self._ctx_token_order in place.
#         """
#         # Group current retained tokens by (layer, head, tier_id)
#         lh_tier: Dict[Tuple[int,int,int], List] = defaultdict(list)
#         for ts in self._retained_tokens:
#             key = (ts.layer, ts.head, ts.new_tier_id)
#             lh_tier[key].append(ts)
 
#         for (layer, head) in changed_lh:
#             K_lh = self._K_all.get((layer, head))
#             if K_lh is None:
#                 continue  # no stored keys (shouldn't happen)
 
#             new_entries = []
#             new_order   = []
 
#             # Process tiers in ascending tier_id order (b1 → b2 → b3)
#             for tier_id in [1, 2, 3]:
#                 toks = lh_tier.get((layer, head, tier_id), [])
#                 if not toks:
#                     continue
#                 toks.sort(key=lambda t: t.index)
#                 tier = self.tiers[tier_id]
 
#                 # Tier-specific codebook
#                 cb = self.codebooks.get((layer, head, tier_id))
#                 if cb is None:
#                     cb = get_codebook(self.codebooks, layer, head, tier_id)
#                 if cb is None:
#                     continue
                
#                 V_lh = self._V_all.get((layer, head))
#                 page_list, K_parts, V_parts = [], [], []
#                 ptable = PointerTable()
#                 offset = 0
 
#                 for i in range(0, len(toks), PAGE_SIZE):
#                     chunk   = toks[i : i + PAGE_SIZE]
#                     N       = len(chunk)
#                     indices = [t.index for t in chunk]
#                     K_chunk = K_lh[indices]
#                     r_g, th_c = _encode_keys_with_codebooks(
#                         K_chunk, cb, tier.g, tier.G)
#                     page = build_page_from_codes(
#                         r_g, th_c, tier, chunk[0].segment_id)
#                     h_sz  = HEADER_BYTES + tier.G * 4
#                     t_sz  = ((N * tier.G * tier.b_theta) + 7) // 8
#                     ptable.add_page(offset, offset + h_sz, offset + h_sz + t_sz)
#                     offset += page.numel()
#                     page_list.append(page)
#                     K_parts.append(K_chunk)
#                     if V_lh is not None:                         # ← ADD
#                         V_parts.append(V_lh[indices].to(self.device))  # ← ADD

#                 V_tensor = torch.cat(V_parts, dim=0) if V_parts else None
                
#                 new_entries.append((
#                     torch.cat(page_list).to(self.device),
#                     ptable.to_tensor().to(self.device),
#                     tier.b_theta,
#                     len(toks),
#                     V_tensor,                        
#                     torch.cat(K_parts),          # CPU, reference path
#                 ))
#                 new_order.extend(toks)
 
#             self.per_head_pages[(layer, head)]   = new_entries
#             self._ctx_token_order[(layer, head)] = new_order
 
#     def uninstall(self) -> None:
#         if self._patched:
#             unpatch_decode(self.model, self._original_forwards)
#             self._patched = False
#             print("[SphericalKVPipeline] Attention layers restored.")
 
#     def _reset_state(self) -> None:
#         self.per_head_pages.clear()
#         self.stg_r.clear()
#         self.stg_theta.clear()
#         self.stg_V.clear()
#         self.reuse          = None
#         self.stability      = None
#         self.seq_len        = 0
#         self._decode_step   = 0
#         self._retained_tokens.clear()
#         self._ctx_token_order.clear()
#         self._K_all.clear()
#         self._V_all.clear()

# BELOW CODE HAS ONLY VECTORIZATION BUT HAS ADA VIOLATIONS 

# from __future__ import annotations
 
# import contextlib
# import math
# import os
# import sys
# from collections import defaultdict
# from typing import Dict, List, Optional, Tuple
 
# import torch
 
 
# @contextlib.contextmanager
# def nvtx_range(name: str):
#     """
#     NVTX range for Nsight Compute / Nsight Systems.
#     Algorithm instrumentation (line 73): page_lookup, kv_read,
#     angle_logits, softmax, proj.  No-ops when CUDA unavailable.
#     """
#     if torch.cuda.is_available():
#         torch.cuda.nvtx.range_push(name)
#     try:
#         yield
#     finally:
#         if torch.cuda.is_available():
#             torch.cuda.nvtx.range_pop()
 
# _HERE = os.path.dirname(os.path.abspath(__file__))
# if _HERE not in sys.path:
#     sys.path.insert(0, _HERE)
 
# from allocation import allocate, online_refresh
# from codebook_loader import get_codebook
# from config import (EPS, GROUP_SIZE, NUM_GROUPS, PAGE_SIZE, HEADER_BYTES,
#                                REFRESH_CADENCE, SINK_TOKENS, RECENT_WINDOW,
#                                COOLDOWN_STEPS)
# from llama_hooks import (
#     aggregate_proxy_to_kv_heads,
#     build_attn_weights_tensor,
#     build_head_outputs_tensor,
#     capture_prefill_pass,
#     patch_for_decode,
#     unpatch_decode,
# )
# from pagebuilder import build_page_from_codes
# from pointer_table import PointerTable
# from resuse_proxy import compute_reuse_proxy
# from spherical_parameterization import spherical_parameterize
# from stability_proxy import compute_stability_proxy
# from tiers import build_tiers
# from token_state import TokenState
 
 
# # ---------------------------------------------------------------------------
# # Reference ADA decode (pure torch -- fallback when CUDA kernel unavailable)
# # Vectorized: no Python loop over groups.
# # ---------------------------------------------------------------------------
 
# def reference_codebook_decode(
#     q:          torch.Tensor,   # [dh]
#     codebooks:  torch.Tensor,   # [G, codebook_size, group_size]
#     K_ctx:      torch.Tensor,   # [T_ctx, dh]
#     group_size: int,
#     num_groups: int,
# ) -> torch.Tensor:              # [T_ctx]
#     """
#     Vectorized reference decode.
#     Matches decode_kernel.cu:
#         dot   = q^(g) . cw^(g)          (raw q group, not normalised)
#         acc   = sum_g r_hat^(g) * dot^(g)
#         logit = acc / sqrt(dh)
#     """
#     T_ctx, dh = K_ctx.shape
#     device    = q.device
 
#     q_groups  = q.view(num_groups, group_size)                          # [G, g]
#     K_grouped = K_ctx.view(T_ctx, num_groups, group_size)               # [T, G, g]
#     r_groups  = K_grouped.norm(dim=-1)                                  # [T, G]
#     K_dir     = K_grouped / (r_groups.unsqueeze(-1) + EPS)              # [T, G, g]
 
#     # vectorized nearest-neighbour: [T, G, cb_size]
#     sims     = torch.einsum('tgi,gci->tgc', K_dir, codebooks)
#     best_idx = sims.argmax(dim=-1)                                      # [T, G]
 
#     # gather codewords: [T, G, g]
#     cw = codebooks[
#         torch.arange(num_groups, device=device).unsqueeze(0),           # [1, G]
#         best_idx                                                         # [T, G]
#     ]
 
#     # dot product and weighted sum
#     dots   = (cw * q_groups.unsqueeze(0)).sum(-1)                       # [T, G]
#     logits = (r_groups * dots).sum(-1) / math.sqrt(dh)                  # [T]
 
#     return logits
 
 
# # ---------------------------------------------------------------------------
# # Fused decode availability + dispatch
# # ---------------------------------------------------------------------------
 
# _fused_ok: Optional[bool] = None
 
# def _fused_available() -> bool:
#     global _fused_ok
#     if _fused_ok is None:
#         try:
#             from fused_decode import fused_decode   # noqa: F401
#             _fused_ok = True
#         except Exception:
#             _fused_ok = False
#     return _fused_ok
 
 
# def _call_fused(
#     pages_tensor:  torch.Tensor,    # packed uint8
#     ptable_tensor: torch.Tensor,    # [P, 3] int32
#     q:             torch.Tensor,    # [dh] OR [num_q, dh]
#     codebooks_lh:  torch.Tensor,    # [G, cb_size, g]
#     b_theta:       int,
#     dh:            int,
#     num_groups:    int,
#     group_size:    int,
# ) -> torch.Tensor:
#     """
#     Dispatches to the batched CUDA kernel.
#     If q is 1D [dh], unsqueezes to [1, dh] and squeezes result back.
#     If q is 2D [num_q, dh], returns [num_q, num_pages * page_size].
#     """
#     from fused_decode import fused_decode
#     squeeze = (q.dim() == 1)
#     if squeeze:
#         q = q.unsqueeze(0)   # [1, dh]
#     result = fused_decode(
#         pages_tensor,
#         ptable_tensor,
#         q.float(),
#         codebooks_lh.float(),
#         dh=dh,
#         groups=num_groups,
#         group_size=group_size,
#         b_theta=b_theta,
#         page_size=PAGE_SIZE,
#     )
#     # result: [num_q, num_pages * page_size]
#     if squeeze:
#         return result.squeeze(0)   # [num_pages * page_size]
#     return result
 
 
# # ---------------------------------------------------------------------------
# # Codebook encoding  (vectorized, no Python loops)
# # ---------------------------------------------------------------------------
 
# def _encode_keys_with_codebooks(
#     K_chunk:    torch.Tensor,   # [N, dh]
#     codebooks:  torch.Tensor,   # [G, codebook_size, group_size]  tier-specific
#     group_size: int,             # g for this tier
#     num_groups: int,             # G for this tier
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Encode N key vectors using tier-specific grouped VQ (Appendix B.1).
 
#     Returns
#     -------
#     r_groups    : [N, G]  float32 -- per-group L2 norms
#     theta_codes : [N, G]  int32   -- nearest codebook index per group
#     """
#     N         = K_chunk.shape[0]
#     K_grouped = K_chunk.view(N, num_groups, group_size)      # [N, G, g]
#     r_groups  = K_grouped.norm(dim=-1)                       # [N, G]
#     K_dir     = K_grouped / (r_groups.unsqueeze(-1) + EPS)   # [N, G, g]
 
#     cb   = codebooks.to(K_dir.device)                        # [G, cb_size, g]
#     sims = torch.einsum('ngi,gci->ngc', K_dir, cb)           # [N, G, cb_size]
#     theta_codes = sims.argmax(dim=-1).to(torch.int32)        # [N, G]
#     return r_groups, theta_codes
 
 
# # ---------------------------------------------------------------------------
# # Per-(layer, kv_head) page builder  (prefill only)
# # ---------------------------------------------------------------------------
 
# def _build_per_head_pages(
#     retained_tokens: List[TokenState],
#     tiers,
#     K_all:      Dict[Tuple[int, int], torch.Tensor],
#     V_all:      Dict[Tuple[int, int], torch.Tensor],
#     codebooks:  Dict[Tuple[int, int], torch.Tensor],
#     group_size: int,
#     num_groups: int,
#     device,
# ) -> Dict[Tuple[int, int], List]:
#     """
#     Build tier-homogeneous, codebook-encoded pages per (layer, kv_head).
 
#     Each entry in the returned list is a 6-tuple:
#         (pages_tensor, ptable_tensor, b_theta, n_tokens, V_tier, K_tier)
 
#     pages_tensor  -- packed uint8 bitstream for all pages in this tier.
#     ptable_tensor -- [P, 3] int32 pointer table.
#     b_theta       -- bits per codebook index for this tier.
#     n_tokens      -- exact count of valid tokens across all pages.
#     V_tier        -- [n_tokens, dh] float on device.
#     K_tier        -- [n_tokens, dh] float on CPU (reference path only).
#     """
#     groups: Dict[Tuple[int, int, int], List[TokenState]] = defaultdict(list)
#     for ts in retained_tokens:
#         groups[(ts.layer, ts.head, ts.new_tier_id)].append(ts)
 
#     per_head: Dict[Tuple[int, int], List] = defaultdict(list)
 
#     for (layer, head, tier_id), tokens in groups.items():
#         tokens.sort(key=lambda t: t.index)
#         tier  = tiers[tier_id]
#         K_lh  = K_all[(layer, head)]
#         V_lh  = V_all[(layer, head)]
#         # ── Tier-specific codebook: keyed by (layer, head, tier_id) ─────
#         # Appendix B.1: b1/b2 use g=16, G=8 codebooks;
#         #               b3 uses g=32, G=4 codebooks.
#         cb_key = (layer, head, tier_id)
#         if cb_key in codebooks:
#             cb_lh = codebooks[cb_key]           # [G, cb_size, g]  tier-aware
#         elif (layer, head) in codebooks:
#             cb_lh = codebooks[(layer, head)]    # legacy format fallback
#         else:
#             continue                            # no codebook, skip tier
 
#         tier_g  = tier.g    # group size for this tier
#         tier_G  = tier.G    # number of groups for this tier
 
#         page_list:   List[torch.Tensor] = []
#         r_parts:     List[torch.Tensor] = []   # [N_chunk, G] float32
#         theta_parts: List[torch.Tensor] = []   # [N_chunk, G] int32
#         V_tier_parts: List[torch.Tensor] = []
#         ptable = PointerTable()
#         offset = 0
 
#         for i in range(0, len(tokens), PAGE_SIZE):
#             chunk   = tokens[i : i + PAGE_SIZE]
#             N       = len(chunk)
#             indices = [t.index for t in chunk]
 
#             K_chunk = K_lh[indices]
#             V_chunk = V_lh[indices]
#             r_groups, theta_codes = _encode_keys_with_codebooks(
#                 K_chunk, cb_lh, tier_g, tier_G   # tier-specific g and G
#             )
 
#             page = build_page_from_codes(
#                 r_groups, theta_codes, tier, chunk[0].segment_id
#             )
 
#             header_size   = HEADER_BYTES + tier_G * 4
#             theta_bytes   = ((N * tier_G * tier.b_theta) + 7) // 8
#             theta_offset  = offset + header_size
#             radius_offset = theta_offset + theta_bytes
 
#             page_list.append(page)
#             # FIX Bug-2: store pre-encoded codes, NOT dense K.
#             # Dense K is never needed at decode time -- the reference
#             # path reads (r_groups, theta_codes) directly, matching
#             # the staging-buffer pattern and satisfying the ADA
#             # 'no dense reconstruction' invariant (Alg.1 §II line 89).
#             r_parts.append(r_groups.cpu())        # [N, G] float32
#             theta_parts.append(theta_codes.cpu()) # [N, G] int32
#             V_tier_parts.append(V_chunk)
#             ptable.add_page(offset, theta_offset, radius_offset)
#             offset += page.numel()
 
#         per_head[(layer, head)].append((
#             torch.cat(page_list).to(device),
#             ptable.to_tensor().to(device),
#             tier.b_theta,
#             len(tokens),
#             torch.cat(V_tier_parts, dim=0).to(device),
#             # Slot [5]: (r_groups, theta_codes) on CPU.
#             # The reference decode path unpacks these directly --
#             # no encode/decode round-trip, no dense K materialisation.
#             (torch.cat(r_parts, dim=0), torch.cat(theta_parts, dim=0)),
#         ))
 
#     return per_head
 
 
# # ---------------------------------------------------------------------------
# # Pipeline
# # ---------------------------------------------------------------------------
 
# class SphericalKVPipeline:
#     """
#     Spherical KV Cache inference pipeline.
 
#     Usage
#     -----
#     pipeline = SphericalKVPipeline(model, tokenizer, codebooks, device)
#     pipeline.prefill(input_ids)
#     out = model.generate(input_ids, ...)
#     pipeline.uninstall()
#     """
 
#     def __init__(
#         self,
#         model,
#         tokenizer,
#         codebooks: Dict[Tuple[int, int], torch.Tensor],
#         device:     torch.device = torch.device("cpu"),
#         head_dim:   Optional[int] = None,
#         group_size: Optional[int] = None,
#         sink_tokens: int = 4,
#         use_fused:   bool = True,
#     ):
#         self.model      = model
#         self.tokenizer  = tokenizer
#         self.codebooks  = {k: v.to(device) for k, v in codebooks.items()}
#         self.device     = device
#         self.sink_tokens = sink_tokens
#         self.use_fused  = use_fused and _fused_available()
 
#         cfg = model.config
#         self.num_layers   = cfg.num_hidden_layers
#         self.num_q_heads  = cfg.num_attention_heads
#         self.num_kv_heads = getattr(cfg, "num_key_value_heads", self.num_q_heads)
#         self.head_dim     = head_dim or getattr(
#             cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
#         )
#         self.group_size = group_size or GROUP_SIZE
#         self.num_groups = self.head_dim // self.group_size
#         self.tiers      = build_tiers(self.head_dim)
 
#         # Prefill compressed pages: per (layer, kv_head) list of 6-tuples
#         self.per_head_pages: Dict[Tuple[int, int], List] = {}
 
#         # Decode staging buffers: compressed codes for recent decode tokens
#         # not yet flushed into a full page.  At most PAGE_SIZE-1 tokens each.
#         # stg_r     : [N_stg, G] float32  -- per-group radii
#         # stg_theta : [N_stg, G] int32    -- codebook indices
#         # stg_V     : [N_stg, dh] float32 -- value vectors
#         self.stg_r:     Dict[Tuple[int, int], torch.Tensor] = {}
#         self.stg_theta: Dict[Tuple[int, int], torch.Tensor] = {}
#         self.stg_V:     Dict[Tuple[int, int], torch.Tensor] = {}
 
#         self.reuse:     Optional[torch.Tensor] = None
#         self.stability: Optional[torch.Tensor] = None
#         self.seq_len:   int = 0
 
#         # ── Online refresh state (Appendix C.3) ──────────────────────
#         # All retained TokenState objects from prefill, kept alive so
#         # record_attn() and the controller refresh can use them.
#         self._retained_tokens: list = []
#         self._decode_step: int = 0   # counts decode steps since prefill
 
#         # ctx_token_order[(layer, head)]: retained TokenState objects in the
#         # same positional order they appear in ctx_logits.  Built at prefill
#         # so record_attn() can map attention position → TokenState in O(1).
#         self._ctx_token_order: Dict[Tuple[int, int], List] = {}
 
#         # Dense prefill keys kept for page rebuild after online refresh.
#         # _K_all[(layer, head)] = Tensor[T, dh] float32 CPU.
#         self._K_all = {}
#         self._V_all = {}
 
#         self._original_forwards: Dict[int, object] = {}
#         self._patched = False
 
#         mode = "fused CUDA" if self.use_fused else "reference (pure-torch)"
#         print(
#             f"[SphericalKVPipeline] layers={self.num_layers}  "
#             f"kv_heads={self.num_kv_heads}  head_dim={self.head_dim}  "
#             f"groups={self.num_groups}  decode_mode={mode}"
#         )
 
#     # ------------------------------------------------------------------
#     # Prefill
#     # ------------------------------------------------------------------
 
#     def prefill(
#         self,
#         input_ids:      torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#     ) -> None:
#         if self._patched:
#             self.uninstall()
#         self._reset_state()
 
#         input_ids = input_ids.to(self.device)
#         B, T = input_ids.shape
#         self.seq_len = T
 
#         print(f"[prefill] Forward pass on {T} tokens ...")
#         kv_pairs, attn_list, ho_list, logits = capture_prefill_pass(
#             self.model, input_ids, attention_mask
#         )
 
#         attn_stacked = build_attn_weights_tensor(attn_list)
#         ho_stacked   = build_head_outputs_tensor(ho_list)
 
#         if attn_stacked is not None:
#             reuse_kv = aggregate_proxy_to_kv_heads(
#                 compute_reuse_proxy(attn_stacked),
#                 self.num_q_heads, self.num_kv_heads,
#             )
#         else:
#             reuse_kv = torch.ones(self.num_layers, self.num_kv_heads) / self.num_kv_heads
 
#         if ho_stacked is not None:
#             stab_kv = aggregate_proxy_to_kv_heads(
#                 compute_stability_proxy(
#                     ho_stacked[:, :, :, -1, :], logits[:, -1, :]
#                 ),
#                 self.num_q_heads, self.num_kv_heads,
#             )
#         else:
#             stab_kv = torch.ones(self.num_layers, self.num_kv_heads) / self.num_kv_heads
 
#         self.reuse     = reuse_kv.cpu()
#         self.stability = stab_kv.cpu()
 
#         del attn_stacked, ho_stacked, attn_list, ho_list
#         torch.cuda.empty_cache()
 
#         print("[prefill] Building TokenStates ...")
#         all_tokens: List[TokenState] = []
#         K_all: Dict[Tuple[int, int], torch.Tensor] = {}
#         V_all: Dict[Tuple[int, int], torch.Tensor] = {}
 
#         for li, (K, V) in enumerate(kv_pairs):
#             for h in range(self.num_kv_heads):
#                 K_lh = K[0, h].float().cpu()    # float32 for encoding
#                 V_lh = V[0, h].half().cpu()     # fp16 to match dense baseline
#                 K_all[(li, h)] = K_lh
#                 V_all[(li, h)] = V_lh
 
#                 r, phi = spherical_parameterize(K_lh)
 
#                 # Appendix B.1 / C.2: compute per-group radii at finest
#                 # granularity (g=16, G=8) for the distortion proxy.
#                 G_fine   = self.num_groups        # 8
#                 g_fine   = self.group_size        # 16
#                 K_grouped = K_lh.view(T, G_fine, g_fine)
#                 r_groups_all = K_grouped.norm(dim=-1)   # [T, G_fine]
 
#                 for t in range(T):
#                     # Segment assignment (Algorithm 1, line 19):
#                     #   seg=0 (prefix)    : default
#                     #   seg=1 (retrieved) : middle block in RAG prompts
#                     #                       (not detectable here without prompt metadata;
#                     #                        set via token.segment_id after construction
#                     #                        if retrieval boundaries are known)
#                     #   seg=2 (recent)    : last RECENT_WINDOW positions
#                     if t >= T - RECENT_WINDOW:
#                         seg = 2   # recent suffix
#                     else:
#                         seg = 0   # prefix (default; caller can override for RAG)
 
#                     all_tokens.append(TokenState(
#                         layer=li, head=h, index=t,
#                         r=r[t], phi=phi[t],
#                         segment_id=seg,
#                         age=T - t,   # Algorithm 1 line 19: a_i = Tp - i
#                         prev_tier_id=0,
#                         protected=(t < self.sink_tokens),
#                         r_groups=r_groups_all[t],
#                     ))
 
#         del kv_pairs
 
#         print(f"[prefill] Allocating tiers for {len(all_tokens)} token-slots ...")
#         # Budget scales with context length (bits per token per head).
#         # A fixed global budget ignores context length — at short T it
#         # may never trigger compression (all tokens fit), at long T it
#         # may be too tight and drop most tokens.
#         import config as _cfg
#         _bpt = getattr(_cfg, 'BITS_PER_TOKEN', 30)
#         _cfg.GLOBAL_BUDGET_BITS = _bpt * T * self.num_layers * self.num_kv_heads
#         print(f"[prefill] Budget: {_cfg.GLOBAL_BUDGET_BITS:,} bits "
#               f"({_bpt} bpt × T={T} × L={self.num_layers} × H={self.num_kv_heads})")
#         retained = allocate(all_tokens, self.tiers, self.reuse, self.stability)
#         for ts in retained:
#             ts.commit_tier()
#         # Keep reference for online refresh and EMA updates
#         self._retained_tokens = retained
#         self._decode_step     = 0
#         pct = 100 * len(retained) / max(len(all_tokens), 1)
#         print(f"[prefill] Retained {len(retained)}/{len(all_tokens)} ({pct:.1f}%)")
 
#         print("[prefill] Encoding keys with codebooks and building pages ...")
#         self.per_head_pages = _build_per_head_pages(
#             retained_tokens=retained,
#             tiers=self.tiers,
#             K_all=K_all,
#             V_all=V_all,
#             codebooks=self.codebooks,
#             group_size=self.group_size,
#             num_groups=self.num_groups,
#             device=self.device,
#         )
 
#         # ── Store dense keys for online refresh page rebuild ─────────
#         self._K_all = {k: v.cpu() for k, v in K_all.items()}
#         self._V_all = {k: v.cpu() for k, v in V_all.items()} 
#         del V_all
#         torch.cuda.empty_cache()
 
#         # ── Build ctx_token_order (positional map for record_attn) ───
#         # Mirrors _build_per_head_pages: tokens grouped by (layer,head,tier_id),
#         # sorted by index within each tier group, then tiers ordered ascending.
#         self._ctx_token_order.clear()
#         lh_tier_map: Dict[Tuple[int,int,int], List] = defaultdict(list)
#         for ts in retained:
#             lh_tier_map[(ts.layer, ts.head, ts.new_tier_id)].append(ts)
#         lh_map: Dict[Tuple[int,int], List] = defaultdict(list)
#         for (layer, head, tid), toks in lh_tier_map.items():
#             toks.sort(key=lambda t: t.index)
#             lh_map[(layer, head)].extend(toks)
#         self._ctx_token_order = dict(lh_map)
 
#         del K_all
 
#         print("[prefill] Patching attention layers for decode ...")
#         self._original_forwards = patch_for_decode(self.model, self)
#         self._patched = True
#         print("[prefill] Done -- model ready for compressed decode.")
 
#     # ------------------------------------------------------------------
#     # Attention: batched over q heads (primary path)
#     # ------------------------------------------------------------------
 
#     def _compressed_head_attention_batched(
#         self,
#         layer_idx: int,
#         kv_head:   int,
#         q_batch:   torch.Tensor,   # [num_q, dh]
#         k_new:     torch.Tensor,   # [dh]
#         v_new:     torch.Tensor,   # [dh]
#     ) -> torch.Tensor:             # [num_q, dh]
#         """
#         Compute compressed attention for ALL q heads sharing this kv_head
#         in a single batched kernel launch.
 
#         Compressed prefill pages -> one _call_fused per tier (one launch).
#         Staging buffer           -> vectorized einsum, zero kernel launches.
#         New token                -> batched dot product.
#         """
#         key    = (layer_idx, kv_head)
#         device = q_batch.device
#         num_q  = q_batch.shape[0]
#         q      = q_batch.float()    # [num_q, dh]
#         kn     = k_new.float().to(device)
#         vn     = v_new.float().to(device)
#         dh     = q.shape[1]
 
#         with nvtx_range("page_lookup"):
#             ph = self.per_head_pages.get(key, [])
#             # Build b_theta → tier_id map once per call
#             _bt_to_tid = {t.b_theta: t.tier_id
#                           for t in self.tiers if t.tier_id != 0}

#         # ctx_logits_parts : each [num_q, n_tokens_for_this_tier]
#         ctx_logits_parts: List[torch.Tensor] = []
#         V_ctx_parts:      List[torch.Tensor] = []

#         with nvtx_range("angle_logits"):
#             for (pt, ptt, b_theta, n_tokens, V_tier, K_tier) in ph:
#                 # Resolve per-tier codebook and geometry
#                 tier_id  = _bt_to_tid.get(b_theta, 1)
#                 cb_lh    = get_codebook(self.codebooks, layer_idx, kv_head, tier_id)
#                 tier_obj = self.tiers[tier_id]
#                 tier_G   = tier_obj.G
#                 tier_g   = tier_obj.g

#                 if self.use_fused:
#                     # one kernel launch for all num_q heads at once
#                     raw = _call_fused(
#                         pt, ptt, q, cb_lh, b_theta,
#                         dh, tier_G, tier_g,
#                     )
#                     # raw: [num_q, num_pages * page_size]
#                     ctx_logits_parts.append(raw[:, :n_tokens])  # [num_q, n_tokens]
#                 else:
#                     # FIX Bug-2 + Bug-3: code-domain reference path.
#                     # K_tier is now (r_codes, th_codes) -- pre-stored at
#                     # prefill time.  We replicate the staging-buffer
#                     # vectorised einsum (Algorithm 1 §II lines 83-85):
#                     #   r_tilde = scale_g * Dr(c_r)  (here: r_codes directly)
#                     #   c       = CosFromAngles(q_hat, c_theta, t)
#                     #   logit   = ||q|| * r_tilde * c / sqrt(dh)
#                     # This avoids: (a) dense K materialisation, (b) the
#                     # Python loop over num_q (Bug-3: was O(num_q) launches).
#                     r_codes, th_codes = K_tier          # [N,G] each, CPU
#                     r_codes  = r_codes.to(device)       # [N, tier_G]
#                     th_codes = th_codes.to(device)      # [N, tier_G]
#                     q_groups = q.view(num_q, tier_G, tier_g)  # [num_q, G, g]
 
#                     # Gather codewords: [tier_G, N, tier_g]
#                     cw = cb_lh[
#                         torch.arange(tier_G, device=device).unsqueeze(1),
#                         th_codes.long().T,
#                     ]
 
#                     # Dots: [num_q, tier_G, N]  -->  logits: [num_q, N]
#                     dots = torch.einsum('qgi,gni->qgn', q_groups, cw)
#                     ctx_logits_parts.append(
#                         (r_codes.T.unsqueeze(0) * dots).sum(1) / math.sqrt(dh)
#                     )  # [num_q, n_tokens]
#                 V_ctx_parts.append(V_tier)  # [n_tokens, dh]

#             # ── staging buffer: decode tokens are always b3 (recent window)
#             stg_r     = self.stg_r.get(key)
#             stg_theta = self.stg_theta.get(key)
#             stg_V     = self.stg_V.get(key)
#             if stg_r is not None and stg_r.shape[0] > 0:
#                 N_stg     = stg_r.shape[0]
#                 # Staging tokens encoded at b3 — use b3 geometry
#                 b3        = self.tiers[3]
#                 stg_G     = b3.G
#                 stg_g     = b3.g
#                 cb_b3     = get_codebook(self.codebooks, layer_idx, kv_head, b3.tier_id)
#                 q_groups  = q.view(num_q, stg_G, stg_g)  # [num_q, G_b3, g_b3]

#                 # gather codewords for all staging tokens: [G_b3, N_stg, g_b3]
#                 cw = cb_b3[
#                     torch.arange(stg_G, device=device).unsqueeze(1),  # [G_b3, 1]
#                     stg_theta.long().T                                  # [G_b3, N_stg]
#                 ]

#                 # dot products: [num_q, G_b3, N_stg]
#                 dots = torch.einsum('qgi,gni->qgn', q_groups, cw)

#                 # weighted sum over groups: [num_q, N_stg]
#                 stg_logits = (stg_r.T.unsqueeze(0) * dots).sum(1) / math.sqrt(dh)

#                 ctx_logits_parts.append(stg_logits)   # [num_q, N_stg]
#                 V_ctx_parts.append(stg_V)
 
#         if ctx_logits_parts:
#             ctx_logits = torch.cat(ctx_logits_parts, dim=1)  # [num_q, total_ctx]
#             V_ctx      = torch.cat(V_ctx_parts, dim=0)       # [total_ctx, dh]
#         else:
#             ctx_logits = torch.zeros(num_q, 0, device=device)
#             V_ctx      = torch.zeros(0, dh, device=device)
 
#         # new token logit for each q head: [num_q]
#         new_logits = (q @ kn) / math.sqrt(dh)
 
#         # full logits: [num_q, total_ctx + 1]
#         all_logits = torch.cat([ctx_logits, new_logits.unsqueeze(1)], dim=1)
 
#         with nvtx_range("softmax"):
#             attn = torch.softmax(all_logits, dim=1)   # [num_q, total_ctx + 1]
 
#         # ── C.2: update EMA importance weights ───────────────────────
#         # attn[:, :-1] are weights over the ctx tokens (all q heads).
#         # We average over q heads (GQA) and record per retained token.
#         # ctx_token_count tells us how many ctx tokens exist.
#         n_ctx = ctx_logits.shape[1] if ctx_logits_parts else 0
#         if n_ctx > 0:
#             # C.2: update EMA importance weights using per-position attention.
#             # Average over q heads (GQA) → [n_ctx] scalar per ctx position.
#             attn_ctx_mean = attn[:, :n_ctx].mean(dim=0).detach().cpu()
#             # _ctx_token_order gives the exact positional ordering used when
#             # pages were built — O(1) lookup, no linear scan over all tokens.
#             order = self._ctx_token_order.get((layer_idx, kv_head), [])
#             for pos, tok in enumerate(order):
#                 if pos < n_ctx:
#                     tok.record_attn(float(attn_ctx_mean[pos]))
 
#         V_full = torch.cat([V_ctx, vn.unsqueeze(0)], dim=0)  # [total_ctx + 1, dh]
 
#         with nvtx_range("kv_read"):
#             attn_out = attn @ V_full   # [num_q, dh]
 
#         return attn_out.to(q_batch.dtype)
 
#     def _compressed_head_attention(
#         self,
#         layer_idx: int,
#         kv_head:   int,
#         q_vec:     torch.Tensor,   # [dh]
#         k_new:     torch.Tensor,   # [dh]
#         v_new:     torch.Tensor,   # [dh]
#     ) -> torch.Tensor:             # [dh]
#         """Single-q wrapper used by evaluate.py."""
#         out = self._compressed_head_attention_batched(
#             layer_idx, kv_head,
#             q_vec.unsqueeze(0),   # [1, dh]
#             k_new, v_new,
#         )
#         return out.squeeze(0)     # [dh]
 
#     # ------------------------------------------------------------------
#     # Decode staging buffer management
#     # ------------------------------------------------------------------
 
#     def _append_decode_kv(
#         self,
#         layer_idx: int,
#         kv_head:   int,
#         k_new:     torch.Tensor,   # [dh]
#         v_new:     torch.Tensor,   # [dh]
#     ) -> None:
#         """
#         Compress the new decode token and append to the staging buffer.
#         Flush to a compressed page when PAGE_SIZE tokens accumulate.
#         Dense K is NEVER stored -- codes are computed immediately.
#         """
#         key    = (layer_idx, kv_head)
#         device = self.device
#         kn     = k_new.float().to(device).unsqueeze(0)   # [1, dh]
#         vn     = v_new.float().to(device).unsqueeze(0)   # [1, dh]
 
#         # Decode tokens are in the recent window → always tier b3 (Low)
#         decode_tier   = self.tiers[3]   # b3 (Low = bmin for recent tokens)
#         cb_key        = (layer_idx, kv_head, decode_tier.tier_id)
#         cb_lh         = self.codebooks.get(cb_key)
#         if cb_lh is None:
#             cb_lh = get_codebook(self.codebooks, layer_idx, kv_head, decode_tier.tier_id)
#         r, theta = _encode_keys_with_codebooks(
#             kn, cb_lh, decode_tier.g, decode_tier.G
#         )   # r: [1, G_b3], theta: [1, G_b3] int32
 
#         if key not in self.stg_r:
#             self.stg_r[key]     = r
#             self.stg_theta[key] = theta
#             self.stg_V[key]     = vn
#         else:
#             self.stg_r[key]     = torch.cat([self.stg_r[key],     r],     dim=0)
#             self.stg_theta[key] = torch.cat([self.stg_theta[key], theta],  dim=0)
#             self.stg_V[key]     = torch.cat([self.stg_V[key],     vn],    dim=0)
 
#         if self.stg_r[key].shape[0] >= PAGE_SIZE:
#             self._flush_decode_page(key)
 
#         # Trigger periodic controller refresh (Appendix C.3)
#         # Only fires on the first (layer=0, head=0) call per decode step
#         # to avoid refreshing N_layers × N_heads times per step.
#         if layer_idx == 0 and kv_head == 0:
#             self._maybe_refresh()
 
#     def _flush_decode_page(self, key: Tuple[int, int]) -> None:
#         """
#         Pack the staging buffer into a compressed page and add it to
#         per_head_pages.  Called automatically when staging buffer fills.
#         """
#         device    = self.device
#         r_buf     = self.stg_r[key]      # [PAGE_SIZE, G]
#         theta_buf = self.stg_theta[key]  # [PAGE_SIZE, G]
#         V_buf     = self.stg_V[key]      # [PAGE_SIZE, dh]
#         n         = r_buf.shape[0]
#         tier      = self.tiers[3]   # b3 (Low) for all decode tokens
 
#         page = build_page_from_codes(
#             r_buf.cpu(), theta_buf.cpu(), tier, segment_id=2
#         )
 
#         header_size   = HEADER_BYTES + tier.G * 4
#         theta_bytes   = ((n * tier.G * tier.b_theta) + 7) // 8
#         theta_offset  = header_size
#         radius_offset = theta_offset + theta_bytes
 
#         ptable = PointerTable()
#         ptable.add_page(0, theta_offset, radius_offset)
 
#         if key not in self.per_head_pages:
#             self.per_head_pages[key] = []
 
#         self.per_head_pages[key].append((
#             page.to(device),
#             ptable.to_tensor().to(device),
#             tier.b_theta,
#             n,
#             V_buf,
#             None,   # no dense K stored — paper compliant
#         ))
 
#         del self.stg_r[key]
#         del self.stg_theta[key]
#         del self.stg_V[key]
 
#     # ------------------------------------------------------------------
#     # Online controller refresh  (Appendix C.3)
#     # ------------------------------------------------------------------
 
#     def _maybe_refresh(self) -> None:
#         """
#         Re-run the RDR controller on the current retained token set.
#         Called every REFRESH_CADENCE decode steps.
#         Skipped if REFRESH_CADENCE == 0 (prefill-only mode).
 
#         What happens:
#           1. allocate() re-scores all retained tokens using updated omega.
#           2. Tokens that now exceed the budget are downgraded or dropped.
#           3. Tokens below the budget get upgrade consideration.
#           4. Frozen tokens (cooldown > 0) are skipped (C.4 hysteresis).
#         """
#         if REFRESH_CADENCE == 0 or not self._retained_tokens:
#             return
#         # _decode_step is incremented in _append_decode_kv (layer=0,head=0).
#         # Here we only check whether we should act.
#         if self._decode_step % REFRESH_CADENCE != 0:
#             return
 
#         # online_refresh from allocation.py: ticks cooldowns, runs downgrade
#         # + upgrade passes, respects frozen tokens (C.3 + C.4).
#         old_tiers = {id(ts): ts.new_tier_id for ts in self._retained_tokens}
#         updated   = online_refresh(self._retained_tokens, self.tiers)
 
#         # Commit and detect which (layer, head) pairs changed tier
#         changed_lh = set()
#         for tok in updated:
#             if tok.new_tier_id != old_tiers.get(id(tok)):
#                 changed_lh.add((tok.layer, tok.head))
#             tok.commit_tier()
 
#         self._retained_tokens = updated
 
#         # Rebuild compressed pages for (layer, head) pairs that changed
#         if changed_lh:
#             self._rebuild_pages_for(changed_lh)
 
#     # ------------------------------------------------------------------
#     # Lifecycle
#     # ------------------------------------------------------------------
 
#     def _rebuild_pages_for(self, changed_lh: set) -> None:
#         """
#         After an online refresh, re-encode and re-page the (layer, head)
#         pairs in changed_lh using the current tier assignments in
#         self._retained_tokens and the stored dense keys in self._K_all.
#         Updates self.per_head_pages and self._ctx_token_order in place.
#         """
#         # Group current retained tokens by (layer, head, tier_id)
#         lh_tier: Dict[Tuple[int,int,int], List] = defaultdict(list)
#         for ts in self._retained_tokens:
#             key = (ts.layer, ts.head, ts.new_tier_id)
#             lh_tier[key].append(ts)
 
#         for (layer, head) in changed_lh:
#             K_lh = self._K_all.get((layer, head))
#             if K_lh is None:
#                 continue  # no stored keys (shouldn't happen)
 
#             new_entries = []
#             new_order   = []
 
#             # Process tiers in ascending tier_id order (b1 → b2 → b3)
#             for tier_id in [1, 2, 3]:
#                 toks = lh_tier.get((layer, head, tier_id), [])
#                 if not toks:
#                     continue
#                 toks.sort(key=lambda t: t.index)
#                 tier = self.tiers[tier_id]
 
#                 # Tier-specific codebook
#                 cb = self.codebooks.get((layer, head, tier_id))
#                 if cb is None:
#                     cb = get_codebook(self.codebooks, layer, head, tier_id)
#                 if cb is None:
#                     continue
                
#                 V_lh = self._V_all.get((layer, head))
#                 page_list, r_parts, theta_parts, V_parts = [], [], [], []
#                 ptable = PointerTable()
#                 offset = 0
 
#                 for i in range(0, len(toks), PAGE_SIZE):
#                     chunk   = toks[i : i + PAGE_SIZE]
#                     N       = len(chunk)
#                     indices = [t.index for t in chunk]
#                     K_chunk = K_lh[indices]
#                     r_g, th_c = _encode_keys_with_codebooks(
#                         K_chunk, cb, tier.g, tier.G)
#                     page = build_page_from_codes(
#                         r_g, th_c, tier, chunk[0].segment_id)
#                     h_sz  = HEADER_BYTES + tier.G * 4
#                     t_sz  = ((N * tier.G * tier.b_theta) + 7) // 8
#                     ptable.add_page(offset, offset + h_sz, offset + h_sz + t_sz)
#                     offset += page.numel()
#                     page_list.append(page)
#                     # FIX Bug-2 (rebuild): store codes, not dense K.
#                     r_parts.append(r_g.cpu())       # [N, G] float32
#                     theta_parts.append(th_c.cpu())  # [N, G] int32
#                     if V_lh is not None:
#                         V_parts.append(V_lh[indices].to(self.device))

#                 V_tensor = torch.cat(V_parts, dim=0) if V_parts else None
 
#                 new_entries.append((
#                     torch.cat(page_list).to(self.device),
#                     ptable.to_tensor().to(self.device),
#                     tier.b_theta,
#                     len(toks),
#                     V_tensor,
#                     # Slot [5]: (r_groups, theta_codes) -- ADA compliant
#                     (torch.cat(r_parts, dim=0), torch.cat(theta_parts, dim=0)),
#                 ))
#                 new_order.extend(toks)
 
#             self.per_head_pages[(layer, head)]   = new_entries
#             self._ctx_token_order[(layer, head)] = new_order
 
#     def uninstall(self) -> None:
#         if self._patched:
#             unpatch_decode(self.model, self._original_forwards)
#             self._patched = False
#             print("[SphericalKVPipeline] Attention layers restored.")
 
#     def _reset_state(self) -> None:
#         self.per_head_pages.clear()
#         self.stg_r.clear()
#         self.stg_theta.clear()
#         self.stg_V.clear()
#         self.reuse          = None
#         self.stability      = None
#         self.seq_len        = 0
#         self._decode_step   = 0
#         self._retained_tokens.clear()
#         self._ctx_token_order.clear()
#         self._K_all.clear()
#         self._V_all.clear()

from __future__ import annotations
 
import contextlib
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
 
import torch
 
 
@contextlib.contextmanager
def nvtx_range(name: str):
    """
    NVTX range for Nsight Compute / Nsight Systems.
    Algorithm instrumentation (line 73): page_lookup, kv_read,
    angle_logits, softmax, proj.  No-ops when CUDA unavailable.
    """
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()
 
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
 
from allocation import allocate, online_refresh
from codebook_loader import get_codebook
from config import (EPS, GROUP_SIZE, NUM_GROUPS, PAGE_SIZE, HEADER_BYTES,
                               REFRESH_CADENCE, SINK_TOKENS, RECENT_WINDOW,
                               COOLDOWN_STEPS)
from llama_hooks import (
    aggregate_proxy_to_kv_heads,
    build_attn_weights_tensor,
    build_head_outputs_tensor,
    capture_prefill_pass,
    patch_for_decode,
    unpatch_decode,
)
from pagebuilder import build_page_from_codes
from pointer_table import PointerTable
from resuse_proxy import compute_reuse_proxy
from spherical_parameterization import spherical_parameterize
from stability_proxy import compute_stability_proxy
from tiers import build_tiers
from token_state import TokenState
 
 
# ---------------------------------------------------------------------------
# Reference ADA decode (pure torch -- fallback when CUDA kernel unavailable)
# Vectorized: no Python loop over groups.
# ---------------------------------------------------------------------------
 
def reference_codebook_decode(
    q:          torch.Tensor,   # [dh]
    codebooks:  torch.Tensor,   # [G, codebook_size, group_size]
    r_codes:    torch.Tensor,   # [T_ctx, G]  pre-stored per-group radii
    theta_codes: torch.Tensor,  # [T_ctx, G]  pre-stored codebook indices (int32)
    group_size: int,
    num_groups: int,
) -> torch.Tensor:              # [T_ctx]
    """
    ADA-compliant code-domain reference decode (Fix B).

    Computes attention logits directly from pre-stored (r_codes, theta_codes)
    — no dense K materialisation anywhere.  Matches Algorithm 1 §II lines 83-85:
        r_tilde = scale_g * Dr(c_r)       (here: r_codes directly)
        c       = CosFromAngles(q_hat, c_theta, t)
        logit   = ||q|| * r_tilde * c / sqrt(dh)

    Replaces the old dense-K version which re-encoded keys at every call,
    violating the ADA "no dense reconstruction" invariant (line 89).
    """
    device     = q.device
    T_ctx, G   = r_codes.shape
    r_c   = r_codes.to(device)    # [T_ctx, G]
    th_c  = theta_codes.to(device).long()  # [T_ctx, G]

    q_groups = q.view(num_groups, group_size)   # [G, g]

    # Gather codewords for all context tokens: [G, T_ctx, g]
    cw = codebooks[
        torch.arange(num_groups, device=device).unsqueeze(1),  # [G, 1]
        th_c.T,                                                 # [G, T_ctx]
    ]

    # Dot products: [G, T_ctx]
    dots = (cw * q_groups.unsqueeze(1)).sum(-1)   # [G, T_ctx]

    # Weighted sum over groups: [T_ctx]
    logits = (r_c.T * dots).sum(0) / math.sqrt(num_groups * group_size)

    return logits
 
 
# ---------------------------------------------------------------------------
# Fused decode availability + dispatch
# ---------------------------------------------------------------------------
 
_fused_ok: Optional[bool] = None
 
def _fused_available() -> bool:
    global _fused_ok
    if _fused_ok is None:
        try:
            from fused_decode import fused_decode   # noqa: F401
            _fused_ok = True
        except Exception:
            _fused_ok = False
    return _fused_ok
 
 
def _call_fused(
    pages_tensor:  torch.Tensor,    # packed uint8
    ptable_tensor: torch.Tensor,    # [P, 3] int32
    q:             torch.Tensor,    # [dh] OR [num_q, dh]
    codebooks_lh:  torch.Tensor,    # [G, cb_size, g]
    b_theta:       int,
    dh:            int,
    num_groups:    int,
    group_size:    int,
) -> torch.Tensor:
    """
    Dispatches to the batched CUDA kernel.
    If q is 1D [dh], unsqueezes to [1, dh] and squeezes result back.
    If q is 2D [num_q, dh], returns [num_q, num_pages * page_size].
    """
    from fused_decode import fused_decode
    squeeze = (q.dim() == 1)
    if squeeze:
        q = q.unsqueeze(0)   # [1, dh]
    result = fused_decode(
        pages_tensor,
        ptable_tensor,
        q.float(),
        codebooks_lh.float(),
        dh=dh,
        groups=num_groups,
        group_size=group_size,
        b_theta=b_theta,
        page_size=PAGE_SIZE,
    )
    # result: [num_q, num_pages * page_size]
    if squeeze:
        return result.squeeze(0)   # [num_pages * page_size]
    return result
 
 
# ---------------------------------------------------------------------------
# Codebook encoding  (vectorized, no Python loops)
# ---------------------------------------------------------------------------
 
def _encode_keys_with_codebooks(
    K_chunk:    torch.Tensor,   # [N, dh]
    codebooks:  torch.Tensor,   # [G, codebook_size, group_size]  tier-specific
    group_size: int,             # g for this tier
    num_groups: int,             # G for this tier
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode N key vectors using tier-specific grouped VQ (Appendix B.1).
 
    Returns
    -------
    r_groups    : [N, G]  float32 -- per-group L2 norms
    theta_codes : [N, G]  int32   -- nearest codebook index per group
    """
    N         = K_chunk.shape[0]
    K_grouped = K_chunk.view(N, num_groups, group_size)      # [N, G, g]
    r_groups  = K_grouped.norm(dim=-1)                       # [N, G]
    K_dir     = K_grouped / (r_groups.unsqueeze(-1) + EPS)   # [N, G, g]
 
    cb   = codebooks.to(K_dir.device)                        # [G, cb_size, g]
    sims = torch.einsum('ngi,gci->ngc', K_dir, cb)           # [N, G, cb_size]
    theta_codes = sims.argmax(dim=-1).to(torch.int32)        # [N, G]
    return r_groups, theta_codes
 
 
# ---------------------------------------------------------------------------
# Per-(layer, kv_head) page builder  (prefill only)
# ---------------------------------------------------------------------------
 
def _build_per_head_pages(
    retained_tokens: List[TokenState],
    tiers,
    K_all:      Dict[Tuple[int, int], torch.Tensor],
    V_all:      Dict[Tuple[int, int], torch.Tensor],
    codebooks:  Dict[Tuple[int, int], torch.Tensor],
    group_size: int,
    num_groups: int,
    device,
) -> Dict[Tuple[int, int], List]:
    """
    Build tier-homogeneous, codebook-encoded pages per (layer, kv_head).
 
    Each entry in the returned list is a 6-tuple:
        (pages_tensor, ptable_tensor, b_theta, n_tokens, V_tier, K_tier)
 
    pages_tensor  -- packed uint8 bitstream for all pages in this tier.
    ptable_tensor -- [P, 3] int32 pointer table.
    b_theta       -- bits per codebook index for this tier.
    n_tokens      -- exact count of valid tokens across all pages.
    V_tier        -- [n_tokens, dh] float on device.
    K_tier        -- tuple (r_groups [n_tokens,G], theta_codes [n_tokens,G]) on CPU.
                      Pre-encoded codes; no dense K stored (ADA compliant).
    """
    groups: Dict[Tuple[int, int, int], List[TokenState]] = defaultdict(list)
    for ts in retained_tokens:
        groups[(ts.layer, ts.head, ts.new_tier_id)].append(ts)
 
    per_head: Dict[Tuple[int, int], List] = defaultdict(list)
 
    for (layer, head, tier_id), tokens in groups.items():
        tokens.sort(key=lambda t: t.index)
        tier  = tiers[tier_id]
        K_lh  = K_all[(layer, head)]
        V_lh  = V_all[(layer, head)]
        # ── Tier-specific codebook: keyed by (layer, head, tier_id) ─────
        # Appendix B.1: b1/b2 use g=16, G=8 codebooks;
        #               b3 uses g=32, G=4 codebooks.
        cb_key = (layer, head, tier_id)
        if cb_key in codebooks:
            cb_lh = codebooks[cb_key]           # [G, cb_size, g]  tier-aware
        elif (layer, head) in codebooks:
            cb_lh = codebooks[(layer, head)]    # legacy format fallback
        else:
            continue                            # no codebook, skip tier
 
        tier_g  = tier.g    # group size for this tier
        tier_G  = tier.G    # number of groups for this tier
 
        page_list:   List[torch.Tensor] = []
        r_parts:     List[torch.Tensor] = []   # [N_chunk, G] float32
        theta_parts: List[torch.Tensor] = []   # [N_chunk, G] int32
        V_tier_parts: List[torch.Tensor] = []
        ptable = PointerTable()
        offset = 0
 
        for i in range(0, len(tokens), PAGE_SIZE):
            chunk   = tokens[i : i + PAGE_SIZE]
            N       = len(chunk)
            indices = [t.index for t in chunk]
 
            K_chunk = K_lh[indices]
            V_chunk = V_lh[indices]
            r_groups, theta_codes = _encode_keys_with_codebooks(
                K_chunk, cb_lh, tier_g, tier_G   # tier-specific g and G
            )
 
            page = build_page_from_codes(
                r_groups, theta_codes, tier, chunk[0].segment_id
            )
 
            header_size   = HEADER_BYTES + tier_G * 4
            theta_bytes   = ((N * tier_G * tier.b_theta) + 7) // 8
            theta_offset  = offset + header_size
            radius_offset = theta_offset + theta_bytes
 
            page_list.append(page)
            # FIX Bug-2: store pre-encoded codes, NOT dense K.
            # Dense K is never needed at decode time -- the reference
            # path reads (r_groups, theta_codes) directly, matching
            # the staging-buffer pattern and satisfying the ADA
            # 'no dense reconstruction' invariant (Alg.1 §II line 89).
            r_parts.append(r_groups.cpu())        # [N, G] float32
            theta_parts.append(theta_codes.cpu()) # [N, G] int32
            V_tier_parts.append(V_chunk)
            ptable.add_page(offset, theta_offset, radius_offset)
            offset += page.numel()
 
        per_head[(layer, head)].append((
            torch.cat(page_list).to(device),
            ptable.to_tensor().to(device),
            tier.b_theta,
            len(tokens),
            torch.cat(V_tier_parts, dim=0).to(device),
            # Slot [5]: (r_groups, theta_codes) on CPU.
            # The reference decode path unpacks these directly --
            # no encode/decode round-trip, no dense K materialisation.
            (torch.cat(r_parts, dim=0), torch.cat(theta_parts, dim=0)),
        ))
 
    return per_head
 
 
# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
 
class SphericalKVPipeline:
    """
    Spherical KV Cache inference pipeline.
 
    Usage
    -----
    pipeline = SphericalKVPipeline(model, tokenizer, codebooks, device)
    pipeline.prefill(input_ids)
    out = model.generate(input_ids, ...)
    pipeline.uninstall()
    """
 
    def __init__(
        self,
        model,
        tokenizer,
        codebooks: Dict[Tuple[int, int], torch.Tensor],
        device:     torch.device = torch.device("cpu"),
        head_dim:   Optional[int] = None,
        group_size: Optional[int] = None,
        sink_tokens: int = 4,
        use_fused:   bool = True,
    ):
        self.model      = model
        self.tokenizer  = tokenizer
        self.codebooks  = {k: v.to(device) for k, v in codebooks.items()}
        self.device     = device
        self.sink_tokens = sink_tokens
        self.use_fused  = use_fused and _fused_available()
 
        cfg = model.config
        self.num_layers   = cfg.num_hidden_layers
        self.num_q_heads  = cfg.num_attention_heads
        self.num_kv_heads = getattr(cfg, "num_key_value_heads", self.num_q_heads)
        self.head_dim     = head_dim or getattr(
            cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
        )
        self.group_size = group_size or GROUP_SIZE
        self.num_groups = self.head_dim // self.group_size
        self.tiers      = build_tiers(self.head_dim)
 
        # Prefill compressed pages: per (layer, kv_head) list of 6-tuples
        self.per_head_pages: Dict[Tuple[int, int], List] = {}
 
        # Decode staging buffers: compressed codes for recent decode tokens
        # not yet flushed into a full page.  At most PAGE_SIZE-1 tokens each.
        # stg_r     : [N_stg, G] float32  -- per-group radii
        # stg_theta : [N_stg, G] int32    -- codebook indices
        # stg_V     : [N_stg, dh] float32 -- value vectors
        self.stg_r:     Dict[Tuple[int, int], torch.Tensor] = {}
        self.stg_theta: Dict[Tuple[int, int], torch.Tensor] = {}
        self.stg_V:     Dict[Tuple[int, int], torch.Tensor] = {}
 
        self.reuse:     Optional[torch.Tensor] = None
        self.stability: Optional[torch.Tensor] = None
        self.seq_len:   int = 0
 
 
        self._retained_tokens: list = []
        self._decode_step: int = 0   # counts decode steps since prefill
 
        self._ctx_token_order: Dict[Tuple[int, int], List] = {}
 
        self._codes_all: Dict[Tuple[int,int], dict] = {}
        self._V_all = {}
 
        self._original_forwards: Dict[int, object] = {}
        self._patched = False
 
        mode = "fused CUDA" if self.use_fused else "reference (pure-torch)"
        print(
            f"[SphericalKVPipeline] layers={self.num_layers}  "
            f"kv_heads={self.num_kv_heads}  head_dim={self.head_dim}  "
            f"groups={self.num_groups}  decode_mode={mode}"
        )
 
    def prefill(
        self,
        input_ids:      torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> None:
        if self._patched:
            self.uninstall()
        self._reset_state()
 
        input_ids = input_ids.to(self.device)
        B, T = input_ids.shape
        self.seq_len = T
 
        print(f"[prefill] Forward pass on {T} tokens ...")
        kv_pairs, attn_list, ho_list, logits = capture_prefill_pass(
            self.model, input_ids, attention_mask
        )
 
        attn_stacked = build_attn_weights_tensor(attn_list)
        ho_stacked   = build_head_outputs_tensor(ho_list)
 
        if attn_stacked is not None:
            reuse_kv = aggregate_proxy_to_kv_heads(
                compute_reuse_proxy(attn_stacked),
                self.num_q_heads, self.num_kv_heads,
            )
        else:
            reuse_kv = torch.ones(self.num_layers, self.num_kv_heads) / self.num_kv_heads
 
        if ho_stacked is not None:
            stab_kv = aggregate_proxy_to_kv_heads(
                compute_stability_proxy(
                    ho_stacked[:, :, :, -1, :], logits[:, -1, :]
                ),
                self.num_q_heads, self.num_kv_heads,
            )
        else:
            stab_kv = torch.ones(self.num_layers, self.num_kv_heads) / self.num_kv_heads
 
        self.reuse     = reuse_kv.cpu()
        self.stability = stab_kv.cpu()
 
        del attn_stacked, ho_stacked, attn_list, ho_list
        torch.cuda.empty_cache()
 
        print("[prefill] Building TokenStates ...")
        all_tokens: List[TokenState] = []
        K_all: Dict[Tuple[int, int], torch.Tensor] = {}
        V_all: Dict[Tuple[int, int], torch.Tensor] = {}
 
        for li, (K, V) in enumerate(kv_pairs):
            for h in range(self.num_kv_heads):
                K_lh = K[0, h].float().cpu()    # float32 for encoding
                V_lh = V[0, h].half().cpu()     # fp16 to match dense baseline
                K_all[(li, h)] = K_lh
                V_all[(li, h)] = V_lh
 
                r, phi = spherical_parameterize(K_lh)
 
                G_fine   = self.num_groups        # 8
                g_fine   = self.group_size        # 16
                K_grouped = K_lh.view(T, G_fine, g_fine)
                r_groups_all = K_grouped.norm(dim=-1)   # [T, G_fine]
 
                for t in range(T):
                   
                    if t >= T - RECENT_WINDOW:
                        seg = 2   # recent suffix
                    else:
                        seg = 0   # prefix (default; caller can override for RAG)
 
                    all_tokens.append(TokenState(
                        layer=li, head=h, index=t,
                        r=r[t], phi=phi[t],
                        segment_id=seg,
                        age=T - t,   # Algorithm 1 line 19: a_i = Tp - i
                        prev_tier_id=0,
                        protected=(t < self.sink_tokens),
                        r_groups=r_groups_all[t],
                    ))
 
        del kv_pairs
 
        print(f"[prefill] Allocating tiers for {len(all_tokens)} token-slots ...")

        import config as _cfg
        # _cfg.GLOBAL_BUDGET_BITS = 5_000_000
        # print(f"[prefill] Budget: {_cfg.GLOBAL_BUDGET_BITS:,} bits (fixed global budget)")
        _bpt = getattr(_cfg, 'BITS_PER_TOKEN', 30)
        _cfg.GLOBAL_BUDGET_BITS = _bpt * T * self.num_layers * self.num_kv_heads
        print(f"[prefill] Budget: {_cfg.GLOBAL_BUDGET_BITS:,} bits "
              f"({_bpt} bpt × T={T} × L={self.num_layers} × H={self.num_kv_heads})")
        retained = allocate(all_tokens, self.tiers, self.reuse, self.stability)
        for ts in retained:
            ts.commit_tier()
        # Keep reference for online refresh and EMA updates
        self._retained_tokens = retained
        self._decode_step     = 0
        pct = 100 * len(retained) / max(len(all_tokens), 1)
        print(f"[prefill] Retained {len(retained)}/{len(all_tokens)} ({pct:.1f}%)")
 
        print("[prefill] Encoding keys with codebooks and building pages ...")
        self.per_head_pages = _build_per_head_pages(
            retained_tokens=retained,
            tiers=self.tiers,
            K_all=K_all,
            V_all=V_all,
            codebooks=self.codebooks,
            group_size=self.group_size,
            num_groups=self.num_groups,
            device=self.device,
        )

        print("[prefill] Pre-encoding all tokens at all tiers for ADA-compliant refresh ...")
        self._codes_all.clear()
        for _li in range(self.num_layers):
            for _h in range(self.num_kv_heads):
                K_lh_cpu = K_all[(_li, _h)]          # [T, dh] float32 CPU
                tier_codes: dict = {}
                for _tid in [1, 2, 3]:
                    _tier = self.tiers[_tid]
                    _cb = self.codebooks.get((_li, _h, _tid))
                    if _cb is None:
                        from codebook_loader import get_codebook as _gcb
                        _cb = _gcb(self.codebooks, _li, _h, _tid)
                    if _cb is None:
                        continue
                    _r, _th = _encode_keys_with_codebooks(
                        K_lh_cpu, _cb, _tier.g, _tier.G
                    )  # [T, G_tier] each; one vectorised call for all T tokens
                    tier_codes[_tid] = (_r.cpu(), _th.cpu())
                self._codes_all[(_li, _h)] = tier_codes
        # Dense K no longer needed — release immediately.
        del K_all
 
        # _V_all kept for online refresh (values stay dense FP16 by design).
        self._V_all = {k: v.cpu() for k, v in V_all.items()}
        del V_all
        torch.cuda.empty_cache()

        self._ctx_token_order.clear()
        lh_tier_map: Dict[Tuple[int,int,int], List] = defaultdict(list)
        for ts in retained:
            lh_tier_map[(ts.layer, ts.head, ts.new_tier_id)].append(ts)
        lh_map: Dict[Tuple[int,int], List] = defaultdict(list)
        for (layer, head, tid), toks in lh_tier_map.items():
            toks.sort(key=lambda t: t.index)
            lh_map[(layer, head)].extend(toks)
        self._ctx_token_order = dict(lh_map)
 
        print("[prefill] Patching attention layers for decode ...")
        self._original_forwards = patch_for_decode(self.model, self)
        self._patched = True
        print("[prefill] Done -- model ready for compressed decode.")

    def _compressed_head_attention_batched(
        self,
        layer_idx: int,
        kv_head:   int,
        q_batch:   torch.Tensor,   # [num_q, dh]
        k_new:     torch.Tensor,   # [dh]
        v_new:     torch.Tensor,   # [dh]
    ) -> torch.Tensor:             # [num_q, dh]
        """
        Compute compressed attention for ALL q heads sharing this kv_head
        in a single batched kernel launch.
 
        Compressed prefill pages -> one _call_fused per tier (one launch).
        Staging buffer           -> vectorized einsum, zero kernel launches.
        New token                -> batched dot product.
        """
        key    = (layer_idx, kv_head)
        device = q_batch.device
        num_q  = q_batch.shape[0]
        q      = q_batch.float()    # [num_q, dh]
        kn     = k_new.float().to(device)
        vn     = v_new.float().to(device)
        dh     = q.shape[1]
 
        with nvtx_range("page_lookup"):
            ph = self.per_head_pages.get(key, [])
            # Build b_theta → tier_id map once per call
            _bt_to_tid = {t.b_theta: t.tier_id
                          for t in self.tiers if t.tier_id != 0}

        # ctx_logits_parts : each [num_q, n_tokens_for_this_tier]
        ctx_logits_parts: List[torch.Tensor] = []
        V_ctx_parts:      List[torch.Tensor] = []

        with nvtx_range("angle_logits"):
            for (pt, ptt, b_theta, n_tokens, V_tier, K_tier) in ph:
                # Resolve per-tier codebook and geometry
                tier_id  = _bt_to_tid.get(b_theta, 1)
                cb_lh    = get_codebook(self.codebooks, layer_idx, kv_head, tier_id)
                tier_obj = self.tiers[tier_id]
                tier_G   = tier_obj.G
                tier_g   = tier_obj.g

                if self.use_fused:
                    # one kernel launch for all num_q heads at once
                    raw = _call_fused(
                        pt, ptt, q, cb_lh, b_theta,
                        dh, tier_G, tier_g,
                    )
                    # raw: [num_q, num_pages * page_size]
                    ctx_logits_parts.append(raw[:, :n_tokens])  # [num_q, n_tokens]
                else:
                    r_codes, th_codes = K_tier          # [N,G] each, CPU
                    r_codes  = r_codes.to(device)       # [N, tier_G]
                    th_codes = th_codes.to(device)      # [N, tier_G]
                    q_groups = q.view(num_q, tier_G, tier_g)  # [num_q, G, g]
 
                    # Gather codewords: [tier_G, N, tier_g]
                    cw = cb_lh[
                        torch.arange(tier_G, device=device).unsqueeze(1),
                        th_codes.long().T,
                    ]
 
                    # Dots: [num_q, tier_G, N]  -->  logits: [num_q, N]
                    dots = torch.einsum('qgi,gni->qgn', q_groups, cw)
                    ctx_logits_parts.append(
                        (r_codes.T.unsqueeze(0) * dots).sum(1) / math.sqrt(dh)
                    )  # [num_q, n_tokens]
                V_ctx_parts.append(V_tier)  # [n_tokens, dh]

            # ── staging buffer: decode tokens are always b3 (recent window)
            stg_r     = self.stg_r.get(key)
            stg_theta = self.stg_theta.get(key)
            stg_V     = self.stg_V.get(key)
            if stg_r is not None and stg_r.shape[0] > 0:
                N_stg     = stg_r.shape[0]
                # Staging tokens encoded at b3 — use b3 geometry
                b3        = self.tiers[3]
                stg_G     = b3.G
                stg_g     = b3.g
                cb_b3     = get_codebook(self.codebooks, layer_idx, kv_head, b3.tier_id)
                q_groups  = q.view(num_q, stg_G, stg_g)  # [num_q, G_b3, g_b3]

                # gather codewords for all staging tokens: [G_b3, N_stg, g_b3]
                cw = cb_b3[
                    torch.arange(stg_G, device=device).unsqueeze(1),  # [G_b3, 1]
                    stg_theta.long().T                                  # [G_b3, N_stg]
                ]

                # dot products: [num_q, G_b3, N_stg]
                dots = torch.einsum('qgi,gni->qgn', q_groups, cw)

                # weighted sum over groups: [num_q, N_stg]
                stg_logits = (stg_r.T.unsqueeze(0) * dots).sum(1) / math.sqrt(dh)

                ctx_logits_parts.append(stg_logits)   # [num_q, N_stg]
                V_ctx_parts.append(stg_V)
 
        if ctx_logits_parts:
            ctx_logits = torch.cat(ctx_logits_parts, dim=1)  # [num_q, total_ctx]
            V_ctx      = torch.cat(V_ctx_parts, dim=0)       # [total_ctx, dh]
        else:
            ctx_logits = torch.zeros(num_q, 0, device=device)
            V_ctx      = torch.zeros(0, dh, device=device)
 
        # new token logit for each q head: [num_q]
        new_logits = (q @ kn) / math.sqrt(dh)
 
        # full logits: [num_q, total_ctx + 1]
        all_logits = torch.cat([ctx_logits, new_logits.unsqueeze(1)], dim=1)
 
        with nvtx_range("softmax"):
            attn = torch.softmax(all_logits, dim=1)   # [num_q, total_ctx + 1]
 
        n_ctx = ctx_logits.shape[1] if ctx_logits_parts else 0
        if n_ctx > 0 and REFRESH_CADENCE > 0 and (self._decode_step % REFRESH_CADENCE == 0):
            attn_ctx_mean = attn[:, :n_ctx].mean(dim=0).detach().cpu()
            order = self._ctx_token_order.get((layer_idx, kv_head), [])
            for pos, tok in enumerate(order):
                if pos < n_ctx:
                    tok.record_attn(float(attn_ctx_mean[pos]))
 
        V_full = torch.cat([V_ctx, vn.unsqueeze(0)], dim=0)  # [total_ctx + 1, dh]
 
        with nvtx_range("kv_read"):
            attn_out = attn @ V_full   # [num_q, dh]
 
        return attn_out.to(q_batch.dtype)
 
    def _compressed_head_attention(
        self,
        layer_idx: int,
        kv_head:   int,
        q_vec:     torch.Tensor,   # [dh]
        k_new:     torch.Tensor,   # [dh]
        v_new:     torch.Tensor,   # [dh]
    ) -> torch.Tensor:             # [dh]
        """Single-q wrapper used by evaluate.py."""
        out = self._compressed_head_attention_batched(
            layer_idx, kv_head,
            q_vec.unsqueeze(0),   # [1, dh]
            k_new, v_new,
        )
        return out.squeeze(0)     # [dh]
 
    def _append_decode_kv(
        self,
        layer_idx: int,
        kv_head:   int,
        k_new:     torch.Tensor,   # [dh]
        v_new:     torch.Tensor,   # [dh]
    ) -> None:
        """
        Compress the new decode token and append to the staging buffer.
        Flush to a compressed page when PAGE_SIZE tokens accumulate.
        Dense K is NEVER stored -- codes are computed immediately.
        """
        key    = (layer_idx, kv_head)
        device = self.device
        kn     = k_new.float().to(device).unsqueeze(0)   # [1, dh]
        vn     = v_new.float().to(device).unsqueeze(0)   # [1, dh]
 
        # Decode tokens are in the recent window → always tier b3 (Low)
        decode_tier   = self.tiers[3]   # b3 (Low = bmin for recent tokens)
        cb_key        = (layer_idx, kv_head, decode_tier.tier_id)
        cb_lh         = self.codebooks.get(cb_key)
        if cb_lh is None:
            cb_lh = get_codebook(self.codebooks, layer_idx, kv_head, decode_tier.tier_id)
        r, theta = _encode_keys_with_codebooks(
            kn, cb_lh, decode_tier.g, decode_tier.G
        )   # r: [1, G_b3], theta: [1, G_b3] int32
 
        if key not in self.stg_r:
            self.stg_r[key]     = r
            self.stg_theta[key] = theta
            self.stg_V[key]     = vn
        else:
            self.stg_r[key]     = torch.cat([self.stg_r[key],     r],     dim=0)
            self.stg_theta[key] = torch.cat([self.stg_theta[key], theta],  dim=0)
            self.stg_V[key]     = torch.cat([self.stg_V[key],     vn],    dim=0)
 
        if self.stg_r[key].shape[0] >= PAGE_SIZE:
            self._flush_decode_page(key)
 
        # Trigger periodic controller refresh (Appendix C.3)
        # Only fires on the first (layer=0, head=0) call per decode step
        # to avoid refreshing N_layers × N_heads times per step.
        if layer_idx == 0 and kv_head == 0:
            self._decode_step += 1
            self._maybe_refresh()
 
    def _flush_decode_page(self, key: Tuple[int, int]) -> None:
        """
        Pack the staging buffer into a compressed page and add it to
        per_head_pages.  Called automatically when staging buffer fills.
        """
        device    = self.device
        r_buf     = self.stg_r[key]      # [PAGE_SIZE, G]
        theta_buf = self.stg_theta[key]  # [PAGE_SIZE, G]
        V_buf     = self.stg_V[key]      # [PAGE_SIZE, dh]
        n         = r_buf.shape[0]
        tier      = self.tiers[3]   # b3 (Low) for all decode tokens
 
        page = build_page_from_codes(
            r_buf.cpu(), theta_buf.cpu(), tier, segment_id=2
        )
 
        header_size   = HEADER_BYTES + tier.G * 4
        theta_bytes   = ((n * tier.G * tier.b_theta) + 7) // 8
        theta_offset  = header_size
        radius_offset = theta_offset + theta_bytes
 
        ptable = PointerTable()
        ptable.add_page(0, theta_offset, radius_offset)
 
        if key not in self.per_head_pages:
            self.per_head_pages[key] = []
 
        self.per_head_pages[key].append((
            page.to(device),
            ptable.to_tensor().to(device),
            tier.b_theta,
            n,
            V_buf,
            None,   # no dense K stored — paper compliant
        ))
 
        del self.stg_r[key]
        del self.stg_theta[key]
        del self.stg_V[key]
 
    def _maybe_refresh(self) -> None:
        """
        Re-run the RDR controller on the current retained token set.
        Called every REFRESH_CADENCE decode steps.
        Skipped if REFRESH_CADENCE == 0 (prefill-only mode).
 
        What happens:
          1. allocate() re-scores all retained tokens using updated omega.
          2. Tokens that now exceed the budget are downgraded or dropped.
          3. Tokens below the budget get upgrade consideration.
          4. Frozen tokens (cooldown > 0) are skipped (C.4 hysteresis).
        """
        if REFRESH_CADENCE == 0 or not self._retained_tokens:
            return
        # _decode_step is incremented in _append_decode_kv (layer=0,head=0).
        # Here we only check whether we should act.
        if self._decode_step % REFRESH_CADENCE != 0:
            return
 
        # online_refresh from allocation.py: ticks cooldowns, runs downgrade
        # + upgrade passes, respects frozen tokens (C.3 + C.4).
        old_tiers = {id(ts): ts.new_tier_id for ts in self._retained_tokens}
        updated   = online_refresh(self._retained_tokens, self.tiers)
 
        # Commit and detect which (layer, head) pairs changed tier
        changed_lh = set()
        for tok in updated:
            if tok.new_tier_id != old_tiers.get(id(tok)):
                changed_lh.add((tok.layer, tok.head))
            tok.commit_tier()
 
        self._retained_tokens = updated
 
        # Rebuild compressed pages for (layer, head) pairs that changed
        if changed_lh:
            self._rebuild_pages_for(changed_lh)
 

    def _rebuild_pages_for(self, changed_lh: set) -> None:
        """
        After an online refresh, re-page the (layer, head) pairs in
        changed_lh using self._codes_all — pre-encoded codes for every
        tier stored at prefill time.  No dense K access anywhere.
        Updates self.per_head_pages and self._ctx_token_order in place.
        """
        # Group current retained tokens by (layer, head, tier_id)
        lh_tier: Dict[Tuple[int,int,int], List] = defaultdict(list)
        for ts in self._retained_tokens:
            key = (ts.layer, ts.head, ts.new_tier_id)
            lh_tier[key].append(ts)
 
        for (layer, head) in changed_lh:
            # Retrieve pre-encoded codes for this (layer, head).
            # _codes_all is populated at prefill for all 3 tiers.
            head_codes = self._codes_all.get((layer, head))
            if head_codes is None:
                continue  # no codes cached (shouldn't happen after prefill)
 
            new_entries = []
            new_order   = []
 
            # Process tiers in ascending tier_id order (b1 → b2 → b3)
            for tier_id in [1, 2, 3]:
                toks = lh_tier.get((layer, head, tier_id), [])
                if not toks:
                    continue
                toks.sort(key=lambda t: t.index)
                tier = self.tiers[tier_id]
 
                # Look up pre-encoded codes for this tier -- no re-encoding.
                code_pair = head_codes.get(tier_id)
                if code_pair is None:
                    continue  # codebook was missing at prefill; skip tier
                r_all_cpu, th_all_cpu = code_pair   # [T, G_tier] each, CPU
 
                V_lh = self._V_all.get((layer, head))
                page_list, r_parts, theta_parts, V_parts = [], [], [], []
                ptable = PointerTable()
                offset = 0
 
                for i in range(0, len(toks), PAGE_SIZE):
                    chunk   = toks[i : i + PAGE_SIZE]
                    N       = len(chunk)
                    indices = [t.index for t in chunk]
                    # Direct index-select from cached codes -- zero dense K.
                    r_g  = r_all_cpu[indices]    # [N, G_tier] float32
                    th_c = th_all_cpu[indices]   # [N, G_tier] int32
                    page = build_page_from_codes(
                        r_g, th_c, tier, chunk[0].segment_id)
                    h_sz  = HEADER_BYTES + tier.G * 4
                    t_sz  = ((N * tier.G * tier.b_theta) + 7) // 8
                    ptable.add_page(offset, offset + h_sz, offset + h_sz + t_sz)
                    offset += page.numel()
                    page_list.append(page)
                    r_parts.append(r_g.cpu())
                    theta_parts.append(th_c.cpu())
                    if V_lh is not None:
                        V_parts.append(V_lh[indices].to(self.device))

                V_tensor = torch.cat(V_parts, dim=0) if V_parts else None
 
                new_entries.append((
                    torch.cat(page_list).to(self.device),
                    ptable.to_tensor().to(self.device),
                    tier.b_theta,
                    len(toks),
                    V_tensor,
                    # Slot [5]: (r_groups, theta_codes) -- ADA compliant
                    (torch.cat(r_parts, dim=0), torch.cat(theta_parts, dim=0)),
                ))
                new_order.extend(toks)
 
            self.per_head_pages[(layer, head)]   = new_entries
            self._ctx_token_order[(layer, head)] = new_order
 
    def uninstall(self) -> None:
        if self._patched:
            unpatch_decode(self.model, self._original_forwards)
            self._patched = False
            print("[SphericalKVPipeline] Attention layers restored.")
 
    def _reset_state(self) -> None:
        self.per_head_pages.clear()
        self.stg_r.clear()
        self.stg_theta.clear()
        self.stg_V.clear()
        self.reuse          = None
        self.stability      = None
        self.seq_len        = 0
        self._decode_step   = 0
        self._retained_tokens.clear()
        self._ctx_token_order.clear()
        self._codes_all.clear()
        self._V_all.clear()