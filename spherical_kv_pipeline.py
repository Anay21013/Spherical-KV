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

 
# def reference_codebook_decode(
#     q:          torch.Tensor,   # [dh]
#     codebooks:  torch.Tensor,   # [G, codebook_size, group_size]
#     r_codes:    torch.Tensor,   # [T_ctx, G]  pre-stored per-group radii
#     theta_codes: torch.Tensor,  # [T_ctx, G]  pre-stored codebook indices (int32)
#     group_size: int,
#     num_groups: int,
# ) -> torch.Tensor:
#     device     = q.device
#     T_ctx, G   = r_codes.shape
#     r_c   = r_codes.to(device)    # [T_ctx, G]
#     th_c  = theta_codes.to(device).long()  # [T_ctx, G]
 
#     q_groups = q.view(num_groups, group_size)   # [G, g]

#     # Per-group query normalization (B.2): q̃(j) = q(j)/(‖q(j)‖₂+ε)
#     q_norms = q_groups.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # [G, 1]
#     q_normed = q_groups / q_norms                                   # [G, g]
 
#     # Gather codewords for all context tokens: [G, T_ctx, g]
#     cw = codebooks[
#         torch.arange(num_groups, device=device).unsqueeze(1),  # [G, 1]
#         th_c.T,                                                 # [G, T_ctx]
#     ]
 
#     # Dot products with normalized q (B.2): [G, T_ctx]
#     dots = (cw * q_normed.unsqueeze(1)).sum(-1)   # [G, T_ctx]

#     # Cosine clipping (B.2): s(j) ← clip(s(j), −1, 1)
#     dots = dots.clamp(-1.0, 1.0)
 
#     # Weighted sum over groups: [T_ctx]
#     logits = (r_c.T * dots).sum(0) / math.sqrt(num_groups * group_size)
 
#     return logits
 
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



# def _encode_keys_with_codebooks(
#     K_chunk:    torch.Tensor,   # [N, dh]
#     codebooks:  torch.Tensor,   # [G, codebook_size, group_size]  tier-specific
#     group_size: int,             # g for this tier
#     num_groups: int,             # G for this tier
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     N         = K_chunk.shape[0]
#     K_grouped = K_chunk.view(N, num_groups, group_size)      # [N, G, g]
#     r_groups  = K_grouped.norm(dim=-1)                       # [N, G]
#     K_dir     = K_grouped / (r_groups.unsqueeze(-1) + EPS)   # [N, G, g]
 
#     cb   = codebooks.to(K_dir.device)                        # [G, cb_size, g]
#     sims = torch.einsum('ngi,gci->ngc', K_dir, cb)           # [N, G, cb_size]
#     theta_codes = sims.argmax(dim=-1).to(torch.int32)        # [N, G]
#     return r_groups, theta_codes
 
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
#     groups: Dict[Tuple[int, int, int], List[TokenState]] = defaultdict(list)
#     for ts in retained_tokens:
#         groups[(ts.layer, ts.head, ts.new_tier_id)].append(ts)
 
#     per_head: Dict[Tuple[int, int], List] = defaultdict(list)
 
#     for (layer, head, tier_id), tokens in groups.items():
#         tokens.sort(key=lambda t: t.index)
#         tier  = tiers[tier_id]
#         K_lh  = K_all[(layer, head)]
#         V_lh  = V_all[(layer, head)]
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
#             # The reference decode path unpacks these directly
#             (torch.cat(r_parts, dim=0), torch.cat(theta_parts, dim=0)),
#         ))
 
#     return per_head
 

 
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
 
#         self.per_head_pages: Dict[Tuple[int, int], List] = {}
 
#         self.stg_r:     Dict[Tuple[int, int], torch.Tensor] = {}
#         self.stg_theta: Dict[Tuple[int, int], torch.Tensor] = {}
#         self.stg_V:     Dict[Tuple[int, int], torch.Tensor] = {}
 
#         self.reuse:     Optional[torch.Tensor] = None
#         self.stability: Optional[torch.Tensor] = None
#         self.seq_len:   int = 0
 
 
#         self._retained_tokens: list = []
#         self._decode_step: int = 0   # counts decode steps since prefill
 
#         self._ctx_token_order: Dict[Tuple[int, int], List] = {}
 
#         self._codes_all: Dict[Tuple[int,int], dict] = {}
#         self._V_all = {}
#         self._decode_pages:  Dict[Tuple[int, int], List] = {}
#         self.stg_rgroups:    Dict[Tuple[int, int], torch.Tensor] = {}
 
#         self._original_forwards: Dict[int, object] = {}
#         self._patched = False
 
#         mode = "fused CUDA" if self.use_fused else "reference (pure-torch)"
#         print(
#             f"[SphericalKVPipeline] layers={self.num_layers}  "
#             f"kv_heads={self.num_kv_heads}  head_dim={self.head_dim}  "
#             f"groups={self.num_groups}  decode_mode={mode}"
#         )
 
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
 
#                 G_fine   = self.num_groups        # 8
#                 g_fine   = self.group_size        # 16
#                 K_grouped = K_lh.view(T, G_fine, g_fine)
#                 r_groups_all = K_grouped.norm(dim=-1)   # [T, G_fine]
 
#                 for t in range(T):
                   
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
 
#         import config as _cfg
#         # _cfg.GLOBAL_BUDGET_BITS = 1_000_000
#         # print(f"[prefill] Budget: {_cfg.GLOBAL_BUDGET_BITS:,} bits (fixed global budget)")
#         _bpt = getattr(_cfg, 'BITS_PER_TOKEN', 30.9)
#         _cfg.GLOBAL_BUDGET_BITS = _bpt * T * self.num_layers * self.num_kv_heads
#         print(f"[prefill] Budget: {_cfg.GLOBAL_BUDGET_BITS:,} bits "
#               f"({_bpt} bpt x T={T} x L={self.num_layers} x H={self.num_kv_heads})")
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
 
#         print("[prefill] Pre-encoding all tokens at all tiers for ADA-compliant refresh ...")
#         self._codes_all.clear()
#         for _li in range(self.num_layers):
#             for _h in range(self.num_kv_heads):
#                 K_lh_cpu = K_all[(_li, _h)]          # [T, dh] float32 CPU
#                 tier_codes: dict = {}
#                 for _tid in [1, 2, 3]:
#                     _tier = self.tiers[_tid]
#                     _cb = self.codebooks.get((_li, _h, _tid))
#                     if _cb is None:
#                         from codebook_loader import get_codebook as _gcb
#                         _cb = _gcb(self.codebooks, _li, _h, _tid)
#                     if _cb is None:
#                         continue
#                     _r, _th = _encode_keys_with_codebooks(
#                         K_lh_cpu, _cb, _tier.g, _tier.G
#                     )  # [T, G_tier] each; one vectorised call for all T tokens
#                     tier_codes[_tid] = (_r.cpu(), _th.cpu())
#                 self._codes_all[(_li, _h)] = tier_codes
#         # Dense K no longer needed — release immediately.
#         del K_all
 
#         # _V_all kept for online refresh (values stay dense FP16 by design).
#         self._V_all = {k: v.cpu() for k, v in V_all.items()}
#         del V_all
#         torch.cuda.empty_cache()
 
#         self._ctx_token_order.clear()
#         lh_tier_map: Dict[Tuple[int,int,int], List] = defaultdict(list)
#         for ts in retained:
#             lh_tier_map[(ts.layer, ts.head, ts.new_tier_id)].append(ts)
#         lh_map: Dict[Tuple[int,int], List] = defaultdict(list)
#         for (layer, head, tid), toks in lh_tier_map.items():
#             toks.sort(key=lambda t: t.index)
#             lh_map[(layer, head)].extend(toks)
#         self._ctx_token_order = dict(lh_map)
 
#         print("[prefill] Patching attention layers for decode ...")
#         self._original_forwards = patch_for_decode(self.model, self)
#         self._patched = True
#         print("[prefill] Done -- model ready for compressed decode.")
 
#     def _compressed_head_attention_batched(
#         self,
#         layer_idx: int,
#         kv_head:   int,
#         q_batch:   torch.Tensor,   # [num_q, dh]
#         k_new:     torch.Tensor,   # [dh]
#         v_new:     torch.Tensor,   # [dh]
#     ) -> torch.Tensor:
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
#                     r_codes, th_codes = K_tier          # [N,G] each, CPU
#                     r_codes  = r_codes.to(device)       # [N, tier_G]
#                     th_codes = th_codes.to(device)      # [N, tier_G]
#                     q_groups = q.view(num_q, tier_G, tier_g)  # [num_q, G, g]

#                     # Per-group query normalization (B.2)
#                     q_norms = q_groups.norm(dim=-1, keepdim=True).clamp(min=1e-6)
#                     q_normed = q_groups / q_norms         # [num_q, G, g]
 
#                     # Gather codewords: [tier_G, N, tier_g]
#                     cw = cb_lh[
#                         torch.arange(tier_G, device=device).unsqueeze(1),
#                         th_codes.long().T,
#                     ]
 
#                     # Dots with normalized q: [num_q, tier_G, N]
#                     dots = torch.einsum('qgi,gni->qgn', q_normed, cw)
#                     # Cosine clipping (B.2)
#                     dots = dots.clamp(-1.0, 1.0)
#                     ctx_logits_parts.append(
#                         (r_codes.T.unsqueeze(0) * dots).sum(1) / math.sqrt(dh)
#                     )  # [num_q, n_tokens]
#                 V_ctx_parts.append(V_tier)  # [n_tokens, dh]
 
#             for (dp, dpt, d_btheta, d_ntok, d_V, _) in self._decode_pages.get(key, []):
#                 b3_dec   = self.tiers[3]
#                 dec_G    = b3_dec.G
#                 dec_g    = b3_dec.g
#                 cb_dec   = get_codebook(self.codebooks, layer_idx, kv_head, b3_dec.tier_id)
#                 q_groups = q.view(num_q, dec_G, dec_g)
#                 if self.use_fused:
#                     raw = _call_fused(dp, dpt, q, cb_dec, d_btheta, dh, dec_G, dec_g)
#                     ctx_logits_parts.append(raw[:, :d_ntok])
#                 else:
#                     r_c, th_c = d_V, None 
#                     _bt_to_tid2 = {t.b_theta: t.tier_id for t in self.tiers if t.tier_id != 0}
#                     tier_id2  = _bt_to_tid2.get(d_btheta, 3)
#                     tier_obj2 = self.tiers[tier_id2]
#                     raw = _call_fused(dp, dpt, q, cb_dec, d_btheta, dh, dec_G, dec_g)
#                     ctx_logits_parts.append(raw[:, :d_ntok])
#                 V_ctx_parts.append(d_V)
 
#             stg_r     = self.stg_r.get(key)
#             stg_theta = self.stg_theta.get(key)
#             stg_V     = self.stg_V.get(key)
#             if stg_r is not None and stg_r.shape[0] > 0:
#                 N_stg     = stg_r.shape[0]
#                 b3        = self.tiers[3]
#                 stg_G     = b3.G
#                 stg_g     = b3.g
#                 cb_b3     = get_codebook(self.codebooks, layer_idx, kv_head, b3.tier_id)
#                 q_groups  = q.view(num_q, stg_G, stg_g)  # [num_q, G_b3, g_b3]

#                 # Per-group query normalization (B.2)
#                 q_norms = q_groups.norm(dim=-1, keepdim=True).clamp(min=1e-6)
#                 q_normed = q_groups / q_norms              # [num_q, G_b3, g_b3]
 
#                 # gather codewords for all staging tokens: [G_b3, N_stg, g_b3]
#                 cw = cb_b3[
#                     torch.arange(stg_G, device=device).unsqueeze(1),  # [G_b3, 1]
#                     stg_theta.long().T                                  # [G_b3, N_stg]
#                 ]
 
#                 # dot products with normalized q: [num_q, G_b3, N_stg]
#                 dots = torch.einsum('qgi,gni->qgn', q_normed, cw)

#                 # Cosine clipping (B.2)
#                 dots = dots.clamp(-1.0, 1.0)
 
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
 
#         n_ctx = ctx_logits.shape[1] if ctx_logits_parts else 0
#         # C.2: update EMA importance weights EVERY decode step, not just
#         # at refresh cadence.  omega_i(t) must track current attention.
#         if n_ctx > 0:
#             attn_ctx_mean = attn[:, :n_ctx].mean(dim=0).detach().cpu()
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
 
#     def _append_decode_kv(
#         self,
#         layer_idx: int,
#         kv_head:   int,
#         k_new:     torch.Tensor,   # [dh]
#         v_new:     torch.Tensor,   # [dh]
#     ) -> None:
#         key    = (layer_idx, kv_head)
#         device = self.device
#         kn     = k_new.float().to(device).unsqueeze(0)   # [1, dh]\r
#         vn     = v_new.float().to(device).unsqueeze(0)   # [1, dh]
 
#         G_fine = self.num_groups
#         g_fine = self.group_size
#         k_grouped     = kn.view(G_fine, g_fine)           # [G_fine, g_fine]
#         r_groups_fine = k_grouped.norm(dim=-1)             # [G_fine] on device
#         if key not in self.stg_rgroups:
#             self.stg_rgroups[key] = r_groups_fine.unsqueeze(0)          # [1, G_fine]
#         else:
#             self.stg_rgroups[key] = torch.cat(
#                 [self.stg_rgroups[key], r_groups_fine.unsqueeze(0)], dim=0
#             )  # [N_stg, G_fine]
 
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
 
#         if layer_idx == 0 and kv_head == 0:
#             self._decode_step += 1
#             self._maybe_refresh()
 
#     def _flush_decode_page(self, key: Tuple[int, int]) -> None:
#         layer_idx, kv_head = key
#         device    = self.device
#         r_buf     = self.stg_r[key]          # [N, G_b3]
#         theta_buf = self.stg_theta[key]      # [N, G_b3]
#         V_buf     = self.stg_V[key]          # [N, dh]
#         rg_buf    = self.stg_rgroups.get(key)  # [N, G_fine] or None
#         n         = r_buf.shape[0]
#         b3        = self.tiers[3]
 
#         page = build_page_from_codes(
#             r_buf.cpu(), theta_buf.cpu(), b3, segment_id=2
#         )
 
#         header_size   = HEADER_BYTES + b3.G * 4
#         theta_bytes   = ((n * b3.G * b3.b_theta) + 7) // 8
#         theta_offset  = header_size
#         radius_offset = theta_offset + theta_bytes
 
#         ptable = PointerTable()
#         ptable.add_page(0, theta_offset, radius_offset)
 
#         if key not in self._decode_pages:
#             self._decode_pages[key] = []
 
#         self._decode_pages[key].append((
#             page.to(device),
#             ptable.to_tensor().to(device),
#             b3.b_theta,
#             n,
#             V_buf,
#             None, 
#         ))
 
#         flush_num = len(self._decode_pages[key]) - 1  # 0-based
#         start_idx = self.seq_len + flush_num * n
 
#         new_token_states: List[TokenState] = []
#         dh = self.head_dim
#         for local_i in range(n):
#             abs_idx = start_idx + local_i
#             if rg_buf is not None:
#                 r_groups = rg_buf[local_i].float() 
#             else:
#                 # Fallback: use b3 radii as approximation
#                 r_groups = r_buf[local_i].cpu().float() 
#             r_scalar = float(r_groups.norm())
 
#             ts = TokenState(
#                 layer      = layer_idx,
#                 head       = kv_head,
#                 index      = abs_idx,
#                 r          = r_scalar,
#                 phi        = torch.zeros(dh - 1, dtype=torch.float32),
#                 segment_id = 2,      # recent
#                 age        = 1,
#                 prev_tier_id = b3.tier_id,
#                 protected  = False,
#                 r_groups   = r_groups,
#                 omega      = 1.0,
#             )
#             ts.assign_tier_protected(b3.tier_id)
#             ts.commit_tier()
#             new_token_states.append(ts)
 
#         # Add to retained set and token-order map (for ω updates in attention)
#         self._retained_tokens.extend(new_token_states)
#         if key not in self._ctx_token_order:
#             self._ctx_token_order[key] = []
#         self._ctx_token_order[key].extend(new_token_states)
 
#         # Clean up staging buffers
#         del self.stg_r[key]
#         del self.stg_theta[key]
#         del self.stg_V[key]
#         if key in self.stg_rgroups:
#             del self.stg_rgroups[key]
 
#     def _maybe_refresh(self) -> None:
#         """
#         Re-run the RDR controller on the current retained token set.
#         Called every REFRESH_CADENCE decode steps.
#         Skipped if REFRESH_CADENCE == 0 (prefill-only mode).
#         """
#         if REFRESH_CADENCE == 0 or not self._retained_tokens:
#             return
#         if self._decode_step % REFRESH_CADENCE != 0:
#             return
#         old_tiers = {id(ts): ts.new_tier_id for ts in self._retained_tokens}
#         updated   = online_refresh(self._retained_tokens, self.tiers,
#                                    prefill_len=self.seq_len)
 
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
 
 
#     def _rebuild_pages_for(self, changed_lh: set) -> None:
#         """
#         After an online refresh, re-page the (layer, head) pairs in
#         changed_lh using self._codes_all — pre-encoded codes for every
#         tier stored at prefill time.  No dense K access anywhere.
#         Updates self.per_head_pages and self._ctx_token_order in place.
#         """
#         lh_tier: Dict[Tuple[int,int,int], List] = defaultdict(list)
#         for ts in self._retained_tokens:
#             if ts.index < self.seq_len:   # prefill tokens only
#                 key2 = (ts.layer, ts.head, ts.new_tier_id)
#                 lh_tier[key2].append(ts)
 
#         for (layer, head) in changed_lh:
#             # Retrieve pre-encoded codes for this (layer, head).
#             # _codes_all is populated at prefill for all 3 tiers.
#             head_codes = self._codes_all.get((layer, head))
#             if head_codes is None:
#                 continue  # no codes cached (shouldn't happen after prefill)
 
#             new_entries = []
#             new_order   = []
 
#             # Process tiers in ascending tier_id order (b1 → b2 → b3)
#             for tier_id in [1, 2, 3]:
#                 toks = lh_tier.get((layer, head, tier_id), [])
#                 if not toks:
#                     continue
#                 toks.sort(key=lambda t: t.index)
#                 tier = self.tiers[tier_id]
 
#                 # Look up pre-encoded codes for this tier -- no re-encoding.
#                 code_pair = head_codes.get(tier_id)
#                 if code_pair is None:
#                     continue  # codebook was missing at prefill; skip tier
#                 r_all_cpu, th_all_cpu = code_pair   # [T, G_tier] each, CPU
 
#                 V_lh = self._V_all.get((layer, head))
#                 page_list, r_parts, theta_parts, V_parts = [], [], [], []
#                 ptable = PointerTable()
#                 offset = 0
 
#                 for i in range(0, len(toks), PAGE_SIZE):
#                     chunk   = toks[i : i + PAGE_SIZE]
#                     N       = len(chunk)
#                     indices = [t.index for t in chunk]
#                     # Direct index-select from cached codes -- zero dense K.
#                     r_g  = r_all_cpu[indices]    # [N, G_tier] float32
#                     th_c = th_all_cpu[indices]   # [N, G_tier] int32
#                     page = build_page_from_codes(
#                         r_g, th_c, tier, chunk[0].segment_id)
#                     h_sz  = HEADER_BYTES + tier.G * 4
#                     t_sz  = ((N * tier.G * tier.b_theta) + 7) // 8
#                     ptable.add_page(offset, offset + h_sz, offset + h_sz + t_sz)
#                     offset += page.numel()
#                     page_list.append(page)
#                     r_parts.append(r_g.cpu())
#                     theta_parts.append(th_c.cpu())
#                     if V_lh is not None:
#                         V_parts.append(V_lh[indices].to(self.device))
 
#                 V_tensor = torch.cat(V_parts, dim=0) if V_parts else None
 
#                 new_entries.append((
#                     torch.cat(page_list).to(self.device),
#                     ptable.to_tensor().to(self.device),
#                     tier.b_theta,
#                     len(toks),
#                     V_tensor,
#                     (torch.cat(r_parts, dim=0), torch.cat(theta_parts, dim=0)),
#                 ))
#                 new_order.extend(toks)
 
#             self.per_head_pages[(layer, head)] = new_entries
 
#             old_order = self._ctx_token_order.get((layer, head), [])
#             decode_order = [tok for tok in old_order if tok.index >= self.seq_len]
#             self._ctx_token_order[(layer, head)] = new_order + decode_order
 
#     def uninstall(self) -> None:
#         if self._patched:
#             unpatch_decode(self.model, self._original_forwards)
#             self._patched = False
#             print("[SphericalKVPipeline] Attention layers restored.")
 
#     def _reset_state(self) -> None:
#         self.per_head_pages.clear()
#         self._decode_pages.clear()
#         self.stg_r.clear()
#         self.stg_theta.clear()
#         self.stg_V.clear()
#         self.stg_rgroups.clear()
#         self.reuse          = None
#         self.stability      = None
#         self.seq_len        = 0
#         self._decode_step   = 0
#         self._retained_tokens.clear()
#         self._ctx_token_order.clear()
#         self._codes_all.clear()
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
                               COOLDOWN_STEPS, EMA_BETA)
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


# ─────────────────────────────────────────────────────────────────────
# Compiled EMA update (paper §C.2 page-level ω).
#
# Out-of-place expression form — no in-place ops, no scratch buffers,
# no mid-chain mutations. This lets Inductor fuse the long pointwise
# chain `(m - m_max).clamp.exp * l_q` plus the normalize/aggregate into
# ~3 Triton kernels instead of ~12 separate ATen launches.
#
# Mathematically identical to the prior in-place version: page-level
# attention mass via online-softmax merge across pages, averaged over
# q_heads in each GQA group, then EMA-blended into the persistent omega.
# Paper-faithful under §C.2's "kernel-side block statistics" clause.
#
# `dynamic=True` so one compiled artifact handles all T values
# (W2 has variable prompt lengths). `fullgraph=True` to fail loudly
# rather than silently fall back to eager on any graph break.
# ─────────────────────────────────────────────────────────────────────
def _omega_ema_eager(m_q, l_q, omega_prev,
                     num_kv: int, kv_groups: int, max_blocks: int,
                     ema_beta: float):
    """
    Inputs (all on device, all float32):
        m_q, l_q   : views into _partial_scratch[..., 0] / [..., 1]
                     shape [num_q, max_blocks]
        omega_prev : prior EMA buffer [num_kv, max_blocks]
        num_kv, kv_groups, max_blocks : GQA + page-table shape constants
        ema_beta   : EMA smoothing factor (0.9)

    Returns:
        Updated omega buffer [num_kv, max_blocks]. Caller assigns
        this back into the per-layer omega dict.
    """
    # Mask invalid pages with -inf so they don't win the amax.
    valid = l_q > 0
    neg_inf = m_q.new_full((), -1e30)
    m_safe = torch.where(valid, m_q, neg_inf)

    # Per-q-head max across pages, then weight = exp(m - max) * l_q.
    # Long pointwise chain — Inductor fuses into one Triton kernel.
    m_max = m_safe.amax(dim=1, keepdim=True)
    w = (m_safe - m_max).clamp_min(-60.0).exp() * l_q

    # Normalize across pages within each q_head.
    Z = w.sum(dim=1, keepdim=True).clamp_min(1e-12)
    page_mass_q = w / Z                                # [num_q, max_blocks]

    # Aggregate q_heads in each GQA group -> per-(kv_head, page) mass.
    page_mass_kv = page_mass_q.view(
        num_kv, kv_groups, max_blocks
    ).mean(dim=1)                                      # [num_kv, max_blocks]

    # EMA blend (out-of-place; caller assigns the result back).
    return omega_prev * ema_beta + page_mass_kv * (1.0 - ema_beta)


try:
    _omega_ema_compiled = torch.compile(
        _omega_ema_eager,
        dynamic=True,
        mode="default",
        fullgraph=True,
    )
except Exception as _compile_err:
    print(f"[SphericalKVPipeline] torch.compile of omega EMA failed: "
          f"{_compile_err}; using eager fallback.")
    _omega_ema_compiled = _omega_ema_eager


 
def _rotate_half(x):
    """Llama RoPE rotate_half: cat(-x[..., half:], x[..., :half], dim=-1)."""
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

def reference_codebook_decode(
    q:          torch.Tensor,   # [dh]
    codebooks:  torch.Tensor,   # [G, codebook_size, group_size]
    r_codes:    torch.Tensor,   # [T_ctx, G]  pre-stored per-group radii
    theta_codes: torch.Tensor,  # [T_ctx, G]  pre-stored codebook indices (int32)
    group_size: int,
    num_groups: int,
) -> torch.Tensor:
    device     = q.device
    T_ctx, G   = r_codes.shape
    r_c   = r_codes.to(device)    # [T_ctx, G]
    th_c  = theta_codes.to(device).long()  # [T_ctx, G]
 
    q_groups = q.view(num_groups, group_size)   # [G, g]

    # Per-group query normalization (B.2): q̃(j) = q(j)/(‖q(j)‖₂+ε)
    q_norms = q_groups.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # [G, 1]
    q_normed = q_groups / q_norms                                   # [G, g]
 
    # Gather codewords for all context tokens: [G, T_ctx, g]
    cw = codebooks[
        torch.arange(num_groups, device=device).unsqueeze(1),  # [G, 1]
        th_c.T,                                                 # [G, T_ctx]
    ]
 
    # Dot products with normalized q (B.2): [G, T_ctx]
    dots = (cw * q_normed.unsqueeze(1)).sum(-1)   # [G, T_ctx]

    # Cosine clipping (B.2): s(j) ← clip(s(j), −1, 1)
    dots = dots.clamp(-1.0, 1.0)
 
    # Weighted sum over groups: [T_ctx]
    logits = (r_c.T * dots).sum(0) / math.sqrt(num_groups * group_size)
 
    return logits
 
_fused_ok: Optional[bool] = None

# If set to True (via env SPHKV_FUSED_STRICT=1), _fused_available raises
# instead of silently falling back.  Useful for diagnosing "why am I on
# the torch path?" — forces the real error to surface.
_FUSED_STRICT = bool(int(os.environ.get("SPHKV_FUSED_STRICT", "0")))


def _fused_available() -> bool:
    global _fused_ok
    if _fused_ok is not None:
        return _fused_ok

    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    print("[SphericalKVPipeline] Attempting to load fused_decode CUDA extension...",
          flush=True)
    try:
        from fused_decode import fused_decode   # noqa: F401
        _fused_ok = True
        print("[SphericalKVPipeline] fused_decode CUDA extension LOADED successfully.",
              flush=True)
    except Exception as exc:
        _fused_ok = False
        import traceback
        # Print loudly on BOTH streams to survive stdout/stderr buffering
        # reordering by the shell.
        for stream in (sys.stdout, sys.stderr):
            print("=" * 72, file=stream, flush=True)
            print("[SphericalKVPipeline] FUSED KERNEL LOAD FAILED", file=stream, flush=True)
            print(f"  {type(exc).__name__}: {exc}", file=stream, flush=True)
            print("=" * 72, file=stream, flush=True)
        traceback.print_exc()
        sys.stderr.flush()
        print("[SphericalKVPipeline] falling back to torch reference path.",
              flush=True)
        if _FUSED_STRICT:
            raise RuntimeError(
                "Fused kernel failed to load and SPHKV_FUSED_STRICT=1 is set."
            ) from exc
    return _fused_ok
 
 
def _call_fused(
    pages_tensor:  torch.Tensor,
    ptable_tensor: torch.Tensor,
    positions:     torch.Tensor,
    cos_table:     torch.Tensor,
    sin_table:     torch.Tensor,
    q:             torch.Tensor,
    codebooks_lh:  torch.Tensor,
    b_theta:       int,
    dh:            int,
    num_groups:    int,
    group_size:    int,
) -> torch.Tensor:
    from fused_decode import fused_decode_ragged
    squeeze = (q.dim() == 1)
    if squeeze:
        q = q.unsqueeze(0)
    num_q = q.shape[0]
    num_pages = ptable_tensor.shape[0]
    head_ids = torch.zeros(num_pages, dtype=torch.int32, device=q.device)
    result = fused_decode_ragged(
        pages_tensor,
        ptable_tensor,
        head_ids,
        positions.to(torch.int32),
        cos_table.float(),
        sin_table.float(),
        q.float(),
        num_q,
        1,
        codebooks_lh.float(),
        dh=dh,
        groups=num_groups,
        group_size=group_size,
        b_theta=b_theta,
        page_size=PAGE_SIZE,
    )
    if squeeze:
        return result.squeeze(0)
    return result

def _encode_keys_with_codebooks(
    K_chunk:    torch.Tensor,   # [N, dh]
    codebooks:  torch.Tensor,   # [G, codebook_size, group_size]  tier-specific
    group_size: int,             # g for this tier
    num_groups: int,             # G for this tier
) -> Tuple[torch.Tensor, torch.Tensor]:
    N         = K_chunk.shape[0]
    K_grouped = K_chunk.view(N, num_groups, group_size)      # [N, G, g]
    r_groups  = K_grouped.norm(dim=-1)                       # [N, G]
    K_dir     = K_grouped / (r_groups.unsqueeze(-1) + EPS)   # [N, G, g]
 
    cb   = codebooks.to(K_dir.device)                        # [G, cb_size, g]
    sims = torch.einsum('ngi,gci->ngc', K_dir, cb)           # [N, G, cb_size]
    theta_codes = sims.argmax(dim=-1).to(torch.int32)        # [N, G]
    return r_groups, theta_codes
 
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
    groups: Dict[Tuple[int, int, int], List[TokenState]] = defaultdict(list)
    for ts in retained_tokens:
        groups[(ts.layer, ts.head, ts.new_tier_id)].append(ts)
 
    per_head: Dict[Tuple[int, int], List] = defaultdict(list)
 
    for (layer, head, tier_id), tokens in groups.items():
        tokens.sort(key=lambda t: t.index)
        tier  = tiers[tier_id]
        K_lh  = K_all[(layer, head)]
        V_lh  = V_all[(layer, head)]
        cb_key = (layer, head, tier_id)
        if cb_key in codebooks:
            cb_lh = codebooks[cb_key]
        elif (layer, head) in codebooks:
            cb_lh = codebooks[(layer, head)]
        else:
            continue

 
        tier_g  = tier.g    # group size for this tier
        tier_G  = tier.G    # number of groups for this tier
 
        page_list:   List[torch.Tensor] = []
        r_parts:     List[torch.Tensor] = []   # [N_chunk, G] float32
        theta_parts: List[torch.Tensor] = []   # [N_chunk, G] int32
        V_tier_parts: List[torch.Tensor] = []
        positions_parts: List[torch.Tensor] = []   # [N_chunk] token positions for RoPE
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
            r_parts.append(r_groups.cpu())        # [N, G] float32
            theta_parts.append(theta_codes.cpu()) # [N, G] int32
            V_tier_parts.append(V_chunk)
            positions_parts.append(torch.tensor(indices, dtype=torch.long))
            ptable.add_page(offset, theta_offset, radius_offset)
            offset += page.numel()
 
        per_head[(layer, head)].append((
            torch.cat(page_list).to(device),
            ptable.to_tensor().to(device),
            tier.b_theta,
            len(tokens),
            torch.cat(V_tier_parts, dim=0).to(device),
            # Slot [5]: (r_groups, theta_codes) on CPU.
            # The reference decode path unpacks these directly
            (torch.cat(r_parts, dim=0), torch.cat(theta_parts, dim=0)),
            # Slot [6]: per-token absolute positions (for RoPE back-rotation)
            torch.cat(positions_parts, dim=0).to(device),
        ))
 
    return per_head
 

 
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
 
        self.per_head_pages: Dict[Tuple[int, int], List] = {}
 
        self.stg_r:     Dict[Tuple[int, int], torch.Tensor] = {}
        self.stg_theta: Dict[Tuple[int, int], torch.Tensor] = {}
        self.stg_V:     Dict[Tuple[int, int], torch.Tensor] = {}
        # Per-key position buffers for RoPE back-rotation in staging path
        self.stg_positions: Dict[Tuple[int, int], torch.Tensor] = {}

        # RoPE cos/sin tables, built lazily on first decode call
        self._rope_cos_table: Optional[torch.Tensor] = None
        self._rope_sin_table: Optional[torch.Tensor] = None
        self._rope_max_pos: int = 0
 
        self.reuse:     Optional[torch.Tensor] = None
        self.stability: Optional[torch.Tensor] = None
        self.seq_len:   int = 0
 
 
        self._retained_tokens: list = []
        self._decode_step: int = 0   # counts decode steps since prefill
 
        self._ctx_token_order: Dict[Tuple[int, int], List] = {}
 

        self._omega_buf:   Dict[Tuple[int, int], torch.Tensor] = {}
        self._window_buf:  Dict[Tuple[int, int], torch.Tensor] = {}
        self._window_ptr:  Dict[Tuple[int, int], int] = {}
        self._codes_all: Dict[Tuple[int,int], dict] = {}
        self._V_all = {}
        self._decode_pages:  Dict[Tuple[int, int], List] = {}
        self.stg_rgroups:    Dict[Tuple[int, int], torch.Tensor] = {}

        # ── GPU-resident page-level omega (paper §C.2 EMA, kernel-block stats) ──
        # Per-layer tensor [num_kv, max_blocks] holding the EMA importance of
        # each page. Updated every decode step from kernel partials WITHOUT
        # any GPU→CPU sync. Read by bounded refresh (occasional sync only).
        self._omega_gpu_per_layer: Dict[int, torch.Tensor] = {}
        self._last_bounded_refresh_step: int = 0
 
        self._original_forwards: Dict[int, object] = {}
        self._patched = False

        # --- Ablation-control flags (installed by ablation_modes / negative_controls) ---
        # If set to a callable ``hook(pipeline)``, it is invoked inside prefill()
        # AFTER allocate() has committed tiers but BEFORE _build_per_head_pages()
        # encodes K with those tiers.  This is the ONLY correct place to override
        # ``TokenState.new_tier_id`` for post-allocation ablations (quant_only,
        # decoupled, uniform_head, sphkv_angle) -- once pages are built, the
        # stored codes are locked to the tier that was active at encode time.
        self._pre_pages_hook = None
        # ``_use_recon`` is checked at decode time by _compressed_head_attention
        # to route through the reconstruct-then-dot negative control path.
        self._use_recon = False

        mode = "fused CUDA" if self.use_fused else "reference (pure-torch)"
        print(
            f"[SphericalKVPipeline] layers={self.num_layers}  "
            f"kv_heads={self.num_kv_heads}  head_dim={self.head_dim}  "
            f"groups={self.num_groups}  decode_mode={mode}"
        )
 
    def prefill(
        self,
        input_ids:      torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        retrieval_boundaries: Optional[List[Tuple[int, int]]] = None,
        # vLLM path: pass pre-captured data directly
        kv_pairs=None, attn_weights=None, head_outputs=None,
        pre_rope_K_list=None, seq_len=None,
        prefill_logits=None,                      # vLLM path: pass logits for s_hat
        skip_patch: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        retrieval_boundaries : list of (start, end) token index pairs
            Marks retrieved evidence spans for segment_id=1 (W2 workload).
            Tokens in these spans get higher protection via SEGMENT_WEIGHTS.
        prefill_logits : Optional[Tensor]
            Final-layer logits captured by the vLLM backend during prefill,
            shape [B, T, vocab]. Required if head_outputs is supplied so the
            stability proxy (s_hat) can compute the next-token margin.
        """
        if self._patched:
            self.uninstall()
        self._reset_state()
        self._retrieval_boundaries = retrieval_boundaries or []

        # 'logits' must be defined in both paths so the stability proxy
        # has something to read. In the vLLM path it comes from kwarg;
        # in the HF path capture_prefill_pass returns it.
        logits = prefill_logits

        if kv_pairs is not None:
            # vLLM path: KV pairs already captured externally
            T = seq_len or kv_pairs[0][0].shape[2]
            self.seq_len = T
            print(f"[prefill] Using pre-captured KV pairs, T={T} ...")
            attn_list = attn_weights or [None] * self.num_layers
            ho_list = head_outputs or [None] * self.num_layers
            pre_rope_K_list_local = pre_rope_K_list or [None] * self.num_layers
        else:
            input_ids = input_ids.to(self.device)
            B, T = input_ids.shape
            self.seq_len = T

            print(f"[prefill] Forward pass on {T} tokens ...")
            kv_pairs, attn_list, ho_list, logits, pre_rope_K_list_local = capture_prefill_pass(
                self.model, input_ids, attention_mask
            )

        valid_attn = [a for a in attn_list if a is not None]
        if valid_attn:
            attn_stacked = torch.stack(valid_attn, dim=0)
            if attn_stacked.dim() == 2:
                total = attn_stacked.sum(dim=1, keepdim=True).clamp(min=1e-8)
                reuse_q = attn_stacked / total
                reuse_kv = aggregate_proxy_to_kv_heads(
                    reuse_q, self.num_q_heads, self.num_kv_heads)
            else:
                reuse_kv = torch.ones(self.num_layers, self.num_kv_heads) / self.num_kv_heads
        else:
            reuse_kv = torch.ones(self.num_layers, self.num_kv_heads) / self.num_kv_heads

        ho_stacked = build_head_outputs_tensor(ho_list)
        if ho_stacked is not None and logits is not None:
            # Both inputs may live on different devices (e.g. vLLM path:
            # head_outputs are CPU after .cpu() in prefill_capture, while
            # logits stays on GPU). Force CPU for the proxy compute —
            # this is a one-off prefill cost; no decode-path impact.
            stab_kv = aggregate_proxy_to_kv_heads(
                compute_stability_proxy(
                    ho_stacked[:, :, :, -1, :].cpu(),
                    logits[:, -1, :].cpu(),
                ),
                self.num_q_heads, self.num_kv_heads,
            )
        else:
            stab_kv = torch.ones(self.num_layers, self.num_kv_heads) / self.num_kv_heads

        self.reuse     = reuse_kv.cpu()
        self.stability = stab_kv.cpu()

        del ho_stacked, attn_list, ho_list
        torch.cuda.empty_cache()
 
        print("[prefill] Building TokenStates ...")
        all_tokens: List[TokenState] = []
        K_all: Dict[Tuple[int, int], torch.Tensor] = {}
        V_all: Dict[Tuple[int, int], torch.Tensor] = {}

        for li, (K, V) in enumerate(kv_pairs):
            # Pre-RoPE K (shape [1, T, num_kv_heads * dh]) captured via hooks.
            # This is what we encode into codebook codes.  Post-RoPE K is
            # position-rotated and cannot cluster below eps_theta ~ 0.1.
            K_pre_layer = pre_rope_K_list_local[li][0].view(
                -1, self.num_kv_heads, self.head_dim
            )  # [T, H_kv, dh]
            for h in range(self.num_kv_heads):
                K_lh = K_pre_layer[:, h, :].float().cpu()    # PRE-RoPE
                V_lh = V[0, h].half().cpu()                   # V is position-invariant
                K_all[(li, h)] = K_lh
                V_all[(li, h)] = V_lh
 
                r, phi = spherical_parameterize(K_lh)
 
                G_fine   = self.num_groups        # 8
                g_fine   = self.group_size        # 16
                K_grouped = K_lh.view(T, G_fine, g_fine)
                r_groups_all = K_grouped.norm(dim=-1)   # [T, G_fine]
 
                for t in range(T):
                    # Segment assignment (Algorithm 1, line 19):
                    #   seg=0 (prefix): default
                    #   seg=1 (retrieved): marked by retrieval_boundaries
                    #   seg=2 (recent): last RECENT_WINDOW positions
                    if t >= T - RECENT_WINDOW:
                        seg = 2   # recent suffix
                    elif retrieval_boundaries:
                        seg = 0   # default prefix
                        for (rb_start, rb_end) in retrieval_boundaries:
                            if rb_start <= t < rb_end:
                                seg = 1   # retrieved evidence
                                break
                    else:
                        seg = 0   # prefix (default)
 
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
        # _cfg.GLOBAL_BUDGET_BITS = 1_000_000
        # print(f"[prefill] Budget: {_cfg.GLOBAL_BUDGET_BITS:,} bits (fixed global budget)")
        _bpt = getattr(_cfg, 'BITS_PER_TOKEN', 30.9)
        _cfg.GLOBAL_BUDGET_BITS = _bpt * T * self.num_layers * self.num_kv_heads
        print(f"[prefill] Budget: {_cfg.GLOBAL_BUDGET_BITS:,} bits "
              f"({_bpt} bpt x T={T} x L={self.num_layers} x H={self.num_kv_heads})")
        retained = allocate(all_tokens, self.tiers, self.reuse, self.stability)
        for ts in retained:
            ts.commit_tier()

        # === Diagnostic: post-allocation tier distribution ===
        from collections import Counter
        tier_counts = Counter(ts.new_tier_id for ts in retained)
        print(f"[ALLOC DIAG] retained={len(retained)} "
              f"tier_counts={dict(tier_counts)} "
              f"(tier_id 0=drop, 1=b1 High 112bits, 2=b2 Mid 96bits, 3=b3 Low 44bits)",
              flush=True)
        # === end diagnostic ===

        # Keep reference for online refresh and EMA updates
        self._retained_tokens = retained
        # -1 so the first patched_forward (layer 0) increments to 0, yielding
        # current_pos = seq_len + 0 = T for the first decode step.
        self._decode_step     = -1
        pct = 100 * len(retained) / max(len(all_tokens), 1)
        print(f"[prefill] Retained {len(retained)}/{len(all_tokens)} ({pct:.1f}%)")

        # ── Post-allocation ablation hook (installed by ablation_modes.py) ──
        # Runs AFTER allocate() and tier commits, but BEFORE key codes are
        # packed into pages.  Hooks may rewrite ``ts.new_tier_id`` to override
        # the tier allocation (e.g. pin-to-b3 for quant_only / sphkv_angle,
        # layer-modal tier for uniform_head).  Any change takes effect in:
        #   (a) _build_per_head_pages() below -- key codes encoded at new tier,
        #   (b) _ctx_token_order rebuild further down -- groups tokens by tier,
        #   (c) tier-homogeneous-pages invariant preserved.
        if self._pre_pages_hook is not None:
            try:
                self._pre_pages_hook(self)
                # Re-commit so prev_tier_id tracks the hook-applied tier.
                # Without this the first online_refresh() would treat every
                # hook-pinned token as "just changed tier" and apply a cooldown.
                for ts in self._retained_tokens:
                    ts.commit_tier()
                print(f"[prefill] Pre-pages hook applied "
                      f"({self._pre_pages_hook.__name__ if hasattr(self._pre_pages_hook, '__name__') else 'hook'})")
            except Exception as e:
                print(f"[prefill] WARNING: pre_pages_hook failed: {e}")

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

        from config import EMA_R
        self._omega_buf.clear()
        self._window_buf.clear()
        self._window_ptr.clear()
        for key, order in self._ctx_token_order.items():
            n = len(order)
            self._omega_buf[key]  = torch.ones(n)
            self._window_buf[key] = torch.zeros(EMA_R, n)
            self._window_ptr[key] = 0
 
        if not skip_patch:
            print("[prefill] Patching attention layers for decode ...")
            self._original_forwards = patch_for_decode(self.model, self)
            self._patched = True
        else:
            print("[prefill] Skipping HF patch (vLLM backend handles decode).")
            self._patched = False
        print("[prefill] Done -- model ready for compressed decode.")
 
    def _ensure_rope_tables(self, max_pos: int, device: torch.device,
                            dtype=torch.float32) -> None:
        """Build or extend RoPE cos/sin tables up to max_pos (inclusive)."""
        if self._rope_cos_table is not None and max_pos < self._rope_max_pos:
            return
        # Find a rotary embedding module on the model
        rotary = None
        if hasattr(self.model.model, "rotary_emb"):
            rotary = self.model.model.rotary_emb
        else:
            rotary = self.model.model.layers[0].self_attn.rotary_emb
 
        target_len = max_pos + 64
        pos = torch.arange(target_len, device=device).unsqueeze(0)  # [1, target_len]
        dummy = torch.zeros(1, 1, 1, self.head_dim, device=device, dtype=dtype)
        cos, sin = rotary(dummy, pos)
        # cos, sin shape: [1, target_len, dh]
        self._rope_cos_table = cos[0].to(dtype)   # [target_len, dh]
        self._rope_sin_table = sin[0].to(dtype)   # [target_len, dh]
        self._rope_max_pos = target_len
 
    def _decode_layer_lut(
        self,
        layer_idx: int,
        q_post:    torch.Tensor,
        k_post:    torch.Tensor,
        v:         torch.Tensor,
    ) -> torch.Tensor:
        from sphkv_lut import LayerPool, make_tier_idx_map

        if not hasattr(self, "_lut_pools"):
            self._lut_pools = {}
            self._lut_G_max = max(t.G for t in self.tiers
                                  if getattr(t, "tier_id", 0) != 0)
            self._lut_cb_size_max = max(2 ** t.b_theta for t in self.tiers
                                        if getattr(t, "tier_id", 0) != 0)
            self._lut_tier_idx_map = make_tier_idx_map(self.tiers)

        if layer_idx not in self._lut_pools:
            num_q  = q_post.shape[0]
            num_kv = k_post.shape[0]
            self._lut_pools[layer_idx] = LayerPool(
                pipeline=self, layer_idx=layer_idx,
                page_size=PAGE_SIZE, G_max=self._lut_G_max,
                cb_size_max=self._lut_cb_size_max,
                decode_capacity=num_kv * 64,
                device=q_post.device, tiers=self.tiers,
                tier_idx_map=self._lut_tier_idx_map,
                num_q=num_q, num_kv=num_kv,
                kv_groups=num_q // num_kv, dh=q_post.shape[1],
            )

        sm_scale = 1.0 / math.sqrt(q_post.shape[1])
        pool = self._lut_pools[layer_idx]
        out = pool.append_and_compute(
            q_post, k_post, v, self.tiers, sm_scale)

        # ── Decode-time controller (paper §C.2 EMA + Algorithm 1 line 94-97) ──
        # 1. EMA omega update — every step, GPU-resident, no host sync.
        #    Page-level granularity is paper-allowed under "kernel-side
        #    block statistics" (§C.2). Cost: ~5 small GPU ops, fully async.
        # 2. Bounded refresh — every C steps, only re-tiers decode tokens
        #    that have aged out of the recent window since last refresh
        #    (Algorithm 1 line 94-97: "new items only; do not rewrite old
        #    pages"). For typical short generations this is a no-op.
        if self._retained_tokens:
            self._update_omega_gpu(layer_idx=layer_idx, pool=pool)

            # Bounded refresh fires after the LAST layer in the step has
            # finished updating omega so the snapshot is coherent.
            if (layer_idx == self.num_layers - 1
                    and self._decode_step > 0
                    and (self._decode_step % REFRESH_CADENCE == 0)):
                self._bounded_refresh()

        return out.to(q_post.dtype)


    def _update_omega_gpu(self, layer_idx: int, pool) -> None:
        """
        EMA importance-weight update (paper §C.2).

        Reads (m, l) per (q_head, page) from the kernel's _partial_scratch
        WITHOUT copying to CPU. Computes per-page normalized attention mass
        via online-softmax merge across pages, aggregates q_heads -> kv_heads
        (mean over GQA group), and EMA-blends into the GPU-resident omega.

        Paper-faithful page-level approximation under §C.2's clause
        "ω_i can be approximated by kernel-side block statistics".

        The actual math runs in `_omega_ema_compiled` (Inductor-fused
        Triton kernels) — typically 2-3 launches instead of ~12.
        """
        # Lazy init the per-layer omega buffer
        if layer_idx not in self._omega_gpu_per_layer:
            self._omega_gpu_per_layer[layer_idx] = torch.ones(
                pool.num_kv, pool.max_blocks,
                device=pool.device, dtype=torch.float32)

        # _partial_scratch: [num_q, max_blocks, dh+2]; channels 0/1 are (m, l)
        ml  = pool._partial_scratch[:, :, :2]
        m_q = ml[..., 0]
        l_q = ml[..., 1]

        # Out-of-place compute returns the new omega; assign back.
        self._omega_gpu_per_layer[layer_idx] = _omega_ema_compiled(
            m_q, l_q,
            self._omega_gpu_per_layer[layer_idx],
            pool.num_kv, pool.kv_groups, pool.max_blocks,
            EMA_BETA,
        )


    def _bounded_refresh(self) -> None:
        """
        Bounded refresh (Algorithm 1 line 94-97 / page 15).

        Per paper:
          - Recompute controller scalars for NEW items only
          - Do not rewrite old pages
          - Update only future allocations and a small protect bitmap
          - Amortized overhead: O(ΔN · |T| / C) per token

        For our deployment, "new items" since the last refresh = decode
        tokens generated in [last_refresh, current_step]. These tokens
        are appended at b1 (recent-window protection). They become
        controllable ONLY when they age out of the recent window.

        With RECENT_WINDOW=256 and short generation, no token ages out
        and this path is essentially a metadata bookkeeping step. For
        long generations (> RECENT_WINDOW + REFRESH_CADENCE tokens),
        we re-tier the aged-out tokens using the current omega.

        Implementation note: rather than triggering a full rebuild for
        aged-out tokens (which would cost O(M)), we update the GPU-side
        bits_table (which controls what tier the kernel reads each page
        as) only for pages that have entirely aged out. Per-token codes
        for those pages are pulled from self._codes_all (pre-encoded at
        prefill for every tier). This matches the paper's "small protect
        bitmap" overhead model.
        """
        # Δ = decode steps since last refresh
        last = self._last_bounded_refresh_step
        self._last_bounded_refresh_step = self._decode_step

        # Position-space: tokens at index >= seq_len + decode_step - RECENT_WINDOW
        # are still in the recent window. Anything below that has aged out.
        oldest_recent = self.seq_len + self._decode_step - RECENT_WINDOW
        # For decode tokens (positions in [seq_len, seq_len + decode_step]),
        # the ones aged out are at positions [seq_len, oldest_recent - 1].
        # If oldest_recent <= seq_len, no decode token has aged out.
        if oldest_recent <= self.seq_len:
            return  # no work — typical for short generations

        # Long generation path: there are decode tokens that have aged out.
        # We will sync omega for these positions and re-tier their pages.
        # This is a rare path (only fires for very long generations) so
        # the one-time GPU→CPU sync of omega is acceptable.
        # NOTE: implementing the full long-gen re-tier requires touching
        # bits_table_gpu and is non-trivial. For now we log and skip;
        # the paper's "do not rewrite old pages" clause means correctness
        # is preserved (those pages just stay at b1 longer than ideal).
        # This is a paper-allowed conservative behavior.
        return


    def _decode_layer_all_heads(
        self,
        layer_idx: int,
        Q:         torch.Tensor,
        K_pre:     torch.Tensor,
        V:         torch.Tensor,
        current_pos: int,
        grp:       int,
    ) -> torch.Tensor:
        device = Q.device
        num_kv = K_pre.shape[0]
        num_q  = Q.shape[0]
        dh     = Q.shape[1]

        self._ensure_rope_tables(
            max_pos=max(current_pos,
                        self.seq_len + getattr(self, '_decode_step', 0) + 8),
            device=device, dtype=torch.float32)
        cos_tbl = self._rope_cos_table.to(device)
        sin_tbl = self._rope_sin_table.to(device)

        _bt_to_tid = {t.b_theta: t.tier_id for t in self.tiers if t.tier_id != 0}
        b3_tier = self.tiers[3]

        pos_new = torch.tensor([current_pos], device=device, dtype=torch.long)
        cos_new = cos_tbl[pos_new]
        sin_new = sin_tbl[pos_new]
        from config import EMA_BETA, EMA_R

        tier_groups = {}
        for kv_h in range(num_kv):
            key = (layer_idx, kv_h)
            for ph_tuple in self.per_head_pages.get(key, []):
                pt, ptt, b_theta, n_tokens, V_tier, K_tier = ph_tuple[:6]
                positions = ph_tuple[6] if len(ph_tuple) > 6 else None
                tier_id = _bt_to_tid.get(b_theta, 1)
                tier_groups.setdefault(tier_id, []).append(
                    (kv_h, pt, ptt, b_theta, n_tokens, V_tier, K_tier, positions))

        q_all = Q.float()

        head_ctx_logits = {h: [] for h in range(num_kv)}
        head_ctx_V      = {h: [] for h in range(num_kv)}
        head_page_offsets = {}

        for tier_id, entries in tier_groups.items():
            tier_obj = self.tiers[tier_id]
            tier_G = tier_obj.G
            tier_g = tier_obj.g
            bt = tier_obj.b_theta

            all_pages = []
            all_ptable = []
            all_positions = []
            all_head_ids = []
            page_owner = []
            cum_page_offset = 0
            cum_page_count  = 0

            for (kv_h, pt, ptt, b_theta, n_tokens, V_tier, K_tier, positions) in entries:
                n_pages = ptt.shape[0]
                adj_ptt = ptt.clone()
                adj_ptt[:, 0] += cum_page_offset
                adj_ptt[:, 1] += cum_page_offset
                adj_ptt[:, 2] += cum_page_offset

                remainder = pt.numel() % 4
                if remainder != 0:
                    pt_padded = torch.cat([pt, torch.zeros(
                        4 - remainder, dtype=torch.uint8, device=pt.device)])
                else:
                    pt_padded = pt
                all_pages.append(pt_padded)
                all_ptable.append(adj_ptt)

                if positions is not None:
                    pos_i32 = positions.to(torch.int32)
                    total_slots = n_pages * PAGE_SIZE
                    if pos_i32.numel() < total_slots:
                        pad = torch.full((total_slots - pos_i32.numel(),), -1,
                                         dtype=torch.int32, device=device)
                        pos_i32 = torch.cat([pos_i32, pad])
                    all_positions.append(pos_i32[:total_slots])
                else:
                    all_positions.append(torch.full((n_pages * PAGE_SIZE,), -1,
                                                    dtype=torch.int32, device=device))

                all_head_ids.append(torch.full((n_pages,), kv_h,
                                               dtype=torch.int32, device=device))
                page_owner.append((kv_h, cum_page_count, n_pages, n_tokens, V_tier))
                cum_page_offset += pt_padded.numel()
                cum_page_count  += n_pages

            if not all_pages:
                continue

            cat_pages = torch.cat(all_pages + [
                torch.zeros(4, dtype=torch.uint8, device=device)])
            cat_ptable = torch.cat(all_ptable)
            cat_positions = torch.cat(all_positions)
            cat_head_ids = torch.cat(all_head_ids)

            cb_list = []
            for kv_h_cb in range(num_kv):
                cb = get_codebook(self.codebooks, layer_idx, kv_h_cb, tier_id)
                cb_list.append(cb.to(device).float())
            cat_codebooks = torch.cat(cb_list, dim=0)

            from fused_decode import fused_decode_ragged
            raw_logits = fused_decode_ragged(
                cat_pages, cat_ptable, cat_head_ids, cat_positions,
                cos_tbl, sin_tbl, q_all, grp, num_kv, cat_codebooks,
                dh, tier_G, tier_g, bt, PAGE_SIZE,
            )

            for (kv_h, pg_start, n_pg, n_tok, V_tier) in page_owner:
                slot_start = pg_start * PAGE_SIZE
                slot_end   = slot_start + n_tok
                extracted = raw_logits[:, slot_start:slot_end]
                if extracted.shape[1] != V_tier.shape[0]:
                    print(f"[RAGGED MISMATCH] tier={tier_id} head={kv_h} "
                          f"pg_start={pg_start} n_pg={n_pg} n_tok={n_tok} "
                          f"slot_start={slot_start} slot_end={slot_end} "
                          f"extracted={extracted.shape} V_tier={V_tier.shape} "
                          f"raw_logits={raw_logits.shape}",
                          flush=True)
                head_ctx_logits[kv_h].append(extracted)
                head_ctx_V[kv_h].append(V_tier)
                if kv_h == 0: print(f"  [TRACE-ragged t={tier_id}] L={extracted.shape} V={V_tier.shape}", flush=True)

        for kv_h in range(num_kv):
            key = (layer_idx, kv_h)
            for dp_tuple in self._decode_pages.get(key, []):
                dp, dpt, d_btheta, d_ntok, d_V = dp_tuple[:5]
                d_K_codes = dp_tuple[5] if len(dp_tuple) > 5 else None
                d_positions = dp_tuple[6] if len(dp_tuple) > 6 else None
                cb_dec = get_codebook(self.codebooks, layer_idx, kv_h, b3_tier.tier_id)
                q_grp = q_all[kv_h * grp:(kv_h + 1) * grp]
                if self.use_fused and d_positions is not None:
                    n_pg = dpt.shape[0]
                    total_slots = n_pg * PAGE_SIZE
                    pos_i32 = d_positions.to(torch.int32)
                    if pos_i32.numel() < total_slots:
                        pad = torch.full((total_slots - pos_i32.numel(),), -1,
                                         dtype=torch.int32, device=device)
                        pos_i32 = torch.cat([pos_i32, pad])
                    hid = torch.full((n_pg,), kv_h, dtype=torch.int32, device=device)
                    cb_one = cb_dec.to(device).float().unsqueeze(0).expand(num_kv, -1, -1, -1)
                    cb_one = cb_one.reshape(num_kv * b3_tier.G, cb_one.shape[2], b3_tier.g)
                    from fused_decode import fused_decode_ragged
                    raw = fused_decode_ragged(
                        dp, dpt, hid, pos_i32[:total_slots],
                        cos_tbl, sin_tbl, q_all, grp, num_kv, cb_one,
                        dh, b3_tier.G, b3_tier.g, b3_tier.b_theta, PAGE_SIZE)
                    head_ctx_logits[kv_h].append(raw[:, :d_ntok])
                    head_ctx_V[kv_h].append(d_V)
                    if kv_h == 0: print(f"  [TRACE-dp] V={d_V.shape}", flush=True)
                    continue
                elif d_K_codes is not None and d_positions is not None:
                    r_d, th_d = d_K_codes
                    r_d = r_d.to(device)
                    th_d = th_d.to(device)
                    pos_d = d_positions.to(device).long()
                    cos_i = cos_tbl[pos_d]
                    sin_i = sin_tbl[pos_d]
                    q_exp = q_grp.unsqueeze(1)
                    q_h = _rotate_half(q_exp)
                    q_rot = q_exp * cos_i.unsqueeze(0) - q_h * sin_i.unsqueeze(0)
                    q_rot_g = q_rot.view(grp, d_ntok, b3_tier.G, b3_tier.g)
                    q_norms = q_rot_g.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                    q_normed = q_rot_g / q_norms
                    cw_d = cb_dec.to(device)[
                        torch.arange(b3_tier.G, device=device).unsqueeze(1),
                        th_d.long().T]
                    cw_dt = cw_d.permute(1, 0, 2)
                    cos_pg_d = (q_normed * cw_dt.unsqueeze(0)).sum(-1).clamp(-1.0, 1.0)
                    logits_d = (q_norms.squeeze(-1) * cos_pg_d
                                * r_d.unsqueeze(0)).sum(-1) / math.sqrt(dh)
                    head_ctx_logits[kv_h].append(logits_d[:, :d_ntok])
                else:
                    head_ctx_logits[kv_h].append(
                        torch.zeros(grp, d_ntok, device=device))
                head_ctx_V[kv_h].append(d_V)
                if kv_h == 0: print(f"  [TRACE-dp] V={d_V.shape}", flush=True)

        head_outs = [None] * num_q

        for kv_h in range(num_kv):
            key = (layer_idx, kv_h)
            q_grp = q_all[kv_h * grp:(kv_h + 1) * grp]
            kn = K_pre[kv_h].float()
            vn = V[kv_h].float()

            stg_r = self.stg_r.get(key)
            stg_theta = self.stg_theta.get(key)
            stg_V = self.stg_V.get(key)
            stg_pos = self.stg_positions.get(key)
            if stg_r is not None and stg_r.shape[0] > 0:
                N_stg = stg_r.shape[0]
                cb_b3 = get_codebook(self.codebooks, layer_idx, kv_h, b3_tier.tier_id)
                if stg_pos is not None and stg_pos.shape[0] >= N_stg:
                    pos_s = stg_pos[:N_stg].to(device).long()
                    cos_i = cos_tbl[pos_s]
                    sin_i = sin_tbl[pos_s]
                    q_exp = q_grp.unsqueeze(1)
                    q_h = _rotate_half(q_exp)
                    q_rot_s = q_exp * cos_i.unsqueeze(0) - q_h * sin_i.unsqueeze(0)
                else:
                    q_rot_s = q_grp.unsqueeze(1).expand(-1, N_stg, -1)
                q_rot_g = q_rot_s.view(grp, N_stg, b3_tier.G, b3_tier.g)
                q_norms = q_rot_g.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                q_normed = q_rot_g / q_norms
                cw = cb_b3.to(device)[
                    torch.arange(b3_tier.G, device=device).unsqueeze(1),
                    stg_theta.long().T]
                cw_t = cw.permute(1, 0, 2)
                cos_pg = (q_normed * cw_t.unsqueeze(0)).sum(-1).clamp(-1.0, 1.0)
                stg_logits = (q_norms.squeeze(-1) * cos_pg
                              * stg_r.unsqueeze(0)).sum(-1) / math.sqrt(dh)
                head_ctx_logits[kv_h].append(stg_logits)
                head_ctx_V[kv_h].append(stg_V)
                if kv_h == 0: print(f"  [TRACE-stg] V={stg_V.shape}", flush=True)

            if head_ctx_logits[kv_h]:
                n_logit_parts = len(head_ctx_logits[kv_h])
                n_V_parts = len(head_ctx_V[kv_h])
                if n_logit_parts != n_V_parts:
                    mn = min(n_logit_parts, n_V_parts)
                    print(f"[WARN] head={kv_h} logit_parts={n_logit_parts} "
                          f"V_parts={n_V_parts}, truncating to {mn}", flush=True)
                    for idx in range(n_logit_parts):
                        print(f"  logit[{idx}]={head_ctx_logits[kv_h][idx].shape}", flush=True)
                    for idx in range(n_V_parts):
                        print(f"  V[{idx}]={head_ctx_V[kv_h][idx].shape}", flush=True)
                    head_ctx_logits[kv_h] = head_ctx_logits[kv_h][:mn]
                    head_ctx_V[kv_h] = head_ctx_V[kv_h][:mn]
                ctx_logits = torch.cat(head_ctx_logits[kv_h], dim=1)
                V_ctx = torch.cat(head_ctx_V[kv_h], dim=0)
            else:
                ctx_logits = torch.zeros(grp, 0, device=device)
                V_ctx = torch.zeros(0, dh, device=device)

            cb_b3_new = get_codebook(self.codebooks, layer_idx, kv_h, b3_tier.tier_id)
            r_kn, th_kn = _encode_keys_with_codebooks(
                kn.unsqueeze(0), cb_b3_new, b3_tier.g, b3_tier.G)
            r_kn = r_kn[0].to(device)
            th_kn = th_kn[0].to(device).long()
            cw_kn = cb_b3_new.to(device)[
                torch.arange(b3_tier.G, device=device), th_kn]
            q_exp = q_grp.unsqueeze(1)
            q_h = _rotate_half(q_exp)
            q_at_new = (q_exp * cos_new.unsqueeze(0)
                        - q_h * sin_new.unsqueeze(0)).squeeze(1)
            q_gn = q_at_new.view(grp, b3_tier.G, b3_tier.g)
            q_nn = q_gn.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            q_nd = q_gn / q_nn
            cos_n = (q_nd * cw_kn.unsqueeze(0)).sum(-1).clamp(-1.0, 1.0)
            new_logits = (q_nn.squeeze(-1) * r_kn.unsqueeze(0)
                          * cos_n).sum(-1) / math.sqrt(dh)

            all_logits = torch.cat([ctx_logits, new_logits.unsqueeze(1)], dim=1)
            attn = torch.softmax(all_logits, dim=1)

            n_ctx = ctx_logits.shape[1]
            if n_ctx > 0:
                key_ema = key
                attn_vals = attn[:, :n_ctx].mean(dim=0).detach().cpu()
                wbuf = self._window_buf.get(key_ema)
                if wbuf is not None and wbuf.shape[1] >= n_ctx:
                    ptr = self._window_ptr[key_ema]
                    wbuf[ptr, :n_ctx] = attn_vals[:n_ctx]
                    self._window_ptr[key_ema] = (ptr + 1) % EMA_R
                    w_max = wbuf[:, :n_ctx].max(dim=0).values
                    omega = self._omega_buf[key_ema]
                    omega[:n_ctx] = EMA_BETA * omega[:n_ctx] + (1.0 - EMA_BETA) * w_max

            V_full = torch.cat([V_ctx, vn.unsqueeze(0)], dim=0)
            if all_logits.shape[1] != V_full.shape[0]:
                print(f"[SHAPE MISMATCH] head={kv_h} "
                      f"all_logits={all_logits.shape} V_full={V_full.shape}\n"
                      f"  ctx_logits={ctx_logits.shape} V_ctx={V_ctx.shape}\n"
                      f"  n_logit_parts={len(head_ctx_logits[kv_h])} "
                      f"n_V_parts={len(head_ctx_V[kv_h])}",
                      flush=True)
                for idx, lp in enumerate(head_ctx_logits[kv_h]):
                    print(f"  logit_part[{idx}]: {lp.shape}", flush=True)
                for idx, vp in enumerate(head_ctx_V[kv_h]):
                    print(f"  V_part[{idx}]: {vp.shape}", flush=True)
            attn_out = attn @ V_full

            for qi in range(grp):
                head_outs[kv_h * grp + qi] = attn_out[qi].to(Q.dtype)

            self._append_decode_kv(layer_idx, kv_h, kn,
                                   vn.squeeze(0) if vn.dim() > 1 else vn,
                                   position=current_pos)

        return torch.stack(head_outs, dim=0)


    def _compressed_head_attention_batched(
        self,
        layer_idx: int,
        kv_head:   int,
        q_batch:   torch.Tensor,   # [num_q, dh]  POST-RoPE at current_pos
        k_new:     torch.Tensor,   # [dh]         PRE-RoPE
        v_new:     torch.Tensor,   # [dh]
        current_pos: int = 0,
    ) -> torch.Tensor:
        # A1/A2-C: reconstruct-then-dot path (negative control)
        if getattr(self, '_use_recon', False):
            from negative_controls import recon_attention_batched
            return recon_attention_batched(
                self, layer_idx, kv_head, q_batch, k_new, v_new)

        key    = (layer_idx, kv_head)
        device = q_batch.device
        num_q  = q_batch.shape[0]
        q      = q_batch.float()    # [num_q, dh]  POST-RoPE at current_pos
        kn     = k_new.float().to(device)           # PRE-RoPE new key
        vn     = v_new.float().to(device)
        dh     = q.shape[1]

        # Ensure RoPE cos/sin tables are ready for back-rotation.
        self._ensure_rope_tables(
            max_pos=max(current_pos,
                        self.seq_len + getattr(self, '_decode_step', 0) + 8),
            device=device, dtype=torch.float32,
        )
        cos_tbl = self._rope_cos_table.to(device)   # [max_pos, dh]
        sin_tbl = self._rope_sin_table.to(device)   # [max_pos, dh]
 
        with nvtx_range("page_lookup"):
            ph = self.per_head_pages.get(key, [])
            # Build b_theta → tier_id map once per call
            _bt_to_tid = {t.b_theta: t.tier_id
                          for t in self.tiers if t.tier_id != 0}
 
        # ctx_logits_parts : each [num_q, n_tokens_for_this_tier]
        ctx_logits_parts: List[torch.Tensor] = []
        V_ctx_parts:      List[torch.Tensor] = []
 
        with nvtx_range("angle_logits"):
            for ph_tuple in ph:
                # Pages now have 7 slots: (pt, ptt, b_theta, n_tokens,
                #                          V_tier, K_tier, positions)
                pt, ptt, b_theta, n_tokens, V_tier, K_tier = ph_tuple[:6]
                positions = ph_tuple[6] if len(ph_tuple) > 6 else None

                # Resolve per-tier codebook and geometry
                tier_id  = _bt_to_tid.get(b_theta, 1)
                cb_lh    = get_codebook(self.codebooks, layer_idx, kv_head, tier_id)
                tier_obj = self.tiers[tier_id]
                tier_G   = tier_obj.G
                tier_g   = tier_obj.g

                # ───── Fused CUDA kernel path (paper-compliant ADA) ─────
                # The kernel performs per-token Q back-rotation internally
                # using the positions array and cos/sin tables.  This is
                # equivalent to the torch path below but ~10x faster for
                # large contexts.
                if self.use_fused and positions is not None:
                    num_pages_tier = ptt.shape[0]
                    total_slots    = num_pages_tier * PAGE_SIZE
                    pos_i32        = positions.to(torch.int32)
                    if pos_i32.numel() < total_slots:
                        pad = torch.full(
                            (total_slots - pos_i32.numel(),), -1,
                            dtype=torch.int32, device=device,
                        )
                        pos_i32 = torch.cat([pos_i32, pad])
                    raw = _call_fused(
                        pt, ptt, pos_i32, cos_tbl, sin_tbl,
                        q, cb_lh, b_theta, dh, tier_G, tier_g,
                    )   # [num_q, num_pages * page_size]
                    ctx_logits_parts.append(raw[:, :n_tokens])
                    V_ctx_parts.append(V_tier)
                    continue

                # ───── Reference torch path ─────
                r_codes, th_codes = K_tier          # [N, G] CPU
                r_codes  = r_codes.to(device)       # [N, tier_G]
                th_codes = th_codes.to(device)      # [N, tier_G]

                # Build back-rotated Q per context token.
                # positions: [n_tokens] long tensor of absolute token positions.
                if positions is not None:
                    positions = positions.to(device).long()
                    cos_i = cos_tbl[positions]      # [n_tokens, dh]
                    sin_i = sin_tbl[positions]      # [n_tokens, dh]
                    # Back-rotate by -t_i: q_rot = q*cos_i - rotate_half(q)*sin_i
                    q_expanded = q.unsqueeze(1)                         # [num_q, 1, dh]
                    q_half     = _rotate_half(q_expanded)               # [num_q, 1, dh]
                    q_rot = q_expanded * cos_i.unsqueeze(0) \
                          - q_half     * sin_i.unsqueeze(0)              # [num_q, n_tokens, dh]
                else:
                    # Legacy pages without positions: no rotation (will give
                    # wrong answers but keeps the code from crashing)
                    q_rot = q.unsqueeze(1).expand(-1, n_tokens, -1)

                # Per-group normalize q_rot: [num_q, n_tokens, G, g]
                q_rot_g  = q_rot.view(num_q, n_tokens, tier_G, tier_g)
                q_norms  = q_rot_g.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                q_normed = q_rot_g / q_norms                            # [num_q, n_tok, G, g]

                # Gather codewords per stored token: [tier_G, n_tokens, tier_g]
                cw = cb_lh[
                    torch.arange(tier_G, device=device).unsqueeze(1),
                    th_codes.long().T,
                ]
                cw_t = cw.permute(1, 0, 2)                               # [n_tok, G, g]

                # Cosine per (q, token, g): [num_q, n_tokens, G]
                cos_pg = (q_normed * cw_t.unsqueeze(0)).sum(-1).clamp(-1.0, 1.0)

                # Logit per (q, token): Σ_g ||q_g|| · r_g · cos(q_g, cw_g) / √d
                # The ||q_g|| factor (q_norms) recovers the true dot product;
                # omitting it normalizes Q per-group and destroys magnitudes.
                logits_tier = (q_norms.squeeze(-1) * cos_pg
                               * r_codes.unsqueeze(0)).sum(-1) / math.sqrt(dh)
                ctx_logits_parts.append(logits_tier[:, :n_tokens])       # [num_q, n_tokens]

                V_ctx_parts.append(V_tier)  # [n_tokens, dh]
 
            for dp_tuple in self._decode_pages.get(key, []):
                # _decode_pages entries now have 7 slots:
                # (dp, dpt, d_btheta, d_ntok, d_V, d_K, d_positions)
                dp, dpt, d_btheta, d_ntok, d_V = dp_tuple[:5]
                d_K_codes = dp_tuple[5] if len(dp_tuple) > 5 else None
                d_positions = dp_tuple[6] if len(dp_tuple) > 6 else None

                b3_dec   = self.tiers[3]
                dec_G    = b3_dec.G
                dec_g    = b3_dec.g
                cb_dec   = get_codebook(self.codebooks, layer_idx, kv_head, b3_dec.tier_id)

                # ───── Fused kernel path ─────
                if self.use_fused and d_positions is not None:
                    num_pages_dec = dpt.shape[0]
                    total_slots   = num_pages_dec * PAGE_SIZE
                    pos_i32       = d_positions.to(torch.int32)
                    if pos_i32.numel() < total_slots:
                        pad = torch.full(
                            (total_slots - pos_i32.numel(),), -1,
                            dtype=torch.int32, device=device,
                        )
                        pos_i32 = torch.cat([pos_i32, pad])
                    raw = _call_fused(
                        dp, dpt, pos_i32, cos_tbl, sin_tbl,
                        q, cb_dec, d_btheta, dh, dec_G, dec_g,
                    )
                    ctx_logits_parts.append(raw[:, :d_ntok])
                    V_ctx_parts.append(d_V)
                    continue

                # ───── Reference torch path ─────
                if d_K_codes is not None and d_positions is not None:
                    r_codes_d, th_codes_d = d_K_codes
                    r_codes_d  = r_codes_d.to(device)     # [N, dec_G]
                    th_codes_d = th_codes_d.to(device)    # [N, dec_G]
                    positions_d = d_positions.to(device).long()

                    cos_i = cos_tbl[positions_d]           # [N, dh]
                    sin_i = sin_tbl[positions_d]
                    q_expanded = q.unsqueeze(1)            # [num_q, 1, dh]
                    q_half     = _rotate_half(q_expanded)
                    q_rot = q_expanded * cos_i.unsqueeze(0) - q_half * sin_i.unsqueeze(0)

                    q_rot_g  = q_rot.view(num_q, d_ntok, dec_G, dec_g)
                    q_norms  = q_rot_g.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                    q_normed = q_rot_g / q_norms

                    cw_d = cb_dec[
                        torch.arange(dec_G, device=device).unsqueeze(1),
                        th_codes_d.long().T,
                    ]
                    cw_dt = cw_d.permute(1, 0, 2)          # [N, dec_G, dec_g]

                    cos_pg_d = (q_normed * cw_dt.unsqueeze(0)).sum(-1).clamp(-1.0, 1.0)
                    logits_d = (q_norms.squeeze(-1) * cos_pg_d
                                * r_codes_d.unsqueeze(0)).sum(-1) / math.sqrt(dh)
                    ctx_logits_parts.append(logits_d[:, :d_ntok])
                else:
                    # Legacy decode pages without positions — can't back-rotate;
                    # fall back to zeros to avoid contaminating the softmax.
                    ctx_logits_parts.append(
                        torch.zeros(num_q, d_ntok, device=device, dtype=torch.float32)
                    )

                V_ctx_parts.append(d_V)
 
            stg_r     = self.stg_r.get(key)
            stg_theta = self.stg_theta.get(key)
            stg_V     = self.stg_V.get(key)
            stg_pos   = self.stg_positions.get(key)
            if stg_r is not None and stg_r.shape[0] > 0:
                N_stg     = stg_r.shape[0]
                b3        = self.tiers[3]
                stg_G     = b3.G
                stg_g     = b3.g
                cb_b3     = get_codebook(self.codebooks, layer_idx, kv_head, b3.tier_id)

                # Build per-stg-token back-rotated Q
                if stg_pos is not None and stg_pos.shape[0] >= N_stg:
                    positions_s = stg_pos[:N_stg].to(device).long()
                    cos_i = cos_tbl[positions_s]       # [N_stg, dh]
                    sin_i = sin_tbl[positions_s]
                    q_expanded = q.unsqueeze(1)         # [num_q, 1, dh]
                    q_half     = _rotate_half(q_expanded)
                    q_rot_s = q_expanded * cos_i.unsqueeze(0) - q_half * sin_i.unsqueeze(0)
                else:
                    q_rot_s = q.unsqueeze(1).expand(-1, N_stg, -1)

                q_rot_g  = q_rot_s.view(num_q, N_stg, stg_G, stg_g)
                q_norms  = q_rot_g.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                q_normed = q_rot_g / q_norms

                cw = cb_b3[
                    torch.arange(stg_G, device=device).unsqueeze(1),
                    stg_theta.long().T,
                ]
                cw_t = cw.permute(1, 0, 2)              # [N_stg, G, g]

                cos_pg = (q_normed * cw_t.unsqueeze(0)).sum(-1).clamp(-1.0, 1.0)
                stg_logits = (q_norms.squeeze(-1) * cos_pg
                              * stg_r.unsqueeze(0)).sum(-1) / math.sqrt(dh)

                ctx_logits_parts.append(stg_logits)     # [num_q, N_stg]
                V_ctx_parts.append(stg_V)
 
        if ctx_logits_parts:
            ctx_logits = torch.cat(ctx_logits_parts, dim=1)  # [num_q, total_ctx]
            V_ctx      = torch.cat(V_ctx_parts, dim=0)       # [total_ctx, dh]
        else:
            ctx_logits = torch.zeros(num_q, 0, device=device)
            V_ctx      = torch.zeros(0, dh, device=device)
 
        # New token (at position current_pos) — kn is PRE-RoPE.
        # Paper-consistent path: encode kn through the SAME codebook used for
        # storage (tier b3, since new decode tokens are classified as recent),
        # then apply the angle-domain formula with the codeword direction.
        # This keeps new_logits and ctx_logits on the same quantization scale.
        #
        # Math: q_post · k_post(t_q)  ==  (R_{-t_q} q_post) · k_pre
        # With k_pre -> codeword (r_code · codebook_dir), we get
        #   new_logit = (1/√d) · Σ_g r_code_g · cos( R_{-t_q}(q)_g , cw_g )

        b3_tier   = self.tiers[3]
        tier_G_n  = b3_tier.G
        tier_g_n  = b3_tier.g
        cb_b3_new = get_codebook(self.codebooks, layer_idx, kv_head, b3_tier.tier_id)

        # 1. Encode kn through the b3 codebook (same path as stored K)
        #    kn shape [dh] -> [1, dh] -> encoder
        r_kn_codes, th_kn_codes = _encode_keys_with_codebooks(
            kn.unsqueeze(0), cb_b3_new, tier_g_n, tier_G_n
        )   # r: [1, tier_G_n],  theta: [1, tier_G_n] int32
        r_kn_codes  = r_kn_codes[0].to(device)                        # [tier_G_n]
        th_kn_codes = th_kn_codes[0].to(device).long()                # [tier_G_n]

        # 2. Gather codeword directions for this new token: [tier_G_n, tier_g_n]
        cw_kn = cb_b3_new[
            torch.arange(tier_G_n, device=device),
            th_kn_codes,
        ]   # [tier_G_n, tier_g_n]  unit-norm

        # 3. Back-rotate q by current_pos (single-token, same formula as ctx)
        pos_new = torch.tensor([current_pos], device=device, dtype=torch.long)
        cos_new = cos_tbl[pos_new]                                   # [1, dh]
        sin_new = sin_tbl[pos_new]                                   # [1, dh]
        q_expanded = q.unsqueeze(1)                                  # [num_q, 1, dh]
        q_half     = _rotate_half(q_expanded)                        # [num_q, 1, dh]
        q_at_new   = (q_expanded * cos_new.unsqueeze(0)
                      - q_half   * sin_new.unsqueeze(0)).squeeze(1)   # [num_q, dh]

        # 4. Per-group normalize q_at_new using b3 geometry
        q_groups_n = q_at_new.view(num_q, tier_G_n, tier_g_n)        # [num_q, G, g]
        q_norms_n  = q_groups_n.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        q_normed_n = q_groups_n / q_norms_n                          # [num_q, G, g]

        # 5. Per-group cosine with codeword, clipped (B.2)
        cos_pg_n   = (q_normed_n * cw_kn.unsqueeze(0)).sum(-1).clamp(-1.0, 1.0)

        # 6. Logit:  Σ_g ||q_g|| · r_g · cos(q_g, cw_g) / √d
        new_logits = (q_norms_n.squeeze(-1) * r_kn_codes.unsqueeze(0)
                      * cos_pg_n).sum(-1) / math.sqrt(dh)

        # full logits: [num_q, total_ctx + 1]
        all_logits = torch.cat([ctx_logits, new_logits.unsqueeze(1)], dim=1)
 
        with nvtx_range("softmax"):
            attn = torch.softmax(all_logits, dim=1)   # [num_q, total_ctx + 1]
 
        n_ctx = ctx_logits.shape[1] if ctx_logits_parts else 0
        if n_ctx > 0:
            from config import EMA_BETA, EMA_R
            key_ema = (layer_idx, kv_head)
            attn_vals = attn[:, :n_ctx].mean(dim=0).detach().cpu()
            wbuf = self._window_buf.get(key_ema)
            if wbuf is not None and wbuf.shape[1] >= n_ctx:
                ptr = self._window_ptr[key_ema]
                wbuf[ptr, :n_ctx] = attn_vals[:n_ctx]
                self._window_ptr[key_ema] = (ptr + 1) % EMA_R
                w_max = wbuf[:, :n_ctx].max(dim=0).values
                omega = self._omega_buf[key_ema]
                omega[:n_ctx] = EMA_BETA * omega[:n_ctx] + (1.0 - EMA_BETA) * w_max
 
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
        current_pos: int = 0,
    ) -> torch.Tensor:             # [dh]
        """Single-q wrapper used by evaluate.py."""
        out = self._compressed_head_attention_batched(
            layer_idx, kv_head,
            q_vec.unsqueeze(0),   # [1, dh]
            k_new, v_new,
            current_pos=current_pos,
        )
        return out.squeeze(0)     # [dh]
 
    def _append_decode_kv(
        self,
        layer_idx: int,
        kv_head:   int,
        k_new:     torch.Tensor,   # [dh]   PRE-RoPE
        v_new:     torch.Tensor,   # [dh]
        position:  Optional[int] = None,
    ) -> None:
        key    = (layer_idx, kv_head)
        device = self.device
        kn     = k_new.float().to(device).unsqueeze(0)   # [1, dh]\r
        vn     = v_new.float().to(device).unsqueeze(0)   # [1, dh]
 
        G_fine = self.num_groups
        g_fine = self.group_size
        k_grouped     = kn.view(G_fine, g_fine)           # [G_fine, g_fine]
        r_groups_fine = k_grouped.norm(dim=-1)             # [G_fine] on device
        if key not in self.stg_rgroups:
            self.stg_rgroups[key] = r_groups_fine.unsqueeze(0)          # [1, G_fine]
        else:
            self.stg_rgroups[key] = torch.cat(
                [self.stg_rgroups[key], r_groups_fine.unsqueeze(0)], dim=0
            )  # [N_stg, G_fine]
 
        # Decode tokens are in the recent window → always tier b3 (Low)
        decode_tier   = self.tiers[3]   # b3 (Low = bmin for recent tokens)
        cb_key        = (layer_idx, kv_head, decode_tier.tier_id)
        cb_lh         = self.codebooks.get(cb_key)
        if cb_lh is None:
            cb_lh = get_codebook(self.codebooks, layer_idx, kv_head, decode_tier.tier_id)
        r, theta = _encode_keys_with_codebooks(
            kn, cb_lh, decode_tier.g, decode_tier.G
        )   # r: [1, G_b3], theta: [1, G_b3] int32
 
        # Compute token position (fall-back to decode-step counter if caller didn't pass)
        pos_val = position if position is not None else (
            self.seq_len + getattr(self, '_decode_step', 0))
        pos_t = torch.tensor([pos_val], dtype=torch.long, device=device)

        if key not in self.stg_r:
            self.stg_r[key]         = r
            self.stg_theta[key]     = theta
            self.stg_V[key]         = vn
            self.stg_positions[key] = pos_t
        else:
            self.stg_r[key]         = torch.cat([self.stg_r[key],     r],     dim=0)
            self.stg_theta[key]     = torch.cat([self.stg_theta[key], theta],  dim=0)
            self.stg_V[key]         = torch.cat([self.stg_V[key],     vn],    dim=0)
            self.stg_positions[key] = torch.cat(
                [self.stg_positions[key], pos_t], dim=0)
 
        if self.stg_r[key].shape[0] >= PAGE_SIZE:
            self._flush_decode_page(key)

        # _decode_step is advanced by patched_forward at the top of each
        # forward pass (layer 0 only).  Here we just fire _maybe_refresh
        # once per decode step at layer 0 head 0 (ADA cadence).
        if layer_idx == 0 and kv_head == 0:
            self._maybe_refresh()
 
    def _flush_decode_page(self, key: Tuple[int, int]) -> None:
        layer_idx, kv_head = key
        device    = self.device
        r_buf     = self.stg_r[key]          # [N, G_b3]
        theta_buf = self.stg_theta[key]      # [N, G_b3]
        V_buf     = self.stg_V[key]          # [N, dh]
        rg_buf    = self.stg_rgroups.get(key)  # [N, G_fine] or None
        n         = r_buf.shape[0]
        b3        = self.tiers[3]
 
        page = build_page_from_codes(
            r_buf.cpu(), theta_buf.cpu(), b3, segment_id=2
        )
 
        header_size   = HEADER_BYTES + b3.G * 4
        theta_bytes   = ((n * b3.G * b3.b_theta) + 7) // 8
        theta_offset  = header_size
        radius_offset = theta_offset + theta_bytes
 
        ptable = PointerTable()
        ptable.add_page(0, theta_offset, radius_offset)
 
        if key not in self._decode_pages:
            self._decode_pages[key] = []
 
        # Pop the n staging positions that correspond to this flush batch.
        pos_buf = self.stg_positions.get(key)
        if pos_buf is not None and pos_buf.shape[0] >= n:
            pos_slice = pos_buf[:n].clone()
            # Keep remainder in the buffer for the next flush (usually empty).
            if pos_buf.shape[0] > n:
                self.stg_positions[key] = pos_buf[n:].clone()
        else:
            # Fallback: reconstruct contiguous positions from decode-step counter.
            flush_num = len(self._decode_pages[key])
            start = self.seq_len + flush_num * n
            pos_slice = torch.arange(start, start + n, dtype=torch.long)

        self._decode_pages[key].append((
            page.to(device),                              # 0: packed page bytes
            ptable.to_tensor().to(device),                # 1: pointer table
            b3.b_theta,                                   # 2: tier b_theta
            n,                                            # 3: n_tokens
            V_buf,                                        # 4: V
            (r_buf.cpu(), theta_buf.cpu()),               # 5: K codes (r, theta) for torch path
            pos_slice.to(device),                         # 6: per-token absolute positions
        ))
 
        flush_num = len(self._decode_pages[key]) - 1  # 0-based
        start_idx = self.seq_len + flush_num * n
 
        new_token_states: List[TokenState] = []
        dh = self.head_dim
        for local_i in range(n):
            abs_idx = start_idx + local_i
            if rg_buf is not None:
                r_groups = rg_buf[local_i].float() 
            else:
                # Fallback: use b3 radii as approximation
                r_groups = r_buf[local_i].cpu().float() 
            r_scalar = float(r_groups.norm())
 
            ts = TokenState(
                layer      = layer_idx,
                head       = kv_head,
                index      = abs_idx,
                r          = r_scalar,
                phi        = torch.zeros(dh - 1, dtype=torch.float32),
                segment_id = 2,      # recent
                age        = 1,
                prev_tier_id = b3.tier_id,
                protected  = False,
                r_groups   = r_groups,
                omega      = 1.0,
            )
            ts.assign_tier_protected(b3.tier_id)
            ts.commit_tier()
            new_token_states.append(ts)
 
        # Add to retained set and token-order map (for ω updates in attention)
        self._retained_tokens.extend(new_token_states)
        if key not in self._ctx_token_order:
            self._ctx_token_order[key] = []
        self._ctx_token_order[key].extend(new_token_states)
 
        # Clean up staging buffers
        del self.stg_r[key]
        del self.stg_theta[key]
        del self.stg_V[key]
        if key in self.stg_rgroups:
            del self.stg_rgroups[key]
        # stg_positions either fully consumed or trimmed; delete if empty
        if key in self.stg_positions and self.stg_positions[key].numel() == 0:
            del self.stg_positions[key]
 
    def _maybe_refresh(self) -> None:
        """
        Re-run the RDR controller on the current retained token set.
        Called every REFRESH_CADENCE decode steps.
        Skipped if REFRESH_CADENCE == 0 (prefill-only mode).
        """
        if REFRESH_CADENCE == 0 or not self._retained_tokens:
            return
        if self._decode_step % REFRESH_CADENCE != 0:
            return

        for key, omega_t in self._omega_buf.items():
            order = self._ctx_token_order.get(key, [])
            n = min(len(order), omega_t.shape[0])
            for pos in range(n):
                order[pos].omega = omega_t[pos].item()

        old_tiers = {id(ts): ts.new_tier_id for ts in self._retained_tokens}
        updated   = online_refresh(self._retained_tokens, self.tiers,
                                   prefill_len=self.seq_len)
 
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
        lh_tier: Dict[Tuple[int,int,int], List] = defaultdict(list)
        for ts in self._retained_tokens:
            if ts.index < self.seq_len:   # prefill tokens only
                key2 = (ts.layer, ts.head, ts.new_tier_id)
                lh_tier[key2].append(ts)
 
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
                positions_parts = []
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
                    positions_parts.append(torch.tensor(indices, dtype=torch.long))
                    if V_lh is not None:
                        V_parts.append(V_lh[indices].to(self.device))
 
                V_tensor = torch.cat(V_parts, dim=0) if V_parts else None
 
                new_entries.append((
                    torch.cat(page_list).to(self.device),
                    ptable.to_tensor().to(self.device),
                    tier.b_theta,
                    len(toks),
                    V_tensor,
                    (torch.cat(r_parts, dim=0), torch.cat(theta_parts, dim=0)),
                    # Slot [6]: per-token absolute positions (RoPE back-rotation)
                    torch.cat(positions_parts, dim=0).to(self.device),
                ))
                new_order.extend(toks)
 
            self.per_head_pages[(layer, head)] = new_entries
 
            old_order = self._ctx_token_order.get((layer, head), [])
            decode_order = [tok for tok in old_order if tok.index >= self.seq_len]
            self._ctx_token_order[(layer, head)] = new_order + decode_order
 
    def uninstall(self) -> None:
        if self._patched:
            unpatch_decode(self.model, self._original_forwards)
            self._patched = False
            print("[SphericalKVPipeline] Attention layers restored.")
 
    def _reset_state(self) -> None:
        self.per_head_pages.clear()
        self._decode_pages.clear()
        self.stg_r.clear()
        self.stg_theta.clear()
        self.stg_V.clear()
        self.stg_rgroups.clear()
        self.stg_positions.clear()
        self.reuse          = None
        self.stability      = None
        self.seq_len        = 0
        # -1 so the first layer-0 patched_forward increments it to 0
        # (→ current_pos = seq_len + 0 = seq_len for decode step 0).
        self._decode_step   = -1
        self._retained_tokens.clear()
        self._ctx_token_order.clear()
        self._omega_buf.clear()
        self._window_buf.clear()
        self._window_ptr.clear()
        self._codes_all.clear()
        self._V_all.clear()
        # GPU-resident omega state (paper §C.2)
        self._omega_gpu_per_layer.clear()
        self._last_bounded_refresh_step = 0
        if hasattr(self, '_lut_pools'):
            self._lut_pools.clear()
