from __future__ import annotations
import contextlib
import types
import math
from typing import Dict, List, Optional, Tuple

import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


@contextlib.contextmanager
def nvtx_range(name: str):
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()


def _extract_kv_pairs(past_key_values):
    if hasattr(past_key_values, "layers"):
        pairs = []
        for layer in past_key_values.layers:
            K = getattr(layer, "keys", None)
            if K is None:
                K = getattr(layer, "key", None)
            V = getattr(layer, "values", None)
            if V is None:
                V = getattr(layer, "value", None)
            if K is not None and V is not None:
                pairs.append((K.float().cpu(), V.float().cpu()))
        if pairs:
            return pairs
    if hasattr(past_key_values, "to_legacy_cache"):
        try:
            legacy = past_key_values.to_legacy_cache()
            return [(kv[0].float().cpu(), kv[1].float().cpu()) for kv in legacy]
        except Exception:
            pass
    if hasattr(past_key_values, "key_cache"):
        return [(K.float().cpu(), V.float().cpu())
                for K, V in zip(past_key_values.key_cache, past_key_values.value_cache)]
    return [(layer[0].float().cpu(), layer[1].float().cpu())
            for layer in past_key_values]


def capture_prefill_pass(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    reuse_chunk_size: int = 256,
):
    """POST-RoPE capture with chunked reuse proxy."""
    num_layers = len(model.model.layers)
    per_head_reuse: Dict[int, torch.Tensor] = {}
    _q_gpu: Dict[int, torch.Tensor] = {}
    _k_gpu: Dict[int, torch.Tensor] = {}

    cfg = model.config
    num_q_heads  = cfg.num_attention_heads
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_q_heads)
    dh           = getattr(cfg, "head_dim", cfg.hidden_size // num_q_heads)
    grp          = num_q_heads // num_kv_heads

    rot = getattr(model.model, "rotary_emb", None) \
        or getattr(model.model.layers[0].self_attn, "rotary_emb", None)

    def _q_hook(li):
        def hook(mod, inp, out):
            _q_gpu[li] = out.detach()
        return hook

    def _k_hook(li):
        def hook(mod, inp, out):
            _k_gpu[li] = out.detach()
        return hook

    def _attn_post_hook(li):
        def hook(mod, inp, out):
            q_raw = _q_gpu.pop(li, None)
            k_raw = _k_gpu.pop(li, None)
            if q_raw is None or k_raw is None:
                return
            bsz, T, _ = q_raw.shape
            Q = q_raw.view(bsz, T, num_q_heads, dh).transpose(1, 2)
            K = k_raw.view(bsz, T, num_kv_heads, dh).transpose(1, 2)
            pos_ids = torch.arange(T, device=Q.device).unsqueeze(0)
            cos, sin = rot(K, pos_ids)
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

            inv_sqrt_dh = 1.0 / math.sqrt(dh)
            col_sum = torch.zeros(num_q_heads, T, device=Q.device, dtype=torch.float32)

            for kv_h in range(num_kv_heads):
                q_grp = Q[:, kv_h * grp:(kv_h + 1) * grp, :, :]
                k_h   = K[:, kv_h:kv_h + 1, :, :]
                for start in range(0, T, reuse_chunk_size):
                    end = min(start + reuse_chunk_size, T)
                    q_c = q_grp[:, :, start:end, :]
                    scores = torch.matmul(q_c, k_h.transpose(2, 3)) * inv_sqrt_dh
                    row_pos = torch.arange(start, end, device=scores.device)
                    col_pos = torch.arange(T, device=scores.device)
                    causal = torch.where(
                        col_pos.unsqueeze(0) <= row_pos.unsqueeze(1),
                        torch.tensor(0.0, device=scores.device),
                        torch.tensor(float('-inf'), device=scores.device))
                    scores = scores + causal.unsqueeze(0).unsqueeze(0)
                    attn = torch.softmax(scores, dim=-1)
                    col_sum[kv_h * grp:(kv_h + 1) * grp] += attn.sum(dim=(0, 2))
                    del scores, attn, q_c
                del q_grp, k_h
            per_head_reuse[li] = (col_sum / T).std(dim=-1).cpu()
            del Q, K, q_raw, k_raw, col_sum
            torch.cuda.empty_cache()
        return hook

    handles = []
    for li in range(num_layers):
        layer = model.model.layers[li]
        handles.append(layer.self_attn.q_proj.register_forward_hook(_q_hook(li)))
        handles.append(layer.self_attn.k_proj.register_forward_hook(_k_hook(li)))
        handles.append(layer.self_attn.register_forward_hook(_attn_post_hook(li)))

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            use_cache=True,
            return_dict=True,
        )
    for h in handles:
        h.remove()

    kv_pairs = _extract_kv_pairs(out.past_key_values)
    logits   = out.logits.float().cpu()
    post_rope_K_list = []
    for (K_post, V_post) in kv_pairs:
        if K_post.dim() == 4:
            B, H, T, D = K_post.shape
            K_t = K_post.permute(0, 2, 1, 3).contiguous().view(B, T, H * D)
        else:
            K_t = K_post
        post_rope_K_list.append(K_t)
    attn_weights = [per_head_reuse.get(li) for li in range(num_layers)]
    head_outputs = [None] * num_layers

    del out
    torch.cuda.empty_cache()
    return kv_pairs, attn_weights, head_outputs, logits, post_rope_K_list


def patch_for_decode(model, pipeline) -> Dict[int, object]:
    """
    Patch attention layers AND pre-compute RoPE tables.
    The decode hot path does ZERO HF module calls.
    """
    rot = getattr(model.model, "rotary_emb", None) \
        or getattr(model.model.layers[0].self_attn, "rotary_emb", None)
    if rot is None:
        raise RuntimeError("Could not locate rotary_emb")

    # Pre-compute RoPE cos/sin tables for all positions we'll need
    device = next(model.parameters()).device
    max_pos = pipeline.seq_len + 1024  # headroom for decode
    dh = pipeline.head_dim
    pos_ids = torch.arange(max_pos, device=device).unsqueeze(0)
    dummy = torch.zeros(1, 1, max_pos, dh, device=device, dtype=torch.float16)
    with torch.no_grad():
        cos_table, sin_table = rot(dummy, pos_ids)
    # cos_table: [1, max_pos, dh] or [1, 1, max_pos, dh]
    if cos_table.dim() == 4:
        cos_table = cos_table.squeeze(0)
    if cos_table.dim() == 3:
        cos_table = cos_table.squeeze(0)  # [max_pos, dh]
    if sin_table.dim() == 4:
        sin_table = sin_table.squeeze(0)
    if sin_table.dim() == 3:
        sin_table = sin_table.squeeze(0)
    pipeline._rope_cos = cos_table.float().to(device)  # [max_pos, dh]
    pipeline._rope_sin = sin_table.float().to(device)
    pipeline._rope_max_pos = max_pos
    del dummy

    originals: Dict[int, object] = {}
    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        originals[layer_idx] = attn.forward
        attn.forward = types.MethodType(
            _make_patched_forward(layer_idx, pipeline), attn)
    return originals


def unpatch_decode(model, originals: Dict[int, object]) -> None:
    for layer_idx, layer in enumerate(model.model.layers):
        if layer_idx in originals:
            layer.self_attn.forward = originals[layer_idx]


def _make_patched_forward(layer_idx: int, pipeline):
    """
    HOT PATH. Zero HF calls, zero allocations, zero imports.
    """
    _sm_scale = 1.0 / math.sqrt(pipeline.head_dim)

    def patched_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position=None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.shape

        if q_len > 1:
            orig = pipeline._original_forwards.get(layer_idx)
            return orig(
                hidden_states, attention_mask=attention_mask,
                position_ids=position_ids, past_key_value=past_key_value,
                output_attentions=output_attentions, use_cache=use_cache,
                cache_position=cache_position, **kwargs,
            )

        dh     = self.head_dim
        num_q  = self.q_proj.weight.shape[0] // dh
        num_kv = self.k_proj.weight.shape[0] // dh

        # Projections (3 matmuls — same as dense)
        q_raw = self.q_proj(hidden_states)
        k_raw = self.k_proj(hidden_states)
        v_raw = self.v_proj(hidden_states)

        Q = q_raw.view(bsz, 1, num_q,  dh).transpose(1, 2)
        K = k_raw.view(bsz, 1, num_kv, dh).transpose(1, 2)
        V = v_raw.view(bsz, 1, num_kv, dh).transpose(1, 2)

        if layer_idx == 0:
            pipeline._decode_step += 1
        current_pos = pipeline.seq_len + pipeline._decode_step - 1

        # RoPE from PRE-CACHED table — NO HF module call, NO torch.tensor alloc
        cos = pipeline._rope_cos[current_pos].unsqueeze(0).unsqueeze(0)  # [1, 1, dh]
        sin = pipeline._rope_sin[current_pos].unsqueeze(0).unsqueeze(0)
        Q_post, K_post = apply_rotary_pos_emb(Q, K, cos, sin)

        # Single pipeline call — all work batched inside
        head_outs = pipeline._decode_layer_lut(
            layer_idx=layer_idx,
            q_post=Q_post[0, :, 0, :].float(),
            k_post=K_post[0, :, 0, :].float(),
            v=V[0, :, 0, :].float(),
        )

        merged = head_outs.to(hidden_states.dtype).view(1, 1, num_q * dh)
        out = self.o_proj(merged)
        return out, None

    return patched_forward


def aggregate_proxy_to_kv_heads(proxy, num_q_heads, num_kv_heads):
    if num_q_heads == num_kv_heads:
        return proxy
    grp = num_q_heads // num_kv_heads
    return proxy.view(proxy.shape[0], num_kv_heads, grp).mean(dim=-1)


def build_attn_weights_tensor(attn_weights):
    valid = [w for w in attn_weights if w is not None]
    if not valid:
        return None
    try:
        return torch.stack(valid, dim=0)
    except Exception:
        return None


def build_head_outputs_tensor(head_outputs):
    return None
