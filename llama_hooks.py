from __future__ import annotations

import contextlib
import types
from typing import Dict, List, Optional, Tuple

import torch


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
            # Use explicit None checks — 'or' on a Tensor calls __bool__ which
            # raises RuntimeError for multi-element tensors.
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
        Ks = past_key_values.key_cache
        Vs = past_key_values.value_cache
        return [(K.float().cpu(), V.float().cpu()) for K, V in zip(Ks, Vs)]

    return [(layer[0].float().cpu(), layer[1].float().cpu())
            for layer in past_key_values]


def capture_prefill_pass(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[
    List[Tuple[torch.Tensor, torch.Tensor]],
    List[Optional[torch.Tensor]],
    List[Optional[torch.Tensor]],
    torch.Tensor,
]:
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=False,
            use_cache=True,
            return_dict=True,
        )

    # Extract KV pairs from any Cache format (Transformers 4.x and 5.x)
    kv_pairs = _extract_kv_pairs(out.past_key_values)

    raw_attn = out.attentions
    attn_weights = (
        [a.float().cpu() if a is not None else None for a in raw_attn]
        if raw_attn is not None
        else [None] * len(kv_pairs)
    )

    head_outputs = _compute_head_outputs(kv_pairs, attn_weights)
    logits = out.logits.float().cpu()

    del out
    torch.cuda.empty_cache()

    return kv_pairs, attn_weights, head_outputs, logits

def patch_for_decode(model, pipeline) -> Dict[int, object]:
    originals: Dict[int, object] = {}
    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        originals[layer_idx] = attn.forward
        patched = _make_patched_forward(layer_idx, pipeline)
        attn.forward = types.MethodType(patched, attn)
    return originals


def unpatch_decode(model, originals: Dict[int, object]) -> None:
    for layer_idx, layer in enumerate(model.model.layers):
        if layer_idx in originals:
            layer.self_attn.forward = originals[layer_idx]

def _make_patched_forward(layer_idx: int, pipeline):

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

        # prefill: fall through to original
        if q_len > 1:
            orig = pipeline._original_forwards.get(layer_idx)
            if orig is None:
                raise RuntimeError(
                    f"[llama_hooks] No original forward saved for layer {layer_idx}."
                )
            return orig(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        # ── (a) projections + RoPE ─────────────────────────────────────
        dh     = self.head_dim
        num_q  = self.q_proj.weight.shape[0] // dh
        num_kv = self.k_proj.weight.shape[0] // dh

        q_raw = self.q_proj(hidden_states)   # [B, 1, num_q*dh]
        k_raw = self.k_proj(hidden_states)   # [B, 1, num_kv*dh]
        v_raw = self.v_proj(hidden_states)   # [B, 1, num_kv*dh]

        Q = q_raw.view(bsz, 1, num_q,  dh).transpose(1, 2)  # [B, H_q,  1, dh]
        K = k_raw.view(bsz, 1, num_kv, dh).transpose(1, 2)  # [B, H_kv, 1, dh]
        V = v_raw.view(bsz, 1, num_kv, dh).transpose(1, 2)  # [B, H_kv, 1, dh]

        try:
            from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
            cos, sin = self.rotary_emb(V, position_ids)
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
        except Exception:
            pass

        # ── (b)+(c) batched compressed attention ───────────────────────
        # Stack all Q heads for this KV group into [grp, dh] and issue
        # ONE kernel launch per KV head instead of one per Q head.
        grp = num_q // num_kv
        head_outs: List[torch.Tensor] = []

        for kv_h in range(num_kv):
            k_new = K[0, kv_h, 0]   # [dh]
            v_new = V[0, kv_h, 0]   # [dh]

            # all Q heads in this GQA group: [grp, dh]
            q_batch = torch.stack(
                [Q[0, kv_h * grp + qi, 0] for qi in range(grp)],
                dim=0,
            )

            # ONE batched call -> [grp, dh]
            attn_outs = pipeline._compressed_head_attention_batched(
                layer_idx=layer_idx,
                kv_head=kv_h,
                q_batch=q_batch,
                k_new=k_new,
                v_new=v_new,
            )

            for qi in range(grp):
                head_outs.append(attn_outs[qi])

            # compress and stage the new KV token ONCE per KV head
            pipeline._append_decode_kv(layer_idx, kv_h, k_new, v_new)

        # ── (d) merge heads -> o_proj ──────────────────────────────────
        merged = torch.stack(head_outs, dim=0).view(1, 1, num_q * dh)
        with nvtx_range("proj"):
            out = self.o_proj(merged)

        return out, None

    return patched_forward

def _compute_head_outputs(
    kv_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    attn_weights: List[Optional[torch.Tensor]],
) -> List[Optional[torch.Tensor]]:
    head_outputs: List[Optional[torch.Tensor]] = []
    for li, (_, V) in enumerate(kv_pairs):
        attn = attn_weights[li] if li < len(attn_weights) else None
        if attn is None:
            head_outputs.append(None)
            continue
        nq, nkv = attn.shape[1], V.shape[1]
        if nq != nkv:
            V = V.repeat_interleave(nq // nkv, dim=1)
        head_outputs.append(torch.matmul(attn.float(), V.float()))
    return head_outputs


def build_attn_weights_tensor(
    attn_weights: List[Optional[torch.Tensor]],
) -> Optional[torch.Tensor]:
    valid = [w for w in attn_weights if w is not None]
    if not valid:
        return None
    try:
        return torch.stack(valid, dim=1)
    except Exception as exc:
        print(f"[llama_hooks] Could not stack attn_weights: {exc}")
        return None


def build_head_outputs_tensor(
    head_outputs: List[Optional[torch.Tensor]],
) -> Optional[torch.Tensor]:
    valid = [h for h in head_outputs if h is not None]
    if not valid:
        return None
    try:
        return torch.stack(valid, dim=1)
    except Exception as exc:
        print(f"[llama_hooks] Could not stack head_outputs: {exc}")
        return None


def aggregate_proxy_to_kv_heads(
    proxy: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
) -> torch.Tensor:
    if num_q_heads == num_kv_heads:
        return proxy
    grp = num_q_heads // num_kv_heads
    return proxy.view(proxy.shape[0], num_kv_heads, grp).mean(dim=-1)