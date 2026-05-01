"""
vllm like backend for decode loop to eliminate hf and python calls
"""
from __future__ import annotations
import time
import math
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
#  Model loading
# ---------------------------------------------------------------------------

def load_vllm_model(model_name: str, dtype: str = "float16",
                    gpu_mem: float = 0.85, max_model_len: int = 131072):
    """Load model via vLLM, return engine + raw model + config."""
    from vllm import LLM
    llm = LLM(
        model=model_name,
        dtype=dtype,
        gpu_memory_utilization=gpu_mem,
        enforce_eager=True,
        max_model_len=max_model_len,
        trust_remote_code=True,
    )
    runner = llm.llm_engine.model_executor.driver_worker.model_runner
    raw = runner.model
    return llm, raw, runner


def _get_model_parts(raw_model):
    """Extract embed, layers, norm, lm_head from vLLM model."""
    inner = getattr(raw_model, 'model', None)
    if inner is None:
        raise RuntimeError("Cannot find inner model")

    embed_tokens = inner.embed_tokens
    layers = inner.layers
    norm = inner.norm

    # lm_head: vLLM blocks direct forward(), use weight matrix
    lm_head = getattr(raw_model, 'lm_head', None)
    if lm_head is None:
        lm_head = getattr(inner, 'lm_head', None)

    # Extract the raw weight for manual matmul
    lm_head_weight = None
    if lm_head is not None:
        if hasattr(lm_head, 'weight'):
            lm_head_weight = lm_head.weight  # [vocab, hidden]
        elif hasattr(lm_head, 'linear_weights'):
            lm_head_weight = lm_head.linear_weights.get('weight')

    # If lm_head shares weights with embed_tokens (common in Llama)
    if lm_head_weight is None:
        lm_head_weight = embed_tokens.weight  # tied weights

    # Get config
    cfg = getattr(raw_model, 'config', None)
    if cfg is None:
        cfg = getattr(inner, 'config', None)

    num_q = cfg.num_attention_heads
    num_kv = getattr(cfg, 'num_key_value_heads', num_q)
    dh = getattr(cfg, 'head_dim', cfg.hidden_size // num_q)
    q_size = num_q * dh
    kv_size = num_kv * dh

    return {
        'embed_tokens': embed_tokens,
        'layers': layers,
        'norm': norm,
        'lm_head_weight': lm_head_weight,
        'num_layers': len(layers),
        'num_q': num_q,
        'num_kv': num_kv,
        'dh': dh,
        'q_size': q_size,
        'kv_size': kv_size,
        'hidden_size': cfg.hidden_size,
    }


# ---------------------------------------------------------------------------
#  Direct model forward (bypasses vLLM scheduling, uses vLLM layers)
# ---------------------------------------------------------------------------

class VLLMDirectForward:
    """
    Runs vLLM model layers directly for decode.
    Uses vLLM's optimized MLP/RMSNorm/embeddings.
    Attention is either dense (SDPA) or compressed (our kernel).
    """
    def __init__(self, raw_model, device):
        parts = _get_model_parts(raw_model)
        self.embed_tokens = parts['embed_tokens']
        self.layers = parts['layers']
        self.norm = parts['norm']
        self.lm_head_weight = parts['lm_head_weight']
        self.num_layers = parts['num_layers']
        self.num_q = parts['num_q']
        self.num_kv = parts['num_kv']
        self.dh = parts['dh']
        self.q_size = parts['q_size']
        self.kv_size = parts['kv_size']
        self.device = device

        # Pre-allocate position tensor for decode (reused every step)
        self._pos_tensor = torch.zeros(1, 1, dtype=torch.long, device=device)
        rot = self.layers[0].self_attn.rotary_emb
        self._rotary_emb = rot

        # Compile MLP + norms for faster decode
        if device.type == "cuda":
            for layer in self.layers:
                layer.mlp = torch.compile(layer.mlp, mode="reduce-overhead")
            print(f"[vLLM] torch.compiled {self.num_layers} MLP layers")

    @torch.no_grad()
    def prefill_capture(self, input_ids: torch.Tensor):
        """
        Run prefill using vLLM layers + dense SDPA attention.
        Captures post-RoPE K and V for each layer.
        """
        B, T = input_ids.shape
        hidden = self.embed_tokens(input_ids)  # [B, T, hidden]
        positions = torch.arange(T, device=self.device).unsqueeze(0)  # [1, T]

        kv_pairs = []

        for i, layer in enumerate(self.layers):
            residual = hidden
            hidden = layer.input_layernorm(hidden)

            # Fused QKV
            qkv, _ = layer.self_attn.qkv_proj(hidden)
            q, k, v = qkv.split(
                [self.q_size, self.kv_size, self.kv_size], dim=-1)

            # Reshape for attention: [B, T, num_heads, dh] → [B, num_heads, T, dh]
            q = q.view(B, T, self.num_q, self.dh).transpose(1, 2)
            k = k.view(B, T, self.num_kv, self.dh).transpose(1, 2)
            v = v.view(B, T, self.num_kv, self.dh).transpose(1, 2)

            # RoPE
            cos, sin = self._get_rope_cos_sin(positions, T)
            q, k = _apply_rotary_emb(q, k, cos, sin)

            # Capture post-RoPE KV (keep in model dtype, not float32)
            kv_pairs.append((
                k.cpu(),   # [1, num_kv, T, dh]
                v.cpu(),
            ))

            # Dense SDPA attention for prefill
            if self.num_q != self.num_kv:
                grp = self.num_q // self.num_kv
                k_exp = k.unsqueeze(2).expand(B, self.num_kv, grp, T, self.dh)
                k_exp = k_exp.reshape(B, self.num_q, T, self.dh)
                v_exp = v.unsqueeze(2).expand(B, self.num_kv, grp, T, self.dh)
                v_exp = v_exp.reshape(B, self.num_q, T, self.dh)
            else:
                k_exp, v_exp = k, v

            attn_out = F.scaled_dot_product_attention(
                q, k_exp, v_exp, is_causal=True)  # [B, num_q, T, dh]

            attn_out = attn_out.transpose(1, 2).reshape(B, T, -1)  # [B, T, hidden]
            attn_out, _ = layer.self_attn.o_proj(attn_out)

            hidden = residual + attn_out

            # MLP (vLLM optimized)
            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = layer.mlp(hidden)
            hidden = residual + hidden

        hidden = self.norm(hidden)
        logits = F.linear(hidden, self.lm_head_weight)  # [B, T, vocab]
        return kv_pairs, logits

    @torch.no_grad()
    def decode_step_dense(self, input_ids: torch.Tensor, position: int,
                          kv_cache: List[Tuple[torch.Tensor, torch.Tensor]]):
        B = input_ids.shape[0]
        hidden = self.embed_tokens(input_ids)  # [B, 1, hidden]
        positions = torch.tensor([[position]], device=self.device)

        new_cache = []
        for i, layer in enumerate(self.layers):
            residual = hidden
            hidden = layer.input_layernorm(hidden)

            qkv, _ = layer.self_attn.qkv_proj(hidden)
            q, k, v = qkv.split(
                [self.q_size, self.kv_size, self.kv_size], dim=-1)

            q = q.view(B, 1, self.num_q, self.dh).transpose(1, 2)
            k = k.view(B, 1, self.num_kv, self.dh).transpose(1, 2)
            v = v.view(B, 1, self.num_kv, self.dh).transpose(1, 2)

            cos, sin = self._get_rope_cos_sin(positions, 1)
            q, k = _apply_rotary_emb(q, k, cos, sin)

            # Append to cache
            k_prev, v_prev = kv_cache[i]
            k_cat = torch.cat([k_prev, k], dim=2)
            v_cat = torch.cat([v_prev, v], dim=2)
            new_cache.append((k_cat, v_cat))

            # GQA expand
            T_full = k_cat.shape[2]
            if self.num_q != self.num_kv:
                grp = self.num_q // self.num_kv
                k_exp = k_cat.unsqueeze(2).expand(B, self.num_kv, grp, T_full, self.dh)
                k_exp = k_exp.reshape(B, self.num_q, T_full, self.dh)
                v_exp = v_cat.unsqueeze(2).expand(B, self.num_kv, grp, T_full, self.dh)
                v_exp = v_exp.reshape(B, self.num_q, T_full, self.dh)
            else:
                k_exp, v_exp = k_cat, v_cat

            attn_out = F.scaled_dot_product_attention(
                q, k_exp, v_exp, is_causal=False)
            attn_out = attn_out.transpose(1, 2).reshape(B, 1, -1)
            attn_out, _ = layer.self_attn.o_proj(attn_out)

            hidden = residual + attn_out
            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = layer.mlp(hidden)
            hidden = residual + hidden

        hidden = self.norm(hidden)
        logits = F.linear(hidden, self.lm_head_weight)
        return logits, new_cache

    @torch.no_grad()
    def decode_step_sphkv(self, input_ids: torch.Tensor, position: int,
                          pipeline):
        B = input_ids.shape[0]
        hidden = self.embed_tokens(input_ids)
        self._pos_tensor[0, 0] = position
        positions = self._pos_tensor

        for i, layer in enumerate(self.layers):
            residual = hidden
            hidden = layer.input_layernorm(hidden)

            # vLLM fused QKV projection
            qkv, _ = layer.self_attn.qkv_proj(hidden)
            q, k, v = qkv.split(
                [self.q_size, self.kv_size, self.kv_size], dim=-1)

            q = q.view(B, 1, self.num_q, self.dh).transpose(1, 2)
            k = k.view(B, 1, self.num_kv, self.dh).transpose(1, 2)
            v = v.view(B, 1, self.num_kv, self.dh).transpose(1, 2)

            # vLLM RoPE
            cos, sin = self._get_rope_cos_sin(positions, 1)
            q, k = _apply_rotary_emb(q, k, cos, sin)

            # OUR compressed attention (replaces PagedAttention)
            if i == 0:
                pipeline._decode_step += 1

            attn_out = pipeline._decode_layer_lut(
                layer_idx=i,
                q_post=q[0, :, 0, :].float(),   # [num_q, dh]
                k_post=k[0, :, 0, :].float(),    # [num_kv, dh]
                v=v[0, :, 0, :].float(),          # [num_kv, dh]
            )

            attn_out = attn_out.to(hidden.dtype).view(B, 1, -1)
            attn_out, _ = layer.self_attn.o_proj(attn_out)

            hidden = residual + attn_out

            # vLLM optimized MLP
            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = layer.mlp(hidden)
            hidden = residual + hidden

        hidden = self.norm(hidden)
        logits = F.linear(hidden, self.lm_head_weight)
        return logits

    def _get_rope_cos_sin(self, positions, seq_len):
        """Get cos/sin from vLLM's rotary_emb, shape [1, 1, seq_len, dh]."""
        rot = self._rotary_emb
        dh = self.dh
        idx = positions.flatten()  # [seq_len]

        cos = sin = None

        if hasattr(rot, 'cos_cached') and rot.cos_cached is not None:
            cos = rot.cos_cached[idx]  # [seq_len, dh/2 or dh]
            sin = rot.sin_cached[idx]
        elif hasattr(rot, 'cos_sin_cache'):
            cache = rot.cos_sin_cache
            cos_sin = cache[idx]
            if cos_sin.dim() == 2:
                half = cos_sin.shape[-1] // 2
                cos = cos_sin[:, :half]
                sin = cos_sin[:, half:]
            else:
                cos = cos_sin[:, 0, :]
                sin = cos_sin[:, 1, :]
        else:
            inv_freq = rot.inv_freq
            freqs = torch.outer(idx.float(), inv_freq)
            cos = freqs.cos()
            sin = freqs.sin()

        if cos.shape[-1] == dh // 2:
            cos = torch.cat([cos, cos], dim=-1)
            sin = torch.cat([sin, sin], dim=-1)

        # Shape: [1, 1, seq_len, dh] for broadcasting with [B, H, T, dh]
        return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)


def _apply_rotary_emb(q, k, cos, sin):
    """Apply rotary embeddings. Handles both full-dim and half-dim cos/sin."""
    # Ensure broadcastable shapes
    if cos.dim() == 3:
        cos = cos.unsqueeze(1)  # [B, 1, T, dim]
        sin = sin.unsqueeze(1)

    rot_dim = cos.shape[-1]
    dh = q.shape[-1]

    if rot_dim == dh:
        # Full-dim RoPE (HF style)
        def _rotate_half(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)
        q_rot = q * cos + _rotate_half(q) * sin
        k_rot = k * cos + _rotate_half(k) * sin
    else:
        # Half-dim RoPE (vLLM style): only rotate first rot_dim dims
        q1, q2 = q[..., :rot_dim], q[..., rot_dim:]
        k1, k2 = k[..., :rot_dim], k[..., rot_dim:]

        # Split rotated part into two halves
        q1a, q1b = q1.chunk(2, dim=-1)
        k1a, k1b = k1.chunk(2, dim=-1)

        cos_half = cos[..., :rot_dim // 2]
        sin_half = sin[..., :rot_dim // 2]

        q_rot_a = q1a * cos_half - q1b * sin_half
        q_rot_b = q1b * cos_half + q1a * sin_half
        k_rot_a = k1a * cos_half - k1b * sin_half
        k_rot_b = k1b * cos_half + k1a * sin_half

        q_rot = torch.cat([q_rot_a, q_rot_b, q2], dim=-1)
        k_rot = torch.cat([k_rot_a, k_rot_b, k2], dim=-1)

    return q_rot, k_rot


# ---------------------------------------------------------------------------
#  Measurement functions
# ---------------------------------------------------------------------------

def measure_dense_vllm(vllm_fwd: VLLMDirectForward, eval_ids, T,
                       n_warm, n_meas, device):
    """Measure dense decode using vLLM layers directly."""
    prefill_ids = eval_ids[:, :T].to(device)

    # Prefill
    kv_pairs, _ = vllm_fwd.prefill_capture(prefill_ids)

    # Move cache to device
    kv_cache = [(k.to(device), v.to(device)) for k, v in kv_pairs]

    # Warmup
    last_tok = eval_ids[:, T:T+1].to(device)
    for wi in range(n_warm):
        logits, kv_cache = vllm_fwd.decode_step_dense(last_tok, T + wi, kv_cache)
        last_tok = eval_ids[:, T + 1 + wi : T + 2 + wi].to(device)

    # Measure
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    nll_sum = 0.0
    for step in range(n_meas):
        logits, kv_cache = vllm_fwd.decode_step_dense(last_tok, T + n_warm + step, kv_cache)
        logits_last = logits[:, -1, :]
        ref_idx = T + n_warm + 1 + step
        ref_tok = eval_ids[0, ref_idx].item()
        log_probs = F.log_softmax(logits_last, dim=-1)
        nll_sum -= log_probs[0, ref_tok].item()
        last_tok = eval_ids[:, ref_idx:ref_idx+1].to(device)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    tok_s = n_meas / elapsed
    nll = nll_sum / n_meas

    # bKV = effective bytes per token (K+V, fp16, all layers all heads)
    bKV = 2 * vllm_fwd.num_layers * vllm_fwd.num_kv * vllm_fwd.dh * 2  # 2 tensors × 2 bytes fp16

    return {
        'mode': 'dense', 'T': T, 'tok_s': tok_s,
        'nll': nll, 'ppl': math.exp(nll) if nll < 20 else float('inf'),
        'bKV': bKV, 'elapsed': elapsed,
    }


def measure_sphkv_vllm(vllm_fwd: VLLMDirectForward, pipeline,
                        eval_ids, T, n_warm, n_meas, budget_bpt, device):
    """Measure SphKV decode using vLLM layers + compressed attention."""
    prefill_ids = eval_ids[:, :T].to(device)

    # Prefill: capture KV pairs
    kv_pairs, prefill_logits = vllm_fwd.prefill_capture(prefill_ids)

    # Build post-RoPE K list in pipeline format: [B, T, H*dh]
    post_rope_K_list = []
    for (K_post, V_post) in kv_pairs:
        B, H, Tlen, D = K_post.shape
        K_t = K_post.permute(0, 2, 1, 3).contiguous().view(B, Tlen, H * D)
        post_rope_K_list.append(K_t)

    # Dummy attention weights (we use real attention in prefill hooks)
    attn_w = [None] * vllm_fwd.num_layers
    ho = [None] * vllm_fwd.num_layers

    # Set budget via config (same as HF path)
    import config as _cfg
    _cfg.BITS_PER_TOKEN = budget_bpt
    _cfg.GLOBAL_BUDGET_BITS = budget_bpt * T * vllm_fwd.num_layers * vllm_fwd.num_kv

    # Run pipeline prefill with pre-captured data (skip HF patching)
    pipeline.prefill(
        kv_pairs=kv_pairs,
        attn_weights=attn_w,
        head_outputs=ho,
        pre_rope_K_list=post_rope_K_list,
        seq_len=T,
        skip_patch=True,
    )

    # Warmup
    last_tok = eval_ids[:, T:T+1].to(device)
    for wi in range(n_warm):
        logits = vllm_fwd.decode_step_sphkv(last_tok, T + wi, pipeline)
        last_tok = eval_ids[:, T + 1 + wi : T + 2 + wi].to(device)

    # Measure
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    nll_sum = 0.0
    for step in range(n_meas):
        logits = vllm_fwd.decode_step_sphkv(
            last_tok, T + n_warm + step, pipeline)
        logits_last = logits[:, -1, :]
        ref_idx = T + n_warm + 1 + step
        ref_tok = eval_ids[0, ref_idx].item()
        log_probs = F.log_softmax(logits_last, dim=-1)
        nll_sum -= log_probs[0, ref_tok].item()
        last_tok = eval_ids[:, ref_idx:ref_idx+1].to(device)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    pipeline.uninstall() if hasattr(pipeline, "uninstall") else None

    tok_s = n_meas / elapsed
    nll = nll_sum / n_meas
    bKV_K = budget_bpt * vllm_fwd.num_layers * vllm_fwd.num_kv / 8
    bKV_V = vllm_fwd.num_layers * vllm_fwd.num_kv * vllm_fwd.dh * 2

    return {
        'mode': 'sphkv', 'T': T, 'tok_s': tok_s,
        'nll': nll, 'ppl': math.exp(nll) if nll < 20 else float('inf'),
        'budget_bpt': budget_bpt, 'elapsed': elapsed,
        'bKV': bKV_K + bKV_V,
    }
