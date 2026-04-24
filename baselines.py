from __future__ import annotations
import math
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


def _get_layer_kv(cache, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (K, V) references for ``layer_idx``.  Works with DynamicCache's
    ``.layers[i].keys/.values`` (HF 5.x) and legacy ``.key_cache[i]``."""
    if hasattr(cache, "layers"):
        layer = cache.layers[layer_idx]
        return layer.keys, layer.values
    # legacy fallback
    return cache.key_cache[layer_idx], cache.value_cache[layer_idx]


def _set_layer_kv(cache, layer_idx: int, K: torch.Tensor, V: torch.Tensor) -> None:
    if hasattr(cache, "layers"):
        layer = cache.layers[layer_idx]
        layer.keys   = K
        layer.values = V
        return
    cache.key_cache[layer_idx]   = K
    cache.value_cache[layer_idx] = V


def _num_layers(cache) -> int:
    if hasattr(cache, "layers"):
        return len(cache.layers)
    return len(cache.key_cache)


def _cache_seq_len(cache) -> int:
    """Actual tensor length (after eviction), not the "true time" length."""
    if hasattr(cache, "get_seq_length"):
        try:
            return int(cache.get_seq_length())
        except Exception:
            pass
    K0, _ = _get_layer_kv(cache, 0)
    return int(K0.shape[-2])


def _make_cache():
    """Construct a fresh DynamicCache that survives across calls."""
    from transformers import DynamicCache
    return DynamicCache()


def _prefill_with_cache(model, prefill_ids: torch.Tensor,
                        want_attn: bool = False):
    """Run prefill with a real KV cache.  Returns (cache, logits, attn_list)."""
    out = model(
        input_ids=prefill_ids,
        use_cache=True,
        output_attentions=want_attn,
        return_dict=True,
    )
    return out.past_key_values, out.logits, (out.attentions if want_attn else None)


def _cuda_timer(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        return ("cuda", s, e)
    return ("cpu", time.perf_counter(), None)


def _cuda_stop(t):
    kind, a, b = t
    if kind == "cuda":
        b.record(); torch.cuda.synchronize()
        return a.elapsed_time(b) / 1e3
    return max(time.perf_counter() - a, 1e-9)


def _config_dims(model):
    cfg = model.config
    num_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    dh     = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    return cfg.num_hidden_layers, num_kv, dh


def _evict_streaming(cache, n_sink: int, window: int) -> None:
    """Keep positions [0..n_sink) ∪ [L-window..L) in every layer."""
    L = _cache_seq_len(cache)
    if L <= n_sink + window:
        return
    idx = torch.cat([
        torch.arange(0, n_sink),
        torch.arange(L - window, L),
    ])
    n_lay = _num_layers(cache)
    for li in range(n_lay):
        K, V = _get_layer_kv(cache, li)
        # K,V shape: [B, H_kv, T, D] -- slice on dim -2.
        idx_ = idx.to(K.device)
        _set_layer_kv(cache, li, K.index_select(-2, idx_),
                                  V.index_select(-2, idx_))


@torch.no_grad()
def run_streaming_llm(model, eval_ids: torch.Tensor, T: int,
                      n_warm: int, n_meas: int, device: torch.device,
                      n_sink: int = 4, window: int = 256) -> dict:
    """StreamingLLM baseline with real cache eviction."""
    prefill_ids = eval_ids[:, :T].to(device)
    true_len    = T   # absolute time counter for cache_position

    # Prefill -- builds full KV cache, then evict to sinks+window.
    cache, logits, _ = _prefill_with_cache(model, prefill_ids)
    _evict_streaming(cache, n_sink, window)
    last_tok = logits[:, -1, :].argmax(-1, keepdim=True)

    # Warmup
    for _ in range(n_warm):
        cache_pos = torch.tensor([true_len], device=device)
        out = model(input_ids=last_tok, past_key_values=cache,
                    cache_position=cache_pos, use_cache=True,
                    return_dict=True)
        cache = out.past_key_values
        true_len += 1
        last_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
        _evict_streaming(cache, n_sink, window)

    # Measurement
    timer = _cuda_timer(device)
    nll_sum = 0.0
    for _ in range(n_meas):
        cache_pos = torch.tensor([true_len], device=device)
        out = model(input_ids=last_tok, past_key_values=cache,
                    cache_position=cache_pos, use_cache=True,
                    return_dict=True)
        cache = out.past_key_values
        logits = out.logits[:, -1, :]
        nid = logits.argmax(-1, keepdim=True)
        nll_sum -= F.log_softmax(logits, dim=-1)[0, nid.item()].item()
        last_tok = nid
        true_len += 1
        _evict_streaming(cache, n_sink, window)
    elapsed = _cuda_stop(timer)

    n_layers, num_kv, dh = _config_dims(model)
    kept = n_sink + window
    # bKV = 2 (K and V) × bytes_per_elem × kept × L × H_kv × D
    bytes_per_elem = 2  # fp16
    bKV = 2 * bytes_per_elem * kept * n_layers * num_kv * dh

    return {
        "mode": "streaming_llm", "tok_s": n_meas / elapsed,
        "nll":  nll_sum / max(n_meas, 1),
        "bKV":  bKV / max(T, 1), "peak_KV_GB": bKV / 1e9, "T": T,
        "config": {"n_sink": n_sink, "window": window},
    }



class _H2OState:
    """Per-layer cumulative attention score aligned with the live cache."""
    def __init__(self, n_layers: int, device: torch.device):
        self.scores: List[Optional[torch.Tensor]] = [None] * n_layers

    def init_from_prefill_attn(self, attns) -> None:
        """``attns[li]`` is [B, H_q, T, T] -- sum over (H_q, T_q) to get
        per-position score of shape [T]."""
        for li, a in enumerate(attns or []):
            if a is None:
                continue
            s = a[0].sum(dim=(0, 1))  # [T]
            self.scores[li] = s.detach().float()

    def reindex(self, li: int, keep_idx: torch.Tensor) -> None:
        if self.scores[li] is not None:
            self.scores[li] = self.scores[li].index_select(0, keep_idx.to(self.scores[li].device))

    def append_decode_score(self, li: int, new_score: torch.Tensor) -> None:
        """``new_score`` is [T_k] (attention from the new query to context keys
        including the just-appended K).  Summed over heads already."""
        if self.scores[li] is None:
            self.scores[li] = new_score.detach().float()
            return
        old = self.scores[li]
        if new_score.shape[0] == old.shape[0] + 1:
            updated = old + new_score[:old.shape[0]].to(old.device)
            self.scores[li] = torch.cat([updated, new_score[-1:].to(old.device)])
        elif new_score.shape[0] == old.shape[0]:
            # no append (e.g. cache full post-eviction and new token also evicted)
            self.scores[li] = old + new_score.to(old.device)
        else:
            # Recompute from scratch if shapes disagree.
            self.scores[li] = new_score.detach().float()


def _evict_h2o(cache, state: _H2OState, n_sink: int,
               budget_tokens: int) -> None:
    """Per-layer: keep sinks + top-(budget - sinks) by cumulative attention."""
    n_lay = _num_layers(cache)
    for li in range(n_lay):
        K, V = _get_layer_kv(cache, li)
        L = K.shape[-2]
        if L <= budget_tokens:
            continue
        score = state.scores[li]
        if score is None or score.shape[0] != L:
            # fallback: score by recency
            score = torch.arange(L, device=K.device, dtype=torch.float32)
        ns = min(n_sink, L)
        non_sink = score.clone()
        non_sink[:ns] = float("-inf")
        k = max(budget_tokens - ns, 0)
        if k == 0:
            keep = torch.arange(ns, device=K.device)
        else:
            k = min(k, L - ns)
            _, top = torch.topk(non_sink, k)
            sinks = torch.arange(ns, device=K.device)
            keep  = torch.cat([sinks, top.sort().values]).sort().values
        _set_layer_kv(cache, li, K.index_select(-2, keep),
                                  V.index_select(-2, keep))
        state.reindex(li, keep)


@torch.no_grad()
def run_h2o(model, eval_ids: torch.Tensor, T: int,
            n_warm: int, n_meas: int, device: torch.device,
            budget_tokens: int = 256, n_sink: int = 4) -> dict:
    """H2O baseline with live heavy-hitter scoring and cache eviction."""
    prefill_ids = eval_ids[:, :T].to(device)

    n_layers, num_kv, dh = _config_dims(model)
    state = _H2OState(n_layers, device)

    # Prefill with output_attentions -- seeds per-position scores.
    cache, logits, attns = _prefill_with_cache(model, prefill_ids, want_attn=True)
    state.init_from_prefill_attn(attns)
    del attns

    _evict_h2o(cache, state, n_sink, budget_tokens)
    last_tok = logits[:, -1, :].argmax(-1, keepdim=True)
    true_len = T

    def _decode_step(tok, cache, t_abs):
        cache_pos = torch.tensor([t_abs], device=device)
        out = model(input_ids=tok, past_key_values=cache,
                    cache_position=cache_pos, use_cache=True,
                    output_attentions=True, return_dict=True)
        # Per-layer attention from new query (shape [B, H_q, 1, T_k]):
        for li, a in enumerate(out.attentions or []):
            if a is None:
                continue
            s = a[0].sum(dim=(0, 1))   # [T_k]
            state.append_decode_score(li, s.detach().float())
        return out

    # Warmup
    for _ in range(n_warm):
        out = _decode_step(last_tok, cache, true_len)
        cache = out.past_key_values
        true_len += 1
        last_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
        _evict_h2o(cache, state, n_sink, budget_tokens)

    # Measurement
    timer = _cuda_timer(device)
    nll_sum = 0.0
    for _ in range(n_meas):
        out = _decode_step(last_tok, cache, true_len)
        cache = out.past_key_values
        true_len += 1
        logits = out.logits[:, -1, :]
        nid = logits.argmax(-1, keepdim=True)
        nll_sum -= F.log_softmax(logits, dim=-1)[0, nid.item()].item()
        last_tok = nid
        _evict_h2o(cache, state, n_sink, budget_tokens)
    elapsed = _cuda_stop(timer)

    bytes_per_elem = 2  # fp16
    bKV = 2 * bytes_per_elem * budget_tokens * n_layers * num_kv * dh

    return {
        "mode": "h2o", "tok_s": n_meas / elapsed,
        "nll":  nll_sum / max(n_meas, 1),
        "bKV":  bKV / max(T, 1), "peak_KV_GB": bKV / 1e9, "T": T,
        "config": {"budget_tokens": budget_tokens, "n_sink": n_sink},
    }



def _quant_dequant(x: torch.Tensor, n_bits: int) -> torch.Tensor:
    if n_bits >= 16:
        return x  # fp16 no-op
    qmax = (1 << (n_bits - 1)) - 1
    orig_dtype = x.dtype
    xf = x.float()
    # per-token max-abs across the last dim
    amax = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = amax / qmax
    xq = (xf / scale).round().clamp_(-qmax - 1, qmax)
    return (xq * scale).to(orig_dtype)


def _quant_cache_inplace(cache, n_bits: int,
                         only_last_n: Optional[int] = None) -> None:
    """Round-trip quantize K and V in place."""
    n_lay = _num_layers(cache)
    for li in range(n_lay):
        K, V = _get_layer_kv(cache, li)
        if only_last_n is not None and only_last_n > 0 and K.shape[-2] > only_last_n:
            # Only quantize newly appended tokens -- the prefill-ones were
            # already quantized earlier.  This keeps quant noise bounded
            # (no repeated quantization of already-dequantized values).
            tail_K = _quant_dequant(K[..., -only_last_n:, :], n_bits)
            tail_V = _quant_dequant(V[..., -only_last_n:, :], n_bits)
            Kq = torch.cat([K[..., :-only_last_n, :], tail_K], dim=-2)
            Vq = torch.cat([V[..., :-only_last_n, :], tail_V], dim=-2)
        else:
            Kq = _quant_dequant(K, n_bits)
            Vq = _quant_dequant(V, n_bits)
        _set_layer_kv(cache, li, Kq, Vq)


@torch.no_grad()
def run_uniform_quant(model, eval_ids: torch.Tensor, T: int,
                      n_warm: int, n_meas: int, device: torch.device,
                      n_bits: int = 4) -> dict:
    """N-bit KV baseline -- quantize+dequantize the cache after each update."""
    prefill_ids = eval_ids[:, :T].to(device)

    cache, logits, _ = _prefill_with_cache(model, prefill_ids)
    _quant_cache_inplace(cache, n_bits)  # quant entire prefill cache
    last_tok = logits[:, -1, :].argmax(-1, keepdim=True)
    true_len = T

    # Warmup
    for _ in range(n_warm):
        cache_pos = torch.tensor([true_len], device=device)
        out = model(input_ids=last_tok, past_key_values=cache,
                    cache_position=cache_pos, use_cache=True,
                    return_dict=True)
        cache = out.past_key_values
        true_len += 1
        last_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
        _quant_cache_inplace(cache, n_bits, only_last_n=1)

    timer = _cuda_timer(device)
    nll_sum = 0.0
    for _ in range(n_meas):
        cache_pos = torch.tensor([true_len], device=device)
        out = model(input_ids=last_tok, past_key_values=cache,
                    cache_position=cache_pos, use_cache=True,
                    return_dict=True)
        cache = out.past_key_values
        logits = out.logits[:, -1, :]
        nid = logits.argmax(-1, keepdim=True)
        nll_sum -= F.log_softmax(logits, dim=-1)[0, nid.item()].item()
        last_tok = nid
        true_len += 1
        _quant_cache_inplace(cache, n_bits, only_last_n=1)
    elapsed = _cuda_stop(timer)

    n_layers, num_kv, dh = _config_dims(model)
    # Storage model: per-token per-channel sym quant = n_bits per element
    # plus ONE fp16 scale per (layer, head, token).
    bits_per_elem   = n_bits
    scale_bits_per_token = 16    # fp16 per-token scale
    # 2 * (K+V) * T * L * H * D * bits + 2 * L * H * T * scale_bits
    bKV_bits  = 2 * T * n_layers * num_kv * dh * bits_per_elem
    bKV_bits += 2 * n_layers * num_kv * T * scale_bits_per_token
    bKV_bytes = bKV_bits / 8

    return {
        "mode": f"quant_{n_bits}bit", "tok_s": n_meas / elapsed,
        "nll":  nll_sum / max(n_meas, 1),
        "bKV":  bKV_bytes / max(T, 1), "peak_KV_GB": bKV_bytes / 1e9, "T": T,
        "config": {"n_bits": n_bits},
    }



BASELINE_MODES = {
    "streaming_llm": "StreamingLLM (sliding window + sinks)",
    "h2o":           "H2O (heavy-hitter eviction)",
    "quant_2bit":    "Uniform 2-bit KV quantization (per-token)",
    "quant_4bit":    "Uniform 4-bit KV quantization (per-token)",
    "quant_8bit":    "Uniform 8-bit KV quantization (per-token)",
}


def run_baseline(mode: str, model, eval_ids: torch.Tensor, T: int,
                 n_warm: int, n_meas: int, device: torch.device,
                 **kwargs) -> dict:
    if mode == "streaming_llm":
        window = kwargs.get("window", min(T // 2, 2048))
        n_sink = kwargs.get("n_sink", 4)
        return run_streaming_llm(model, eval_ids, T, n_warm, n_meas, device,
                                 n_sink=n_sink, window=window)
    if mode == "h2o":
        budget = kwargs.get("budget_tokens", min(T // 2, 2048))
        n_sink = kwargs.get("n_sink", 4)
        return run_h2o(model, eval_ids, T, n_warm, n_meas, device,
                       budget_tokens=budget, n_sink=n_sink)
    if mode.startswith("quant_"):
        bits = int(mode.split("_")[1].replace("bit", ""))
        return run_uniform_quant(model, eval_ids, T, n_warm, n_meas, device,
                                 n_bits=bits)
    raise ValueError(f"Unknown baseline: {mode}")
