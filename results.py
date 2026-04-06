from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

SPHKV_COLOR = "#4361EE"
DENSE_COLOR = "#F72585"
TIER_COLORS = {
    "Tier-16": "#2DC653",
    "Tier-8":  "#4361EE",
    "Tier-4":  "#F77F00",
    "Tier-2":  "#FCBF49",
    "Dropped": "#AAAAAA",
}
FIG_DPI   = 180
LW        = 2.4
MS        = 8
FONT_T    = 14
FONT_L    = 11
FONT_TK   = 10
FONT_LEG  = 10
GA        = 0.22

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       GA,
    "grid.linestyle":   "--",
})


def _cuda_timer(is_cuda: bool):
    if is_cuda:
        import torch
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        return s, e
    return None, None


def measure_dense_decode(
    model,
    prefill_ids,   # [1, T] on device
    n_warm: int,
    n_meas: int,
    n_trials: int,
    device,
) -> dict:
    """
    Time standard HuggingFace decode (use_cache=True, dense KV).
    No SphericalKV patches.  Returns tok/s median over n_trials.
    Also measures peak VRAM during prefill.
    """
    import torch
    is_cuda = device.type == "cuda"

    # ── Peak VRAM during dense prefill ───────────────────────────────────
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        mem_before_prefill = torch.cuda.memory_allocated(device)

    with torch.no_grad():
        out = model(prefill_ids, use_cache=True, return_dict=True)
    past_kv = out.past_key_values

    if is_cuda:
        torch.cuda.synchronize(device)
        peak_vram_prefill = torch.cuda.max_memory_allocated(device)
        mem_after_prefill = torch.cuda.memory_allocated(device)
    else:
        peak_vram_prefill = 0
        mem_after_prefill = 0
        mem_before_prefill = 0

    dense_kv_bytes = sum(
        k.numel() * k.element_size() + v.numel() * v.element_size()
        for (k, v) in (past_kv if isinstance(past_kv, (list, tuple))
                       else [(past_kv.layers[i].keys, past_kv.layers[i].values)
                             for i in range(len(past_kv.layers))])
    )

    del out, past_kv
    if is_cuda:
        torch.cuda.empty_cache()

    tps_list = []
    for _ in range(n_trials):
        # Re-run prefill fresh each trial so cache state is identical
        with torch.no_grad():
            out_t = model(prefill_ids, use_cache=True, return_dict=True)
        pkv = out_t.past_key_values
        current_ids = prefill_ids.clone()

        # Warmup
        for _ in range(n_warm):
            next_id = out_t.logits[:, -1, :].argmax(-1, keepdim=True)
            current_ids = torch.cat([current_ids, next_id], dim=-1)
            with torch.no_grad():
                out_t = model(next_id, use_cache=True,
                              past_key_values=pkv, return_dict=True)
            pkv = out_t.past_key_values

        if is_cuda:
            torch.cuda.synchronize(device)

        # Measurement
        if is_cuda:
            t_s = torch.cuda.Event(enable_timing=True)
            t_e = torch.cuda.Event(enable_timing=True)
            t_s.record()
        else:
            t0 = time.perf_counter()

        for _ in range(n_meas):
            next_id = out_t.logits[:, -1, :].argmax(-1, keepdim=True)
            with torch.no_grad():
                out_t = model(next_id, use_cache=True,
                              past_key_values=pkv, return_dict=True)
            pkv = out_t.past_key_values

        if is_cuda:
            t_e.record()
            torch.cuda.synchronize(device)
            elapsed = t_s.elapsed_time(t_e) / 1e3
        else:
            elapsed = time.perf_counter() - t0

        tps_list.append(n_meas / elapsed)
        # Free KV cache and activations before next trial
        del out_t, pkv, current_ids
        if is_cuda:
            torch.cuda.empty_cache()

    return {
        "tok_s_median":         statistics.median(tps_list),
        "tok_s_all":            tps_list,
        "dense_kv_bytes":       dense_kv_bytes,
        "peak_vram_prefill":    peak_vram_prefill,
        "mem_after_prefill":    mem_after_prefill,
        "mem_before_prefill":   mem_before_prefill,
        # HBM bytes per generated token = full KV cache read once
        "hbm_bytes_per_tok":    dense_kv_bytes,
    }


def measure_sphkv_decode(
    pipeline,
    prefill_ids,
    n_warm: int,
    n_meas: int,
    n_trials: int,
    device,
) -> dict:
    """
    Time SphericalKV decode.  Also measures peak VRAM during prefill and
    computes CORRECT HBM bytes/token from actual data-structure sizes.
    """
    import torch
    is_cuda = device.type == "cuda"

    # ── Peak VRAM during SphericalKV prefill ──────────────────────────────
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        mem_before_prefill = torch.cuda.memory_allocated(device)

    pipeline.prefill(prefill_ids.to(device))

    if is_cuda:
        torch.cuda.synchronize(device)
        peak_vram_prefill = torch.cuda.max_memory_allocated(device)
        mem_after_prefill = torch.cuda.memory_allocated(device)
    else:
        peak_vram_prefill = 0
        mem_after_prefill = 0
        mem_before_prefill = 0

    kv_bytes = 0
    tier_counts = defaultdict(int)  # b_theta → token count
    total_retained = 0
    _bt_to_label = {6: "High (6-bit)", 4: "Mid (4-bit)", 3: "Low (3-bit)"}

    for (layer, head), tier_list in pipeline.per_head_pages.items():
        for (pt, ptt, b_theta, n_tokens, V_tier, K_tier) in tier_list:
            kv_bytes += pt.numel()                              # compressed K
            kv_bytes += ptt.numel() * 4                        # pointer table
            if V_tier is not None:
                kv_bytes += V_tier.numel() * V_tier.element_size() # dense V
            tier_counts[b_theta] += n_tokens
            total_retained += n_tokens

    codebook_bytes = sum(
        cb.numel() * cb.element_size()
        for cb in pipeline.codebooks.values()
    )
    hbm_sphkv = kv_bytes + codebook_bytes

    # ── Throughput trials ─────────────────────────────────────────────────
    model = pipeline.model
    tps_list = []
    for _ in range(n_trials):
        current_ids = prefill_ids.to(device).clone()

        for _ in range(n_warm):
            with torch.no_grad():
                out = model(input_ids=current_ids[:, -1:],
                            use_cache=False, return_dict=True)
            next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
            current_ids = torch.cat([current_ids, next_id], dim=-1)

        if is_cuda:
            torch.cuda.synchronize(device)

        if is_cuda:
            t_s = torch.cuda.Event(enable_timing=True)
            t_e = torch.cuda.Event(enable_timing=True)
            t_s.record()
        else:
            t0 = time.perf_counter()

        for _ in range(n_meas):
            with torch.no_grad():
                out = model(input_ids=current_ids[:, -1:],
                            use_cache=False, return_dict=True)
            next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
            current_ids = torch.cat([current_ids, next_id], dim=-1)

        if is_cuda:
            t_e.record()
            torch.cuda.synchronize(device)
            elapsed = t_s.elapsed_time(t_e) / 1e3
        else:
            elapsed = time.perf_counter() - t0

        tps_list.append(n_meas / elapsed)
        # Free activations before next trial
        del out, current_ids
        if is_cuda:
            torch.cuda.empty_cache()
    from config import SINK_TOKENS as _SINK_N
    _tid_to_btheta = {t.tier_id: t.b_theta
                      for t in pipeline.tiers if t.tier_id != 0}

    decode_tier_counts = defaultdict(int)
    decode_sink_count  = 0
    for tok in pipeline._retained_tokens:
        bt = _tid_to_btheta.get(tok.new_tier_id, 0)
        if bt > 0:
            decode_tier_counts[bt] += 1
        if tok.protected or tok.index < _SINK_N:
            decode_sink_count += 1

    # Staging buffer: unflushed decode tokens, always stored at b3
    b3_btheta = pipeline.tiers[3].b_theta   # = 3
    stg_token_count = sum(
        r.shape[0] for r in pipeline.stg_r.values()
    )
    decode_tier_counts[b3_btheta] += stg_token_count

    decode_total = sum(decode_tier_counts.values())

    return {
        "tok_s_median":         statistics.median(tps_list),
        "tok_s_all":            tps_list,
        "hbm_bytes_per_tok":    hbm_sphkv,
        "kv_bytes_no_codebooks":kv_bytes,
        "codebook_bytes":       codebook_bytes,
        "tier_counts":          dict(tier_counts),
        "total_retained":       total_retained,
        "peak_vram_prefill":    peak_vram_prefill,
        "mem_after_prefill":    mem_after_prefill,
        "mem_before_prefill":   mem_before_prefill,
        "decode_tier_counts":   dict(decode_tier_counts),
        "decode_total":         decode_total,
        "decode_sink_count":    decode_sink_count,
    }


def eval_memory_from_pipeline(pipeline, T: int) -> dict:
    """Extract memory metrics from a prefilled pipeline."""
    L  = pipeline.num_layers
    H  = pipeline.num_kv_heads
    dh = pipeline.head_dim

    dense_K = 2 * T * L * H * dh   # FP16
    dense_V = 2 * T * L * H * dh

    comp_K_pages = 0
    comp_ptable  = 0
    ret_V        = 0
    tier_token_counts = defaultdict(int)
    total_retained    = 0

    for (layer, head), tier_list in pipeline.per_head_pages.items():
        for (pt, ptt, b_theta, n_tokens, V_tier, K_tier) in tier_list:
            comp_K_pages += pt.numel()
            comp_ptable  += ptt.numel() * 4
            if V_tier is not None:
                ret_V += V_tier.numel() * V_tier.element_size()
            tier_token_counts[b_theta] += n_tokens
            total_retained += n_tokens

    comp_K_total = comp_K_pages + comp_ptable

    return {
        "T":                         T,
        "dense_K_bytes":             dense_K,
        "dense_V_bytes":             dense_V,
        "dense_KV_bytes":            dense_K + dense_V,
        "comp_K_pages_bytes":        comp_K_pages,
        "comp_K_ptable_bytes":       comp_ptable,
        "comp_K_total_bytes":        comp_K_total,
        "retained_V_bytes":          ret_V,
        "sphkv_KV_bytes":            comp_K_total + ret_V,
        "kv_bytes_per_tok_dense":    (dense_K + dense_V) / max(T, 1),
        "kv_bytes_per_tok_sphkv":    (comp_K_total + ret_V) / max(T, 1),
        "compression_ratio_K":       dense_K / max(comp_K_total, 1),
        "compression_ratio_total":   (dense_K + dense_V) / max(comp_K_total + ret_V, 1),
        "peak_KV_MB_dense":          (dense_K + dense_V) / 1e6,
        "peak_KV_MB_sphkv":          (comp_K_total + ret_V) / 1e6,
        "tier_token_counts":         dict(tier_token_counts),
        "total_retained":            total_retained,
        "retention_pct":             100 * total_retained / max(T * L * H, 1),
    }


# ══════════════════════════════════════════════════════════════════════════
#  SYNTHETIC DATA  (--dry_run)
# ══════════════════════════════════════════════════════════════════════════

def synthetic_run(T: int, model_cfg: dict) -> dict:
    """Realistic synthetic metrics for dry-run / CI testing."""
    L   = model_cfg.get("L", 16)
    H   = model_cfg.get("H", 8)
    dh  = model_cfg.get("dh", 64)

    dense_K = 2 * T * L * H * dh
    dense_V = 2 * T * L * H * dh
    # K compressed ~7× at medium context; grows slowly with T
    comp_ratio = 6.0 + 1.5 * math.log2(max(T / 512, 1))
    comp_K  = int(dense_K / comp_ratio)
    ptable  = max(comp_K // 25, 64)
    ret_V   = dense_V   # V stays dense
    cb_bytes = L * H * 8 * 16 * 4  # 8 groups, codebook_size=16 guess

    # Tier distribution: ~90% Tier-8, rest Tier-4
    retained = int(T * L * H * 0.15)
    tier_counts = {8: int(retained * 0.9), 4: int(retained * 0.1)}

    dense_kv  = dense_K + dense_V
    sphkv_kv  = comp_K + ptable + ret_V

    t_python_dense  = 0.060   # 60 ms fixed Python overhead per dense step
    t_python_sphkv  = 0.090   # 90 ms fixed Python overhead per SphKV step
    hbm_bw          = 2e12
    t_kv_dense  = dense_kv  / hbm_bw
    t_kv_sphkv  = (sphkv_kv + cb_bytes) / hbm_bw
    tps_dense   = 1.0 / (t_python_dense  + t_kv_dense)
    tps_sphkv   = 1.0 / (t_python_sphkv + t_kv_sphkv)

    w_bytes = 2.48e9  # 1.24B params FP16
    dense_peak_vram  = w_bytes + dense_K + dense_V          # model + full KV
    sphkv_peak_vram  = w_bytes + dense_K + dense_V + sphkv_kv + cb_bytes
    # The SPIKE is: intermediate dense KV is still live while compression runs
    vram_change_pct  = 100 * (sphkv_peak_vram - dense_peak_vram) / dense_peak_vram

    return {
        "T": T,
        # Memory
        "dense_K_bytes":           dense_K,
        "dense_V_bytes":           dense_V,
        "dense_KV_bytes":          dense_K + dense_V,
        "comp_K_total_bytes":      comp_K + ptable,
        "retained_V_bytes":        ret_V,
        "sphkv_KV_bytes":          comp_K + ptable + ret_V,
        "kv_bytes_per_tok_dense":  (dense_K + dense_V) / T,
        "kv_bytes_per_tok_sphkv":  (comp_K + ptable + ret_V) / T,
        "compression_ratio_K":     dense_K / max(comp_K + ptable, 1),
        "compression_ratio_total": dense_kv / max(comp_K + ptable + ret_V, 1),
        "peak_KV_MB_dense":        (dense_K + dense_V) / 1e6,
        "peak_KV_MB_sphkv":        (comp_K + ptable + ret_V) / 1e6,
        "tier_token_counts":       tier_counts,
        "total_retained":          retained,
        "retention_pct":           100 * retained / max(T * L * H, 1),
        # Throughput (both measured with same Python-overhead model)
        "dense_tok_s":             tps_dense,
        "sphkv_tok_s":             tps_sphkv,
        "speedup":                 tps_sphkv / max(tps_dense, 1e-9),
        # HBM bytes/token (from actual tensor sizes, not proxy)
        "dense_hbm_bytes_per_tok": dense_kv,
        "sphkv_hbm_bytes_per_tok": sphkv_kv + cb_bytes,
        # VRAM (peak during prefill)
        "dense_peak_vram_MB":      dense_peak_vram / 1e6,
        "sphkv_peak_vram_MB":      sphkv_peak_vram / 1e6,
        "vram_change_pct":         vram_change_pct,
    }


# ══════════════════════════════════════════════════════════════════════════
#  REAL RUN
# ══════════════════════════════════════════════════════════════════════════

def real_run(
    model,
    pipeline,
    eval_ids,      # pre-loaded corpus tensor, shape [num_eval_tokens] on CPU
    context_len: int,
    n_warm: int,
    n_meas: int,
    n_trials: int,
    device,
) -> dict:
    """
    Run one context-length point.

    eval_ids is loaded ONCE in main() at max(context_lengths) + decode_buffer.
    We slice [:context_len] so every T uses the same text prefix —
    same document, growing context window, matching the paper benchmark.

    This correctly decouples the two parameters:
        num_eval_tokens = max_context + decode_buffer  (loaded once from dataset)
        prefill_len     = context_len                  (sliced per sweep point)
    """
    import torch

    prefill_ids = eval_ids[:context_len].unsqueeze(0).to(device)
    T = context_len

    # ── Dense decode (unpatched) ─────────────────────────────────────────
    print(f"\n  [T={T}] Measuring DENSE decode ...")
    dense = measure_dense_decode(model, prefill_ids, n_warm, n_meas,
                                 n_trials, device)
    # Free everything dense left on GPU before SphericalKV prefill
    if device.type == "cuda":
        import torch as _torch
        _torch.cuda.empty_cache()

    # ── SphericalKV decode ───────────────────────────────────────────────
    print(f"  [T={T}] Measuring SphericalKV decode ...")
    if pipeline._patched:
        pipeline.uninstall()
    sphkv = measure_sphkv_decode(pipeline, prefill_ids, n_warm, n_meas,
                                  n_trials, device)

    # ── Memory from pipeline state (authoritative) ───────────────────────
    mem = eval_memory_from_pipeline(pipeline, T)

    # ── Combine ──────────────────────────────────────────────────────────
    speedup = sphkv["tok_s_median"] / max(dense["tok_s_median"], 1e-9)

    vram_dense_MB  = dense["peak_vram_prefill"] / 1e6
    vram_sphkv_MB  = sphkv["peak_vram_prefill"] / 1e6
    vram_change    = 100 * (vram_sphkv_MB - vram_dense_MB) / max(vram_dense_MB, 1)

    # Uninstall patches and free compressed KV before next context length
    pipeline.uninstall()
    if device.type == "cuda":
        import torch as _torch
        _torch.cuda.empty_cache()

    return {
        "T": T,
        **mem,
        "prefill_tier_counts": sphkv["tier_counts"],
        "dense_tok_s":             dense["tok_s_median"],
        "sphkv_tok_s":             sphkv["tok_s_median"],
        "speedup":                 speedup,
        "dense_hbm_bytes_per_tok": dense["hbm_bytes_per_tok"],
        "sphkv_hbm_bytes_per_tok": sphkv["hbm_bytes_per_tok"],
        "dense_peak_vram_MB":      vram_dense_MB,
        "sphkv_peak_vram_MB":      vram_sphkv_MB,
        "vram_change_pct":         vram_change,
        "decode_tier_counts":      sphkv["decode_tier_counts"],
        "decode_total":            sphkv["decode_total"],
        "decode_sink_count":       sphkv["decode_sink_count"],
    }


def _fmtx(ax):
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.tick_params(labelsize=FONT_TK)
    ax.set_xlabel("Context length (tokens)", fontsize=FONT_L)


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path.name}")


def plot_kv_bytes_per_tok(tc, results, out):
    dense = [r["kv_bytes_per_tok_dense"] for r in results]
    sphkv = [r["kv_bytes_per_tok_sphkv"] for r in results]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(tc, dense, "o-", color=DENSE_COLOR,  lw=LW, ms=MS, label="Dense FP16")
    ax.plot(tc, sphkv, "s-", color=SPHKV_COLOR,  lw=LW, ms=MS, label="SphericalKV")
    ax.set_ylabel("KV bytes / stored token", fontsize=FONT_L)
    ax.set_title("Memory: KV Bytes per Stored Token", fontsize=FONT_T, fontweight="bold")
    ax.legend(fontsize=FONT_LEG)
    _fmtx(ax)
    _save(fig, out)


def plot_peak_kv_mb(tc, results, out):
    dense = [r["peak_KV_MB_dense"] for r in results]
    sphkv = [r["peak_KV_MB_sphkv"] for r in results]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(tc, dense, "o-", color=DENSE_COLOR, lw=LW, ms=MS, label="Dense FP16")
    ax.plot(tc, sphkv, "s-", color=SPHKV_COLOR, lw=LW, ms=MS, label="SphericalKV")
    ax.fill_between(tc, sphkv, dense, color=SPHKV_COLOR, alpha=0.13, label="Savings")
    ax.set_ylabel("Steady-state KV footprint (MB)", fontsize=FONT_L)
    ax.set_title("Steady-State KV Memory Footprint", fontsize=FONT_T, fontweight="bold")
    ax.legend(fontsize=FONT_LEG)
    _fmtx(ax)
    _save(fig, out)


def plot_peak_vram(tc, results, out):
    dense = [r["dense_peak_vram_MB"] for r in results]
    sphkv = [r["sphkv_peak_vram_MB"] for r in results]
    pcts  = [r["vram_change_pct"]    for r in results]
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax2 = ax1.twinx()
    ax1.plot(tc, dense, "o-", color=DENSE_COLOR, lw=LW, ms=MS, label="Dense FP16 (peak)")
    ax1.plot(tc, sphkv, "s-", color=SPHKV_COLOR, lw=LW, ms=MS, label="SphericalKV (peak)")
    ax2.plot(tc, pcts, "^--", color="gray", lw=1.5, ms=6, label="VRAM Δ%")
    ax2.axhline(0, color="gray", lw=1, ls=":")
    ax2.set_ylabel("Peak VRAM change %\n(positive = SphericalKV uses more)", fontsize=9)
    for x, y, p in zip(tc, pcts, pcts):
        ax2.annotate(f"{p:+.1f}%", (x, y), textcoords="offset points",
                     xytext=(0, 7), ha="center", fontsize=8, color="gray")
    ax1.set_ylabel("Peak VRAM during prefill (MB)", fontsize=FONT_L)
    ax1.set_title("Peak VRAM During Prefill\n"
                  "(SphericalKV higher due to intermediate tensors + codebooks)",
                  fontsize=FONT_T, fontweight="bold")
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, fontsize=FONT_LEG)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax1.tick_params(labelsize=FONT_TK)
    ax1.set_xlabel("Context length (tokens)", fontsize=FONT_L)
    _save(fig, out)


def plot_compression_ratio(tc, results, out):
    ratios_K   = [r["compression_ratio_K"]     for r in results]
    ratios_tot = [r["compression_ratio_total"]  for r in results]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(tc, ratios_K,   "s-", color=SPHKV_COLOR, lw=LW, ms=MS, label="K compression (dense K / compressed K)")
    ax.plot(tc, ratios_tot, "D-", color=DENSE_COLOR,  lw=LW, ms=MS, label="Total KV compression")
    ax.axhline(1.0, color="gray", lw=1.5, ls="--", label="No compression (1×)")
    ax.set_ylabel("Compression ratio  (dense / compressed)", fontsize=FONT_L)
    ax.set_title("K-Cache Compression Ratio", fontsize=FONT_T, fontweight="bold")
    ax.legend(fontsize=FONT_LEG)
    _fmtx(ax)
    _save(fig, out)


def plot_throughput(tc, results, out):
    dense = [r["dense_tok_s"] for r in results]
    sphkv = [r["sphkv_tok_s"] for r in results]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(tc, dense, "o-", color=DENSE_COLOR, lw=LW, ms=MS, label="Dense FP16 (measured)")
    ax.plot(tc, sphkv, "s-", color=SPHKV_COLOR, lw=LW, ms=MS,
            label="SphericalKV (measured, Python ref)")
    ax.set_ylabel("Decode throughput (tok/s)", fontsize=FONT_L)
    ax.set_title("Measured Decode Throughput\n"
                 "(SphericalKV Python impl; fused kernel needed for hardware speed)",
                 fontsize=FONT_T, fontweight="bold")
    ax.legend(fontsize=FONT_LEG)
    _fmtx(ax)
    _save(fig, out)


def plot_speedup(tc, results, out):
    speedups = [r["speedup"] for r in results]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(range(len(tc)), speedups, color=SPHKV_COLOR, alpha=0.85)
    ax.axhline(1.0, color=DENSE_COLOR, lw=2, ls="--", label="Dense baseline (1×)")
    ax.set_xticks(range(len(tc)))
    ax.set_xticklabels([f"{t:,}" for t in tc], fontsize=FONT_TK)
    ax.set_xlabel("Context length (tokens)", fontsize=FONT_L)
    ax.set_ylabel("Speedup  (SphericalKV tok/s  /  Dense tok/s)", fontsize=FONT_L)
    ax.set_title("Measured Decode Speedup\n(< 1 = slower due to Python overhead; "
                 "fused CUDA kernel recovers this)",
                 fontsize=FONT_T, fontweight="bold")
    for i, v in enumerate(speedups):
        ax.text(i, v + 0.01, f"{v:.2f}×", ha="center", fontsize=FONT_TK, fontweight="bold")
    ax.legend(fontsize=FONT_LEG)
    _save(fig, out)


def plot_hbm_bytes(tc, results, out):
    """
    HBM bytes read per GENERATED token = full KV cache read once.
    Dense:   dense_K + dense_V  (from past_key_values tensor sizes)
    SphKV:   compressed_K + retained_V + codebooks  (from per_head_pages sizes)
    """
    dense_mb = [r["dense_hbm_bytes_per_tok"] / 1e6 for r in results]
    sphkv_mb = [r["sphkv_hbm_bytes_per_tok"] / 1e6 for r in results]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(tc, dense_mb, "o-", color=DENSE_COLOR, lw=LW, ms=MS, label="Dense FP16")
    ax.plot(tc, sphkv_mb, "s-", color=SPHKV_COLOR, lw=LW, ms=MS, label="SphericalKV")
    ax.fill_between(tc, sphkv_mb, dense_mb, color=SPHKV_COLOR, alpha=0.13,
                    label="BW savings")
    ax.set_ylabel("HBM read per generated token (MB)\n[full KV cache streamed once]",
                  fontsize=FONT_L)
    ax.set_title("HBM Bandwidth Pressure per Decode Token\n"
                 "(from actual tensor sizes, not proxy)",
                 fontsize=FONT_T, fontweight="bold")
    ax.legend(fontsize=FONT_LEG)
    _fmtx(ax)
    _save(fig, out)

# ── Tier color / label constants (shared by both pie functions) ────────────
_B_TO_LABEL  = {6: "High (6-bit)", 4: "Mid (4-bit)", 3: "Low (3-bit)"}
_TIER_LABELS = ["Sink (High/b1)", "High (6-bit)", "Mid (4-bit)",
                "Low (3-bit)", "Dropped"]
_TIER_PIE_COLORS = {
    "Sink (High/b1)": "#00B050",   # distinct green for sinks
    "High (6-bit)":   "#2DC653",
    "Mid (4-bit)":    "#4361EE",
    "Low (3-bit)":    "#F77F00",
    "Dropped":        "#AAAAAA",
}


def _tier_pie_slices(tc_counts: dict, total_slots: int,
                     sink_count: int) -> dict:
    """
    Build {label: count} for one context-length data point.
    Sinks are a sub-slice of High — they are pulled out separately so
    their marginal size is visible even when High is otherwise tiny.
    """
    high_total = tc_counts.get(6, 0)         # all b1 tokens
    sink_n     = min(int(sink_count), high_total)
    high_non_sink = max(0, high_total - sink_n)
    mid     = tc_counts.get(4, 0)
    low     = tc_counts.get(3, 0)
    retained = high_total + mid + low
    dropped  = max(0, total_slots - retained)
    return {
        "Sink (High/b1)": sink_n,
        "High (6-bit)":   high_non_sink,
        "Mid (4-bit)":    mid,
        "Low (3-bit)":    low,
        "Dropped":        dropped,
    }


def plot_tier_distribution(tc, results, out):
    from config import SINK_TOKENS, NUM_GROUPS
    n = len(tc)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, T, r in zip(axes, tc, results):
        tc_r           = r.get("prefill_tier_counts", r.get("tier_token_counts", {}))
        total_retained = r.get("total_retained", sum(tc_r.values()))
        pct = r.get("retention_pct", None)
        if pct and pct > 0:
            total_slots = total_retained / (pct / 100.0)
        else:
            total_slots = total_retained

        # Sink count: SINK_TOKENS positions × L × H
        L = r.get("T", T)   # fall back
        # We derive L*H from total_slots / T
        lh = int(round(total_slots / T)) if T > 0 else 1
        sink_count = SINK_TOKENS * lh

        slices = _tier_pie_slices(tc_r, int(total_slots), sink_count)

        labels, sizes, colors = [], [], []
        for lbl in _TIER_LABELS:
            v = slices[lbl]
            if v > 0:
                labels.append(lbl)
                sizes.append(v)
                colors.append(_TIER_PIE_COLORS[lbl])

        total = sum(sizes)
        def _autopct(pct_val):
            count = int(round(pct_val / 100 * total))
            if pct_val < 1.5:
                return f"{pct_val:.1f}%\n({count:,})"
            return f"{pct_val:.1f}%"

        wedges, texts, autotexts = ax.pie(
            sizes, labels=None, colors=colors,
            autopct=_autopct, startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 1.2},
            pctdistance=0.78,
        )
        for at in autotexts:
            at.set_fontsize(8)

        ax.legend(wedges, labels, loc="lower center",
                  bbox_to_anchor=(0.5, -0.18),
                  fontsize=8, ncol=2, frameon=False)

        # Annotate sink count explicitly below title
        ax.set_title(
            f"T = {T:,}\n"
            f"(sinks: {sink_count:,} / {int(total_slots):,} slots)",
            fontsize=FONT_T, fontweight="bold", pad=10
        )

    fig.suptitle("Prefill-Time Tier Allocation (pie per context length)",
                 fontsize=FONT_T + 1, fontweight="bold", y=1.02)
    _save(fig, out)


def plot_tier_distribution_decode(tc, results, out):
    """
    Decode-time tier allocation: one pie chart per context length.
    Reads the tier state of _retained_tokens after the full warm+meas window,
    so it reflects any online_refresh updates that ran during decode.
    Staging buffer tokens (unflushed recent decode KV) are included as b3.
    """
    n = len(tc)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, T, r in zip(axes, tc, results):
        dtc        = r.get("decode_tier_counts", {})
        dtotal     = r.get("decode_total", sum(dtc.values()))
        sink_count = r.get("decode_sink_count", 0)

        if dtotal == 0:
            ax.text(0.5, 0.5, "No decode data\n(dry_run?)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=FONT_L, color="gray")
            ax.set_title(f"T = {T:,}", fontsize=FONT_T, fontweight="bold")
            continue

        slices = _tier_pie_slices(dtc, dtotal, sink_count)

        labels, sizes, colors = [], [], []
        for lbl in _TIER_LABELS:
            v = slices[lbl]
            if v > 0:
                labels.append(lbl)
                sizes.append(v)
                colors.append(_TIER_PIE_COLORS[lbl])

        total = sum(sizes)
        def _autopct(pct_val):
            count = int(round(pct_val / 100 * total))
            if pct_val < 1.5:
                return f"{pct_val:.1f}%\n({count:,})"
            return f"{pct_val:.1f}%"

        wedges, texts, autotexts = ax.pie(
            sizes, labels=None, colors=colors,
            autopct=_autopct, startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 1.2},
            pctdistance=0.78,
        )
        for at in autotexts:
            at.set_fontsize(8)

        ax.legend(wedges, labels, loc="lower center",
                  bbox_to_anchor=(0.5, -0.18),
                  fontsize=8, ncol=2, frameon=False)

        ax.set_title(
            f"T = {T:,}\n"
            f"(sinks: {sink_count:,}, decode tokens incl.)",
            fontsize=FONT_T, fontweight="bold", pad=10
        )

    fig.suptitle("Decode-Time Tier Allocation  (after warm+meas window)",
                 fontsize=FONT_T + 1, fontweight="bold", y=1.02)
    _save(fig, out)


def plot_summary_grid(tc, results, out):
    """4-panel presentation-ready summary."""
    dense_kbpt  = [r["kv_bytes_per_tok_dense"]     for r in results]
    sphkv_kbpt  = [r["kv_bytes_per_tok_sphkv"]     for r in results]
    dense_mb    = [r["peak_KV_MB_dense"]            for r in results]
    sphkv_mb    = [r["peak_KV_MB_sphkv"]            for r in results]
    dense_tps   = [r["dense_tok_s"]                 for r in results]
    sphkv_tps   = [r["sphkv_tok_s"]                 for r in results]
    dense_hbm   = [r["dense_hbm_bytes_per_tok"]/1e6 for r in results]
    sphkv_hbm   = [r["sphkv_hbm_bytes_per_tok"]/1e6 for r in results]
    speedups    = [r["speedup"]                     for r in results]
    vram_dense  = [r["dense_peak_vram_MB"]          for r in results]
    vram_sphkv  = [r["sphkv_peak_vram_MB"]          for r in results]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("SphericalKV Cache  vs  Dense FP16 Baseline",
                 fontsize=16, fontweight="bold", y=1.01)

    kd = dict(color=DENSE_COLOR, lw=LW, ms=MS, label="Dense FP16")
    ks = dict(color=SPHKV_COLOR, lw=LW, ms=MS, label="SphericalKV")

    # 1. KV bytes / stored token
    ax = axes[0, 0]
    ax.plot(tc, dense_kbpt, "o-", **kd); ax.plot(tc, sphkv_kbpt, "s-", **ks)
    ax.set_title("KV Bytes / Stored Token", fontsize=FONT_T, fontweight="bold")
    ax.set_ylabel("bytes / token"); ax.legend(fontsize=FONT_LEG)

    # 2. Steady-state KV footprint
    ax = axes[0, 1]
    ax.plot(tc, dense_mb, "o-", **kd); ax.plot(tc, sphkv_mb, "s-", **ks)
    ax.fill_between(tc, sphkv_mb, dense_mb, color=SPHKV_COLOR, alpha=0.12)
    ax.set_title("Steady-State KV Footprint (MB)", fontsize=FONT_T, fontweight="bold")
    ax.set_ylabel("MB"); ax.legend(fontsize=FONT_LEG)

    # 3. Peak VRAM (prefill)
    ax = axes[0, 2]
    ax.plot(tc, vram_dense, "o-", **kd); ax.plot(tc, vram_sphkv, "s-", **ks)
    ax.set_title("Peak VRAM During Prefill (MB)\n"
                 "(SphKV higher = intermediate tensors)",
                 fontsize=FONT_T, fontweight="bold")
    ax.set_ylabel("MB"); ax.legend(fontsize=FONT_LEG)

    # 4. Throughput (measured)
    ax = axes[1, 0]
    ax.plot(tc, dense_tps, "o-", **kd); ax.plot(tc, sphkv_tps, "s-", **ks)
    ax.set_title("Decode Throughput  (tok/s)\n[measured]",
                 fontsize=FONT_T, fontweight="bold")
    ax.set_ylabel("tok/s"); ax.legend(fontsize=FONT_LEG)

    # 5. HBM bytes / generated token (from tensor sizes)
    ax = axes[1, 1]
    ax.plot(tc, dense_hbm, "o-", **kd); ax.plot(tc, sphkv_hbm, "s-", **ks)
    ax.fill_between(tc, sphkv_hbm, dense_hbm, color=SPHKV_COLOR, alpha=0.12)
    ax.set_title("HBM Read / Generated Token (MB)\n[from actual tensor sizes]",
                 fontsize=FONT_T, fontweight="bold")
    ax.set_ylabel("MB / token"); ax.legend(fontsize=FONT_LEG)

    # 6. Speedup bar
    ax = axes[1, 2]
    colors = [SPHKV_COLOR if s >= 1.0 else "#FF6B6B" for s in speedups]
    ax.bar(range(len(tc)), speedups, color=colors, alpha=0.85)
    ax.axhline(1.0, color=DENSE_COLOR, lw=2, ls="--")
    ax.set_xticks(range(len(tc)))
    ax.set_xticklabels([f"{t:,}" for t in tc], fontsize=FONT_TK)
    for i, v in enumerate(speedups):
        ax.text(i, v + 0.01, f"{v:.2f}×", ha="center",
                fontsize=FONT_TK, fontweight="bold")
    ax.set_title("Measured Speedup  (SphKV / Dense)\n"
                 "[red = slower due to Python overhead]",
                 fontsize=FONT_T, fontweight="bold")
    ax.set_ylabel("Speedup"); ax.set_xlabel("Context length (tokens)", fontsize=FONT_L)

    for ax in axes.flat:
        if ax != axes[1, 2]:
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
            ax.set_xlabel("Context length (tokens)", fontsize=FONT_L)
        ax.tick_params(labelsize=FONT_TK)

    _save(fig, out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--context_lengths", type=int, nargs="+",
                   default=[512, 1024, 2048, 4096],
                   help="Context lengths to sweep (= prefill_len). "
                        "THESE are the x-axis values.")
    p.add_argument("--output_dir",          default="sweep_results")
    p.add_argument("--dry_run",             action="store_true")
    # Model / data
    p.add_argument("--model_name_or_path",  default="meta-llama/Llama-3.2-1B")
    p.add_argument("--codebook_dir",
                   default="llama_3.2_1B_codebooks/codebooks_llama_1b")
    p.add_argument("--dataset",             default="pg19")
    p.add_argument("--device",              default="cuda")
    # Timing
    p.add_argument("--n_warm",   type=int,  default=2)
    p.add_argument("--n_meas",   type=int,  default=16)
    p.add_argument("--n_trials", type=int,  default=3)
    # Dry-run model config (ignored when --dry_run is False)
    p.add_argument("--dry_run_L",  type=int, default=16)
    p.add_argument("--dry_run_H",  type=int, default=8)
    p.add_argument("--dry_run_dh", type=int, default=64)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tc      = sorted(set(args.context_lengths))

    results: List[dict] = []

    if args.dry_run:
        model_cfg = {"L": args.dry_run_L, "H": args.dry_run_H, "dh": args.dry_run_dh}
        print("\n[dry_run] Generating synthetic metrics ...")
        for T in tc:
            r = synthetic_run(T, model_cfg)
            results.append(r)
            print(f"  T={T:5d}  kv/tok dense={r['kv_bytes_per_tok_dense']:.0f}"
                  f"  sphkv={r['kv_bytes_per_tok_sphkv']:.0f}"
                  f"  ratio={r['compression_ratio_K']:.1f}x"
                  f"  tps_dense={r['dense_tok_s']:.0f}"
                  f"  tps_sphkv={r['sphkv_tok_s']:.1f}"
                  f"  speedup={r['speedup']:.2f}x"
                  f"  vram_Δ={r['vram_change_pct']:+.1f}%")
    else:
        import torch
        import sys
        _HERE = str(Path(__file__).parent)
        if _HERE not in sys.path:
            sys.path.insert(0, _HERE)

        from evaluate import load_model_and_tokenizer
        from codebook_loader import load_codebooks
        from spherical_kv_pipeline import SphericalKVPipeline
        from config import GROUP_SIZE

        device = torch.device(args.device)
        model, tokenizer = load_model_and_tokenizer(
            args.model_name_or_path, device)
        cfg          = model.config
        num_layers   = cfg.num_hidden_layers
        num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        head_dim     = getattr(cfg, "head_dim",
                               cfg.hidden_size // cfg.num_attention_heads)
        num_groups   = head_dim // GROUP_SIZE

        from tiers import build_tiers
        tiers_list   = build_tiers(head_dim)
        codebooks    = load_codebooks(
            args.codebook_dir, num_layers, num_kv_heads, tiers_list)

        pipeline = SphericalKVPipeline(
            model=model, tokenizer=tokenizer,
            codebooks=codebooks, device=device,
            head_dim=head_dim, group_size=GROUP_SIZE,
            sink_tokens=4,
            use_fused=(device.type == "cuda"),
        )

        # Load corpus ONCE at max context length + decode buffer
        # num_eval_tokens = max context + n_warm + n_meas (decode buffer)
        # prefill_len     = T  (sliced per run inside real_run)
        # Same text prefix is used for every context length: [:512], [:1024], etc.
        from evaluate import get_eval_tokens
        max_context    = max(tc)
        num_eval_tokens = max_context + args.n_warm + args.n_meas + 16
        print(f"\nLoading corpus: {num_eval_tokens} tokens from '{args.dataset}' "
              f"(max context={max_context}, decode buffer={args.n_warm + args.n_meas + 16})")
        eval_ids = get_eval_tokens(tokenizer, None, args.dataset, num_eval_tokens)
        print(f"Corpus loaded: {eval_ids.numel()} tokens  "
              f"(sweep will slice [:T] for each T in {tc})\n")

        for T in tc:
            print(f"\n{'='*60}\n  Context length: T={T}  (prefill_len={T}, "
                  f"num_eval_tokens={eval_ids.numel()})\n{'='*60}")
            r = real_run(model, pipeline, eval_ids, T,
                         args.n_warm, args.n_meas, args.n_trials, device)
            results.append(r)
            print(f"  kv/tok  dense={r['kv_bytes_per_tok_dense']:.0f}"
                  f"  sphkv={r['kv_bytes_per_tok_sphkv']:.0f}")
            print(f"  ratio_K={r['compression_ratio_K']:.1f}×"
                  f"  ratio_total={r['compression_ratio_total']:.1f}×")
            print(f"  tps  dense={r['dense_tok_s']:.1f}"
                  f"  sphkv={r['sphkv_tok_s']:.1f}"
                  f"  speedup={r['speedup']:.3f}×")
            print(f"  HBM/tok  dense={r['dense_hbm_bytes_per_tok']/1e6:.2f} MB"
                  f"  sphkv={r['sphkv_hbm_bytes_per_tok']/1e6:.2f} MB")
            print(f"  VRAM  dense={r['dense_peak_vram_MB']:.0f} MB"
                  f"  sphkv={r['sphkv_peak_vram_MB']:.0f} MB"
                  f"  Δ={r['vram_change_pct']:+.1f}%")

    with open(out_dir / "results.json", "w") as f:
        json.dump({"context_lengths": tc, "results": results}, f, indent=2)
    print(f"\nSaved results.json")

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\nRendering plots …")
    plot_kv_bytes_per_tok(tc, results, out_dir / "plot_memory_kv_bytes_per_tok.png")
    plot_peak_kv_mb(tc, results,       out_dir / "plot_memory_peak_mb.png")
    plot_peak_vram(tc, results,        out_dir / "plot_peak_vram_mb.png")
    plot_compression_ratio(tc, results,out_dir / "plot_compression_ratio.png")
    plot_throughput(tc, results,       out_dir / "plot_throughput_tokps.png")
    plot_speedup(tc, results,          out_dir / "plot_speedup.png")
    plot_hbm_bytes(tc, results,        out_dir / "plot_hbm_bytes_per_tok.png")
    # plot_tier_distribution(tc, results,out_dir / "plot_tier_distribution.png")
    plot_tier_distribution(tc, results, out_dir / "plot_tier_distribution.png")
    plot_tier_distribution_decode(tc, results, out_dir / "plot_tier_distribution_decode.png")  # NEW
    plot_summary_grid(tc, results,     out_dir / "summary_grid.png")

    print(f"\n{'='*60}")
    print(f"  All outputs → {out_dir.resolve()}")
    for p2 in sorted(out_dir.glob("*.png")):
        print(f"    {p2.name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()