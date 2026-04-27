from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import config as _cfg
from evaluate import load_model_and_tokenizer, _load_dataset_tokens
from codebook_loader import load_codebooks
from tiers import build_tiers
from negative_controls import (
    MODES, recon_attention_batched, reconstruct_dense_K_from_codes,
)


class HBMProfiler:
    """
    Measures HBM bytes/token during decode.
    Uses torch.cuda memory stats as a proxy.
    For hardware DRAM counters, use ncu externally:
        ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum ...
    """
    def __init__(self, device, enabled=True):
        self.device = device
        self.enabled = enabled and device.type == "cuda"

    def start(self):
        if not self.enabled:
            return
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self._allocated_before = torch.cuda.memory_allocated(self.device)
        self._peak_before = torch.cuda.max_memory_allocated(self.device)

    def stop(self, n_tokens: int) -> dict:
        if not self.enabled:
            return {"hbm_bytes_per_tok": 0.0, "peak_delta_MB": 0.0}
        torch.cuda.synchronize(self.device)
        peak_after = torch.cuda.max_memory_allocated(self.device)
        alloc_after = torch.cuda.memory_allocated(self.device)
        peak_delta = peak_after - self._peak_before
        alloc_delta = alloc_after - self._allocated_before
        return {
            "hbm_bytes_per_tok": peak_delta / max(n_tokens, 1),
            "peak_delta_MB":     peak_delta / 1e6,
            "alloc_delta_MB":    alloc_delta / 1e6,
        }



@torch.no_grad()
def measure_dense_decode(model, eval_ids, T, n_warm, n_meas, n_trials, device):

    from transformers import DynamicCache

    prefill_ids = eval_ids[:, :T].to(device)
    all_tps = []
    all_nll = []
    all_generated = []

    def _fresh_cache():
        # Prefill with the full prompt -> primed KV cache.
        out = model(input_ids=prefill_ids, use_cache=True, return_dict=True)
        return out.past_key_values, out.logits[:, -1, :]

    for _ in range(n_trials):
        cache, last_logits = _fresh_cache()
        last_tok = eval_ids[:, T:T+1].to(device)
        true_len = T

        for wi in range(n_warm):
            out = model(input_ids=last_tok, past_key_values=cache,
                        use_cache=True, return_dict=True)
            cache = out.past_key_values
            last_tok = eval_ids[:, T + 1 + wi : T + 2 + wi].to(device)
            true_len += 1

        if device.type == "cuda":
            torch.cuda.synchronize()
            ts = torch.cuda.Event(enable_timing=True)
            te = torch.cuda.Event(enable_timing=True)
            ts.record()

        nll_sum = 0.0
        gen_ids = []
        for step in range(n_meas):
            out = model(input_ids=last_tok, past_key_values=cache,
                        use_cache=True, return_dict=True)
            cache = out.past_key_values
            logits = out.logits[:, -1, :]

            ref_idx = T + n_warm + 1 + step
            ref_tok = eval_ids[0, ref_idx].item()

            log_probs = F.log_softmax(logits, dim=-1)
            nll_sum -= log_probs[0, ref_tok].item()

            gen_ids.append(ref_tok)
            last_tok = eval_ids[:, ref_idx:ref_idx + 1].to(device)
            true_len += 1

        if device.type == "cuda":
            te.record()
            torch.cuda.synchronize()
            elapsed = ts.elapsed_time(te) / 1e3
        else:
            elapsed = 1.0

        tps = n_meas / max(elapsed, 1e-9)
        all_tps.append(tps)
        all_nll.append(nll_sum / n_meas)
        all_generated.append(gen_ids)

        # Free cache between trials so VRAM accounting is clean.
        del cache
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if device.type == "cuda":
        vram_mb = torch.cuda.max_memory_allocated(device) / 1e6
    else:
        vram_mb = 0.0

    cfg = model.config
    num_kv  = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = getattr(cfg, "head_dim",
                       cfg.hidden_size // cfg.num_attention_heads)
    # Dense bKV (bits per token): 2 tensors (K, V) x 16 bits fp16 x num_layers
    # x num_kv_heads x head_dim.  This is the per-token KV footprint.
    bKV_per_tok_bits = 2 * 16 * cfg.num_hidden_layers * num_kv * head_dim
    return {
        "mode":      "dense",
        "tok_s":     statistics.median(all_tps),
        "tok_s_all": all_tps,
        "nll":       statistics.median(all_nll),
        "ppl":       math.exp(statistics.median(all_nll)),
        "nll_all":   all_nll,
        "generated": all_generated,
        "vram_MB":   vram_mb,
        "T":         T,
        "bKV":       bKV_per_tok_bits / 8,
    }


@torch.no_grad()
def measure_sphkv_decode(
    mode:     str,
    model,
    pipeline, 
    eval_ids: torch.Tensor,
    T:        int,
    n_warm:   int,
    n_meas:   int,
    n_trials: int,
    device:   torch.device,
    bpt:      float = 30.9,
):
    """
    Measure SphericalKV variants: tok/s, NLL, memory, tier distribution.
    Handles modes: sphkv, sphkv_recon, sphkv_angle, sphkv_rd
    """
    prefill_ids = eval_ids[:, :T].to(device)

    # Set budget
    _cfg.BITS_PER_TOKEN = bpt
    _cfg.GLOBAL_BUDGET_BITS = bpt * T * pipeline.num_layers * pipeline.num_kv_heads
    # _cfg.GLOBAL_BUDGET_BITS = bpt * T * pipeline.num_layers * pipeline.num_kv_heads

    all_tps = []
    all_nll = []
    all_generated = []

    for trial in range(n_trials):
        # Fresh prefill each trial
        pipeline.prefill(prefill_ids)

        # First decode token must be eval_ids[T] at position T — exactly
        # mirroring the dense baseline.  Re-feeding prefill_ids[:, -1:]
        # would (a) duplicate the last prefill token and (b) shift every
        # subsequent token one position forward in the spherical cache,
        # creating a corrupted context the model has never seen.
        last_tok = eval_ids[:, T:T + 1].to(device)

        for wi in range(n_warm):
            out = model(input_ids=last_tok,
                        use_cache=False, return_dict=True)
            last_tok = eval_ids[:, T + 1 + wi : T + 2 + wi].to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
            ts = torch.cuda.Event(enable_timing=True)
            te = torch.cuda.Event(enable_timing=True)
            ts.record()

        nll_sum = 0.0
        gen_ids = []
        for step in range(n_meas):
            out = model(input_ids=last_tok,
                        use_cache=False, return_dict=True)
            logits = out.logits[:, -1, :]

            ref_idx = T + n_warm + 1 + step
            ref_tok_id = eval_ids[0, ref_idx].item()

            log_probs = F.log_softmax(logits, dim=-1)
            nll_sum -= log_probs[0, ref_tok_id].item()

            gen_ids.append(ref_tok_id)
            last_tok = eval_ids[:, ref_idx:ref_idx + 1].to(device)

        if device.type == "cuda":
            te.record()
            torch.cuda.synchronize()
            elapsed = ts.elapsed_time(te) / 1e3
        else:
            elapsed = 1.0

        tps = n_meas / max(elapsed, 1e-9)
        all_tps.append(tps)
        all_nll.append(nll_sum / n_meas)
        all_generated.append(gen_ids)

    # Memory measurement
    mem = _measure_memory(pipeline, T)

    # Tier distribution
    tier_counts = _count_tiers(pipeline)

    # HBM proxy
    hbm_prof = HBMProfiler(device)
    hbm_prof.start()
    # One more measurement pass for HBM
    current_ids = prefill_ids.clone()
    for _ in range(min(n_meas, 16)):
        out = model(input_ids=current_ids[:, -1:],
                    use_cache=False, return_dict=True)
        nid = out.logits[:, -1, :].argmax(-1, keepdim=True)
        current_ids = torch.cat([current_ids, nid], dim=-1)
    hbm = hbm_prof.stop(min(n_meas, 16))

    if device.type == "cuda":
        vram_mb = torch.cuda.max_memory_allocated(device) / 1e6
    else:
        vram_mb = 0.0

    pipeline.uninstall()

    return {
        "mode":          mode,
        "budget_bpt":    bpt,
        "tok_s":         statistics.median(all_tps),
        "tok_s_all":     all_tps,
        "tok_s_p50":     sorted(all_tps)[len(all_tps) // 2] if all_tps else 0,
        "tok_s_p95":     sorted(all_tps)[int(len(all_tps) * 0.95)] if len(all_tps) > 1 else (all_tps[0] if all_tps else 0),
        "nll":           statistics.median(all_nll),
        "ppl":           math.exp(statistics.median(all_nll)),
        "nll_all":       all_nll,
        "generated":     all_generated,
        "vram_MB":       vram_mb,
        "T":             T,
        "bKV":           mem["bKV"],
        "peak_KV_GB":    mem["peak_KV_GB"],
        "page_overhead_bytes": mem["page_overhead"],
        "hbm_bytes_per_tok":  hbm["hbm_bytes_per_tok"],
        "tier_counts":   tier_counts,
    }


def _measure_memory(pipeline, T: int) -> dict:
    """Compute effective bKV including all metadata (App A.2)."""
    total_payload = 0
    total_overhead = 0
    total_V = 0

    for (layer, head), tier_list in pipeline.per_head_pages.items():
        for entry in tier_list:
            # Entry layout (>=6 slots, 7 with positions):
            #   [0] pages_tensor, [1] ptable_tensor, [2] b_theta,
            #   [3] n_tokens, [4] V_tier, [5] (r,theta) codes, [6] positions
            pages_tensor  = entry[0]
            ptable_tensor = entry[1]
            V_tier        = entry[4]
            total_payload  += pages_tensor.numel()
            total_overhead += ptable_tensor.numel() * 4  # int32
            total_V        += V_tier.numel() * 2 if V_tier is not None else 0

    total_bytes = total_payload + total_overhead + total_V
    return {
        "bKV":          total_bytes / max(T, 1),
        "peak_KV_GB":   total_bytes / 1e9,
        "page_overhead": total_overhead,
        "payload_bytes": total_payload,
        "V_bytes":       total_V,
    }


def _count_tiers(pipeline) -> dict:
    """Count tokens per tier from retained tokens."""
    counts = defaultdict(int)
    for ts in pipeline._retained_tokens:
        counts[ts.new_tier_id] += 1
    # Add sinks
    counts["sink"] = pipeline.sink_tokens * pipeline.num_layers * pipeline.num_kv_heads
    return dict(counts)



def compute_stability_metrics(
    dense_results: dict,
    sphkv_results: dict,
    n_seeds:       int = 1,
) -> dict:
    """
    Compute stability metrics from multi-seed results.

    S_traj: variance of NLL across seeds
    DeltaT: mean |len(sphkv_generated) - len(dense_generated)| across seeds
    """
    # S_traj: variance of quality across seeds
    if len(sphkv_results.get("nll_all", [])) > 1:
        s_traj = statistics.variance(sphkv_results["nll_all"])
    else:
        s_traj = 0.0

    # DeltaT: length drift
    dense_gens = dense_results.get("generated", [[]])
    sphkv_gens = sphkv_results.get("generated", [[]])

    length_drifts = []
    for dg, sg in zip(dense_gens, sphkv_gens):
        length_drifts.append(abs(len(sg) - len(dg)))
    delta_t = sum(length_drifts) / max(len(length_drifts), 1)

    # NLL drift per seed
    dense_nlls = dense_results.get("nll_all", [0.0])
    sphkv_nlls = sphkv_results.get("nll_all", [0.0])
    nll_gaps = []
    for dn, sn in zip(dense_nlls, sphkv_nlls):
        nll_gaps.append(abs(sn - dn))
    mean_nll_gap = sum(nll_gaps) / max(len(nll_gaps), 1)

    return {
        "S_traj":      s_traj,
        "DeltaT":      delta_t,
        "mean_nll_gap": mean_nll_gap,
        "nll_gaps":    nll_gaps,
    }



def build_iso_quality_frontier(
    all_results: List[dict],
    delta:       float = 0.8,
) -> dict:
    # Find dense baseline
    dense_pts = [r for r in all_results if r["mode"] == "dense"]
    if not dense_pts:
        return {"error": "no dense baseline found"}

    # Best dense quality (lowest NLL)
    best_dense = min(dense_pts, key=lambda r: r["nll"])
    nll_dense = best_dense["nll"]
    tps_dense = best_dense["tok_s"]
    bkv_dense = best_dense["bKV"]

    # Quality threshold: NLL <= nll_dense + delta
    nll_threshold = nll_dense + delta

    # Filter quality-matched points
    retained = []
    excluded = []
    for r in all_results:
        q_gap = r["nll"] - nll_dense
        r["q_gap"] = q_gap
        r["quality_matched"] = (r["nll"] <= nll_threshold)
        if r["quality_matched"]:
            retained.append(r)
        else:
            excluded.append(r)

    # Group by mode for Pareto computation
    mode_points = defaultdict(list)
    for r in retained:
        mode_points[r["mode"]].append(r)

    # Compute Pareto envelope per mode: non-dominated in (bKV↓, tok/s↑)
    frontiers = {}
    for mode, pts in mode_points.items():
        pts.sort(key=lambda p: p["bKV"])
        pareto = []
        best_tps = -1
        for p in pts:
            if p["tok_s"] > best_tps:
                pareto.append(p)
                best_tps = p["tok_s"]
        frontiers[mode] = pareto

    # Iso-quality speedup: max(s_sphkv) / s_dense among matched points
    sphkv_retained = [r for r in retained if r["mode"] != "dense"]
    if sphkv_retained:
        best_sphkv_tps = max(r["tok_s"] for r in sphkv_retained)
        iso_quality_speedup = best_sphkv_tps / max(tps_dense, 1e-9)
    else:
        iso_quality_speedup = 1.0

    # Iso-throughput memory reduction: min bKV among points with tok/s >= tps_dense
    fast_enough = [r for r in sphkv_retained if r["tok_s"] >= tps_dense]
    if fast_enough:
        best_bkv = min(r["bKV"] for r in fast_enough)
        iso_throughput_mem_reduction = bkv_dense / max(best_bkv, 1)
    else:
        iso_throughput_mem_reduction = 1.0

    # Synergy gap (A2 non-additivity test)
    synergy = _compute_synergy_gap(all_results, nll_dense, delta)

    return {
        "nll_dense":                nll_dense,
        "tps_dense":                tps_dense,
        "bKV_dense":                bkv_dense,
        "delta":                    delta,
        "n_retained":               len(retained),
        "n_excluded":               len(excluded),
        "iso_quality_speedup":      iso_quality_speedup,
        "iso_throughput_mem_reduction": iso_throughput_mem_reduction,
        "frontiers":                {k: [(p["bKV"], p["tok_s"], p["nll"], p["mode"])
                                         for p in v]
                                     for k, v in frontiers.items()},
        "synergy_gap":              synergy,
        "all_retained":             [(r["mode"], r.get("budget_bpt", 0),
                                      r["bKV"], r["tok_s"], r["nll"], r["q_gap"])
                                     for r in retained],
    }


def _compute_synergy_gap(results, nll_dense, delta, beta=1.0):
    nll_thresh = nll_dense + delta

    def psi(r):
        q = -r["nll"]  # higher is better
        s = r["tok_s"]
        return q + beta * math.log(max(s, 1e-9))

    # Group by mode
    by_mode = defaultdict(list)
    for r in results:
        if r["nll"] <= nll_thresh:
            by_mode[r["mode"]].append(r)

    psi_joint = max((psi(r) for r in by_mode.get("sphkv", [])), default=float("-inf"))
    psi_angle = max((psi(r) for r in by_mode.get("sphkv_angle", [])), default=float("-inf"))
    psi_rd    = max((psi(r) for r in by_mode.get("sphkv_rd", [])), default=float("-inf"))
    psi_recon = max((psi(r) for r in by_mode.get("sphkv_recon", [])), default=float("-inf"))

    return {
        "psi_joint":  psi_joint,
        "psi_angle":  psi_angle,
        "psi_rd":     psi_rd,
        "psi_recon":  psi_recon,
        "synergy":    psi_joint - max(psi_angle, psi_rd) if psi_joint > float("-inf") else 0.0,
    }



def plot_frontier(frontier: dict, output_dir: Path):
    """Plot iso-quality Pareto frontier."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not available, skipping plots")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    colors = {
        "dense": "#2196F3", "sphkv": "#FF5722", "sphkv_recon": "#4CAF50",
        "sphkv_angle": "#FF9800", "sphkv_rd": "#9C27B0",
    }
    markers = {
        "dense": "o", "sphkv": "D", "sphkv_recon": "s",
        "sphkv_angle": "^", "sphkv_rd": "v",
    }

    for mode, points in frontier.get("frontiers", {}).items():
        if not points:
            continue
        bkvs = [p[0] for p in points]
        tpss = [p[1] for p in points]
        label = MODES.get(mode, mode)
        ax.plot(bkvs, tpss, '-o', color=colors.get(mode, "#666"),
                marker=markers.get(mode, "o"), label=label, markersize=8)

    ax.set_xlabel("Effective KV bytes/token (lower is better)", fontsize=12)
    ax.set_ylabel("Decode throughput (tok/s, higher is better)", fontsize=12)
    ax.set_title(f"Iso-quality Pareto Frontier (delta={frontier.get('delta', 0.8)})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add annotation
    speedup = frontier.get("iso_quality_speedup", 1.0)
    mem_red = frontier.get("iso_throughput_mem_reduction", 1.0)
    ax.text(0.02, 0.02,
            f"Iso-Q speedup: {speedup:.2f}x\n"
            f"Iso-S mem reduction: {mem_red:.1f}x",
            transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.tight_layout()
    path = output_dir / "plot_pareto_frontier.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  -> {path}")


def plot_budget_sweep(all_results: List[dict], output_dir: Path):
    """Plot budget vs quality and budget vs throughput."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Group by mode
    by_mode = defaultdict(list)
    for r in all_results:
        if "budget_bpt" in r:
            by_mode[r["mode"]].append(r)

    colors = {
        "sphkv": "#FF5722", "sphkv_recon": "#4CAF50",
        "sphkv_angle": "#FF9800", "sphkv_rd": "#9C27B0",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for mode, pts in by_mode.items():
        pts.sort(key=lambda p: p["budget_bpt"])
        budgets = [p["budget_bpt"] for p in pts]
        nlls    = [p["nll"] for p in pts]
        tpss    = [p["tok_s"] for p in pts]

        c = colors.get(mode, "#666")
        ax1.plot(budgets, nlls, '-o', color=c, label=mode)
        ax2.plot(budgets, tpss, '-o', color=c, label=mode)

    ax1.set_xlabel("Budget (bits/token)")
    ax1.set_ylabel("NLL (lower is better)")
    ax1.set_title("Budget vs Quality")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Budget (bits/token)")
    ax2.set_ylabel("tok/s (higher is better)")
    ax2.set_title("Budget vs Throughput")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "plot_budget_sweep.png", dpi=150)
    plt.close(fig)
    print(f"  -> plot_budget_sweep.png")


def parse_args():
    p = argparse.ArgumentParser(
        description="SphericalKV experiment runner (Sections 3-4)")

    # Models (Section 3.2: Llama-3.1-8B, Qwen2.5-14B, gpt-oss-20b)
    p.add_argument("--models", nargs="+",
                   default=["meta-llama/Llama-3.1-8B-Instruct"],
                   help="Model name(s) or path(s). Paper uses: "
                        "meta-llama/Llama-3.1-8B-Instruct "
                        "Qwen/Qwen2.5-14B-Instruct "
                        "gpt-oss-20b")
    p.add_argument("--codebook_dirs", nargs="+",
                   default=["codebooks/codebooks_llama_8b"],
                   help="Codebook dir per model (same order as --models)")
    p.add_argument("--device", default="cuda")

    # Workload selection
    p.add_argument("--workloads", nargs="+",
                   default=["w1"],
                   choices=["w1", "w2", "w3"],
                   help="W1=PG-19 LM, W2=LongBench QA, W3=Agentic rollouts")
    p.add_argument("--dataset", default="pg19",
                   help="Dataset for W1 (pg19 or wikitext)")
    p.add_argument("--context_lengths", type=int, nargs="+",
                   default=[8192, 32768],
                   help="Paper uses: 8192 32768 131072")
    p.add_argument("--num_eval_tokens", type=int, default=0,
                   help="0 = auto from context_length + decode buffer")

    # Modes to run (Section 4.2 ablations + Section 3.6 baselines)
    ALL_MODES = list(MODES.keys()) + [
        "streaming_llm", "h2o", "quant_2bit", "quant_4bit", "quant_8bit",
        "keepdrop", "quant_only", "decoupled",
        "uniform_head", "noseg", "nogate",
    ]
    p.add_argument("--modes", nargs="+",
                   default=["dense", "sphkv", "sphkv_recon",
                            "sphkv_angle", "sphkv_rd"],
                   choices=ALL_MODES)

    # Budget sweep (bits per token)
    p.add_argument("--budgets", type=float, nargs="+",
                   default=[20, 25, 30, 35, 40, 50, 60],
                   help="Bits-per-token budget sweep points")

    # Measurement
    p.add_argument("--n_warm",   type=int, default=8)
    p.add_argument("--n_meas",   type=int, default=32)
    p.add_argument("--n_trials", type=int, default=3)
    p.add_argument("--n_seeds",  type=int, default=1,
                   help="Seeds for stability analysis (>1 enables S_traj/DeltaT)")

    # Frontier
    p.add_argument("--delta", type=float, default=0.8,
                   help="Iso-quality tolerance for Pareto frontier")

    # Output
    p.add_argument("--output_dir", default="experiment_results")

    # W2 options
    p.add_argument("--w2_task", default="hotpotqa",
                   choices=["hotpotqa", "2wikimqa"])
    p.add_argument("--w2_max_samples", type=int, default=50)
    p.add_argument("--w2_distractors", type=int, nargs="+", default=[0, 3, 5],
                   help="Distractor counts for W2 sweep")
    p.add_argument("--w2_positions", nargs="+",
                   default=["early", "middle", "late"],
                   help="Answer positions for W2 sweep")

    # W3 options
    p.add_argument("--w3_source", default="toolbench",
                   choices=["toolbench", "agentbench"])
    p.add_argument("--w3_max_episodes", type=int, default=30)
    p.add_argument("--w3_max_steps", type=int, default=10)
    p.add_argument("--w3_seeds", type=int, default=3,
                   help="Seeds per episode for trajectory stability")

    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pad codebook_dirs to match models
    codebook_dirs = list(args.codebook_dirs)
    while len(codebook_dirs) < len(args.models):
        codebook_dirs.append(codebook_dirs[-1])

    grand_results = []   # all results across all models

    for model_idx, model_path in enumerate(args.models):
        cb_dir = codebook_dirs[model_idx]
        model_tag = model_path.split("/")[-1]

        print(f"\n{'#'*70}")
        print(f"  Model: {model_path}")
        print(f"  Codebooks: {cb_dir}")
        print(f"{'#'*70}")

        model, tokenizer = load_model_and_tokenizer(model_path, device)
        cfg = model.config
        num_layers   = cfg.num_hidden_layers
        num_kv_heads = getattr(cfg, "num_key_value_heads",
                               cfg.num_attention_heads)
        head_dim     = getattr(cfg, "head_dim",
                               cfg.hidden_size // cfg.num_attention_heads)

        tiers_list = build_tiers(head_dim)
        codebooks  = load_codebooks(cb_dir, num_layers,
                                    num_kv_heads, tiers_list)

        all_results = []   # results for this model

        # Table 2: Hardware audit
        from hardware_audit import generate_hardware_table, print_hardware_table
        hw_info = generate_hardware_table(device)
        hw_info["model"] = model_tag
        hw_info["num_layers"] = num_layers
        hw_info["num_kv_heads"] = num_kv_heads
        hw_info["head_dim"] = head_dim
        print_hardware_table(hw_info)
        all_results.append({"type": "hardware_audit", "model": model_tag, **hw_info})

        # all_results = []   # results for this model

        if "w1" in args.workloads:
            print(f"\n{'='*60}")
            print(f"  W1: Long-context LM ({args.dataset})")
            print(f"{'='*60}")

            max_T = max(args.context_lengths)
            needed = max_T + args.n_warm + args.n_meas + 64
            eval_ids_1d = _load_dataset_tokens(tokenizer, args.dataset, needed)
            eval_ids = eval_ids_1d[:needed].unsqueeze(0)
            print(f"  Loaded {eval_ids.shape[1]} tokens")

            for T in sorted(args.context_lengths):
                print(f"\n  --- Context T={T} ---")

                # Dense baseline
                if "dense" in args.modes:
                    print(f"  [T={T}] Dense ...")
                    dr = measure_dense_decode(
                        model, eval_ids, T,
                        args.n_warm, args.n_meas, args.n_trials, device)
                    dr["model"] = model_tag
                    dr["workload"] = "w1"
                    all_results.append(dr)
                    print(f"    tok/s={dr['tok_s']:.1f}  NLL={dr['nll']:.4f}  PPL={dr['ppl']:.2f}  bKV={dr['bKV']:.0f}")

                # External baselines (streaming_llm, h2o, quant_Xbit)
                from baselines import BASELINE_MODES, run_baseline
                baseline_modes = [m for m in args.modes if m in BASELINE_MODES]
                for bmode in baseline_modes:
                    print(f"  [T={T}] {bmode} ...")
                    try:
                        br = run_baseline(bmode, model, eval_ids, T,
                                          args.n_warm, args.n_meas, device)
                        br["model"] = model_tag
                        br["workload"] = "w1"
                        all_results.append(br)
                        print(f"    tok/s={br['tok_s']:.1f}  NLL={br['nll']:.4f}  "
                              f"bKV={br['bKV']:.0f}")
                    except Exception as e:
                        print(f"    ERROR: {e}")

                # SphericalKV modes + ablation modes x budgets
                from ablation_modes import (ABLATION_MODES, apply_ablation_mode,
                                            restore_ablation_mode)
                sphkv_modes = [m for m in args.modes
                               if m not in ("dense",) and m not in BASELINE_MODES]
                # Modes that need an effectively unbounded budget so RDR does
                # not drop tokens (the ablation hook does the real work).
                _RETAIN_ALL_MODES = {"quant_only", "sphkv_angle"}
                for mode in sphkv_modes:
                    for bpt in args.budgets:
                        print(f"  [T={T}] {mode} @ {bpt:.0f} bpt ...")
                        from spherical_kv_pipeline import SphericalKVPipeline
                        pipeline = SphericalKVPipeline(
                            model=model, tokenizer=tokenizer,
                            codebooks=codebooks, device=device,
                            head_dim=head_dim,
                            group_size=_cfg.GROUP_SIZE,
                            sink_tokens=_cfg.SINK_TOKENS,
                            use_fused=(device.type == "cuda"))

                        # sphkv_recon / sphkv_rd use reconstruct-then-dot at
                        # decode time -- the flag is read inside the pipeline.
                        if mode in ("sphkv_recon", "sphkv_rd"):
                            pipeline._use_recon = True

                        bpt_eff = 9999.0 if mode in _RETAIN_ALL_MODES else bpt

                        saved_abl = {}
                        if mode in ABLATION_MODES:
                            saved_abl = apply_ablation_mode(mode, pipeline)

                        try:
                            r = measure_sphkv_decode(
                                mode, model, pipeline, eval_ids, T,
                                args.n_warm, args.n_meas, args.n_trials,
                                device, bpt_eff)

                            r["model"] = model_tag
                            r["workload"] = "w1"
                            all_results.append(r)
                            print(f"    tok/s={r['tok_s']:.1f}  "
                                  f"NLL={r['nll']:.4f}  PPL={r['ppl']:.2f}  bKV={r['bKV']:.0f}")
                        except Exception as e:
                            print(f"    ERROR: {e}")
                        finally:
                            if saved_abl:
                                restore_ablation_mode(saved_abl, pipeline)
                            if hasattr(pipeline, '_patched') and pipeline._patched:
                                pipeline.uninstall()

        if "w2" in args.workloads:
            print(f"\n{'='*60}")
            print(f"  W2: Retrieval QA ({args.w2_task})")
            print(f"{'='*60}")

            from dataset_w2 import load_longbench_dataset, evaluate_w2

            try:
                w2_samples = load_longbench_dataset(
                    task=args.w2_task, max_samples=args.w2_max_samples)

                for n_dist in args.w2_distractors:
                    for pos in args.w2_positions:
                        print(f"\n  distractors={n_dist}  position={pos}")

                        # Dense
                        if "dense" in args.modes:
                            w2r = evaluate_w2(
                                model, tokenizer, None, w2_samples,
                                device, mode="dense",
                                n_distractors=n_dist,
                                answer_position=pos)
                            w2r["model"] = model_tag
                            w2r["workload"] = "w2"
                            all_results.append(w2r)
                            print(f"    Dense:  EM={w2r['em']:.3f} "
                                  f"F1={w2r['f1']:.3f}")

                        # SphKV
                        if "sphkv" in args.modes:
                            from spherical_kv_pipeline import SphericalKVPipeline
                            pip_w2 = SphericalKVPipeline(
                                model=model, tokenizer=tokenizer,
                                codebooks=codebooks, device=device,
                                head_dim=head_dim,
                                group_size=_cfg.GROUP_SIZE,
                                sink_tokens=_cfg.SINK_TOKENS,
                                use_fused=(device.type == "cuda"))
                            w2r = evaluate_w2(
                                model, tokenizer, pip_w2, w2_samples,
                                device, mode="sphkv",
                                n_distractors=n_dist,
                                answer_position=pos)
                            w2r["model"] = model_tag
                            w2r["workload"] = "w2"
                            all_results.append(w2r)
                            print(f"    SphKV:  EM={w2r['em']:.3f} "
                                  f"F1={w2r['f1']:.3f}  "
                                  f"seg_ret={w2r.get('seg_retention',{})}")

            except Exception as e:
                print(f"  W2 failed: {e}")
                import traceback; traceback.print_exc()

        # ── W3: Agentic rollouts  ─────────────────────────────────────
        if "w3" in args.workloads:
            print(f"\n{'='*60}")
            print(f"  W3: Agentic rollouts ({args.w3_source})")
            print(f"{'='*60}")

            from dataset_w3 import (
                load_toolbench_dataset, load_agentbench_dataset,
                evaluate_w3, compute_w3_length_drift)

            try:
                if args.w3_source == "toolbench":
                    w3_episodes = load_toolbench_dataset(
                        max_samples=args.w3_max_episodes)
                else:
                    w3_episodes = load_agentbench_dataset(
                        max_samples=args.w3_max_episodes)

                # Dense
                w3_dense = None
                if "dense" in args.modes:
                    w3_dense = evaluate_w3(
                        model, tokenizer, None, w3_episodes,
                        device, mode="dense",
                        max_steps=args.w3_max_steps,
                        n_seeds=args.w3_seeds)
                    w3_dense["model"] = model_tag
                    w3_dense["workload"] = "w3"
                    all_results.append(w3_dense)
                    print(f"  Dense:  success={w3_dense['success_rate']:.3f}  "
                          f"S_traj={w3_dense['S_traj']:.4f}  "
                          f"disagree={w3_dense['disagree_rate']:.3f}")

                # SphKV
                if "sphkv" in args.modes:
                    from spherical_kv_pipeline import SphericalKVPipeline
                    pip_w3 = SphericalKVPipeline(
                        model=model, tokenizer=tokenizer,
                        codebooks=codebooks, device=device,
                        head_dim=head_dim,
                        group_size=_cfg.GROUP_SIZE,
                        sink_tokens=_cfg.SINK_TOKENS,
                        use_fused=(device.type == "cuda"))
                    w3_sphkv = evaluate_w3(
                        model, tokenizer, pip_w3, w3_episodes,
                        device, mode="sphkv",
                        max_steps=args.w3_max_steps,
                        n_seeds=args.w3_seeds)
                    w3_sphkv["model"] = model_tag
                    w3_sphkv["workload"] = "w3"
                    all_results.append(w3_sphkv)
                    print(f"  SphKV:  success={w3_sphkv['success_rate']:.3f}  "
                          f"S_traj={w3_sphkv['S_traj']:.4f}  "
                          f"disagree={w3_sphkv['disagree_rate']:.3f}")

                    # Length drift
                    if w3_dense is not None:
                        drift = compute_w3_length_drift(w3_dense, w3_sphkv)
                        print(f"  DeltaT: {drift['DeltaT']:.1f} tokens")
                        all_results.append({
                            "type": "w3_drift", "model": model_tag,
                            **drift})

            except Exception as e:
                print(f"  W3 failed: {e}")
                import traceback; traceback.print_exc()

        # ── Analysis: TTFT, head stats, segments, drift ───────────────
        from analysis import (measure_ttft, compute_head_allocation_stats,
                              compute_segment_profiles, compute_failure_rates,
                              compute_drift_auroc, plot_head_allocation,
                              plot_segment_profiles, plot_failure_rates)

        model_dir = out_dir / model_tag
        model_dir.mkdir(parents=True, exist_ok=True)

        # TTFT
        if "w1" in args.workloads:
            print(f"\n  TTFT measurement ...")
            for T in args.context_lengths[:1]:  # first context length
                max_T = T
                needed = max_T + 64
                eval_ids_1d = _load_dataset_tokens(tokenizer, args.dataset, needed)
                pfill = eval_ids_1d[:T].unsqueeze(0).to(device)

                ttft_d = measure_ttft(model, None, pfill, device,
                                      n_trials=args.n_trials, mode="dense")
                print(f"    Dense TTFT: {ttft_d['ttft_ms_median']:.1f} ms")
                all_results.append({"type": "ttft", "mode": "dense",
                                    "model": model_tag, "T": T, **ttft_d})

                if "sphkv" in args.modes:
                    from spherical_kv_pipeline import SphericalKVPipeline
                    pip_t = SphericalKVPipeline(
                        model=model, tokenizer=tokenizer,
                        codebooks=codebooks, device=device,
                        head_dim=head_dim,
                        group_size=_cfg.GROUP_SIZE,
                        sink_tokens=_cfg.SINK_TOKENS, use_fused=(device.type == "cuda"))
                    ttft_s = measure_ttft(model, pip_t, pfill, device,
                                          n_trials=args.n_trials, mode="sphkv")
                    print(f"    SphKV TTFT: {ttft_s['ttft_ms_median']:.1f} ms")
                    all_results.append({"type": "ttft", "mode": "sphkv",
                                        "model": model_tag, "T": T, **ttft_s})

        # Head allocation (A3) + Segment profiles (A4)
        sphkv_w1 = [r for r in all_results
                    if r.get("mode") == "sphkv" and r.get("workload") == "w1"]
        if sphkv_w1:
            T_last = args.context_lengths[0]
            needed = T_last + 64
            eval_ids_1d = _load_dataset_tokens(tokenizer, args.dataset, needed)
            pfill = eval_ids_1d[:T_last].unsqueeze(0).to(device)
            from spherical_kv_pipeline import SphericalKVPipeline
            pip_a = SphericalKVPipeline(
                model=model, tokenizer=tokenizer,
                codebooks=codebooks, device=device,
                head_dim=head_dim,
                group_size=_cfg.GROUP_SIZE,
                sink_tokens=_cfg.SINK_TOKENS, use_fused=(device.type == "cuda"))
            pip_a.prefill(pfill)

            # A3
            hs = compute_head_allocation_stats(pip_a)
            print(f"  Head Gini: {hs['gini']:.4f}")
            all_results.append({"type": "head_alloc", "model": model_tag,
                                **{k: v for k, v in hs.items()
                                   if not isinstance(v, dict)}})
            plot_head_allocation(hs, model_dir)

            # A4
            sp = compute_segment_profiles(pip_a)
            for sn, sv in sp.items():
                print(f"  Seg {sn}: rho={sv['rho']:.3f}  "
                      f"b_bar={sv['b_bar_bytes']:.3f}")
            all_results.append({"type": "segments", "model": model_tag,
                                "profiles": sp})
            plot_segment_profiles(sp, model_dir)

            # A5 drift AUROC
            pip_a.prefill(pfill)
            dr = compute_drift_auroc(pip_a, model, pfill,
                                      n_decode_steps=min(16, args.n_meas),
                                      device=device)
            print(f"  Drift AUROC: {dr['auroc']:.3f}")
            all_results.append({"type": "drift", "model": model_tag, **dr})

            pip_a.uninstall()

        # Full cost accounting + decode breakdown (Section 3.5 / 4.3)
        if sphkv_w1:
            from hardware_audit import measure_full_cost, measure_decode_breakdown
            T_fc = args.context_lengths[0]
            needed_fc = T_fc + 64
            eval_ids_fc = _load_dataset_tokens(tokenizer, args.dataset, needed_fc)
            pfill_fc = eval_ids_fc[:T_fc].unsqueeze(0).to(device)

            from spherical_kv_pipeline import SphericalKVPipeline
            pip_fc = SphericalKVPipeline(
                model=model, tokenizer=tokenizer,
                codebooks=codebooks, device=device,
                head_dim=head_dim, group_size=_cfg.GROUP_SIZE,
                sink_tokens=_cfg.SINK_TOKENS, use_fused=(device.type == "cuda"))

            # Full cost
            fc = measure_full_cost(model, pip_fc, pfill_fc,
                                    n_meas=args.n_meas, device=device)
            print(f"  Full cost: prefill={fc['t_prefill_ms']:.0f}ms "
                  f"decode={fc['t_decode_ms']:.0f}ms "
                  f"({fc['prefill_pct']:.1f}% prefill)")
            all_results.append({"type": "full_cost", "model": model_tag, **fc})

            # Decode breakdown
            pip_fc2 = SphericalKVPipeline(
                model=model, tokenizer=tokenizer,
                codebooks=codebooks, device=device,
                head_dim=head_dim, group_size=_cfg.GROUP_SIZE,
                sink_tokens=_cfg.SINK_TOKENS, use_fused=(device.type == "cuda"))
            bd = measure_decode_breakdown(model, pip_fc2, pfill_fc,
                                          n_steps=min(16, args.n_meas), device=device)
            print(f"  Decode breakdown: {bd}")
            all_results.append({"type": "decode_breakdown", "model": model_tag, **bd})

        # Failure rates
        measurable = [r for r in all_results if "mode" in r and "nll" in r]
        if measurable:
            fr = compute_failure_rates(measurable)
            if fr:
                all_results.append({"type": "failure", "model": model_tag,
                                    **{k: v for k, v in fr.items()
                                       if not isinstance(v, dict)}})
                plot_failure_rates(fr, model_dir)

        # Stability (multi-seed)
        if args.n_seeds > 1 or args.n_trials > 1:
            dense_r = [r for r in all_results
                       if r.get("mode") == "dense" and "nll" in r]
            sphkv_r = [r for r in all_results
                       if r.get("mode") == "sphkv" and "nll" in r]
            if dense_r and sphkv_r:
                stab = compute_stability_metrics(dense_r[0], sphkv_r[0])
                print(f"  S_traj={stab['S_traj']:.6f}  "
                      f"DeltaT={stab['DeltaT']:.1f}")
                all_results.append({"type": "stability",
                                    "model": model_tag, **stab})

        # Frontier
        if measurable:
            frontier = build_iso_quality_frontier(measurable, args.delta)
            print(f"  Frontier: speedup={frontier.get('iso_quality_speedup',1):.2f}x  "
                  f"mem_red={frontier.get('iso_throughput_mem_reduction',1):.1f}x")
            plot_frontier(frontier, model_dir)
            plot_budget_sweep(measurable, model_dir)

        grand_results.extend(all_results)

        # Free model memory before loading next
        del model, tokenizer, codebooks
        torch.cuda.empty_cache()

    serializable = []
    for r in grand_results:
        sr = {}
        for k, v in r.items():
            if isinstance(v, (int, float, str, bool, type(None))):
                sr[k] = v
            elif isinstance(v, list):
                sr[k] = [x if isinstance(x, (int, float, str)) else str(x)
                         for x in v]
            elif isinstance(v, dict):
                sr[k] = {str(kk): vv for kk, vv in v.items()
                         if isinstance(vv, (int, float, str))}
            else:
                sr[k] = str(v)
        serializable.append(sr)

    results_path = out_dir / "all_experiment_results.json"
    with open(results_path, "w") as f:
        json.dump({"args": vars(args), "results": serializable},
                  f, indent=2, default=str)
    print(f"\nSaved: {results_path}")

    # ── Generate ALL paper figures ────────────────────────────────────
    from paper_plots import generate_all_paper_figures
    model_tags = [m.split("/")[-1] for m in args.models]
    generate_all_paper_figures(
        grand_results, model_tags, args.context_lengths,
        str(out_dir), args.delta)

    # ── ncu command guidance ──────────────────────────────────────────
    from hardware_audit import generate_ncu_command
    print(f"\n  For hardware DRAM counters, run:")
    print(f"    {generate_ncu_command('experiment_runner.py', '--modes sphkv --budgets 30')}")

    # ── Summary table ─────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  Summary (Table 5 format)")
    print(f"{'='*80}")
    print(f"{'Model':<25} {'Mode':<16} {'WL':>3} {'Budget':>7} {'NLL':>8} "
          f"{'PPL':>8} {'tok/s':>7} {'bKV':>8}")
    print("-" * 88)

    for r in sorted(grand_results,
                    key=lambda x: (x.get("model",""), x.get("mode",""),
                                   x.get("budget_bpt",0))):
        if "nll" not in r or "mode" not in r:
            continue
        bpt_s = f"{r.get('budget_bpt',0):>7.0f}" if "budget_bpt" in r else "    ---"
        ppl_val = r.get("ppl", math.exp(r["nll"]))
        print(f"{r.get('model',''):25s} {r['mode']:<16} "
              f"{r.get('workload',''):>3} {bpt_s} "
              f"{r['nll']:>8.4f} {ppl_val:>8.2f} "
              f"{r['tok_s']:>7.1f} {r.get('bKV',0):>8.0f}")

    print(f"\nAll outputs -> {out_dir}")


if __name__ == "__main__":
    main()
