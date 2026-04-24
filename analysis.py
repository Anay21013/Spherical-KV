import math
import statistics
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np


@torch.no_grad()
def measure_ttft(
    model,
    pipeline,       # SphericalKVPipeline or None for dense
    input_ids:      torch.Tensor,   # [1, T]
    device:         torch.device,
    n_trials:       int = 5,
    mode:           str = "dense",
) -> dict:
    """
    Measure prefill latency = time to first generated token.
    Includes controller allocation time for SphericalKV.
    """
    is_cuda = device.type == "cuda"
    times = []

    for _ in range(n_trials):
        if is_cuda:
            torch.cuda.synchronize()
            ts = torch.cuda.Event(enable_timing=True)
            te = torch.cuda.Event(enable_timing=True)
            ts.record()

        if mode == "dense" or pipeline is None:
            # Dense: just the forward pass
            out = model(input_ids=input_ids.to(device),
                        use_cache=False, return_dict=True)
            _ = out.logits[:, -1, :].argmax(-1)
        else:
            # SphericalKV: prefill includes controller + page building
            pipeline.prefill(input_ids.to(device))
            # Generate first token
            out = model(input_ids=input_ids[:, -1:].to(device),
                        use_cache=False, return_dict=True)
            _ = out.logits[:, -1, :].argmax(-1)

        if is_cuda:
            te.record()
            torch.cuda.synchronize()
            times.append(ts.elapsed_time(te))  # ms
        else:
            times.append(0.0)

    if pipeline is not None and hasattr(pipeline, 'uninstall') and pipeline._patched:
        pipeline.uninstall()

    return {
        "ttft_ms_median": statistics.median(times) if times else 0.0,
        "ttft_ms_p95":    sorted(times)[int(len(times) * 0.95)] if len(times) > 1 else (times[0] if times else 0.0),
        "ttft_ms_all":    times,
    }


def compute_head_allocation_stats(pipeline) -> dict:
    """
    Compute per-head average bytes/token and Gini coefficient.
    Paper A3: tests whether gains rely on head-adaptive rate allocation.

    a_h = (1/T) * sum_i bytes(z_{i,l,h}, t_{i,l,h})
    Gini measures concentration of allocation across heads.
    """
    from tiers import build_tiers
    tiers = build_tiers(pipeline.head_dim)
    tier_by_id = {t.tier_id: t for t in tiers}

    # Count tokens per (layer, head) and their tier assignments
    head_bits = defaultdict(float)   # (layer, head) -> total bits
    head_count = defaultdict(int)    # (layer, head) -> token count

    for ts in pipeline._retained_tokens:
        tier = tier_by_id.get(ts.new_tier_id)
        bits = tier.token_bits() if tier else 0
        head_bits[(ts.layer, ts.head)] += bits
        head_count[(ts.layer, ts.head)] += 1

    # Compute a_h = total_bytes / T per head
    T = pipeline.seq_len
    per_head_bytes = {}
    for (layer, head), total_bits in head_bits.items():
        per_head_bytes[(layer, head)] = (total_bits / 8.0) / max(T, 1)

    # Flatten for Gini
    values = sorted(per_head_bytes.values())
    n = len(values)

    if n == 0:
        return {"gini": 0.0, "per_head_bytes": {}, "entropy": 0.0}

    # Gini coefficient
    total = sum(values)
    if total == 0:
        gini = 0.0
    else:
        cum = 0.0
        for i, v in enumerate(values):
            cum += (2 * (i + 1) - n - 1) * v
        gini = cum / (n * total)

    # Entropy of normalized allocation
    probs = [v / total for v in values if v > 0] if total > 0 else []
    entropy = -sum(p * math.log(p) for p in probs) if probs else 0.0

    # Per-layer stats
    layer_means = defaultdict(list)
    for (layer, head), bpt in per_head_bytes.items():
        layer_means[layer].append(bpt)

    return {
        "gini":           gini,
        "entropy":        entropy,
        "mean_bytes_per_tok_per_head": total / (8.0 * n) if n > 0 else 0.0,
        "std_bytes":      statistics.stdev(values) if len(values) > 1 else 0.0,
        "per_head_bytes": {f"L{l}_H{h}": v for (l, h), v in per_head_bytes.items()},
        "n_heads":        n,
        "per_layer_mean": {l: statistics.mean(vs) for l, vs in layer_means.items()},
    }



def compute_segment_profiles(pipeline) -> dict:
    from tiers import build_tiers
    tiers = build_tiers(pipeline.head_dim)
    tier_by_id = {t.tier_id: t for t in tiers}

    SEGMENT_NAMES = {0: "prefix", 1: "retrieved", 2: "recent"}

    # Count total tokens per segment (before retention)
    total_per_seg = defaultdict(int)
    retained_per_seg = defaultdict(int)
    bits_per_seg = defaultdict(float)
    tier_counts_per_seg = defaultdict(lambda: defaultdict(int))

    # We need all tokens (retained + dropped).  The pipeline only keeps
    # retained tokens.  For total counts, use T * L * H and segment boundaries.
    T = pipeline.seq_len
    L = pipeline.num_layers
    H = pipeline.num_kv_heads
    from config import SINK_TOKENS, RECENT_WINDOW

    # Estimate total per segment (all L*H heads see the same T tokens)
    n_recent = min(RECENT_WINDOW, T)
    n_prefix = T - n_recent
    total_per_seg[0] = n_prefix * L * H   # prefix
    total_per_seg[2] = n_recent * L * H   # recent
    # segment 1 (retrieved) only if explicitly set -- default is 0
    total_per_seg[1] = 0

    # Count retained tokens per segment
    for ts in pipeline._retained_tokens:
        seg = ts.segment_id
        retained_per_seg[seg] += 1
        tier = tier_by_id.get(ts.new_tier_id)
        bits = tier.token_bits() if tier else 0
        bits_per_seg[seg] += bits
        tier_counts_per_seg[seg][ts.new_tier_id] += 1

    profiles = {}
    for seg_id, seg_name in SEGMENT_NAMES.items():
        total = total_per_seg[seg_id]
        retained = retained_per_seg.get(seg_id, 0)

        if total > 0:
            rho = retained / total
            b_bar = (bits_per_seg.get(seg_id, 0) / 8.0) / total  # bytes per total token
        else:
            rho = 0.0
            b_bar = 0.0

        profiles[seg_name] = {
            "rho":         rho,
            "b_bar_bytes": b_bar,
            "total":       total,
            "retained":    retained,
            "tier_dist":   dict(tier_counts_per_seg.get(seg_id, {})),
        }

    return profiles



def compute_failure_rates(
    all_results: List[dict],
    nll_threshold_factor: float = 2.0,
) -> dict:
    # Find dense baseline NLL
    dense_pts = [r for r in all_results if r.get("mode") == "dense"]
    if not dense_pts:
        return {}

    nll_dense = min(r["nll"] for r in dense_pts)
    threshold = nll_dense * nll_threshold_factor

    # Group by (mode, budget)
    by_mode_budget = defaultdict(list)
    for r in all_results:
        mode = r.get("mode", "")
        bpt = r.get("budget_bpt", 0)
        if mode and "nll" in r:
            by_mode_budget[(mode, bpt)].append(r)

    failure_rates = {}
    for (mode, bpt), pts in by_mode_budget.items():
        n_total = len(pts)
        # Check all trials within each point
        n_fail = 0
        for r in pts:
            nll_trials = r.get("nll_all", [r.get("nll", 0)])
            for nll_val in nll_trials:
                if nll_val > threshold:
                    n_fail += 1
        n_total_trials = sum(len(r.get("nll_all", [r.get("nll", 0)])) for r in pts)
        rate = n_fail / max(n_total_trials, 1)
        failure_rates[(mode, bpt)] = {
            "failure_rate": rate,
            "n_failures":   n_fail,
            "n_total":      n_total_trials,
            "threshold":    threshold,
        }

    return {
        "nll_dense":    nll_dense,
        "threshold":    threshold,
        "by_mode_budget": {f"{m}@{b}": v for (m, b), v in failure_rates.items()},
    }


@torch.no_grad()
def compute_drift_auroc(
    pipeline,
    model,
    prefill_ids:    torch.Tensor,
    n_decode_steps: int = 16,
    device:         torch.device = None,
) -> dict:
    if device is None:
        device = pipeline.device

    pipeline.prefill(prefill_ids.to(device))

    # Collect per-step drift scores and NLL gaps
    drift_scores = []
    nll_gaps = []

    current_sphkv = prefill_ids.to(device).clone()
    current_dense = prefill_ids.to(device).clone()

    for step in range(n_decode_steps):
        # SphericalKV decode
        out_sphkv = model(input_ids=current_sphkv[:, -1:],
                          use_cache=False, return_dict=True)
        logits_sphkv = out_sphkv.logits[:, -1, :]
        nid_sphkv = logits_sphkv.argmax(-1, keepdim=True)

        # Dense decode (separate forward, no pipeline hooks)
        # We approximate by using the same model without hooks
        pipeline.uninstall()
        out_dense = model(input_ids=current_dense[:, -1:],
                          use_cache=False, return_dict=True)
        logits_dense = out_dense.logits[:, -1, :]
        nid_dense = logits_dense.argmax(-1, keepdim=True)

        # Re-install hooks for next step
        from llama_hooks import patch_for_decode
        pipeline._original_forwards = patch_for_decode(model, pipeline)
        pipeline._patched = True

        # NLL gap for this step
        lp_sphkv = F.log_softmax(logits_sphkv, dim=-1)
        lp_dense = F.log_softmax(logits_dense, dim=-1)
        nll_sphkv = -lp_sphkv[0, nid_dense.item()].item()
        nll_dense = -lp_dense[0, nid_dense.item()].item()
        nll_gap = abs(nll_sphkv - nll_dense)
        nll_gaps.append(nll_gap)

        # Drift score: L2 of logit difference (proxy for bound violation)
        drift = (logits_sphkv - logits_dense).pow(2).mean().sqrt().item()
        drift_scores.append(drift)

        current_sphkv = torch.cat([current_sphkv, nid_sphkv], dim=-1)
        current_dense = torch.cat([current_dense, nid_dense], dim=-1)

    pipeline.uninstall()

    # Compute AUROC: drift_score predicts nll_gap > median
    if len(drift_scores) < 4:
        return {"auroc": 0.5, "n_steps": len(drift_scores)}

    threshold = statistics.median(nll_gaps)
    labels = [1 if g > threshold else 0 for g in nll_gaps]

    # Simple AUROC via sorted pairs
    auroc = _compute_auroc(drift_scores, labels)

    return {
        "auroc":          auroc,
        "n_steps":        len(drift_scores),
        "mean_drift":     statistics.mean(drift_scores),
        "max_drift":      max(drift_scores),
        "mean_nll_gap":   statistics.mean(nll_gaps),
        "failure_threshold": threshold,
    }


def _compute_auroc(scores: List[float], labels: List[int]) -> float:
    """Compute AUROC from score-label pairs (no sklearn dependency)."""
    n = len(scores)
    if n == 0:
        return 0.5

    # Sort by score descending
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])

    n_pos = sum(labels)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    tp = 0
    fp = 0
    auc = 0.0
    prev_score = None

    for score, label in pairs:
        if prev_score is not None and score != prev_score:
            pass  # score changed, nothing special needed for trapezoidal
        if label == 1:
            tp += 1
        else:
            fp += 1
            auc += tp  # each FP contributes tp to the sum

    return auc / (n_pos * n_neg)



def plot_head_allocation(head_stats: dict, output_dir):
    """Plot per-head allocation histogram and mark Gini."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    from pathlib import Path
    out = Path(output_dir)

    per_head = head_stats.get("per_head_bytes", {})
    if not per_head:
        return

    values = sorted(per_head.values())
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(values)), values, color="#4CAF50", alpha=0.7)
    ax.axhline(y=statistics.mean(values), color="red", linestyle="--",
               label=f"mean={statistics.mean(values):.2f}")
    ax.set_xlabel("Head index (sorted by allocation)")
    ax.set_ylabel("Bytes/token")
    ax.set_title(f"Per-head KV allocation (Gini={head_stats['gini']:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "plot_head_allocation.png", dpi=150)
    plt.close(fig)
    print(f"  -> plot_head_allocation.png")


def plot_segment_profiles(profiles: dict, output_dir):
    """Plot segment-wise retention and rate bars."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    from pathlib import Path
    out = Path(output_dir)

    segments = list(profiles.keys())
    rhos = [profiles[s]["rho"] for s in segments]
    b_bars = [profiles[s]["b_bar_bytes"] for s in segments]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.bar(segments, rhos, color=["#2196F3", "#FF9800", "#4CAF50"])
    ax1.set_ylabel("Retention rate (rho_S)")
    ax1.set_title("Segment retention rates (A4)")
    ax1.set_ylim(0, 1.1)

    ax2.bar(segments, b_bars, color=["#2196F3", "#FF9800", "#4CAF50"])
    ax2.set_ylabel("Avg bytes/token (b_bar_S)")
    ax2.set_title("Segment byte rates (A4)")

    fig.tight_layout()
    fig.savefig(out / "plot_segment_profiles.png", dpi=150)
    plt.close(fig)
    print(f"  -> plot_segment_profiles.png")


def plot_failure_rates(failure_data: dict, output_dir):
    """Plot failure rate vs budget for each mode."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    from pathlib import Path
    out = Path(output_dir)

    by_mb = failure_data.get("by_mode_budget", {})
    if not by_mb:
        return

    # Parse mode@budget keys
    mode_data = defaultdict(lambda: ([], []))
    for key, val in by_mb.items():
        parts = key.split("@")
        if len(parts) == 2:
            mode, bpt = parts[0], float(parts[1])
            mode_data[mode][0].append(bpt)
            mode_data[mode][1].append(val["failure_rate"])

    colors = {
        "sphkv": "#FF5722", "sphkv_recon": "#4CAF50",
        "sphkv_angle": "#FF9800", "sphkv_rd": "#9C27B0",
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for mode, (budgets, rates) in mode_data.items():
        order = sorted(range(len(budgets)), key=lambda i: budgets[i])
        ax.plot([budgets[i] for i in order],
                [rates[i] for i in order],
                '-o', color=colors.get(mode, "#666"), label=mode)

    ax.set_xlabel("Budget (bits/token)")
    ax.set_ylabel("Failure rate")
    ax.set_title("Failure rate vs budget")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "plot_failure_rates.png", dpi=150)
    plt.close(fig)
    print(f"  -> plot_failure_rates.png")